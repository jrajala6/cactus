#include "test_utils.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kGroupSize = 32;
constexpr size_t kBlockSize = 4;
constexpr size_t kDefaultLayers = 18;

enum class PrecisionMode {
    INT8,
    INT4
};

const char* mode_name(PrecisionMode mode) {
    return mode == PrecisionMode::INT8 ? "INT8" : "INT4";
}

struct ProjectionSpec {
    const char* name;
    size_t K;
    size_t N;
};

constexpr std::array<ProjectionSpec, 7> kProjectionSpecs = {{
    {"attn_q",   640, 1024},
    {"attn_k",   640,  256},
    {"attn_v",   640,  256},
    {"attn_o",  1024,  640},
    {"ffn_gate", 640, 2048},
    {"ffn_up",   640, 2048},
    {"ffn_down",2048,  640},
}};

struct ProjectionWeights {
    size_t K = 0;
    size_t N = 0;
    std::vector<int8_t> int8_weights_interleaved;
    std::vector<__fp16> int8_scales_interleaved;
    std::vector<uint8_t> int4_weights_packed;
    std::vector<__fp16> int4_scales_interleaved;
};

using LayerWeights = std::array<ProjectionWeights, kProjectionSpecs.size()>;

struct ProjectionTiming {
    double node_ms = 0.0;    // quantize + matmul
    double quant_ms = 0.0;   // activation quantization only
    double kernel_ms = 0.0;  // cactus_matmul_int{8,4} only
    size_t calls = 0;
};

struct ScenarioResult {
    std::array<ProjectionTiming, kProjectionSpecs.size()> projection_stats{};
    double step_ms_total = 0.0;
    double checksum = 0.0;
};

struct BenchOptions {
    int warmup = 8;
    int iterations = 30;
    size_t layers = kDefaultLayers;
    std::vector<size_t> batch_sizes = {1, 13, 34};
    bool run_int8 = true;
    bool run_int4 = true;
};

double now_ms() {
    using clock = std::chrono::steady_clock;
    auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

float clamp_f32(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}

std::vector<int8_t> interleave_weights_nk4(const std::vector<int8_t>& rowmajor, size_t N, size_t K) {
    if (N % kBlockSize != 0 || K % kBlockSize != 0) {
        throw std::runtime_error("Interleave requires N and K divisible by 4");
    }
    const size_t N_blocks = N / kBlockSize;
    const size_t K_blocks = K / kBlockSize;

    std::vector<int8_t> interleaved(N * K);
    for (size_t n_blk = 0; n_blk < N_blocks; ++n_blk) {
        for (size_t k_blk = 0; k_blk < K_blocks; ++k_blk) {
            for (size_t ni = 0; ni < kBlockSize; ++ni) {
                for (size_t ki = 0; ki < kBlockSize; ++ki) {
                    const size_t src_n = n_blk * kBlockSize + ni;
                    const size_t src_k = k_blk * kBlockSize + ki;
                    const size_t dst_idx =
                        (n_blk * K_blocks + k_blk) * (kBlockSize * kBlockSize) + ni * kBlockSize + ki;
                    interleaved[dst_idx] = rowmajor[src_n * K + src_k];
                }
            }
        }
    }
    return interleaved;
}

std::vector<__fp16> interleave_scales_n4(const std::vector<float>& rowmajor_scales, size_t N, size_t num_groups) {
    if (N % kBlockSize != 0) {
        throw std::runtime_error("Scale interleave requires N divisible by 4");
    }
    const size_t N_blocks = N / kBlockSize;
    std::vector<__fp16> interleaved(N * num_groups);

    for (size_t n_blk = 0; n_blk < N_blocks; ++n_blk) {
        for (size_t g = 0; g < num_groups; ++g) {
            for (size_t ni = 0; ni < kBlockSize; ++ni) {
                const size_t src_n = n_blk * kBlockSize + ni;
                const size_t dst_idx = (n_blk * num_groups + g) * kBlockSize + ni;
                interleaved[dst_idx] = static_cast<__fp16>(rowmajor_scales[src_n * num_groups + g]);
            }
        }
    }
    return interleaved;
}

std::vector<uint8_t> pack_int4_pairs(const std::vector<int8_t>& interleaved_int4) {
    if (interleaved_int4.size() % 32 != 0) {
        throw std::runtime_error("INT4 packing requires size divisible by 32");
    }
    std::vector<uint8_t> packed(interleaved_int4.size() / 2);
    for (size_t i = 0; i < interleaved_int4.size(); i += 32) {
        for (size_t j = 0; j < 16; ++j) {
            const uint8_t lo = static_cast<uint8_t>(interleaved_int4[i + j]) & 0x0F;
            const uint8_t hi = (static_cast<uint8_t>(interleaved_int4[i + 16 + j]) & 0x0F) << 4;
            packed[i / 2 + j] = lo | hi;
        }
    }
    return packed;
}

void quantize_weight_int8_grouped(const std::vector<float>& src, size_t N, size_t K,
                                  std::vector<int8_t>& dst, std::vector<float>& scales) {
    if (K % kGroupSize != 0) {
        throw std::runtime_error("INT8 grouped quantization requires K divisible by group_size");
    }
    const size_t num_groups = K / kGroupSize;
    dst.resize(N * K);
    scales.resize(N * num_groups);

    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            float max_abs = 0.0f;
            const size_t group_start = g * kGroupSize;
            for (size_t k = 0; k < kGroupSize; ++k) {
                const float v = std::abs(src[n * K + group_start + k]);
                if (v > max_abs) max_abs = v;
            }
            float scale = max_abs / 127.0f;
            if (scale < 1e-10f) scale = 1e-10f;
            scales[n * num_groups + g] = scale;

            for (size_t k = 0; k < kGroupSize; ++k) {
                float qf = src[n * K + group_start + k] / scale;
                int32_t q = static_cast<int32_t>(std::round(qf));
                q = std::max(-128, std::min(127, q));
                dst[n * K + group_start + k] = static_cast<int8_t>(q);
            }
        }
    }
}

void quantize_weight_int4_grouped(const std::vector<float>& src, size_t N, size_t K,
                                  std::vector<int8_t>& dst, std::vector<float>& scales) {
    if (K % kGroupSize != 0) {
        throw std::runtime_error("INT4 grouped quantization requires K divisible by group_size");
    }
    const size_t num_groups = K / kGroupSize;
    dst.resize(N * K);
    scales.resize(N * num_groups);

    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            float max_abs = 0.0f;
            const size_t group_start = g * kGroupSize;
            for (size_t k = 0; k < kGroupSize; ++k) {
                const float v = std::abs(src[n * K + group_start + k]);
                if (v > max_abs) max_abs = v;
            }
            float scale = max_abs / 7.0f;
            if (scale < 1e-10f) scale = 1e-10f;
            scales[n * num_groups + g] = scale;

            for (size_t k = 0; k < kGroupSize; ++k) {
                float qf = src[n * K + group_start + k] / scale;
                int32_t q = static_cast<int32_t>(std::round(qf));
                q = std::max(-8, std::min(7, q));
                dst[n * K + group_start + k] = static_cast<int8_t>(q);
            }
        }
    }
}

ProjectionWeights build_projection_weights(const ProjectionSpec& spec, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    ProjectionWeights out;
    out.K = spec.K;
    out.N = spec.N;

    std::vector<float> w_fp32(spec.N * spec.K);
    for (float& v : w_fp32) {
        v = dist(gen);
    }

    std::vector<int8_t> w_int8_rowmajor;
    std::vector<float> s_int8_rowmajor;
    quantize_weight_int8_grouped(w_fp32, spec.N, spec.K, w_int8_rowmajor, s_int8_rowmajor);
    out.int8_weights_interleaved = interleave_weights_nk4(w_int8_rowmajor, spec.N, spec.K);
    out.int8_scales_interleaved = interleave_scales_n4(s_int8_rowmajor, spec.N, spec.K / kGroupSize);

    std::vector<int8_t> w_int4_rowmajor;
    std::vector<float> s_int4_rowmajor;
    quantize_weight_int4_grouped(w_fp32, spec.N, spec.K, w_int4_rowmajor, s_int4_rowmajor);
    const auto w_int4_interleaved = interleave_weights_nk4(w_int4_rowmajor, spec.N, spec.K);
    out.int4_weights_packed = pack_int4_pairs(w_int4_interleaved);
    out.int4_scales_interleaved = interleave_scales_n4(s_int4_rowmajor, spec.N, spec.K / kGroupSize);

    return out;
}

std::vector<LayerWeights> build_model_weights(size_t layers, uint32_t seed) {
    std::mt19937 gen(seed);
    std::vector<LayerWeights> model_weights(layers);
    for (size_t l = 0; l < layers; ++l) {
        for (size_t p = 0; p < kProjectionSpecs.size(); ++p) {
            model_weights[l][p] = build_projection_weights(kProjectionSpecs[p], gen);
        }
    }
    return model_weights;
}

void quantize_activation_rows(const __fp16* src, size_t M, size_t K,
                              std::vector<int8_t>& q, std::vector<float>& scales) {
    q.resize(M * K);
    scales.resize(M);
    for (size_t m = 0; m < M; ++m) {
        float max_abs = cactus_fp16_max_abs(src + m * K, K);
        float scale = max_abs / 127.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        scales[m] = scale;
        cactus_fp16_to_int8(src + m * K, q.data() + m * K, K, scale);
    }
}

void run_projection(const ProjectionWeights& w, PrecisionMode mode, size_t M,
                    const __fp16* input_fp16, __fp16* output_fp16,
                    std::vector<int8_t>& a_quant, std::vector<float>& a_scales,
                    ProjectionTiming& stats, bool measure) {
    if (!measure) {
        quantize_activation_rows(input_fp16, M, w.K, a_quant, a_scales);
        if (mode == PrecisionMode::INT8) {
            cactus_matmul_int8(a_quant.data(), a_scales.data(),
                               w.int8_weights_interleaved.data(),
                               w.int8_scales_interleaved.data(),
                               output_fp16, M, w.K, w.N, kGroupSize);
        } else {
            cactus_matmul_int4(a_quant.data(), a_scales.data(),
                               reinterpret_cast<const int8_t*>(w.int4_weights_packed.data()),
                               w.int4_scales_interleaved.data(),
                               output_fp16, M, w.K, w.N, kGroupSize);
        }
        return;
    }

    const double t0 = now_ms();
    quantize_activation_rows(input_fp16, M, w.K, a_quant, a_scales);
    const double t1 = now_ms();

    if (mode == PrecisionMode::INT8) {
        cactus_matmul_int8(a_quant.data(), a_scales.data(),
                           w.int8_weights_interleaved.data(),
                           w.int8_scales_interleaved.data(),
                           output_fp16, M, w.K, w.N, kGroupSize);
    } else {
        cactus_matmul_int4(a_quant.data(), a_scales.data(),
                           reinterpret_cast<const int8_t*>(w.int4_weights_packed.data()),
                           w.int4_scales_interleaved.data(),
                           output_fp16, M, w.K, w.N, kGroupSize);
    }
    const double t2 = now_ms();

    stats.quant_ms += (t1 - t0);
    stats.kernel_ms += (t2 - t1);
    stats.node_ms += (t2 - t0);
    stats.calls++;
}

ScenarioResult run_scenario(const std::vector<LayerWeights>& model_weights, PrecisionMode mode,
                            size_t M, int warmup, int iterations) {
    ScenarioResult out;
    const int total_iters = warmup + iterations;

    std::mt19937 gen(static_cast<uint32_t>(9001 + M + (mode == PrecisionMode::INT4 ? 100 : 0)));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<__fp16> hidden(M * 640);
    std::vector<__fp16> hidden_next(M * 640);
    std::vector<__fp16> q(M * 1024), k(M * 256), v(M * 256);
    std::vector<__fp16> attn(M * 1024), o(M * 640);
    std::vector<__fp16> gate(M * 2048), up(M * 2048), ffn(M * 2048), down(M * 640);

    for (__fp16& x : hidden) x = static_cast<__fp16>(dist(gen));

    std::vector<int8_t> a_quant;
    std::vector<float> a_scales;

    for (int iter = 0; iter < total_iters; ++iter) {
        const bool measure = iter >= warmup;
        const double step_t0 = measure ? now_ms() : 0.0;

        for (size_t l = 0; l < model_weights.size(); ++l) {
            const auto& w = model_weights[l];
            run_projection(w[0], mode, M, hidden.data(), q.data(), a_quant, a_scales, out.projection_stats[0], measure);
            run_projection(w[1], mode, M, hidden.data(), k.data(), a_quant, a_scales, out.projection_stats[1], measure);
            run_projection(w[2], mode, M, hidden.data(), v.data(), a_quant, a_scales, out.projection_stats[2], measure);

            const size_t q_count = attn.size();
            const size_t kv_count = k.size();
            for (size_t i = 0; i < q_count; ++i) {
                const float qv = static_cast<float>(q[i]);
                const float kv = static_cast<float>(k[i % kv_count]);
                const float vv = static_cast<float>(v[(i * 7) % kv_count]);
                const float mixed = qv * 0.25f + kv * 0.03125f - vv * 0.015625f;
                attn[i] = static_cast<__fp16>(clamp_f32(mixed, -8.0f, 8.0f));
            }

            run_projection(w[3], mode, M, attn.data(), o.data(), a_quant, a_scales, out.projection_stats[3], measure);

            for (size_t i = 0; i < hidden_next.size(); ++i) {
                const float mixed =
                    static_cast<float>(o[i]) * 0.2f + static_cast<float>(hidden[i]) * 0.8f;
                hidden_next[i] = static_cast<__fp16>(clamp_f32(mixed, -8.0f, 8.0f));
            }

            run_projection(w[4], mode, M, hidden_next.data(), gate.data(), a_quant, a_scales, out.projection_stats[4], measure);
            run_projection(w[5], mode, M, hidden_next.data(), up.data(), a_quant, a_scales, out.projection_stats[5], measure);

            for (size_t i = 0; i < ffn.size(); ++i) {
                const float g = std::tanh(static_cast<float>(gate[i]) * 0.25f);
                const float u = static_cast<float>(up[i]) * 0.25f;
                ffn[i] = static_cast<__fp16>(clamp_f32(g * u, -8.0f, 8.0f));
            }

            run_projection(w[6], mode, M, ffn.data(), down.data(), a_quant, a_scales, out.projection_stats[6], measure);

            for (size_t i = 0; i < hidden.size(); ++i) {
                const float mixed =
                    static_cast<float>(down[i]) * 0.2f + static_cast<float>(hidden_next[i]) * 0.8f;
                hidden[i] = static_cast<__fp16>(clamp_f32(mixed, -8.0f, 8.0f));
            }
        }

        if (measure) {
            out.step_ms_total += (now_ms() - step_t0);
            out.checksum += static_cast<double>(hidden[(iter + 17) % hidden.size()]);
        }
    }

    return out;
}

double gops(size_t M, size_t K, size_t N, size_t calls, double kernel_ms) {
    if (kernel_ms <= 0.0) return 0.0;
    const double ops = static_cast<double>(2ULL * M * K * N) * static_cast<double>(calls);
    return ops / (kernel_ms * 1e6);
}

void log_projection_tables(TestUtils::TestRunner& runner, const ScenarioResult& r,
                           PrecisionMode mode, size_t M, int iterations, size_t layers) {
    const double step_ms = r.step_ms_total / static_cast<double>(iterations);
    const double rows_per_sec = (static_cast<double>(iterations) * static_cast<double>(M)) / (r.step_ms_total / 1000.0);

    {
        std::ostringstream details;
        details << std::fixed << std::setprecision(3)
                << "layers=" << layers
                << ", step=" << step_ms << "ms"
                << ", rows/s=" << std::setprecision(2) << rows_per_sec
                << ", checksum=" << std::setprecision(6) << r.checksum;
        runner.log_performance(
            std::string("Gemma270m stack ") + mode_name(mode) + " M=" + std::to_string(M),
            details.str());
    }

    for (size_t i = 0; i < kProjectionSpecs.size(); ++i) {
        const auto& s = kProjectionSpecs[i];
        const auto& st = r.projection_stats[i];
        if (st.calls == 0) continue;

        const double avg_node_us = (st.node_ms * 1000.0) / static_cast<double>(st.calls);
        const double avg_quant_us = (st.quant_ms * 1000.0) / static_cast<double>(st.calls);
        const double avg_kernel_us = (st.kernel_ms * 1000.0) / static_cast<double>(st.calls);
        const double proj_gops = gops(M, s.K, s.N, st.calls, st.kernel_ms);

        std::ostringstream details;
        details << std::fixed << std::setprecision(2)
                << "avg(node/quant/kernel)="
                << avg_node_us << "/" << avg_quant_us << "/" << avg_kernel_us << "us"
                << ", calls=" << st.calls
                << ", kernel_gops=" << std::setprecision(2) << proj_gops;
        runner.log_performance(
            std::string("  ") + mode_name(mode) + " " + s.name + " MxKxN="
                + std::to_string(M) + "x" + std::to_string(s.K) + "x" + std::to_string(s.N),
            details.str());
    }

    struct ShapeAgg {
        double node_ms = 0.0;
        double kernel_ms = 0.0;
        size_t calls = 0;
    };

    std::array<ShapeAgg, 4> shape_aggs{};
    const std::array<size_t, 4> shape_N = {1024, 256, 640, 2048};
    for (size_t i = 0; i < kProjectionSpecs.size(); ++i) {
        const auto& s = kProjectionSpecs[i];
        for (size_t si = 0; si < shape_N.size(); ++si) {
            if (s.N == shape_N[si]) {
                shape_aggs[si].node_ms += r.projection_stats[i].node_ms;
                shape_aggs[si].kernel_ms += r.projection_stats[i].kernel_ms;
                shape_aggs[si].calls += r.projection_stats[i].calls;
            }
        }
    }

    for (size_t si = 0; si < shape_N.size(); ++si) {
        const auto& agg = shape_aggs[si];
        if (agg.calls == 0) continue;
        const double avg_node_us = (agg.node_ms * 1000.0) / static_cast<double>(agg.calls);
        const double avg_kernel_us = (agg.kernel_ms * 1000.0) / static_cast<double>(agg.calls);
        std::ostringstream details;
        details << std::fixed << std::setprecision(2)
                << "avg(node/kernel)=" << avg_node_us << "/" << avg_kernel_us << "us"
                << ", calls=" << agg.calls;
        runner.log_performance(
            std::string("  ") + mode_name(mode) + " shape MxN="
                + std::to_string(M) + "x" + std::to_string(shape_N[si]),
            details.str());
    }
}

bool parse_args(int argc, char** argv, BenchOptions& opt, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--iterations") {
            if (i + 1 >= argc) { err = "Missing value for --iterations"; return false; }
            opt.iterations = std::max(1, std::stoi(argv[++i]));
        } else if (a == "--warmup") {
            if (i + 1 >= argc) { err = "Missing value for --warmup"; return false; }
            opt.warmup = std::max(0, std::stoi(argv[++i]));
        } else if (a == "--layers") {
            if (i + 1 >= argc) { err = "Missing value for --layers"; return false; }
            opt.layers = static_cast<size_t>(std::max(1, std::stoi(argv[++i])));
        } else if (a == "--m") {
            if (i + 1 >= argc) { err = "Missing value for --m"; return false; }
            opt.batch_sizes = {static_cast<size_t>(std::max(1, std::stoi(argv[++i])))};
        } else if (a == "--precision") {
            if (i + 1 >= argc) { err = "Missing value for --precision"; return false; }
            const std::string p(argv[++i]);
            if (p == "int8") {
                opt.run_int8 = true;
                opt.run_int4 = false;
            } else if (p == "int4") {
                opt.run_int8 = false;
                opt.run_int4 = true;
            } else if (p == "both") {
                opt.run_int8 = true;
                opt.run_int4 = true;
            } else {
                err = "Invalid --precision, expected int8|int4|both";
                return false;
            }
        } else {
            err = "Unknown argument: " + a;
            return false;
        }
    }
    return true;
}

bool run_microbench(TestUtils::TestRunner& runner, const BenchOptions& opt) {
    runner.log_performance("Config",
        "Deterministic Gemma-3 270M matmul stack benchmark (LM head excluded)");
    {
        std::ostringstream details;
        details << "layers=" << opt.layers
                << ", warmup=" << opt.warmup
                << ", iterations=" << opt.iterations
                << ", group_size=" << kGroupSize;
        runner.log_performance("Config", details.str());
    }

    const auto model_weights = build_model_weights(opt.layers, 270270u);

    for (size_t M : opt.batch_sizes) {
        if (opt.run_int8) {
            const auto r8 = run_scenario(model_weights, PrecisionMode::INT8, M, opt.warmup, opt.iterations);
            log_projection_tables(runner, r8, PrecisionMode::INT8, M, opt.iterations, opt.layers);
        }
        if (opt.run_int4) {
            const auto r4 = run_scenario(model_weights, PrecisionMode::INT4, M, opt.warmup, opt.iterations);
            log_projection_tables(runner, r4, PrecisionMode::INT4, M, opt.iterations, opt.layers);
        }
    }

    return true;
}

} // namespace

int main(int argc, char** argv) {
    TestUtils::TestRunner runner("Gemma3 270M MatMul Microbench");

    BenchOptions opt;
    std::string parse_error;
    if (!parse_args(argc, argv, opt, parse_error)) {
        std::cerr << "Argument error: " << parse_error << "\n"
                  << "Usage: " << argv[0]
                  << " [--iterations N] [--warmup N] [--layers N] [--m N] [--precision int8|int4|both]\n";
        return 1;
    }

    runner.run_test("Deterministic MatMul Stack", run_microbench(runner, opt));
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}

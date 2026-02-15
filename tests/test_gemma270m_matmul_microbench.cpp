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

namespace ggml_q4 {

static constexpr int QK4_0 = 32;
static constexpr int QK8_0 = 32;

struct block_q4_0 {
    __fp16 d;
    uint8_t qs[QK4_0 / 2];
};

struct block_q8_0 {
    __fp16 d;
    int8_t qs[QK8_0];
};

static inline int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    int16x8_t prod_lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t prod_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vpadalq_s16(vpadalq_s16(acc, prod_lo), prod_hi);
#endif
}

void quantize_row_q4_0(const float* src, block_q4_0* dst, size_t k) {
    const size_t nb = k / QK4_0;
    for (size_t b = 0; b < nb; b++) {
        const float* s = src + b * QK4_0;
        float amax = 0;
        for (int j = 0; j < QK4_0; j++)
            amax = std::max(amax, std::abs(s[j]));
        float d = amax / 8.0f;
        if (d < 1e-10f) d = 1e-10f;
        dst[b].d = (__fp16)d;
        float id = 1.0f / d;
        for (int j = 0; j < QK4_0 / 2; j++) {
            int x0 = (int)std::round(s[j] * id) + 8;
            int x1 = (int)std::round(s[j + QK4_0/2] * id) + 8;
            x0 = std::max(0, std::min(15, x0));
            x1 = std::max(0, std::min(15, x1));
            dst[b].qs[j] = (uint8_t)(x0 | (x1 << 4));
        }
    }
}

void quantize_activation_q8_0(const __fp16* src, block_q8_0* dst, size_t k) {
    const size_t nb = k / QK8_0;
    for (size_t b = 0; b < nb; b++) {
        const __fp16* s = src + b * QK8_0;
        float amax = 0;
        for (int j = 0; j < QK8_0; j++)
            amax = std::max(amax, std::abs((float)s[j]));
        float d = amax / 127.0f;
        if (d < 1e-10f) d = 1e-10f;
        dst[b].d = (__fp16)d;
        float id = 1.0f / d;
        for (int j = 0; j < QK8_0; j++) {
            float v = (float)s[j] * id;
            dst[b].qs[j] = (int8_t)std::round(std::max(-128.0f, std::min(127.0f, v)));
        }
    }
}

float vec_dot_q4_0_q8_0(const block_q4_0* x, const block_q8_0* y, size_t nb) {
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    size_t ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const block_q4_0* x0 = &x[ib];
        const block_q4_0* x1 = &x[ib + 1];
        const block_q8_0* y0 = &y[ib];
        const block_q8_0* y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), (float)x0->d * (float)y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), (float)x1->d * (float)y1->d);
    }

    float sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

    for (; ib < nb; ib++) {
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < QK4_0/2; j++) {
            int v0 = (x[ib].qs[j] & 0x0F) - 8;
            int v1 = (x[ib].qs[j] >> 4) - 8;
            sumi0 += v0 * y[ib].qs[j];
            sumi1 += v1 * y[ib].qs[j + QK4_0/2];
        }
        sumf += (sumi0 + sumi1) * (float)x[ib].d * (float)y[ib].d;
    }

    return sumf;
}

void gemv_q4_0(
    const block_q8_0* A_q8,
    const block_q4_0* B_q4,
    __fp16* C,
    size_t K, size_t N
) {
    const size_t nb = K / QK4_0;
    const size_t N_blocks = (N + 3) / 4;

    auto& pool = CactusThreading::get_thread_pool();
    size_t num_threads = CactusThreading::GemmThreading::get_gemv_threads(N_blocks, pool.num_workers());
    num_threads = std::min(num_threads, N);

    auto process = [=](size_t n_start, size_t n_end) {
        for (size_t n = n_start; n < n_end; n++) {
            C[n] = (__fp16)vec_dot_q4_0_q8_0(B_q4 + n * nb, A_q8, nb);
        }
    };

    if (num_threads <= 1) {
        process(0, N);
    } else {
        pool.enqueue_n_threads(N, num_threads, process);
        pool.wait_all();
    }
}

void gemm_q4_0(
    const block_q8_0* A_q8,
    const block_q4_0* B_q4,
    __fp16* C,
    size_t M, size_t K, size_t N
) {
    const size_t nb = K / QK4_0;
    const size_t total = M * N;

    CactusThreading::parallel_gemm_tiles(M, total,
        [=](size_t idx_start, size_t idx_end) {
            for (size_t idx = idx_start; idx < idx_end; idx++) {
                size_t m = idx / N;
                size_t n = idx % N;
                C[m * N + n] = (__fp16)vec_dot_q4_0_q8_0(B_q4 + n * nb, A_q8 + m * nb, nb);
            }
        });
}

void matmul_q4_0(
    const block_q8_0* A_q8,
    const block_q4_0* B_q4,
    __fp16* C,
    size_t M, size_t K, size_t N
) {
    if (M == 1) {
        gemv_q4_0(A_q8, B_q4, C, K, N);
    } else {
        gemm_q4_0(A_q8, B_q4, C, M, K, N);
    }
}

} // namespace ggml_q4

namespace {

constexpr size_t kGroupSize = 32;
constexpr size_t kBlockSize = 4;
constexpr size_t kDefaultLayers = 18;

enum class PrecisionMode {
    INT8,
    INT4,
    GGML_Q4_0,
    INT4_COLOC
};

const char* mode_name(PrecisionMode mode) {
    switch (mode) {
        case PrecisionMode::INT8: return "INT8";
        case PrecisionMode::INT4: return "INT4";
        case PrecisionMode::GGML_Q4_0: return "GGML_Q4";
        case PrecisionMode::INT4_COLOC: return "INT4_COLOC";
    }
    return "UNKNOWN";
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
    std::vector<ggml_q4::block_q4_0> ggml_q4_weights;
    std::vector<uint8_t> int4_colocated;
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
    bool run_ggml = false;
    bool run_coloc = false;
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

std::vector<uint8_t> pack_colocated_int4(
    const std::vector<uint8_t>& int4_packed,
    const std::vector<__fp16>& int4_scales,
    size_t K, size_t N, size_t group_size
) {
    const size_t N_blocks = N / 4;
    const size_t num_groups = K / group_size;
    const size_t group_stride = 8 + group_size * 2;

    std::vector<uint8_t> result(N_blocks * num_groups * group_stride);

    for (size_t nb = 0; nb < N_blocks; nb++) {
        for (size_t g = 0; g < num_groups; g++) {
            uint8_t* dst = result.data() + (nb * num_groups + g) * group_stride;

            const __fp16* scale_src = int4_scales.data() + (nb * num_groups + g) * 4;
            std::memcpy(dst, scale_src, 8);

            const uint8_t* data_src = int4_packed.data() + (nb * K + g * group_size) * 2;
            std::memcpy(dst + 8, data_src, group_size * 2);
        }
    }

    return result;
}

std::vector<uint8_t> pack_int4_pairs(const std::vector<int8_t>& interleaved_int4) {
    if (interleaved_int4.size() % 32 != 0) {
        throw std::runtime_error("INT4 packing requires size divisible by 32");
    }
    std::vector<uint8_t> packed(interleaved_int4.size() / 2);
    for (size_t i = 0; i < interleaved_int4.size(); i += 32) {
        for (size_t j = 0; j < 16; ++j) {
            const uint8_t lo = static_cast<uint8_t>((static_cast<int16_t>(interleaved_int4[i + j]) + 8) & 0x0F);
            const uint8_t hi = static_cast<uint8_t>(((static_cast<int16_t>(interleaved_int4[i + 16 + j]) + 8) & 0x0F) << 4);
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

    out.int4_colocated = pack_colocated_int4(
        out.int4_weights_packed, out.int4_scales_interleaved,
        spec.K, spec.N, kGroupSize);

    const size_t nb_per_row = spec.K / ggml_q4::QK4_0;
    out.ggml_q4_weights.resize(spec.N * nb_per_row);
    for (size_t n = 0; n < spec.N; n++) {
        ggml_q4::quantize_row_q4_0(w_fp32.data() + n * spec.K,
                                    out.ggml_q4_weights.data() + n * nb_per_row, spec.K);
    }

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
                    std::vector<ggml_q4::block_q8_0>& a_q8,
                    ProjectionTiming& stats, bool measure) {
    const bool is_ggml = (mode == PrecisionMode::GGML_Q4_0);

    if (!measure) {
        if (is_ggml) {
            const size_t nb_per_row = w.K / ggml_q4::QK8_0;
            a_q8.resize(M * nb_per_row);
            for (size_t m = 0; m < M; m++)
                ggml_q4::quantize_activation_q8_0(input_fp16 + m * w.K, a_q8.data() + m * nb_per_row, w.K);
            ggml_q4::matmul_q4_0(a_q8.data(), w.ggml_q4_weights.data(), output_fp16, M, w.K, w.N);
        } else {
            quantize_activation_rows(input_fp16, M, w.K, a_quant, a_scales);
            if (mode == PrecisionMode::INT8) {
                cactus_matmul_int8(a_quant.data(), a_scales.data(),
                                   w.int8_weights_interleaved.data(),
                                   w.int8_scales_interleaved.data(),
                                   output_fp16, M, w.K, w.N, kGroupSize);
            } else if (mode == PrecisionMode::INT4_COLOC) {
                cactus_matmul_int4_colocated(a_quant.data(), a_scales.data(),
                                              w.int4_colocated.data(),
                                              output_fp16, M, w.K, w.N, kGroupSize);
            } else {
                cactus_matmul_int4(a_quant.data(), a_scales.data(),
                                   reinterpret_cast<const int8_t*>(w.int4_weights_packed.data()),
                                   w.int4_scales_interleaved.data(),
                                   output_fp16, M, w.K, w.N, kGroupSize);
            }
        }
        return;
    }

    const double t0 = now_ms();
    if (is_ggml) {
        const size_t nb_per_row = w.K / ggml_q4::QK8_0;
        a_q8.resize(M * nb_per_row);
        for (size_t m = 0; m < M; m++)
            ggml_q4::quantize_activation_q8_0(input_fp16 + m * w.K, a_q8.data() + m * nb_per_row, w.K);
    } else {
        quantize_activation_rows(input_fp16, M, w.K, a_quant, a_scales);
    }
    const double t1 = now_ms();

    if (is_ggml) {
        ggml_q4::matmul_q4_0(a_q8.data(), w.ggml_q4_weights.data(), output_fp16, M, w.K, w.N);
    } else if (mode == PrecisionMode::INT8) {
        cactus_matmul_int8(a_quant.data(), a_scales.data(),
                           w.int8_weights_interleaved.data(),
                           w.int8_scales_interleaved.data(),
                           output_fp16, M, w.K, w.N, kGroupSize);
    } else if (mode == PrecisionMode::INT4_COLOC) {
        cactus_matmul_int4_colocated(a_quant.data(), a_scales.data(),
                                      w.int4_colocated.data(),
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
    std::vector<ggml_q4::block_q8_0> a_q8;

    for (int iter = 0; iter < total_iters; ++iter) {
        const bool measure = iter >= warmup;
        const double step_t0 = measure ? now_ms() : 0.0;

        for (size_t l = 0; l < model_weights.size(); ++l) {
            const auto& w = model_weights[l];
            run_projection(w[0], mode, M, hidden.data(), q.data(), a_quant, a_scales, a_q8, out.projection_stats[0], measure);
            run_projection(w[1], mode, M, hidden.data(), k.data(), a_quant, a_scales, a_q8, out.projection_stats[1], measure);
            run_projection(w[2], mode, M, hidden.data(), v.data(), a_quant, a_scales, a_q8, out.projection_stats[2], measure);

            const size_t q_count = attn.size();
            const size_t kv_count = k.size();
            for (size_t i = 0; i < q_count; ++i) {
                const float qv = static_cast<float>(q[i]);
                const float kv = static_cast<float>(k[i % kv_count]);
                const float vv = static_cast<float>(v[(i * 7) % kv_count]);
                const float mixed = qv * 0.25f + kv * 0.03125f - vv * 0.015625f;
                attn[i] = static_cast<__fp16>(clamp_f32(mixed, -8.0f, 8.0f));
            }

            run_projection(w[3], mode, M, attn.data(), o.data(), a_quant, a_scales, a_q8, out.projection_stats[3], measure);

            for (size_t i = 0; i < hidden_next.size(); ++i) {
                const float mixed =
                    static_cast<float>(o[i]) * 0.2f + static_cast<float>(hidden[i]) * 0.8f;
                hidden_next[i] = static_cast<__fp16>(clamp_f32(mixed, -8.0f, 8.0f));
            }

            run_projection(w[4], mode, M, hidden_next.data(), gate.data(), a_quant, a_scales, a_q8, out.projection_stats[4], measure);
            run_projection(w[5], mode, M, hidden_next.data(), up.data(), a_quant, a_scales, a_q8, out.projection_stats[5], measure);

            for (size_t i = 0; i < ffn.size(); ++i) {
                const float g = std::tanh(static_cast<float>(gate[i]) * 0.25f);
                const float u = static_cast<float>(up[i]) * 0.25f;
                ffn[i] = static_cast<__fp16>(clamp_f32(g * u, -8.0f, 8.0f));
            }

            run_projection(w[6], mode, M, ffn.data(), down.data(), a_quant, a_scales, a_q8, out.projection_stats[6], measure);

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
                opt.run_int8 = true; opt.run_int4 = false; opt.run_ggml = false; opt.run_coloc = false;
            } else if (p == "int4") {
                opt.run_int8 = false; opt.run_int4 = true; opt.run_ggml = false; opt.run_coloc = false;
            } else if (p == "coloc") {
                opt.run_int8 = false; opt.run_int4 = false; opt.run_ggml = false; opt.run_coloc = true;
            } else if (p == "int4+coloc") {
                opt.run_int8 = false; opt.run_int4 = true; opt.run_ggml = false; opt.run_coloc = true;
            } else if (p == "ggml") {
                opt.run_int8 = false; opt.run_int4 = false; opt.run_ggml = true; opt.run_coloc = false;
            } else if (p == "int4+ggml") {
                opt.run_int8 = false; opt.run_int4 = true; opt.run_ggml = true; opt.run_coloc = false;
            } else if (p == "both") {
                opt.run_int8 = true; opt.run_int4 = true; opt.run_ggml = false; opt.run_coloc = false;
            } else if (p == "all") {
                opt.run_int8 = true; opt.run_int4 = true; opt.run_ggml = true; opt.run_coloc = true;
            } else {
                err = "Invalid --precision, expected int8|int4|coloc|int4+coloc|ggml|int4+ggml|both|all";
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
        if (opt.run_coloc) {
            const auto rc = run_scenario(model_weights, PrecisionMode::INT4_COLOC, M, opt.warmup, opt.iterations);
            log_projection_tables(runner, rc, PrecisionMode::INT4_COLOC, M, opt.iterations, opt.layers);
        }
        if (opt.run_ggml) {
            const auto rg = run_scenario(model_weights, PrecisionMode::GGML_Q4_0, M, opt.warmup, opt.iterations);
            log_projection_tables(runner, rg, PrecisionMode::GGML_Q4_0, M, opt.iterations, opt.layers);
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
                  << " [--iterations N] [--warmup N] [--layers N] [--m N] [--precision int8|int4|coloc|int4+coloc|ggml|int4+ggml|both|all]\n";
        return 1;
    }

    runner.run_test("Deterministic MatMul Stack", run_microbench(runner, opt));
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}

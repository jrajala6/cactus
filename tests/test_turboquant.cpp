#include "test_utils.h"
#include "../cactus/kernel/kernel.h"
#include <vector>
#include <cmath>
#include <iostream>

static float compute_mse(const __fp16* a, const __fp16* b, size_t n) {
    float mse = 0;
    for (size_t i = 0; i < n; ++i) {
        float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        mse += diff * diff;
    }
    return mse / n;
}

bool test_turboquant_roundtrip_2bit() {
    size_t seq_len = 16;
    size_t kv_heads = 4;
    size_t head_dim = 128;
    size_t angle_bits = 2;
    size_t projection_dim = 128;

    std::vector<__fp16> src(seq_len * kv_heads * head_dim);
    TestUtils::fill_random_fp16(src);

    std::vector<uint8_t> rotation_signs(head_dim / 8);
    std::vector<uint8_t> projection_matrix(projection_dim * head_dim / 8);
    cactus_turboquant_init(rotation_signs.data(), projection_matrix.data(), head_dim, projection_dim, 42);

    size_t num_vectors = seq_len * kv_heads;
    std::vector<float> radii(num_vectors);
    std::vector<uint8_t> angles(num_vectors * turboquant_angles_bytes_per_head(head_dim, angle_bits));
    std::vector<float> error_norms(num_vectors);
    std::vector<uint8_t> qjl_bits(num_vectors * turboquant_qjl_bytes_per_head(projection_dim));

    cactus_turboquant_encode_kv_fp16(
        src.data(), radii.data(), angles.data(), error_norms.data(), qjl_bits.data(),
        rotation_signs.data(), projection_matrix.data(), seq_len, kv_heads, head_dim, angle_bits, projection_dim);

    std::vector<__fp16> dst(seq_len * kv_heads * head_dim);
    cactus_turboquant_decode_kv_fp16(radii.data(), angles.data(), rotation_signs.data(), dst.data(), seq_len, kv_heads, head_dim, angle_bits);

    float mse = compute_mse(src.data(), dst.data(), src.size());
    std::cout << "  [2-bit roundtrip MSE: " << mse << ", expected ~0.117] ";
    return mse < 0.25f;
}

bool test_turboquant_roundtrip_4bit() {
    size_t seq_len = 16;
    size_t kv_heads = 4;
    size_t head_dim = 128;
    size_t angle_bits = 4;
    size_t projection_dim = 128;

    std::vector<__fp16> src(seq_len * kv_heads * head_dim);
    TestUtils::fill_random_fp16(src);

    std::vector<uint8_t> rotation_signs(head_dim / 8);
    std::vector<uint8_t> projection_matrix(projection_dim * head_dim / 8);
    cactus_turboquant_init(rotation_signs.data(), projection_matrix.data(), head_dim, projection_dim, 42);

    size_t num_vectors = seq_len * kv_heads;
    std::vector<float> radii(num_vectors);
    std::vector<uint8_t> angles(num_vectors * turboquant_angles_bytes_per_head(head_dim, angle_bits));
    std::vector<float> error_norms(num_vectors);
    std::vector<uint8_t> qjl_bits(num_vectors * turboquant_qjl_bytes_per_head(projection_dim));

    cactus_turboquant_encode_kv_fp16(
        src.data(), radii.data(), angles.data(), error_norms.data(), qjl_bits.data(),
        rotation_signs.data(), projection_matrix.data(), seq_len, kv_heads, head_dim, angle_bits, projection_dim);

    std::vector<__fp16> dst(seq_len * kv_heads * head_dim);
    cactus_turboquant_decode_kv_fp16(radii.data(), angles.data(), rotation_signs.data(), dst.data(), seq_len, kv_heads, head_dim, angle_bits);

    float mse = compute_mse(src.data(), dst.data(), src.size());
    std::cout << "  [4-bit roundtrip MSE: " << mse << ", expected ~0.009] ";
    return mse < 0.05f;
}

static bool run_accuracy_config(size_t cache_len, size_t num_q_heads, size_t num_kv_heads, size_t head_dim) {
    size_t batch_size = 1;
    size_t seq_len = 1;
    size_t new_len = 0;
    size_t key_angle_bits = 2;
    size_t value_angle_bits = 4;
    size_t projection_dim = head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<__fp16> queries(num_q_heads * head_dim);
    TestUtils::fill_random_fp16(queries);
    std::vector<__fp16> k_cache(cache_len * num_kv_heads * head_dim);
    TestUtils::fill_random_fp16(k_cache);
    std::vector<__fp16> v_cache(cache_len * num_kv_heads * head_dim);
    TestUtils::fill_random_fp16(v_cache);

    std::vector<__fp16> out_fp16(num_q_heads * head_dim);
    cactus_attention_f16(queries.data(), k_cache.data(), v_cache.data(), out_fp16.data(),
                         batch_size, seq_len, cache_len, num_q_heads, num_kv_heads, head_dim, scale, nullptr, 0, 0, true);

    std::vector<int8_t> k_cache_int8(cache_len * num_kv_heads * head_dim);
    std::vector<int8_t> v_cache_int8(cache_len * num_kv_heads * head_dim);
    std::vector<float> k_scales(kv_scales_count(cache_len, num_kv_heads, head_dim));
    std::vector<float> v_scales(kv_scales_count(cache_len, num_kv_heads, head_dim));
    cactus_quantize_kv_fp16_to_int8(k_cache.data(), k_cache_int8.data(), k_scales.data(), cache_len, num_kv_heads, head_dim);
    cactus_quantize_kv_fp16_to_int8(v_cache.data(), v_cache_int8.data(), v_scales.data(), cache_len, num_kv_heads, head_dim);

    std::vector<__fp16> out_int8(num_q_heads * head_dim);
    cactus_attention_hybrid_int8_fp16(queries.data(), k_cache_int8.data(), v_cache_int8.data(), k_scales.data(), v_scales.data(),
                                      nullptr, nullptr, out_int8.data(), batch_size, seq_len, cache_len, new_len, num_q_heads, num_kv_heads, head_dim, scale, 0, true, 0);

    std::vector<uint8_t> rot_signs(head_dim / 8);
    std::vector<uint8_t> proj_mat(projection_dim * head_dim / 8);
    cactus_turboquant_init(rot_signs.data(), proj_mat.data(), head_dim, projection_dim, 42);

    size_t num_vectors = cache_len * num_kv_heads;
    std::vector<float> k_radii(num_vectors), v_radii(num_vectors);
    std::vector<uint8_t> k_angles(num_vectors * turboquant_angles_bytes_per_head(head_dim, key_angle_bits));
    std::vector<uint8_t> v_angles(num_vectors * turboquant_angles_bytes_per_head(head_dim, value_angle_bits));
    std::vector<float> k_err_norms(num_vectors);
    std::vector<uint8_t> k_qjl(num_vectors * turboquant_qjl_bytes_per_head(projection_dim));

    cactus_turboquant_encode_kv_fp16(k_cache.data(), k_radii.data(), k_angles.data(), k_err_norms.data(), k_qjl.data(), rot_signs.data(), proj_mat.data(), cache_len, num_kv_heads, head_dim, key_angle_bits, projection_dim);
    cactus_turboquant_encode_kv_fp16(v_cache.data(), v_radii.data(), v_angles.data(), nullptr, nullptr, rot_signs.data(), proj_mat.data(), cache_len, num_kv_heads, head_dim, value_angle_bits, projection_dim);

    std::vector<__fp16> out_tq(num_q_heads * head_dim);
    cactus_attention_hybrid_turboquant_fp16(queries.data(), k_radii.data(), k_angles.data(), k_err_norms.data(), k_qjl.data(),
                                            v_radii.data(), v_angles.data(),
                                            rot_signs.data(), proj_mat.data(),
                                            nullptr, nullptr, out_tq.data(), batch_size, seq_len, cache_len, new_len,
                                            num_q_heads, num_kv_heads, head_dim, scale, key_angle_bits, value_angle_bits, projection_dim, 0, true, 0);

    float mse_int8 = compute_mse(out_fp16.data(), out_int8.data(), out_fp16.size());
    float mse_tq   = compute_mse(out_fp16.data(), out_tq.data(), out_fp16.size());

    std::cout << "  q=" << num_q_heads << " kv=" << num_kv_heads << " d=" << head_dim
              << " cache=" << cache_len << " -> INT8: " << mse_int8 << ", TQ: " << mse_tq << std::endl;
    return mse_tq < 0.05f;
}

bool test_attention_accuracy() {
    std::cout << std::endl;
    bool pass = true;
    pass &= run_accuracy_config(64,   8,  4, 64);
    pass &= run_accuracy_config(128,  8,  4, 128);
    pass &= run_accuracy_config(512, 32,  8, 128);
    pass &= run_accuracy_config(1024, 32,  8, 128);
    pass &= run_accuracy_config(256, 16,  4, 64);
    pass &= run_accuracy_config(128, 28,  4, 128);
    std::cout << "  ";
    return pass;
}

bool test_attention_performance() {
    size_t batch_size = 1;
    size_t seq_len = 1;
    size_t cache_len = 4096;
    size_t new_len = 0;
    size_t num_q_heads = 32;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;
    size_t key_angle_bits = 2;
    size_t value_angle_bits = 4;
    size_t projection_dim = 128;
    float scale = 1.0f / std::sqrt(head_dim);

    std::vector<__fp16> queries(num_q_heads * head_dim);
    TestUtils::fill_random_fp16(queries);
    std::vector<__fp16> k_cache(cache_len * num_kv_heads * head_dim);
    TestUtils::fill_random_fp16(k_cache);
    std::vector<__fp16> v_cache(cache_len * num_kv_heads * head_dim);
    TestUtils::fill_random_fp16(v_cache);

    std::vector<__fp16> out_fp16(num_q_heads * head_dim);
    std::vector<__fp16> out_int8(num_q_heads * head_dim);
    std::vector<__fp16> out_tq(num_q_heads * head_dim);

    std::vector<int8_t> k_cache_int8(cache_len * num_kv_heads * head_dim);
    std::vector<int8_t> v_cache_int8(cache_len * num_kv_heads * head_dim);
    std::vector<float> k_scales(kv_scales_count(cache_len, num_kv_heads, head_dim));
    std::vector<float> v_scales(kv_scales_count(cache_len, num_kv_heads, head_dim));
    cactus_quantize_kv_fp16_to_int8(k_cache.data(), k_cache_int8.data(), k_scales.data(), cache_len, num_kv_heads, head_dim);
    cactus_quantize_kv_fp16_to_int8(v_cache.data(), v_cache_int8.data(), v_scales.data(), cache_len, num_kv_heads, head_dim);

    std::vector<uint8_t> rot_signs(head_dim / 8);
    std::vector<uint8_t> proj_mat(projection_dim * head_dim / 8);
    cactus_turboquant_init(rot_signs.data(), proj_mat.data(), head_dim, projection_dim, 42);

    size_t num_vectors = cache_len * num_kv_heads;
    std::vector<float> k_radii(num_vectors), v_radii(num_vectors);
    std::vector<uint8_t> k_angles(num_vectors * turboquant_angles_bytes_per_head(head_dim, key_angle_bits));
    std::vector<uint8_t> v_angles(num_vectors * turboquant_angles_bytes_per_head(head_dim, value_angle_bits));
    std::vector<float> k_err_norms(num_vectors);
    std::vector<uint8_t> k_qjl(num_vectors * turboquant_qjl_bytes_per_head(projection_dim));

    cactus_turboquant_encode_kv_fp16(k_cache.data(), k_radii.data(), k_angles.data(), k_err_norms.data(), k_qjl.data(), rot_signs.data(), proj_mat.data(), cache_len, num_kv_heads, head_dim, key_angle_bits, projection_dim);
    cactus_turboquant_encode_kv_fp16(v_cache.data(), v_radii.data(), v_angles.data(), nullptr, nullptr, rot_signs.data(), proj_mat.data(), cache_len, num_kv_heads, head_dim, value_angle_bits, projection_dim);

    int iterations = 10;

    double f16_time = TestUtils::time_function([&]() {
        cactus_attention_f16(queries.data(), k_cache.data(), v_cache.data(), out_fp16.data(),
                             batch_size, seq_len, cache_len, num_q_heads, num_kv_heads, head_dim, scale, nullptr, 0, 0, true);
    }, iterations);

    double int8_time = TestUtils::time_function([&]() {
        cactus_attention_hybrid_int8_fp16(queries.data(), k_cache_int8.data(), v_cache_int8.data(), k_scales.data(), v_scales.data(),
                                          nullptr, nullptr, out_int8.data(), batch_size, seq_len, cache_len, new_len, num_q_heads, num_kv_heads, head_dim, scale, 0, true, 0);
    }, iterations);

    double tq_time = TestUtils::time_function([&]() {
        cactus_attention_hybrid_turboquant_fp16(queries.data(), k_radii.data(), k_angles.data(), k_err_norms.data(), k_qjl.data(),
                                                v_radii.data(), v_angles.data(),
                                                rot_signs.data(), proj_mat.data(),
                                                nullptr, nullptr, out_tq.data(), batch_size, seq_len, cache_len, new_len,
                                                num_q_heads, num_kv_heads, head_dim, scale, key_angle_bits, value_angle_bits, projection_dim, 0, true, 0);
    }, iterations);

    size_t fp16_bytes = head_dim * 2;
    size_t int8_bytes = head_dim + ((head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE) * 4;
    size_t tq_k_bytes = 4 + turboquant_angles_bytes_per_head(head_dim, key_angle_bits) + 4 + turboquant_qjl_bytes_per_head(projection_dim);
    size_t tq_v_bytes = 4 + turboquant_angles_bytes_per_head(head_dim, value_angle_bits);
    float tq_compression = static_cast<float>(fp16_bytes * 2) / static_cast<float>(tq_k_bytes + tq_v_bytes);

    std::cout << "\n  Latency  -> FP16: " << (f16_time/iterations) << " ms, INT8: " << (int8_time/iterations) << " ms, TQ K2V4: " << (tq_time/iterations) << " ms";
    std::cout << "\n  Memory   -> FP16: " << fp16_bytes << "B, INT8: " << int8_bytes << "B, TQ(K): " << tq_k_bytes << "B, TQ(V): " << tq_v_bytes << "B";
    std::cout << "\n  Compress -> TQ K2V4: " << tq_compression << "x vs FP16, " << static_cast<float>(int8_bytes * 2) / static_cast<float>(tq_k_bytes + tq_v_bytes) << "x vs INT8\n  ";

    return true;
}

int main() {
    TestUtils::TestRunner runner("TurboQuant");
    runner.run_test("2-bit PolarQuant roundtrip", test_turboquant_roundtrip_2bit());
    runner.run_test("4-bit PolarQuant roundtrip", test_turboquant_roundtrip_4bit());
    runner.run_test("K2V4 attention accuracy vs FP16/INT8", test_attention_accuracy());
    runner.run_test("K2V4 attention performance", test_attention_performance());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}

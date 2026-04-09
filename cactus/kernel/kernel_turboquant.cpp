#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>


namespace {

static constexpr float PI_F = 3.14159265358979323846f;

struct Xoshiro256 {
    uint64_t s[4];
    explicit Xoshiro256(uint64_t seed) {
        for (int i = 0; i < 4; i++)
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }
    
    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t next() {
        uint64_t result = rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }
};


static void hadamard_inplace(float* __restrict data, size_t dim) {
    if (dim < 4 || (dim & (dim - 1)) != 0) return;

    // Phase 1: size-4 in-register butterfly.
    // vtrn1q_f32(sum, sub) produces [a+b, a-b, c+d, c-d] (correct H4 stage 1).
    // vcombine_f32(low(sum2), low(sub2)) produces the correct H4 stage 2 output.
    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        float32x4_t v0 = vld1q_f32(data + i);
        float32x4_t v1 = vld1q_f32(data + i + 4);
        float32x4_t v2 = vld1q_f32(data + i + 8);
        float32x4_t v3 = vld1q_f32(data + i + 12);
        float32x4_t r0 = vrev64q_f32(v0), r1 = vrev64q_f32(v1);
        float32x4_t r2 = vrev64q_f32(v2), r3 = vrev64q_f32(v3);
        float32x4_t p0 = vtrn1q_f32(vaddq_f32(v0, r0), vsubq_f32(v0, r0));
        float32x4_t p1 = vtrn1q_f32(vaddq_f32(v1, r1), vsubq_f32(v1, r1));
        float32x4_t p2 = vtrn1q_f32(vaddq_f32(v2, r2), vsubq_f32(v2, r2));
        float32x4_t p3 = vtrn1q_f32(vaddq_f32(v3, r3), vsubq_f32(v3, r3));
        float32x4_t s0 = vcombine_f32(vget_high_f32(p0), vget_low_f32(p0));
        float32x4_t s1 = vcombine_f32(vget_high_f32(p1), vget_low_f32(p1));
        float32x4_t s2 = vcombine_f32(vget_high_f32(p2), vget_low_f32(p2));
        float32x4_t s3 = vcombine_f32(vget_high_f32(p3), vget_low_f32(p3));
        vst1q_f32(data + i,      vcombine_f32(vget_low_f32(vaddq_f32(p0, s0)), vget_low_f32(vsubq_f32(p0, s0))));
        vst1q_f32(data + i + 4,  vcombine_f32(vget_low_f32(vaddq_f32(p1, s1)), vget_low_f32(vsubq_f32(p1, s1))));
        vst1q_f32(data + i + 8,  vcombine_f32(vget_low_f32(vaddq_f32(p2, s2)), vget_low_f32(vsubq_f32(p2, s2))));
        vst1q_f32(data + i + 12, vcombine_f32(vget_low_f32(vaddq_f32(p3, s3)), vget_low_f32(vsubq_f32(p3, s3))));
    }
    for (; i < dim; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t r = vrev64q_f32(v);
        float32x4_t p = vtrn1q_f32(vaddq_f32(v, r), vsubq_f32(v, r));
        float32x4_t s = vcombine_f32(vget_high_f32(p), vget_low_f32(p));
        vst1q_f32(data + i, vcombine_f32(vget_low_f32(vaddq_f32(p, s)), vget_low_f32(vsubq_f32(p, s))));
    }

    // Phase 2a: dedicated half=4 pass (only when more passes follow).
    if (dim >= 16) {
        for (size_t k = 0; k < dim; k += 8) {
            float32x4_t a = vld1q_f32(data + k);
            float32x4_t b = vld1q_f32(data + k + 4);
            vst1q_f32(data + k,     vaddq_f32(a, b));
            vst1q_f32(data + k + 4, vsubq_f32(a, b));
        }
    }

    // Phase 2b: dedicated half=8 pass (only when more passes follow).
    if (dim >= 32) {
        for (size_t k = 0; k < dim; k += 16) {
            float32x4_t a0 = vld1q_f32(data + k);
            float32x4_t a1 = vld1q_f32(data + k + 4);
            float32x4_t b0 = vld1q_f32(data + k + 8);
            float32x4_t b1 = vld1q_f32(data + k + 12);
            vst1q_f32(data + k,      vaddq_f32(a0, b0));
            vst1q_f32(data + k + 8,  vsubq_f32(a0, b0));
            vst1q_f32(data + k + 4,  vaddq_f32(a1, b1));
            vst1q_f32(data + k + 12, vsubq_f32(a1, b1));
        }
    }

    // Phase 2c: general outer butterfly for half=16..dim/4.
    // The 16-wide fast path is never dead here since half >= 16.
    for (size_t half = 16; half * 2 < dim; half <<= 1) {
        for (size_t g = 0; g < dim; g += half * 2) {
            size_t j = g;
            for (; j + 15 < g + half; j += 16) {
                float32x4_t a0 = vld1q_f32(data + j);
                float32x4_t a1 = vld1q_f32(data + j + 4);
                float32x4_t a2 = vld1q_f32(data + j + 8);
                float32x4_t a3 = vld1q_f32(data + j + 12);
                float32x4_t b0 = vld1q_f32(data + j + half);
                float32x4_t b1 = vld1q_f32(data + j + half + 4);
                float32x4_t b2 = vld1q_f32(data + j + half + 8);
                float32x4_t b3 = vld1q_f32(data + j + half + 12);
                vst1q_f32(data + j,             vaddq_f32(a0, b0));
                vst1q_f32(data + j + half,      vsubq_f32(a0, b0));
                vst1q_f32(data + j + 4,         vaddq_f32(a1, b1));
                vst1q_f32(data + j + half + 4,  vsubq_f32(a1, b1));
                vst1q_f32(data + j + 8,         vaddq_f32(a2, b2));
                vst1q_f32(data + j + half + 8,  vsubq_f32(a2, b2));
                vst1q_f32(data + j + 12,        vaddq_f32(a3, b3));
                vst1q_f32(data + j + half + 12, vsubq_f32(a3, b3));
            }
            for (; j < g + half; j += 4) {
                float32x4_t a = vld1q_f32(data + j);
                float32x4_t b = vld1q_f32(data + j + half);
                vst1q_f32(data + j,        vaddq_f32(a, b));
                vst1q_f32(data + j + half, vsubq_f32(a, b));
            }
        }
    }

    // Final pass: half=dim/2, fused with normalization to save one full array read/write.
    float32x4_t norm_vec = vdupq_n_f32(1.0f / sqrtf(static_cast<float>(dim)));
    if (dim >= 8) {
        size_t half = dim / 2;
        size_t j = 0;
        for (; j + 15 < half; j += 16) {
            float32x4_t a0 = vld1q_f32(data + j);
            float32x4_t a1 = vld1q_f32(data + j + 4);
            float32x4_t a2 = vld1q_f32(data + j + 8);
            float32x4_t a3 = vld1q_f32(data + j + 12);
            float32x4_t b0 = vld1q_f32(data + j + half);
            float32x4_t b1 = vld1q_f32(data + j + half + 4);
            float32x4_t b2 = vld1q_f32(data + j + half + 8);
            float32x4_t b3 = vld1q_f32(data + j + half + 12);
            vst1q_f32(data + j,             vmulq_f32(vaddq_f32(a0, b0), norm_vec));
            vst1q_f32(data + j + half,      vmulq_f32(vsubq_f32(a0, b0), norm_vec));
            vst1q_f32(data + j + 4,         vmulq_f32(vaddq_f32(a1, b1), norm_vec));
            vst1q_f32(data + j + half + 4,  vmulq_f32(vsubq_f32(a1, b1), norm_vec));
            vst1q_f32(data + j + 8,         vmulq_f32(vaddq_f32(a2, b2), norm_vec));
            vst1q_f32(data + j + half + 8,  vmulq_f32(vsubq_f32(a2, b2), norm_vec));
            vst1q_f32(data + j + 12,        vmulq_f32(vaddq_f32(a3, b3), norm_vec));
            vst1q_f32(data + j + half + 12, vmulq_f32(vsubq_f32(a3, b3), norm_vec));
        }
        for (; j < half; j += 4) {
            float32x4_t a = vld1q_f32(data + j);
            float32x4_t b = vld1q_f32(data + j + half);
            vst1q_f32(data + j,        vmulq_f32(vaddq_f32(a, b), norm_vec));
            vst1q_f32(data + j + half, vmulq_f32(vsubq_f32(a, b), norm_vec));
        }
    } else {
        // dim == 4: no outer butterfly needed, just normalize
        vst1q_f32(data, vmulq_f32(vld1q_f32(data), norm_vec));
    }
}

// Signs are bit-packed: bit=1 means sign=-1, bit=0 means sign=+1.
// Convention: bit k of byte (i/8) corresponds to element i, where k = i%8 (LSB first).
// kBitMasks[k] = 1<<k, used by vtst_u8 to isolate each bit position.
// vtst_u8 returns 0xFF when the bit is set (sign=-1) and 0x00 otherwise.
// After widening to int32 and shifting to bit 31 we get 0x80000000 or 0x00000000,
// which XOR'd with the float bits negates or preserves the value.
static const uint8x8_t kBitMasks = vcreate_u8(0x8040201008040201ULL);

static inline void apply_signs(float* __restrict data, const uint8_t* __restrict signs_packed, size_t dim) {
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        // Each byte covers 8 elements; two bytes cover 16.
        int8x8_t sm0 = vreinterpret_s8_u8(vtst_u8(vdup_n_u8(signs_packed[i / 8]),     kBitMasks));
        int8x8_t sm1 = vreinterpret_s8_u8(vtst_u8(vdup_n_u8(signs_packed[i / 8 + 1]), kBitMasks));
        int16x8_t sm16_0 = vmovl_s8(sm0);
        int16x8_t sm16_1 = vmovl_s8(sm1);
        int32x4_t f0 = vshlq_n_s32(vmovl_s16(vget_low_s16(sm16_0)),  31);
        int32x4_t f1 = vshlq_n_s32(vmovl_high_s16(sm16_0),            31);
        int32x4_t f2 = vshlq_n_s32(vmovl_s16(vget_low_s16(sm16_1)),  31);
        int32x4_t f3 = vshlq_n_s32(vmovl_high_s16(sm16_1),            31);
        vst1q_f32(data + i,      vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(vld1q_f32(data + i)),      f0)));
        vst1q_f32(data + i + 4,  vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(vld1q_f32(data + i + 4)),  f1)));
        vst1q_f32(data + i + 8,  vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(vld1q_f32(data + i + 8)),  f2)));
        vst1q_f32(data + i + 12, vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(vld1q_f32(data + i + 12)), f3)));
    }
    for (; i < dim; i++) {
        if ((signs_packed[i / 8] >> (i % 8)) & 1) data[i] = -data[i];
    }
}

// XOR each vec[d] with the sign mask to negate sign=-1 entries, then accumulate.
// Avoids float conversion and FMA; vaddq_f32 on the conditionally negated values suffices.
static float signed_dot(const uint8_t* __restrict signs_packed, const float* __restrict vec, size_t dim) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t d = 0;
    for (; d + 8 <= dim; d += 8) {
        int8x8_t sm = vreinterpret_s8_u8(vtst_u8(vdup_n_u8(signs_packed[d / 8]), kBitMasks));
        int16x8_t sm16 = vmovl_s8(sm);
        int32x4_t f0 = vshlq_n_s32(vmovl_s16(vget_low_s16(sm16)),  31);
        int32x4_t f1 = vshlq_n_s32(vmovl_high_s16(sm16),            31);
        acc0 = vaddq_f32(acc0, vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(vld1q_f32(&vec[d])),     f0)));
        acc1 = vaddq_f32(acc1, vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(vld1q_f32(&vec[d + 4])), f1)));
    }
    float result = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; d < dim; d++) {
        float v = vec[d];
        result += ((signs_packed[d / 8] >> (d % 8)) & 1) ? -v : v;
    }
    return result;
}

static void rotate_forward(float* data, const uint8_t* signs_packed, size_t dim) {
    apply_signs(data, signs_packed, dim);
    hadamard_inplace(data, dim);
}

static void rotate_inverse(float* data, const uint8_t* signs_packed, size_t dim) {
    hadamard_inplace(data, dim);
    apply_signs(data, signs_packed, dim);
}

static void quantize_2bit(const float* src, uint8_t* dst, size_t dim) {
    const float32x4_t two = vdupq_n_f32(2.0f);
    const int32x4_t zero_i = vdupq_n_s32(0);
    const int32x4_t three_i = vdupq_n_s32(3);

    const int32_t shift_arr[4] = {0, 2, 4, 6};
    const int32x4_t bit_shifts = vld1q_s32(shift_arr);
    size_t packed = 0;
    size_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t v = vld1q_f32(&src[i]);
        float32x4_t scaled = vfmaq_f32(two, v, two);
        int32x4_t codes = vcvtq_s32_f32(scaled);
        codes = vmaxq_s32(vminq_s32(codes, three_i), zero_i);
        codes = vshlq_s32(codes, bit_shifts);
        dst[packed++] = static_cast<uint8_t>(vaddvq_s32(codes));
    }

    if (i < dim) {
        uint8_t byte = 0;
        for (size_t j = 0; j < dim - i; j++) {
            int code = static_cast<int>(floorf((src[i + j] + 1.0f) * 2.0f));
            code = std::max(0, std::min(3, code));
            byte |= (static_cast<uint8_t>(code) << (j * 2));
        }
        dst[packed] = byte;
    }
}

static void dequantize_2bit(const uint8_t* src, float* dst, size_t dim) {
    const uint32x4_t mask = vdupq_n_u32(3);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t offset = vdupq_n_f32(-0.75f);

    const int32_t shift_arr[4] = {0, -2, -4, -6};
    const int32x4_t bit_shifts = vld1q_s32(shift_arr);
    size_t packed = 0;
    size_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        uint32x4_t byte_vec = vdupq_n_u32(src[packed++]);
        uint32x4_t shifted = vshlq_u32(byte_vec, bit_shifts);
        uint32x4_t codes_int = vandq_u32(shifted, mask);
        float32x4_t codes_float = vcvtq_f32_u32(codes_int);
        float32x4_t decoded = vfmaq_f32(offset, codes_float, half);
        vst1q_f32(&dst[i], decoded);
    }

    if (i < dim) {
        uint8_t byte = src[packed];
        for (size_t j = 0; j < dim - i; j++) {
            uint8_t code = (byte >> (j * 2)) & 0x03;
            dst[i + j] = static_cast<float>(2 * static_cast<int>(code) - 3) * 0.25f;
        }
    }
}

void cactus_turboquant_init(
    uint8_t* rotation_signs, uint8_t* projection_matrix,
    size_t head_dim, size_t projection_dim, uint64_t seed
) {
    assert((head_dim & (head_dim - 1)) == 0 && head_dim >= 8);
    Xoshiro256 rng(seed);

    // Pack rotation_signs: head_dim bits = head_dim/8 bytes.
    // Use all 8 bytes of each next() call to avoid wasting entropy.
    const size_t rot_bytes = head_dim / 8;
    for (size_t i = 0; i < rot_bytes; ) {
        uint64_t w = rng.next();
        size_t n = std::min(size_t(8), rot_bytes - i);
        std::memcpy(rotation_signs + i, &w, n);
        i += n;
    }

    // Pack projection_matrix: projection_dim * head_dim bits = projection_dim * head_dim / 8 bytes.
    const size_t proj_bytes = projection_dim * head_dim / 8;
    for (size_t i = 0; i < proj_bytes; ) {
        uint64_t w = rng.next();
        size_t n = std::min(size_t(8), proj_bytes - i);
        std::memcpy(projection_matrix + i, &w, n);
        i += n;
    }
}

void cactus_turboquant_encode_kv_fp16(
    const __fp16* src, float* dst_radii, uint8_t* dst_angles,
    float* dst_error_norms, uint8_t* dst_qjl_bits,
    const uint8_t* rotation_signs, const uint8_t* projection_matrix,
    size_t seq_len, size_t kv_heads, size_t head_dim,
    size_t angle_bits, size_t projection_dim
) {
    assert(angle_bits == TURBOQUANT_ANGLE_BITS);
    assert((head_dim & (head_dim - 1)) == 0);
    assert(head_dim <= 512);

    const size_t angles_bytes = turboquant_angles_bytes_per_head(head_dim, angle_bits);
    const size_t qjl_bytes = turboquant_qjl_bytes_per_head(projection_dim);
    const size_t rot_row_bytes = head_dim / 8;  // bytes per projection row

    CactusThreading::parallel_for(
        seq_len * kv_heads, CactusThreading::Thresholds::ELEMENT_WISE,
        [=](size_t start, size_t end) {
            float buf[512];
            float dq[512];
            float residual[512];

            for (size_t idx = start; idx < end; idx++) {
                const __fp16* in = src + idx * head_dim;

                float32x4_t norm_acc = vdupq_n_f32(0.0f);
                size_t d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    float16x8_t h = vld1q_f16(in + d);
                    float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
                    float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
                    vst1q_f32(buf + d, lo);
                    vst1q_f32(buf + d + 4, hi);
                    norm_acc = vfmaq_f32(norm_acc, lo, lo);
                    norm_acc = vfmaq_f32(norm_acc, hi, hi);
                }
                float norm_sq = vaddvq_f32(norm_acc);
                for (; d < head_dim; d++) {
                    float v = static_cast<float>(in[d]);
                    buf[d] = v;
                    norm_sq += v * v;
                }

                float radius = sqrtf(norm_sq);
                dst_radii[idx] = radius;

                if (radius > 1e-10f) {
                    float inv_r = 1.0f / radius;
                    float32x4_t inv_r_vec = vdupq_n_f32(inv_r);
                    d = 0;
                    for (; d + 4 <= head_dim; d += 4)
                        vst1q_f32(buf + d, vmulq_f32(vld1q_f32(buf + d), inv_r_vec));
                    for (; d < head_dim; d++)
                        buf[d] *= inv_r;
                }

                rotate_forward(buf, rotation_signs, head_dim);
                quantize_2bit(buf, dst_angles + idx * angles_bytes, head_dim);
                dequantize_2bit(dst_angles + idx * angles_bytes, dq, head_dim);

                float32x4_t err_acc = vdupq_n_f32(0.0f);
                float32x4_t r_vec = vdupq_n_f32(radius);
                float sum_abs_res = 1e-10f;
                d = 0;
                for (; d + 4 <= head_dim; d += 4) {
                    float32x4_t err = vmulq_f32(vsubq_f32(vld1q_f32(&buf[d]), vld1q_f32(&dq[d])), r_vec);
                    vst1q_f32(&residual[d], err);
                    err_acc = vfmaq_f32(err_acc, err, err);
                }
                float err_sq = vaddvq_f32(err_acc);
                for (; d < head_dim; d++) {
                    float err = (buf[d] - dq[d]) * radius;
                    residual[d] = err;
                    err_sq += err * err;
                }
                for (size_t k = 0; k < head_dim; k++) {
                    sum_abs_res += std::abs(residual[k]);
                }
                dst_error_norms[idx] = sqrtf(err_sq);

                uint8_t* bits = dst_qjl_bits + idx * qjl_bytes;
                std::memset(bits, 0, qjl_bytes);
                Xoshiro256 rng(1337 + idx); // localized stochastic PRNG

                for (size_t p = 0; p < projection_dim; p++) {
                    float dot = signed_dot(projection_matrix + p * rot_row_bytes, residual, head_dim);
                    float prob = (dot / sum_abs_res + 1.0f) * 0.5f;
                    prob = std::max(0.0f, std::min(1.0f, prob));
                    float rand_f = (rng.next() & 0xFFFFFF) / static_cast<float>(0x1000000);
                    if (rand_f < prob) {
                        bits[p / 8] |= (1u << (p % 8));
                    }
                }
            }
        });
}

void cactus_turboquant_decode_kv_fp16(
    const float* radii, const uint8_t* angles, const uint8_t* rotation_signs,
    __fp16* dst, size_t seq_len, size_t kv_heads, size_t head_dim, size_t angle_bits
) {
    const size_t angles_bytes = turboquant_angles_bytes_per_head(head_dim, angle_bits);
    CactusThreading::parallel_for(
        seq_len * kv_heads, CactusThreading::Thresholds::ELEMENT_WISE,
        [=](size_t start, size_t end) {
            float buf[512];
            for (size_t idx = start; idx < end; idx++) {
                float radius = radii[idx];
                dequantize_2bit(angles + idx * angles_bytes, buf, head_dim);
                float32x4_t r_vec = vdupq_n_f32(radius);
                size_t d = 0;
                for (; d + 4 <= head_dim; d += 4) {
                    vst1q_f32(&buf[d], vmulq_f32(vld1q_f32(&buf[d]), r_vec));
                }
                for (; d < head_dim; d++) buf[d] *= radius;

                rotate_inverse(buf, rotation_signs, head_dim);
                __fp16* out = dst + idx * head_dim;
                d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    vst1q_f16(&out[d], vcombine_f16(vcvt_f16_f32(vld1q_f32(&buf[d])), vcvt_f16_f32(vld1q_f32(&buf[d + 4]))));
                }
                for (; d < head_dim; d++) out[d] = static_cast<__fp16>(buf[d]);
            }
        });
}

void cactus_turboquant_decode_dot_fp16(
    const __fp16* query_rotated, const float* query_norms, const uint8_t* query_qjl_bits,
    const float* cached_radii, const uint8_t* cached_angles,
    const float* cached_error_norms, const uint8_t* cached_qjl_bits,
    __fp16* output, size_t seq_len, size_t kv_heads, size_t head_dim,
    size_t angle_bits, size_t projection_dim
) {
    const size_t angles_bytes = turboquant_angles_bytes_per_head(head_dim, angle_bits);
    const size_t qjl_bytes = turboquant_qjl_bytes_per_head(projection_dim);
    float m = static_cast<float>(projection_dim);

    CactusThreading::parallel_for(
        seq_len * kv_heads, CactusThreading::Thresholds::ELEMENT_WISE,
        [=](size_t start, size_t end) {
            float dq[256];
            for (size_t idx = start; idx < end; idx++) {
                const __fp16* q_rot = query_rotated + idx * head_dim;
                float q_norm = query_norms[idx];
                float radius = cached_radii[idx];
                float error_norm = cached_error_norms[idx];
                const uint8_t* angles = cached_angles + idx * angles_bytes;
                const uint8_t* qjl_bits = cached_qjl_bits + idx * qjl_bytes;
                const uint8_t* q_qjl_bits = query_qjl_bits + idx * qjl_bytes;

                dequantize_2bit(angles, dq, head_dim);

                float32x4_t dot_acc = vdupq_n_f32(0.0f);
                size_t d = 0;
                for (; d + 4 <= head_dim; d += 4) {
                    float32x4_t q_vec = vcvt_f32_f16(vld1_f16(&q_rot[d]));
                    dot_acc = vfmaq_f32(dot_acc, q_vec, vld1q_f32(&dq[d]));
                }
                float polar_dot = vaddvq_f32(dot_acc);
                for (; d < head_dim; d++) {
                    polar_dot += static_cast<float>(q_rot[d]) * dq[d];
                }
                polar_dot *= radius;

                size_t matches = 0;
                for (size_t b = 0; b < qjl_bytes; b++) {
                    // __builtin_popcount works precisely up to int boundary. 8-bit masks safe if &0xFF
                    matches += __builtin_popcount(~(qjl_bits[b] ^ q_qjl_bits[b]) & 0xFF);
                }

                float avg_sign = (2.0f * static_cast<float>(matches) - m) / m;
                float qjl_correction = q_norm * error_norm * sinf(PI_F * 0.5f * avg_sign);

                output[idx] = static_cast<__fp16>(polar_dot + qjl_correction);
            }
        });
}

float cactus_turboquant_dot(
    const __fp16* q_rot, float q_norm, float radius, const uint8_t* angles, float error_norm,
    const uint8_t* qjl_bits, const uint8_t* query_qjl_bits, size_t head_dim, size_t projection_dim
) {
    float dq[256];
    dequantize_2bit(angles, dq, head_dim);
    float polar_dot = 0.0f;
    for (size_t d = 0; d < head_dim; d++) polar_dot += static_cast<float>(q_rot[d]) * dq[d];
    polar_dot *= radius;

    size_t matches = 0;
    const size_t qjl_bytes = turboquant_qjl_bytes_per_head(projection_dim);
    for (size_t b = 0; b < qjl_bytes; b++) matches += __builtin_popcount(~(qjl_bits[b] ^ query_qjl_bits[b]) & 0xFF);

    float m = static_cast<float>(projection_dim);
    float avg_sign = (2.0f * static_cast<float>(matches) - m) / m;
    float qjl_correction = q_norm * error_norm * sinf(PI_F * 0.5f * avg_sign);
    return polar_dot + qjl_correction;
}

} // namespace

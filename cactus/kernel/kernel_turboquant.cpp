#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>



static constexpr float PI_F = 3.14159265358979323846f;

struct Xoshiro256 {
    uint64_t s[4];
    explicit Xoshiro256(uint64_t seed) {
        for (int i = 0; i < 4; i++) {
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

    if (dim >= 16) {
        for (size_t k = 0; k < dim; k += 8) {
            float32x4_t a = vld1q_f32(data + k);
            float32x4_t b = vld1q_f32(data + k + 4);
            vst1q_f32(data + k,     vaddq_f32(a, b));
            vst1q_f32(data + k + 4, vsubq_f32(a, b));
        }
    }

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
        vst1q_f32(data, vmulq_f32(vld1q_f32(data), norm_vec));
    }
}


static inline void apply_signs(float* __restrict data, const uint8_t* __restrict signs_packed, size_t dim) {
    const uint8x8_t kBitMasks = vcreate_u8(0x8040201008040201ULL);
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
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
    for (; i < dim; i++) 
        if ((signs_packed[i / 8] >> (i % 8)) & 1) data[i] = -data[i];
}

// Compute 8 signed dot products simultaneously.
// proj_group points to the interleaved group base for these 8 rows:
//   projection_matrix + (p / 8) * rot_row_bytes * 8
// Layout: proj_group[byte_idx * 8 + r] = byte for row r at bit-position byte_idx.
// This allows a single vld1_u8 to load all 8 row-bytes per d-step instead of 8 scalar loads.
static void signed_dot(
    const uint8_t* __restrict proj_group,
    size_t rot_row_bytes,
    const float* __restrict vec,
    size_t dim,
    float* __restrict out  // length 8
) {
    const uint8x8_t kBitMasks = vcreate_u8(0x8040201008040201ULL);
    float32x4_t acc0[8], acc1[8];
    for (int r = 0; r < 8; r++) {
        acc0[r] = vdupq_n_f32(0.0f);
        acc1[r] = vdupq_n_f32(0.0f);
    }

    // dim is always a power-of-2 multiple of 8 (enforced by hadamard_inplace).
    for (size_t d = 0; d < dim; d += 8) {
        const size_t byte_idx = d / 8;
        float32x4_t v0 = vld1q_f32(vec + d);
        float32x4_t v1 = vld1q_f32(vec + d + 4);
        // Hoist reinterpret outside r-loop: same value used by every row.
        int32x4_t iv0 = vreinterpretq_s32_f32(v0);
        int32x4_t iv1 = vreinterpretq_s32_f32(v1);

        // One 8-byte load: all 8 row-bytes for this bit-position are contiguous.
        // vget_lane requires a compile-time constant, so extract to a stack array;
        // these 8 reads hit the store buffer / L1 immediately.
        uint8_t byte_arr[8];
        vst1_u8(byte_arr, vld1_u8(proj_group + byte_idx * 8));

        for (int r = 0; r < 8; r++) {
            int8x8_t sm   = vreinterpret_s8_u8(vtst_u8(vdup_n_u8(byte_arr[r]), kBitMasks));
            int16x8_t sm16 = vmovl_s8(sm);
            int32x4_t f0  = vshlq_n_s32(vmovl_s16(vget_low_s16(sm16)), 31);
            int32x4_t f1  = vshlq_n_s32(vmovl_high_s16(sm16),           31);
            acc0[r] = vaddq_f32(acc0[r], vreinterpretq_f32_s32(veorq_s32(iv0, f0)));
            acc1[r] = vaddq_f32(acc1[r], vreinterpretq_f32_s32(veorq_s32(iv1, f1)));
        }
    }

    for (int r = 0; r < 8; r++)
        out[r] = vaddvq_f32(vaddq_f32(acc0[r], acc1[r]));
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
    static const int32_t shift_arr[4] = {0, 2, 4, 6};
    const int32x4_t shift4 = vld1q_s32(shift_arr);
    size_t packed = 0;
    size_t i = 0;

    for (; i + 16 <= dim; i += 16) {
        int32x4_t c0 = vcvtmq_s32_f32(vfmaq_f32(two, vld1q_f32(&src[i]),      two));
        int32x4_t c1 = vcvtmq_s32_f32(vfmaq_f32(two, vld1q_f32(&src[i + 4]),  two));
        int32x4_t c2 = vcvtmq_s32_f32(vfmaq_f32(two, vld1q_f32(&src[i + 8]),  two));
        int32x4_t c3 = vcvtmq_s32_f32(vfmaq_f32(two, vld1q_f32(&src[i + 12]), two));
        c0 = vmaxq_s32(vminq_s32(c0, three_i), zero_i);
        c1 = vmaxq_s32(vminq_s32(c1, three_i), zero_i);
        c2 = vmaxq_s32(vminq_s32(c2, three_i), zero_i);
        c3 = vmaxq_s32(vminq_s32(c3, three_i), zero_i);

        uint32_t b0 = (uint32_t)vaddvq_s32(vshlq_s32(c0, shift4));
        uint32_t b1 = (uint32_t)vaddvq_s32(vshlq_s32(c1, shift4));
        uint32_t b2 = (uint32_t)vaddvq_s32(vshlq_s32(c2, shift4));
        uint32_t b3 = (uint32_t)vaddvq_s32(vshlq_s32(c3, shift4));
        uint32_t word = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | (b3 << 24);
        memcpy(dst + packed, &word, 4);
        packed += 4;
    }
    for (; i + 4 <= dim; i += 4) {
        int32x4_t c = vcvtmq_s32_f32(vfmaq_f32(two, vld1q_f32(&src[i]), two));
        c = vmaxq_s32(vminq_s32(c, three_i), zero_i);
        dst[packed++] = (uint8_t)vaddvq_s32(vshlq_s32(c, shift4));
    }
    if (i < dim) {
        uint8_t byte = 0;
        for (size_t j = 0; j < dim - i; j++) {
            int code = (int)floorf((src[i + j] + 1.0f) * 2.0f);
            code = std::max(0, std::min(3, code));
            byte |= (uint8_t)(code << (j * 2));
        }
        dst[packed] = byte;
    }
}

static void dequantize_2bit(const uint8_t* src, float* dst, size_t dim) {
    const uint32x4_t mask = vdupq_n_u32(3);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t offset = vdupq_n_f32(-0.75f);
    static const int32_t shift_arr[4] = {0, -2, -4, -6};
    const int32x4_t bit_shifts = vld1q_s32(shift_arr);
    size_t packed = 0;
    size_t i = 0;

    for (; i + 16 <= dim; i += 16) {
        uint32_t word;
        memcpy(&word, src + packed, 4); packed += 4;
        uint32x4_t bv0 = vdupq_n_u32( word        & 0xFF);
        uint32x4_t bv1 = vdupq_n_u32((word >>  8) & 0xFF);
        uint32x4_t bv2 = vdupq_n_u32((word >> 16) & 0xFF);
        uint32x4_t bv3 = vdupq_n_u32( word >> 24);
        vst1q_f32(&dst[i],      vfmaq_f32(offset, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv0, bit_shifts), mask)), half));
        vst1q_f32(&dst[i + 4],  vfmaq_f32(offset, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv1, bit_shifts), mask)), half));
        vst1q_f32(&dst[i + 8],  vfmaq_f32(offset, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv2, bit_shifts), mask)), half));
        vst1q_f32(&dst[i + 12], vfmaq_f32(offset, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv3, bit_shifts), mask)), half));
    }
    for (; i + 4 <= dim; i += 4) {
        uint32x4_t bv = vdupq_n_u32(src[packed++]);
        vst1q_f32(&dst[i], vfmaq_f32(offset, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv, bit_shifts), mask)), half));
    }
    if (i < dim) {
        uint8_t byte = src[packed];
        for (size_t j = 0; j < dim - i; j++) {
            uint8_t code = (byte >> (j * 2)) & 0x03;
            dst[i + j] = (float)(2 * (int)code - 3) * 0.25f;
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

    // Pack projection_matrix in interleaved groups of 8 rows:
    // layout[group * rot_row_bytes * 8 + byte_idx * 8 + row_in_group]
    // All 8 row-bytes for a given byte_idx are contiguous, so signed_dot
    // can replace 8 scalar loads with one vld1_u8 per outer loop iteration.
    const size_t num_groups = (projection_dim + 7) / 8;
    for (size_t g = 0; g < num_groups; g++) {
        for (size_t byte_idx = 0; byte_idx < rot_bytes; byte_idx++) {
            uint64_t w = rng.next();
            std::memcpy(projection_matrix + (g * rot_bytes + byte_idx) * 8, &w, 8);
        }
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
    const size_t qjl_bytes    = turboquant_qjl_bytes_per_head(projection_dim);
    const size_t rot_row_bytes = head_dim / 8;

    CactusThreading::parallel_for(
        seq_len * kv_heads, CactusThreading::Thresholds::ELEMENT_WISE,
        [=](size_t start, size_t end) {
            alignas(16) float buf[512]; // Static stack arrays used over std::vector for lock-free parallel performance
            alignas(16) float residual[512];

            for (size_t idx = start; idx < end; idx++) {
                const __fp16* in = src + idx * head_dim;

                float32x4_t norm_acc0 = vdupq_n_f32(0.0f);
                float32x4_t norm_acc1 = vdupq_n_f32(0.0f);
                size_t d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    float16x8_t h0 = vld1q_f16(in + d);
                    float16x8_t h1 = vld1q_f16(in + d + 8);
                    float32x4_t a = vcvt_f32_f16(vget_low_f16(h0));
                    float32x4_t b = vcvt_f32_f16(vget_high_f16(h0));
                    float32x4_t c = vcvt_f32_f16(vget_low_f16(h1));
                    float32x4_t e = vcvt_f32_f16(vget_high_f16(h1));
                    vst1q_f32(buf + d,      a);
                    vst1q_f32(buf + d + 4,  b);
                    vst1q_f32(buf + d + 8,  c);
                    vst1q_f32(buf + d + 12, e);
                    norm_acc0 = vfmaq_f32(vfmaq_f32(norm_acc0, a, a), b, b);
                    norm_acc1 = vfmaq_f32(vfmaq_f32(norm_acc1, c, c), e, e);
                }
                for (; d + 8 <= head_dim; d += 8) {
                    float16x8_t h = vld1q_f16(in + d);
                    float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
                    float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
                    vst1q_f32(buf + d,     lo);
                    vst1q_f32(buf + d + 4, hi);
                    norm_acc0 = vfmaq_f32(vfmaq_f32(norm_acc0, lo, lo), hi, hi);
                }
                float norm_sq = vaddvq_f32(vaddq_f32(norm_acc0, norm_acc1));
                for (; d < head_dim; d++) {
                    float v = static_cast<float>(in[d]);
                    buf[d] = v;
                    norm_sq += v * v;
                }

                float radius = sqrtf(norm_sq);
                dst_radii[idx] = radius;

                if (radius > 1e-10f) {
                    const float inv_r = 1.0f / radius;
                    float32x4_t inv_r_vec = vdupq_n_f32(inv_r);
                    d = 0;
                    for (; d + 16 <= head_dim; d += 16) {
                        vst1q_f32(buf + d,      vmulq_f32(vld1q_f32(buf + d),      inv_r_vec));
                        vst1q_f32(buf + d + 4,  vmulq_f32(vld1q_f32(buf + d + 4),  inv_r_vec));
                        vst1q_f32(buf + d + 8,  vmulq_f32(vld1q_f32(buf + d + 8),  inv_r_vec));
                        vst1q_f32(buf + d + 12, vmulq_f32(vld1q_f32(buf + d + 12), inv_r_vec));
                    }
                    for (; d < head_dim; d++) buf[d] *= inv_r;
                }

                rotate_forward(buf, rotation_signs, head_dim);
                uint8_t* q_angles = dst_angles + idx * angles_bytes;
                quantize_2bit(buf, q_angles, head_dim);

                {
                    const uint32x4_t mask2   = vdupq_n_u32(3);
                    const float32x4_t half   = vdupq_n_f32(0.5f);
                    const float32x4_t offset = vdupq_n_f32(-0.75f);
                    static const int32_t shift_arr[4] = {0, -2, -4, -6};
                    const int32x4_t bit_shifts = vld1q_s32(shift_arr);
                    float32x4_t r_vec    = vdupq_n_f32(radius);
                    float32x4_t err_acc0 = vdupq_n_f32(0.0f);
                    float32x4_t err_acc1 = vdupq_n_f32(0.0f);
                    float32x4_t abs_acc0 = vdupq_n_f32(0.0f);
                    float32x4_t abs_acc1 = vdupq_n_f32(0.0f);
                    size_t packed = 0;
                    d = 0;

                    for (; d + 16 <= head_dim; d += 16) {
                        uint32_t word;
                        memcpy(&word, q_angles + packed, 4); packed += 4;
                        uint32x4_t bv0 = vdupq_n_u32( word        & 0xFF);
                        uint32x4_t bv1 = vdupq_n_u32((word >>  8) & 0xFF);
                        uint32x4_t bv2 = vdupq_n_u32((word >> 16) & 0xFF);
                        uint32x4_t bv3 = vdupq_n_u32( word >> 24);
                        float32x4_t dq0 = vfmaq_f32(offset, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv0, bit_shifts), mask2)), half);
                        float32x4_t dq1 = vfmaq_f32(offset, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv1, bit_shifts), mask2)), half);
                        float32x4_t dq2 = vfmaq_f32(offset, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv2, bit_shifts), mask2)), half);
                        float32x4_t dq3 = vfmaq_f32(offset, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv3, bit_shifts), mask2)), half);
                        float32x4_t e0 = vmulq_f32(vsubq_f32(vld1q_f32(buf + d),      dq0), r_vec);
                        float32x4_t e1 = vmulq_f32(vsubq_f32(vld1q_f32(buf + d + 4),  dq1), r_vec);
                        float32x4_t e2 = vmulq_f32(vsubq_f32(vld1q_f32(buf + d + 8),  dq2), r_vec);
                        float32x4_t e3 = vmulq_f32(vsubq_f32(vld1q_f32(buf + d + 12), dq3), r_vec);
                        vst1q_f32(residual + d,      e0);
                        vst1q_f32(residual + d + 4,  e1);
                        vst1q_f32(residual + d + 8,  e2);
                        vst1q_f32(residual + d + 12, e3);
                        err_acc0 = vfmaq_f32(vfmaq_f32(err_acc0, e0, e0), e1, e1);
                        err_acc1 = vfmaq_f32(vfmaq_f32(err_acc1, e2, e2), e3, e3);
                        abs_acc0 = vaddq_f32(abs_acc0, vabsq_f32(e0));
                        abs_acc0 = vaddq_f32(abs_acc0, vabsq_f32(e1));
                        abs_acc1 = vaddq_f32(abs_acc1, vabsq_f32(e2));
                        abs_acc1 = vaddq_f32(abs_acc1, vabsq_f32(e3));
                    }
                    for (; d + 4 <= head_dim; d += 4) {
                        uint32x4_t bv = vdupq_n_u32(q_angles[packed++]);
                        float32x4_t dq0 = vfmaq_f32(offset, vcvtq_f32_u32(vandq_u32(vshlq_u32(bv, bit_shifts), mask2)), half);
                        float32x4_t e = vmulq_f32(vsubq_f32(vld1q_f32(buf + d), dq0), r_vec);
                        vst1q_f32(residual + d, e);
                        err_acc0 = vfmaq_f32(err_acc0, e, e);
                        abs_acc0 = vaddq_f32(abs_acc0, vabsq_f32(e));
                    }

                    float err_sq = vaddvq_f32(vaddq_f32(err_acc0, err_acc1));
                    float sum_abs_res = vaddvq_f32(vaddq_f32(abs_acc0, abs_acc1)) + 1e-10f;

                    // Can delete (for cases where head_dim is not a multiple of 4 (not possible for hadamard))
                    for (; d < head_dim; d++) {
                        uint8_t code = (q_angles[d / 4] >> ((d % 4) * 2)) & 0x3;
                        float dq_v = (float)(2 * (int)code - 3) * 0.25f;
                        float e = (buf[d] - dq_v) * radius;
                        residual[d] = e;
                        err_sq      += e * e;
                        sum_abs_res += std::abs(e);
                    }

                    dst_error_norms[idx] = sqrtf(err_sq);

                    uint8_t* bits = dst_qjl_bits + idx * qjl_bytes;
                    std::memset(bits, 0, qjl_bytes);
                    Xoshiro256 rng(1337 + idx);

                    const float inv_sum = 1.0f / sum_abs_res;

                    size_t p = 0;
                    for (; p + 8 <= projection_dim; p += 8) {
                        float dots[8];
                        signed_dot(projection_matrix + (p / 8) * rot_row_bytes * 8, rot_row_bytes, residual, head_dim, dots);

                        uint64_t rand_word = rng.next();
                        for (int r = 0; r < 8; r++) {
                            float prob = (dots[r] * inv_sum + 1.0f) * 0.5f;
                            prob = prob < 0.0f ? 0.0f : (prob > 1.0f ? 1.0f : prob);
                            uint32_t raw = (uint32_t)(rand_word >> (r * 8)) & 0xFF;
                            float rand_f = raw * (1.0f / 256.0f);
                            if (rand_f < prob) {
                                size_t slot = p + r;
                                bits[slot / 8] |= (1u << (slot % 8));
                            }
                        }
                    }
                    for (; p < projection_dim; p++) {
                        // Interleaved layout: proj[group * rot_row_bytes * 8 + byte_idx * 8 + row_in_group]
                        const size_t group        = p / 8;
                        const size_t row_in_group = p % 8;
                        const uint8_t* grp        = projection_matrix + group * rot_row_bytes * 8;
                        float dot = 0.0f;
                        for (size_t d = 0; d < head_dim; d += 8) {
                            uint8_t byte = grp[(d / 8) * 8 + row_in_group];
                            for (int b = 0; b < 8; b++)
                                dot += ((byte >> b) & 1) ? -residual[d + b] : residual[d + b];
                        }
                        float prob = (dot * inv_sum + 1.0f) * 0.5f;
                        prob = prob < 0.0f ? 0.0f : (prob > 1.0f ? 1.0f : prob);
                        float rand_f = (rng.next() & 0xFFFFFF) / static_cast<float>(0x1000000);
                        if (rand_f < prob) bits[p / 8] |= (1u << (p % 8));
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
            alignas(16) float buf[512];
            for (size_t idx = start; idx < end; idx++) {
                float radius = radii[idx];
                dequantize_2bit(angles + idx * angles_bytes, buf, head_dim);
                float32x4_t r_vec = vdupq_n_f32(radius);
                size_t d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    vst1q_f32(&buf[d],      vmulq_f32(vld1q_f32(&buf[d]),      r_vec));
                    vst1q_f32(&buf[d + 4],  vmulq_f32(vld1q_f32(&buf[d + 4]),  r_vec));
                    vst1q_f32(&buf[d + 8],  vmulq_f32(vld1q_f32(&buf[d + 8]),  r_vec));
                    vst1q_f32(&buf[d + 12], vmulq_f32(vld1q_f32(&buf[d + 12]), r_vec));
                }
                for (; d + 4 <= head_dim; d += 4) {
                    vst1q_f32(&buf[d], vmulq_f32(vld1q_f32(&buf[d]), r_vec));
                }
                for (; d < head_dim; d++) buf[d] *= radius;

                rotate_inverse(buf, rotation_signs, head_dim);
                __fp16* out = dst + idx * head_dim;
                d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    vst1q_f16(&out[d],     vcombine_f16(vcvt_f16_f32(vld1q_f32(&buf[d])),      vcvt_f16_f32(vld1q_f32(&buf[d + 4]))));
                    vst1q_f16(&out[d + 8], vcombine_f16(vcvt_f16_f32(vld1q_f32(&buf[d + 8])), vcvt_f16_f32(vld1q_f32(&buf[d + 12]))));
                }
                for (; d + 8 <= head_dim; d += 8) {
                    vst1q_f16(&out[d], vcombine_f16(vcvt_f16_f32(vld1q_f32(&buf[d])), vcvt_f16_f32(vld1q_f32(&buf[d + 4]))));
                }
                for (; d < head_dim; d++) out[d] = static_cast<__fp16>(buf[d]);
            }
        });
}

static inline float fast_sin_half_pi(float x) {
    float x2 = x * x;
    return x * (1.0f - x2 * (1.0f/6.0f - x2 * (1.0f/120.0f)));
}

static inline size_t xnor_popcount(const uint8_t* a, const uint8_t* b, size_t nbytes) {
    uint32x4_t acc = vdupq_n_u32(0);
    size_t i = 0;
    for (; i + 16 <= nbytes; i += 16) {
        uint8x16_t xnor = vmvnq_u8(veorq_u8(vld1q_u8(a + i), vld1q_u8(b + i)));
        acc = vpadalq_u16(acc, vpaddlq_u8(vcntq_u8(xnor)));
    }
    size_t result = vaddvq_u32(acc);
    for (; i < nbytes; i++)
        result += __builtin_popcount(~(a[i] ^ b[i]) & 0xFF);
    return result;
}

static inline float turboquant_score(
    const __fp16* q_rot, float q_norm,
    const uint8_t* q_qjl, float radius,
    const uint8_t* angs, float err_norm,
    const uint8_t* k_qjl,
    float* dq, size_t head_dim, size_t qjl_bytes, size_t projection_dim
) {
    dequantize_2bit(angs, dq, head_dim);

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t d = 0;
    for (; d + 16 <= head_dim; d += 16) {
        float16x8_t q0 = vld1q_f16(q_rot + d);
        float16x8_t q1 = vld1q_f16(q_rot + d + 8);
        acc0 = vfmaq_f32(vfmaq_f32(acc0,
            vcvt_f32_f16(vget_low_f16(q0)),  vld1q_f32(dq + d)),
            vcvt_f32_f16(vget_high_f16(q0)), vld1q_f32(dq + d + 4));
        acc1 = vfmaq_f32(vfmaq_f32(acc1,
            vcvt_f32_f16(vget_low_f16(q1)),  vld1q_f32(dq + d + 8)),
            vcvt_f32_f16(vget_high_f16(q1)), vld1q_f32(dq + d + 12));
    }
    for (; d + 8 <= head_dim; d += 8) {
        float16x8_t q = vld1q_f16(q_rot + d);
        acc0 = vfmaq_f32(vfmaq_f32(acc0,
            vcvt_f32_f16(vget_low_f16(q)),  vld1q_f32(dq + d)),
            vcvt_f32_f16(vget_high_f16(q)), vld1q_f32(dq + d + 4));
    }
    float polar_dot = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; d < head_dim; d++) polar_dot += static_cast<float>(q_rot[d]) * dq[d];
    polar_dot *= radius;

    size_t matches  = xnor_popcount(k_qjl, q_qjl, qjl_bytes);
    float avg_sign  = (2.0f * static_cast<float>(matches) - static_cast<float>(projection_dim))
                      / static_cast<float>(projection_dim);
    float correction = q_norm * err_norm * fast_sin_half_pi(PI_F * 0.5f * avg_sign);
    return polar_dot + correction;
}

void cactus_turboquant_decode_dot_fp16(
    const __fp16* query_rotated, const float* query_norms, const uint8_t* query_qjl_bits,
    const float* cached_radii, const uint8_t* cached_angles,
    const float* cached_error_norms, const uint8_t* cached_qjl_bits,
    float* output, size_t kv_seq_len, size_t num_q_heads, size_t num_kv_heads,
    size_t head_dim, size_t angle_bits, size_t projection_dim, float scale
) {
    const size_t angles_bytes = turboquant_angles_bytes_per_head(head_dim, angle_bits);
    const size_t qjl_bytes    = turboquant_qjl_bytes_per_head(projection_dim);
    const size_t group_size   = num_q_heads / num_kv_heads;

    CactusThreading::parallel_for(
        num_q_heads, CactusThreading::Thresholds::ELEMENT_WISE,
        [=](size_t start, size_t end) {
            alignas(16) float dq[512];

            for (size_t q_head = start; q_head < end; q_head++) {
                const size_t   kv_head = q_head / group_size;
                const __fp16*  q_rot   = query_rotated  + q_head * head_dim;
                const float    q_norm  = query_norms[q_head];
                const uint8_t* q_qjl  = query_qjl_bits + q_head * qjl_bytes;
                float*         out_row = output          + q_head * kv_seq_len;

                for (size_t kv_pos = 0; kv_pos < kv_seq_len; kv_pos++) {
                    const size_t idx = kv_pos * num_kv_heads + kv_head;
                    out_row[kv_pos] = turboquant_score(
                        q_rot, q_norm, q_qjl,
                        cached_radii[idx],
                        cached_angles      + idx * angles_bytes,
                        cached_error_norms[idx],
                        cached_qjl_bits    + idx * qjl_bytes,
                        dq, head_dim, qjl_bytes, projection_dim
                    ) * scale;
                }
            }
        });
}

float cactus_turboquant_dot(
    const __fp16* q_rot, float q_norm, float radius, const uint8_t* angles, float error_norm,
    const uint8_t* qjl_bits, const uint8_t* query_qjl_bits, size_t head_dim, size_t projection_dim
) {
    alignas(16) float dq[512];
    const size_t qjl_bytes = turboquant_qjl_bytes_per_head(projection_dim);
    return turboquant_score(
        q_rot, q_norm, query_qjl_bits,
        radius, angles, error_norm, qjl_bits,
        dq, head_dim, qjl_bytes, projection_dim
    );
}


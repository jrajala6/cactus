#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

/**
 * --- SIMD MOCK LAYER ---
 * Because we are on a standard computer and not necessarily an ARM CPU,
 * we will "Mock" (fake) the ARM NEON types. This lets us write code that
 * LOOKS like a kernel but runs as a standard C++ program for study.
 */

// 128-bit 'Buckets' (Vectors)
struct float32x4_t {
  float val[4];
};
struct int32x4_t {
  int32_t val[4];
};
struct int8x16_t {
  int8_t val[16];
};
struct float16x4_t {
  uint16_t val[4];
};

// A simple mock for __fp16 (standard C++ doesn't always have this natively)
typedef uint16_t __fp16;

// Mock Functions (Intrinsics)
// vdupq_n_f32: "Duplicate" (Fill a bucket with one value)
inline float32x4_t vdupq_n_f32(float v) { return {v, v, v, v}; }
inline int32x4_t vdupq_n_s32(int32_t v) { return {v, v, v, v}; }

// vld1q_s8: "Load" (Grab 16 signed bytes from memory)
inline int8x16_t vld1q_s8(const int8_t *p) {
  int8x16_t r;
  for (int i = 0; i < 16; ++i)
    r.val[i] = p[i];
  return r;
}
inline float16x4_t vld1_f16(const __fp16 *p) {
  float16x4_t r;
  for (int i = 0; i < 4; ++i)
    r.val[i] = p[i];
  return r;
}

// vcvt: "Convert" (Change data types, like Integer to Float)
inline float32x4_t vcvt_f32_f16(float16x4_t v) {
  return {(float)v.val[0], (float)v.val[1], (float)v.val[2], (float)v.val[3]};
}
inline float32x4_t vcvtq_f32_s32(int32x4_t v) {
  return {(float)v.val[0], (float)v.val[1], (float)v.val[2], (float)v.val[3]};
}

// vmlaq: "Multiply-Accumulate" (Multiply B*C and add to A)
inline float32x4_t vmlaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  return {a.val[0] + b.val[0] * c.val[0], a.val[1] + b.val[1] * c.val[1],
          a.val[2] + b.val[2] * c.val[2], a.val[3] + b.val[3] * c.val[3]};
}
inline float32x4_t vmulq_n_f32(float32x4_t a, float b) {
  return {a.val[0] * b, a.val[1] * b, a.val[2] * b, a.val[3] * b};
}
inline float16x4_t vcvt_f16_f32(float32x4_t v) {
  return {(uint16_t)v.val[0], (uint16_t)v.val[1], (uint16_t)v.val[2],
          (uint16_t)v.val[3]};
}

// vst1: "Store" (Write the bucket back to RAM)
inline void vst1_f16(__fp16 *p, float16x4_t v) {
  for (int i = 0; i < 4; ++i)
    p[i] = v.val[i];
}

// Define the "Magic Stamp" (Dot Product)
// In a real ARM CPU, this is a single instruction. Here we mock it with a loop.
#define CACTUS_DOTQ_LANE(acc, b, a, lane) mock_dot_product(acc, b, a, lane)
inline int32x4_t mock_dot_product(int32x4_t acc, int8x16_t b, int8x16_t a,
                                  int lane) {
  for (int i = 0; i < 4; ++i) {   // For each of our 4 results
    for (int k = 0; k < 4; ++k) { // Multiply 4 numbers in this lane
      acc.val[i] += (int32_t)b.val[k] * (int32_t)a.val[lane * 4 + k];
    }
  }
  return acc;
}

// Mock Threading Namespace
namespace CactusThreading {
struct MockPool {
  int num_workers() { return 1; }
  template <typename F> void enqueue_n_threads(size_t n, int t, F f) {
    f(0, n);
  }
  void wait_all() {}
};
MockPool &get_thread_pool() {
  static MockPool p;
  return p;
}
namespace GemmThreading {
inline int get_gemv_threads(size_t n, int w) { return 1; }
} // namespace GemmThreading
} // namespace CactusThreading

/**
 * STEP 1: THE NIBBLE EXTRACTOR (Standard C++)
 * ----------------------------------------
 * This shows how a computer "sees" into the 4-bit numbers.
 */
void explain_nibbles(uint8_t byte) {
  int8_t low = byte & 0x0F;
  int8_t high = byte >> 4;

  // Sign extension: Make sure negative 4-bit numbers stay negative in 8-bit
  if (low & 0x08)
    low |= 0xF0;
  if (high & 0x08)
    high |= 0xF0;

  std::cout << "Byte: 0x" << std::hex << (int)byte << std::dec
            << " -> Low: " << (int)low << ", High: " << (int)high << std::endl;
}

/**
 * THE DOCUMENTED KERNEL: cactus_gemv_int8
 * --------------------------------------
 * This is the high-performance assembly room. Follow the numbers!
 */
void cactus_gemv_int8(
    const int8_t *A,        // [INPUT] A long vector of activations
    const float A_scale,    // [INPUT] Scale factor for A
    const int8_t *B,        // [INPUT] The heavy weight matrix (INT8)
    const __fp16 *B_scales, // [INPUT] Scales for every group of weights
    __fp16 *C,              // [OUTPUT] Where the result goes
    size_t K, size_t N,     // K=Vector Length, N=Number of Outputs
    size_t group_size       // Group size for quantization (e.g. 32)
) {
  if (K == 0 || N == 0)
    return;

  // [1] DIVIDE AND CONQUER
  // N is the number of outputs. Since we calculate 4 outputs at once
  // using SIMD (128-bit), we divide the work into "blocks" of 4.
  const size_t num_groups = K / group_size;
  const size_t N_blocks = (N + 3) / 4;

  // [2] THE PARALLEL ENGINE
  // This lambda defines what EACH CPU core will do.
  auto process_blocks = [=](size_t block_start, size_t block_end) {
    // Loop through the blocks assigned to this specific core
    for (size_t n_block = block_start; n_block < block_end; ++n_block) {

      const size_t n_start = n_block * 4;
      const size_t actual_n = std::min(size_t(4), N - n_start);

      // 'running_sum' is our 128-bit bucket for the 4 final decimal results.
      float32x4_t running_sum = vdupq_n_f32(0.0f);

      size_t g = 0;
      // [3] INNER MATH LOOP: Moving down the Vector (K)
      for (; g + 1 < num_groups; g += 2) {

        // Memory math: Where do A and B start for this group?
        const size_t k_base0 = g * group_size;
        const int8_t *a_ptr0 = A + k_base0;

        // B is stored interleaved: 16 weights for col 1, then 16 for col 2...
        const int8_t *b_base0 = B + (n_block * K + k_base0) * 4;

        // PREFETCH: "Order the data" from RAM so it's ready before we need it.
        __builtin_prefetch(b_base0 + group_size * 8, 0, 3);

        // 'acc0' is our temporary bucket for the INTEGER sums.
        int32x4_t acc0 = vdupq_n_s32(0);

        {
          // LOAD: Grab 16 units of A and 64 units of B weights
          int8x16_t a_vec = vld1q_s8(a_ptr0);
          int8x16_t b0 = vld1q_s8(b_base0);      // weights for output #1
          int8x16_t b1 = vld1q_s8(b_base0 + 16); // weights for output #2
          int8x16_t b2 = vld1q_s8(b_base0 + 32); // weights for output #3
          int8x16_t b3 = vld1q_s8(b_base0 + 48); // weights for output #4

          // DOT PRODUCT (The "Stamp"):
          // Multiply 16 numbers and add them to the sum in ONE STEP.
          acc0 = CACTUS_DOTQ_LANE(acc0, b0, a_vec, 0);
          acc0 = CACTUS_DOTQ_LANE(acc0, b1, a_vec, 1);
          acc0 = CACTUS_DOTQ_LANE(acc0, b2, a_vec, 2);
          acc0 = CACTUS_DOTQ_LANE(acc0, b3, a_vec, 3);
        }

        // [4] DE-QUANTIZATION (Decimal Recovery)
        // RealValue = IntegerResult * WeightScale
        const __fp16 *scale_ptr0 = B_scales + (n_block * num_groups + g) * 4;
        float16x4_t scales0_f16 = vld1_f16(scale_ptr0);
        float32x4_t scales0 = vcvt_f32_f16(scales0_f16);

        // Multiply and Accumulate (MLA) into our running float total.
        running_sum = vmlaq_f32(running_sum, vcvtq_f32_s32(acc0), scales0);
      }

      // [5] THE FINAL FINISH
      // After summing everything, we multiply by the A_scale.
      float32x4_t result = vmulq_n_f32(running_sum, A_scale);
      float16x4_t result_f16 = vcvt_f16_f32(result);

      // STORE: Write the final 4 numbers to C.
      if (actual_n == 4) {
        vst1_f16(C + n_start, result_f16);
      }
    }
  };

  // [6] RUN IT!
  auto &pool = CactusThreading::get_thread_pool();
  pool.enqueue_n_threads(N_blocks, 1, process_blocks);
  pool.wait_all();
}

int main() {
  std::cout << "--- Kernel Learning Lab ---" << std::endl;
  explain_nibbles(0x1F); // Try a test nibble

  std::cout
      << "\nSuccess! The documented kernel above is now a valid C++ snippet."
      << std::endl;
  return 0;
}

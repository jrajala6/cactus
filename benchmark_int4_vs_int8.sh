#!/bin/bash
# Gemma Model INT4 vs INT8 Performance Benchmark Script
# This script builds and runs the performance tests using real Gemma models

echo "🚀 Gemma Model INT4 vs INT8 Performance Benchmark"
echo "================================================="
echo ""

# Check if we're in the cactus directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ Please run this script from the cactus root directory"
    exit 1
fi

# Check if Gemma model path is set
if [ -z "$GEMMA_MODEL_PATH" ]; then
    echo "❌ GEMMA_MODEL_PATH environment variable not set"
    echo "Please set it to the path of your Gemma 270M model directory"
    echo ""
    echo "Example:"
    echo "  export GEMMA_MODEL_PATH=/path/to/your/gemma-270m-model"
    echo "  ./benchmark_int4_vs_int8.sh"
    echo ""
    echo "Or run with:"
    echo "  GEMMA_MODEL_PATH=/path/to/model ./benchmark_int4_vs_int8.sh"
    echo ""
    exit 1
fi

# Check if model exists
if [ ! -d "$GEMMA_MODEL_PATH" ]; then
    echo "❌ Gemma model directory not found: $GEMMA_MODEL_PATH"
    echo "Please check that the path is correct"
    exit 1
fi

echo "📁 Using Gemma model: $GEMMA_MODEL_PATH"
echo ""

echo "📋 Building Cactus with optimizations..."
cd cactus
mkdir -p build
cd build

# Configure with optimizations
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native -DNDEBUG"

# Build
echo "🔨 Building..."
if ! make -j$(nproc) cactus_ffi; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build successful"
echo ""

# Check if test executable exists
if [ ! -f "../../tests/test_performance" ]; then
    echo "📋 Building performance tests..."
    cd ../../tests
    if ! make test_performance; then
        echo "❌ Test build failed"
        exit 1
    fi
    cd ../cactus/build
fi

echo "🎯 Running Gemma Model INT4 vs INT8 Performance Tests..."
echo "Expected Results:"
echo "  • INT4 should show 1.3-1.5x speedup over INT8"
echo "  • INT4 should use ~50% less memory for weights"
echo "  • Model loading should be faster with INT4"
echo ""

# Run the performance tests and filter for Gemma model results
echo "🏃 Running benchmarks..."
echo ""

../../tests/test_performance 2>&1 | grep -E "(Gemma|GEMMA|Inference|Loading|INT4|INT8|Speedup|tok/s|Performance Benchmarks|====)"

echo ""
echo "🎉 Benchmark completed!"
echo ""
echo "📊 Key Metrics to Look For:"
echo "   • Inference Speedup: INT4 should be 1.3-1.5x faster than INT8"
echo "   • Tokens/second: Higher for INT4 mode"
echo "   • Model Loading: Faster with INT4 due to less unpacking"
echo "   • Memory Usage: 50% reduction in weight storage"
echo ""
echo "💡 To test with different model sizes:"
echo "   • Set GEMMA_MODEL_PATH to different Gemma variants"
echo "   • Models tested: Gemma 270M, 2B, 9B, 27B"
echo ""
echo "🔧 If performance is lower than expected:"
echo "   • Check model has INT4 quantized weights"
echo "   • Verify ARM64 with dotprod instructions"
echo "   • Ensure model files are on fast storage (SSD)"
cmake_minimum_required(VERSION 3.21)
project(LunaUltimate VERSION 0.1.0 LANGUAGES CXX CUDA)

# ==================== Build Options ====================
# Control which backends to compile:
#   cmake -DUSE_CUDA=ON -DUSE_TRITON=ON ..
#   cmake -DUSE_ROCM=ON -DUSE_TRITON=ON ..
#   cmake -DUSE_APPLE=ON ..
#   cmake -DUSE_CANN=ON ..
#   cmake -DUSE_DML=ON ..
#   cmake -DUSE_CPU=ON ..

option(USE_CUDA   "Build with CUDA backend"   OFF)
option(USE_ROCM   "Build with ROCm backend"   OFF)
option(USE_TRITON "Build Triton kernels"      OFF)
option(USE_APPLE  "Build with Metal/MLX"      OFF)
option(USE_CANN   "Build with CANN backend"   OFF)
option(USE_DML    "Build with DirectML"       OFF)
option(USE_CPU    "Build CPU-only backend"    ON)

# ==================== Auto-detect ====================
if(NOT DEFINED USE_CUDA AND NOT DEFINED USE_ROCM AND NOT DEFINED USE_APPLE
   AND NOT DEFINED USE_CANN)
    # Auto-detect CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(USE_CUDA ON)
        set(USE_TRITON ON)
        message(STATUS "Auto-detected CUDA: ${CUDAToolkit_VERSION}")
    elseif(APPLE)
        set(USE_APPLE ON)
        message(STATUS "Auto-detected Apple Silicon")
    else()
        set(USE_CPU ON)
        message(STATUS "No GPU detected, using CPU backend")
    endif()
endif()

# ==================== C++ Standard ====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ==================== Python ====================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ==================== CUDA Backend ====================
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Custom CUDA kernels for Mamba2 scan
    add_library(luna_cuda_kernels SHARED
        csrc/cuda/mamba2_scan.cu
        csrc/cuda/flash_attention_mla.cu
        csrc/cuda/moe_topk_gate.cu
    )
    target_include_directories(luna_cuda_kernels PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(luna_cuda_kernels
        ${CUDAToolkit_LIBRARIES}
    )
    set_target_properties(luna_cuda_kernels PROPERTIES
        CUDA_ARCHITECTURES "80;86;89;90"  # Ampere, Hopper, Ada
    )

    # PyBind11 bridge
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.0
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(luna_cuda_bridge
        csrc/bridge/cuda_bridge.cpp
    )
    target_link_libraries(luna_cuda_bridge PRIVATE luna_cuda_kernels)
endif()

# ==================== ROCm Backend ====================
if(USE_ROCM)
    # ROCm uses HIP compiler, nearly identical to CUDA source
    enable_language(HIP)
    find_package(hip REQUIRED)

    add_library(luna_rocm_kernels SHARED
        csrc/rocm/mamba2_scan.hip
        csrc/rocm/flash_attention_mla.hip
    )
    target_link_libraries(luna_rocm_kernels
        hip::device
    )
endif()

# ==================== Apple Metal Backend ====================
if(USE_APPLE)
    # Metal shaders for Mamba2 and attention
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)

    add_library(luna_metal_kernels SHARED
        csrc/metal/mamba2_scan.mm
        csrc/metal/flash_attention.mm
    )
    target_link_libraries(luna_metal_kernels
        ${METAL_LIBRARY}
        ${FOUNDATION_LIBRARY}
    )
    set_target_properties(luna_metal_kernels PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
endif()

# ==================== CPU Backend ====================
if(USE_CPU)
    # Vectorized CPU kernels (AVX2/AVX512/NEON)
    add_library(luna_cpu_kernels SHARED
        csrc/cpu/mamba2_scan.cpp
        csrc/cpu/flash_attention.cpp
    )
    target_compile_options(luna_cpu_kernels PRIVATE
        -O3 -march=native
    )
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        target_compile_definitions(luna_cpu_kernels PRIVATE USE_AVX2)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        target_compile_definitions(luna_cpu_kernels PRIVATE USE_NEON)
    endif()
endif()

# ==================== Install ====================
install(TARGETS luna_cuda_kernels luna_cuda_bridge
    LIBRARY DESTINATION lib
    OPTIONAL
)
install(TARGETS luna_rocm_kernels
    LIBRARY DESTINATION lib
    OPTIONAL
)
install(TARGETS luna_metal_kernels
    LIBRARY DESTINATION lib
    OPTIONAL
)
install(TARGETS luna_cpu_kernels
    LIBRARY DESTINATION lib
    OPTIONAL
)

# ==================== Build Info ====================
message(STATUS "")
message(STATUS "Luna-Ultimate Build Configuration:")
message(STATUS "  CUDA:     ${USE_CUDA}")
message(STATUS "  ROCm:     ${USE_ROCM}")
message(STATUS "  Triton:   ${USE_TRITON}")
message(STATUS "  Apple:    ${USE_APPLE}")
message(STATUS "  CANN:     ${USE_CANN}")
message(STATUS "  DML:      ${USE_DML}")
message(STATUS "  CPU:      ${USE_CPU}")
message(STATUS "")
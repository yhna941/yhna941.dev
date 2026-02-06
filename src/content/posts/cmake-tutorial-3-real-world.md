---
title: "Modern CMake Tutorial 3편: 유명 오픈소스 프로젝트는 CMake를 어떻게 쓸까?"
description: "LLVM, PyTorch, OpenCV 등 대규모 C++ 프로젝트들의 CMake 구조를 분석하고, 실전 팁을 배워봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["cmake", "cpp", "opensource", "best-practices"]
draft: false
---

# Modern CMake Tutorial 3편: 유명 프로젝트는 어떻게 쓸까?

이론은 충분히 배웠습니다. 이제 **실제 세계**를 봅시다. LLVM, PyTorch, OpenCV 같은 거대 프로젝트들은 CMake를 어떻게 쓰고 있을까요?

---

## 1. PyTorch: 거대 ML 프레임워크

PyTorch는 C++ 백엔드(libtorch)와 Python 프론트엔드로 구성된 복잡한 프로젝트입니다.

### 프로젝트 구조

```
pytorch/
├── CMakeLists.txt          # 루트
├── cmake/
│   ├── Dependencies.cmake   # 의존성 관리
│   ├── public/             # 사용자용 CMake 모듈
│   └── Modules/            # 내부 CMake 모듈
├── torch/                  # C++ 라이브러리
├── caffe2/                 # 레거시 백엔드
├── aten/                   # Tensor 라이브러리
└── third_party/            # 서브모듈
```

### 핵심 패턴 1: 옵션 관리

**CMakeLists.txt:**
```cmake
# 빌드 옵션들
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_CAFFE2 "Build Caffe2" ON)
option(USE_CUDA "Use CUDA" ON)
option(USE_ROCM "Use ROCm for AMD GPUs" OFF)
option(USE_MKLDNN "Use MKLDNN" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# 플랫폼별 기본값
if(ANDROID)
  set(BUILD_PYTHON OFF)
endif()

# 옵션에 따라 컴파일 정의
if(USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()
```

**사용:**
```bash
# CUDA 없이 빌드
cmake -B build -DUSE_CUDA=OFF

# Python 없이 C++만
cmake -B build -DBUILD_PYTHON=OFF

# 정적 라이브러리로
cmake -B build -DBUILD_SHARED_LIBS=OFF
```

### 핵심 패턴 2: 의존성 관리

**cmake/Dependencies.cmake:**
```cmake
# 조건부 의존성
if(USE_CUDA)
  find_package(CUDA 10.0 REQUIRED)
  
  # CUDA 아키텍처 설정
  if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "3.5;5.0;6.0;7.0;7.5;8.0;8.6")
  endif()
  
  foreach(ARCH ${TORCH_CUDA_ARCH_LIST})
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${ARCH},code=sm_${ARCH}")
  endforeach()
endif()

if(USE_MKLDNN)
  find_package(MKLDNN QUIET)
  if(NOT MKLDNN_FOUND)
    # 없으면 직접 빌드
    add_subdirectory(third_party/ideep)
  endif()
endif()
```

### 핵심 패턴 3: 조건부 소스 추가

```cmake
# 기본 소스
set(TORCH_SRCS
  torch/csrc/autograd/engine.cpp
  torch/csrc/autograd/function.cpp
  # ...
)

# CUDA 소스 추가
if(USE_CUDA)
  list(APPEND TORCH_SRCS
    torch/csrc/cuda/comm.cpp
    torch/csrc/cuda/nccl.cpp
  )
endif()

# ROCm 소스 추가
if(USE_ROCM)
  list(APPEND TORCH_SRCS
    torch/csrc/hip/comm.cpp
  )
endif()

add_library(torch ${TORCH_SRCS})
```

### 핵심 패턴 4: Python 바인딩

```cmake
if(BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  
  # pybind11 사용
  add_subdirectory(third_party/pybind11)
  
  pybind11_add_module(_C
    torch/csrc/Module.cpp
    torch/csrc/autograd/python_autograd.cpp
    # ...
  )
  
  target_link_libraries(_C PRIVATE torch)
  
  # Python 패키지 경로에 설치
  install(TARGETS _C
    LIBRARY DESTINATION torch/lib
  )
endif()
```

### 배울 점

1. **옵션 활용**: 사용자가 원하는 기능만 빌드
2. **조건부 컴파일**: 플랫폼/기능별 소스 관리
3. **의존성 Fallback**: 없으면 직접 빌드
4. **명확한 구조**: `cmake/` 디렉토리로 모듈화

---

## 2. LLVM: 컴파일러 인프라

LLVM은 컴파일러, 링커, 디버거 등 수십 개의 도구를 포함한 거대 프로젝트입니다.

### 프로젝트 구조

```
llvm-project/
├── llvm/
│   ├── CMakeLists.txt
│   ├── cmake/modules/       # CMake 모듈들
│   ├── lib/                 # 라이브러리들
│   └── tools/               # 도구들
├── clang/                   # C++ 컴파일러
├── lld/                     # 링커
└── lldb/                    # 디버거
```

### 핵심 패턴 1: 모듈식 빌드

**llvm/CMakeLists.txt:**
```cmake
# 빌드할 프로젝트 선택
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING
  "Semicolon-separated list of projects to build")

# 각 프로젝트를 서브디렉토리로 추가
foreach(proj ${LLVM_ENABLE_PROJECTS})
  if(EXISTS ${CMAKE_SOURCE_DIR}/${proj})
    add_subdirectory(${proj})
  endif()
endforeach()
```

**사용:**
```bash
# Clang만 빌드
cmake -B build -DLLVM_ENABLE_PROJECTS="clang"

# Clang + LLD + LLDB
cmake -B build -DLLVM_ENABLE_PROJECTS="clang;lld;lldb"
```

### 핵심 패턴 2: 타겟별 옵션

```cmake
# 빌드할 타겟 아키텍처
set(LLVM_TARGETS_TO_BUILD "X86;ARM;AArch64" CACHE STRING
  "Semicolon-separated list of targets to build")

foreach(target ${LLVM_TARGETS_TO_BUILD})
  add_subdirectory(lib/Target/${target})
endforeach()
```

**사용:**
```bash
# x86만
cmake -B build -DLLVM_TARGETS_TO_BUILD="X86"

# ARM만 (크로스 컴파일용)
cmake -B build -DLLVM_TARGETS_TO_BUILD="ARM;AArch64"
```

### 핵심 패턴 3: TableGen (코드 생성)

LLVM은 빌드 타임에 코드를 생성합니다.

```cmake
# TableGen 도구
add_executable(llvm-tblgen
  utils/TableGen/TableGen.cpp
  utils/TableGen/CodeGenTarget.cpp
)

# 코드 생성 함수
function(tablegen output_file)
  add_custom_command(
    OUTPUT ${output_file}
    COMMAND llvm-tblgen ${ARGN}
    DEPENDS llvm-tblgen
    COMMENT "Building ${output_file}"
  )
endfunction()

# 사용 예
tablegen(IntrinsicsX86.h
  -gen-intrinsic-enums
  -intrinsic-prefix=x86
  X86.td
)
```

### 핵심 패턴 4: LLVM 라이브러리 매크로

**cmake/modules/AddLLVM.cmake:**
```cmake
# LLVM 라이브러리 추가 헬퍼
function(add_llvm_library name)
  cmake_parse_arguments(ARG
    "SHARED;STATIC"
    ""
    "LINK_LIBS;DEPENDS"
    ${ARGN}
  )
  
  add_library(${name} ${ARG_UNPARSED_ARGUMENTS})
  
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} ${ARG_LINK_LIBS})
  endif()
  
  # LLVM 공통 설정 적용
  llvm_update_compile_flags(${name})
  
  install(TARGETS ${name}
    EXPORT LLVMExports
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  )
endfunction()
```

**사용:**
```cmake
add_llvm_library(LLVMCore
  IR/BasicBlock.cpp
  IR/Function.cpp
  IR/Module.cpp
  LINK_LIBS LLVMSupport
)
```

### 배울 점

1. **모듈식 빌드**: 필요한 것만 선택
2. **코드 생성**: `add_custom_command`로 빌드타임 코드 생성
3. **헬퍼 함수**: 반복 작업을 함수로 추상화
4. **명명 규칙**: `LLVM` 접두사로 일관성

---

## 3. OpenCV: 컴퓨터 비전 라이브러리

OpenCV는 300개 이상의 모듈로 구성된 거대 프로젝트입니다.

### 프로젝트 구조

```
opencv/
├── CMakeLists.txt
├── cmake/                   # CMake 스크립트들
├── modules/
│   ├── core/               # 기본 모듈
│   ├── imgproc/            # 이미지 처리
│   ├── dnn/                # 딥러닝
│   └── ...
└── 3rdparty/               # 서드파티 라이브러리
```

### 핵심 패턴 1: 모듈 시스템

**cmake/OpenCVModule.cmake:**
```cmake
# 모듈 정의 매크로
macro(ocv_define_module name)
  project(opencv_${name})
  
  # 소스 파일 자동 수집
  file(GLOB_RECURSE sources src/*.cpp)
  file(GLOB_RECURSE headers include/*.hpp)
  
  add_library(opencv_${name} ${sources})
  
  # 헤더 경로
  target_include_directories(opencv_${name}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include/opencv4>
  )
endmacro()
```

**modules/core/CMakeLists.txt:**
```cmake
ocv_define_module(core
  DEPENDS
    opencv_hal
  OPTIONAL
    TBB  # Threading Building Blocks
)

# TBB가 있으면 멀티스레딩 활성화
if(HAVE_TBB)
  target_compile_definitions(opencv_core PRIVATE CV_PARALLEL_FRAMEWORK=1)
  target_link_libraries(opencv_core TBB::tbb)
endif()
```

### 핵심 패턴 2: 플랫폼 감지

**cmake/OpenCVDetectPlatform.cmake:**
```cmake
# CPU 아키텍처
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(X86_64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(AARCH64 1)
endif()

# SIMD 지원
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-msse4.2" HAVE_SSE42)
check_cxx_compiler_flag("-mavx2" HAVE_AVX2)
check_cxx_compiler_flag("-mfma" HAVE_FMA)

# ARM NEON
if(AARCH64)
  set(HAVE_NEON ON)
endif()
```

### 핵심 패턴 3: 최적화 옵션

```cmake
# SSE 최적화 소스
if(HAVE_SSE42)
  add_library(opencv_core_sse4
    src/mathfuncs_core.sse4.cpp
  )
  target_compile_options(opencv_core_sse4 PRIVATE -msse4.2)
  target_link_libraries(opencv_core PRIVATE opencv_core_sse4)
endif()

# AVX2 최적화 소스
if(HAVE_AVX2)
  add_library(opencv_core_avx2
    src/mathfuncs_core.avx2.cpp
  )
  target_compile_options(opencv_core_avx2 PRIVATE -mavx2 -mfma)
  target_link_libraries(opencv_core PRIVATE opencv_core_avx2)
endif()
```

런타임에 CPU를 감지해서 최적화된 코드를 선택합니다!

### 핵심 패턴 4: 의존성 관리

```cmake
# 선택적 의존성
ocv_option(WITH_JPEG "JPEG support" ON)
ocv_option(WITH_PNG "PNG support" ON)
ocv_option(WITH_CUDA "CUDA support" OFF)

if(WITH_JPEG)
  find_package(JPEG)
  if(NOT JPEG_FOUND)
    # 없으면 번들된 버전 사용
    add_subdirectory(3rdparty/libjpeg-turbo)
    set(JPEG_LIBRARIES jpeg)
  endif()
endif()

if(WITH_CUDA)
  find_package(CUDA 10.0 REQUIRED)
  add_subdirectory(modules/cudaarithm)
endif()
```

### 배울 점

1. **모듈 시스템**: 매크로로 일관된 구조
2. **플랫폼 최적화**: 아키텍처별 코드 분리
3. **의존성 Fallback**: 번들 버전 제공
4. **세밀한 옵션**: 기능별로 켜고 끄기

---

## 4. gRPC: RPC 프레임워크

구글의 gRPC는 Protobuf 코드 생성과 C++ 라이브러리를 결합한 복잡한 프로젝트입니다.

### 핵심 패턴: Protobuf 코드 생성

**cmake/protobuf-generate.cmake:**
```cmake
function(protobuf_generate_cpp SRCS HDRS)
  cmake_parse_arguments(protobuf "" "EXPORT_MACRO" "" ${ARGN})
  
  set(${SRCS})
  set(${HDRS})
  
  foreach(FIL ${protobuf_UNPARSED_ARGUMENTS})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    
    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")
    
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h"
      COMMAND protoc
      ARGS --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
           -I${CMAKE_CURRENT_SOURCE_DIR}
           ${ABS_FIL}
      DEPENDS ${ABS_FIL} protoc
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
    )
  endforeach()
  
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()
```

**사용:**
```cmake
# .proto 파일로 C++ 코드 생성
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
  protos/hello.proto
  protos/user.proto
)

add_library(myservice ${PROTO_SRCS})
target_include_directories(myservice PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
```

### 배울 점

1. **코드 생성**: `add_custom_command`로 빌드 파이프라인 통합
2. **생성된 파일 관리**: `CMAKE_CURRENT_BINARY_DIR` 활용
3. **의존성 추적**: `DEPENDS`로 재생성 트리거

---

## 5. Abseil: 구글 C++ 라이브러리

Abseil은 "올바른 CMake 사용법"의 교과서입니다.

### 핵심 패턴 1: 작은 타겟들

```cmake
# 각 기능을 별도 라이브러리로
absl_cc_library(
  NAME strings
  HDRS "string_view.h"
  SRCS "string_view.cc"
  DEPS
    absl::base
    absl::throw_delegate
  PUBLIC
)

absl_cc_library(
  NAME str_format
  HDRS "str_format.h"
  SRCS "str_format.cc"
  DEPS
    absl::strings
    absl::numeric
  PUBLIC
)
```

### 핵심 패턴 2: 헬퍼 함수

**cmake/AbseilHelpers.cmake:**
```cmake
function(absl_cc_library)
  cmake_parse_arguments(ABSL_CC_LIB
    "PUBLIC;TESTONLY"
    "NAME"
    "HDRS;SRCS;DEPS;COPTS;LINKOPTS"
    ${ARGN}
  )
  
  set(target "absl_${ABSL_CC_LIB_NAME}")
  
  add_library(${target} "")
  target_sources(${target} PRIVATE ${ABSL_CC_LIB_SRCS})
  target_link_libraries(${target} PUBLIC ${ABSL_CC_LIB_DEPS})
  target_compile_options(${target} PRIVATE ${ABSL_CC_LIB_COPTS})
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${ABSL_COMMON_INCLUDE_DIRS}>
  )
  
  # Alias
  add_library(absl::${ABSL_CC_LIB_NAME} ALIAS ${target})
endfunction()
```

### 배울 점

1. **세밀한 타겟**: 작은 단위로 쪼개기
2. **일관된 인터페이스**: 헬퍼 함수로 통일
3. **명명 규칙**: `absl::` 네임스페이스

---

## 실전 팁 모음

### 1. 빌드 시간 단축

```cmake
# Precompiled headers (CMake 3.16+)
target_precompile_headers(mylib PRIVATE
  <vector>
  <string>
  <memory>
)

# Unity builds
set_target_properties(mylib PROPERTIES
  UNITY_BUILD ON
  UNITY_BUILD_BATCH_SIZE 16
)

# ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()
```

### 2. 디버그 정보

```cmake
# 컴파일 명령어 출력
set(CMAKE_VERBOSE_MAKEFILE ON)

# 또는 빌드 시
cmake --build build -- VERBOSE=1

# 모든 변수 출력
cmake -LAH build/
```

### 3. 크로스 컴파일

```cmake
# toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

**사용:**
```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
```

### 4. 캐시 변수 vs 일반 변수

```cmake
# 캐시 변수 (사용자가 변경 가능)
set(MY_OPTION "default" CACHE STRING "Description")

# 일반 변수 (내부용)
set(MY_INTERNAL_VAR "value")

# 캐시 변수 강제 업데이트
set(MY_OPTION "new_value" CACHE STRING "" FORCE)
```

### 5. Generator Expression 고급

```cmake
# 설정별 소스
target_sources(mylib PRIVATE
  common.cpp
  $<$<CONFIG:Debug>:debug_utils.cpp>
  $<$<CONFIG:Release>:release_utils.cpp>
)

# 컴파일러별 옵션
target_compile_options(mylib PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-march=native>
  $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
  $<$<CXX_COMPILER_ID:Clang>:-march=native>
)

# 타겟 속성 사용
target_compile_definitions(mylib PRIVATE
  VERSION="$<TARGET_PROPERTY:mylib,VERSION>"
)
```

---

## 안티패턴 (하지 말 것)

### 1. file(GLOB) 남용

```cmake
# ❌ 나쁜 예
file(GLOB_RECURSE SOURCES "src/*.cpp")
```

**문제:**
- 새 파일 추가 시 CMake 재실행 필요
- CI에서 누락 가능

### 2. 글로벌 설정

```cmake
# ❌ 나쁜 예
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(/usr/local/lib)
add_definitions(-DMY_DEFINE)
```

**대신:**
```cmake
# ✅ 좋은 예
target_include_directories(mylib PUBLIC include)
target_link_libraries(mylib /usr/local/lib/libfoo.a)
target_compile_definitions(mylib PRIVATE MY_DEFINE)
```

### 3. 하드코딩된 경로

```cmake
# ❌ 나쁜 예
set(CUDA_PATH "/usr/local/cuda-11.0")
```

**대신:**
```cmake
# ✅ 좋은 예
find_package(CUDAToolkit 11.0 REQUIRED)
```

### 4. 변수 오염

```cmake
# ❌ 나쁜 예
set(SOURCES file1.cpp file2.cpp)
add_subdirectory(subdir)  # subdir에서 SOURCES 변경
add_library(mylib ${SOURCES})  # 의도와 다른 파일들
```

**대신:**
```cmake
# ✅ 좋은 예
function(add_my_library name)
  set(SOURCES file1.cpp file2.cpp)  # 함수 스코프
  add_library(${name} ${SOURCES})
endfunction()
```

---

## 요약

유명 프로젝트들의 공통 패턴:

1. **모듈화**: 작은 타겟들 + 명확한 의존성
2. **옵션 제공**: 사용자가 필요한 것만 빌드
3. **헬퍼 함수**: 반복 작업 추상화
4. **플랫폼 대응**: 조건부 컴파일
5. **코드 생성**: 빌드타임 자동화

이제 여러분도 대규모 프로젝트를 관리할 수 있습니다! 🚀

---

## 다음 글

4편부터는 **LLM 시리즈** 시작!
- **Paged Attention**: vLLM이 어떻게 메모리를 효율적으로 쓰나?

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*

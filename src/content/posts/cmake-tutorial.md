---
title: "Modern CMake Tutorial: From Zero to Production"
description: "CMake를 처음 배우는 사람도 이해할 수 있도록 기초부터 실전까지 차근차근 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["cmake", "cpp", "build-systems", "tutorial"]
draft: false
---

# Modern CMake Tutorial: From Zero to Production

CMake는 어렵다는 인식이 있습니다. 하지만 2015년 이후의 Modern CMake는 생각보다 직관적이고 강력합니다. 차근차근 배워봅시다.

## CMake가 뭐죠?

간단히 말하면, **C++ 프로젝트를 빌드하는 도구**입니다.

```bash
# 직접 컴파일하려면...
g++ main.cpp utils.cpp -I./include -lpthread -o myapp

# CMake를 사용하면
cmake -B build
cmake --build build
```

CMake는 여러분의 프로젝트 구조를 파악하고, 적절한 컴파일 명령어를 자동으로 생성해줍니다.

## 왜 CMake를 써야 하나요?

### 1. 크로스 플랫폼 빌드

같은 `CMakeLists.txt` 파일로 여러 플랫폼에서 빌드할 수 있습니다.

```cmake
# 이 파일 하나로
cmake -B build           # macOS/Linux
cmake -B build -G "Visual Studio 17 2022"  # Windows
```

### 2. 의존성 관리

라이브러리를 쓸 때, 헤더 경로와 링크 순서를 자동으로 관리해줍니다.

```cmake
find_package(OpenCV REQUIRED)
target_link_libraries(myapp OpenCV::OpenCV)
# 헤더 경로, 링크 옵션 등이 자동으로 설정됨
```

### 3. 대규모 프로젝트 관리

여러 디렉토리, 여러 라이브러리를 깔끔하게 구성할 수 있습니다.

---

## 예제로 배우는 CMake

이론보다 실전! 간단한 프로젝트를 만들어보면서 배워봅시다.

### 1단계: Hello World

가장 간단한 예제입니다.

**프로젝트 구조:**
```
hello/
├── CMakeLists.txt
└── main.cpp
```

**main.cpp:**
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, CMake!" << std::endl;
    return 0;
}
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(hello)

add_executable(hello main.cpp)
```

**빌드하기:**
```bash
cmake -B build
cmake --build build
./build/hello
```

간단하죠? 이제 하나씩 뜯어봅시다.

---

## CMakeLists.txt 파헤치기

### cmake_minimum_required

```cmake
cmake_minimum_required(VERSION 3.15)
```

"이 프로젝트는 CMake 3.15 이상에서만 동작합니다"라는 뜻입니다.

왜 필요할까요? CMake는 버전마다 기능이 다릅니다. 최소 버전을 명시하면:
- 예전 버전에서 빌드하려고 할 때 에러를 미리 방지
- 어떤 기능을 쓸 수 있는지 명확히 알 수 있음

**권장:** 3.15 이상 (Modern CMake 기능 사용 가능)

### project

```cmake
project(hello)
```

프로젝트 이름을 정의합니다. 이걸 쓰면:
- `PROJECT_NAME` 변수가 `hello`로 설정됨
- `PROJECT_SOURCE_DIR` 같은 변수들이 자동 생성됨

**더 자세히 쓰기:**
```cmake
project(hello
    VERSION 1.0.0
    DESCRIPTION "My first CMake project"
    LANGUAGES CXX
)
```

### add_executable

```cmake
add_executable(hello main.cpp)
```

"main.cpp를 컴파일해서 hello라는 실행 파일을 만들어라"는 의미입니다.

**여러 파일:**
```cmake
add_executable(myapp
    main.cpp
    utils.cpp
    helper.cpp
)
```

---

## 2단계: 헤더 파일 추가

이제 좀 더 실전적인 구조를 만들어봅시다.

**프로젝트 구조:**
```
myproject/
├── CMakeLists.txt
├── include/
│   └── utils.h
└── src/
    ├── main.cpp
    └── utils.cpp
```

**utils.h:**
```cpp
#pragma once
#include <string>

std::string greet(const std::string& name);
```

**utils.cpp:**
```cpp
#include "utils.h"

std::string greet(const std::string& name) {
    return "Hello, " + name + "!";
}
```

**main.cpp:**
```cpp
#include <iostream>
#include "utils.h"  // 이 헤더를 찾아야 함

int main() {
    std::cout << greet("CMake") << std::endl;
    return 0;
}
```

### 문제: 헤더를 못 찾음

```cmake
add_executable(myapp
    src/main.cpp
    src/utils.cpp
)
```

이렇게만 쓰면 컴파일 에러가 납니다:
```
main.cpp:2:10: fatal error: utils.h: No such file or directory
```

왜? 컴파일러가 `include/` 디렉토리를 모르니까요.

### 해결책: include 디렉토리 알려주기

```cmake
add_executable(myapp
    src/main.cpp
    src/utils.cpp
)

target_include_directories(myapp PRIVATE include)
```

이제 컴파일러가 `include/` 디렉토리에서 헤더를 찾습니다.

**PRIVATE가 뭐죠?**
- `PRIVATE`: 이 타겟(myapp)만 사용
- `PUBLIC`: 이 타겟 + 이 타겟을 사용하는 다른 타겟도 사용
- `INTERFACE`: 이 타겟을 사용하는 다른 타겟만 사용

지금은 실행 파일이라 `PRIVATE`만 있으면 됩니다.

---

## 3단계: 라이브러리 만들기

실행 파일 대신 **라이브러리**를 만들어봅시다.

라이브러리는 재사용 가능한 코드 묶음입니다. 다른 프로젝트에서 가져다 쓸 수 있죠.

**새로운 구조:**
```
mylib/
├── CMakeLists.txt
├── include/
│   └── mylib/
│       └── math.h
├── src/
│   └── math.cpp
└── examples/
    └── example.cpp
```

**math.h:**
```cpp
#pragma once

namespace mylib {
    int add(int a, int b);
    int multiply(int a, int b);
}
```

**math.cpp:**
```cpp
#include "mylib/math.h"

namespace mylib {
    int add(int a, int b) {
        return a + b;
    }

    int multiply(int a, int b) {
        return a * b;
    }
}
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(mylib VERSION 1.0.0)

# 라이브러리 생성
add_library(mylib
    src/math.cpp
)

# 헤더 경로 설정
target_include_directories(mylib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 예제 프로그램
add_executable(example examples/example.cpp)
target_link_libraries(example mylib)
```

### 하나씩 설명

**add_library:**
```cmake
add_library(mylib src/math.cpp)
```

라이브러리를 만듭니다. 기본은 `STATIC` 라이브러리 (`.a` 또는 `.lib` 파일).

**target_include_directories - PUBLIC:**
```cmake
target_include_directories(mylib PUBLIC ...)
```

여기서 `PUBLIC`을 쓴 이유:
- 라이브러리 자체도 `include/`가 필요함
- 이 라이브러리를 쓰는 다른 프로그램도 `include/`가 필요함

**Generator Expression:**
```cmake
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include>
```

상황에 따라 다른 경로를 쓰는 마법입니다:
- 빌드할 때: 소스 디렉토리의 `include/` 사용
- 설치 후: 설치된 위치의 `include/` 사용

**target_link_libraries:**
```cmake
target_link_libraries(example mylib)
```

"example 프로그램이 mylib 라이브러리를 사용한다"는 뜻입니다.

이렇게 하면:
- `mylib`의 `PUBLIC` include 디렉토리가 자동으로 `example`에도 적용됨
- 링크 순서도 자동으로 처리됨

---

## 4단계: 외부 라이브러리 사용

이제 다른 사람이 만든 라이브러리를 써봅시다.

예를 들어, JSON 파싱을 위해 `nlohmann/json`을 사용한다면:

```cmake
find_package(nlohmann_json REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp nlohmann_json::nlohmann_json)
```

### find_package가 하는 일

1. 시스템에서 라이브러리를 찾음
2. 라이브러리의 헤더 경로, 링크 옵션 등을 자동으로 설정
3. `nlohmann_json::nlohmann_json` 같은 "타겟"을 만들어줌

**REQUIRED:**
- 라이브러리를 못 찾으면 즉시 에러
- 선택 사항이면 빼면 됨

### 실전 예제: CURL 사용

HTTP 요청을 하려면 CURL 라이브러리가 필요합니다.

```cmake
cmake_minimum_required(VERSION 3.15)
project(http_client)

# CURL 찾기
find_package(CURL REQUIRED)

# 실행 파일
add_executable(client main.cpp)

# CURL 링크
target_link_libraries(client CURL::libcurl)
```

**main.cpp:**
```cpp
#include <curl/curl.h>
#include <iostream>

int main() {
    CURL* curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "https://example.com");
        curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }
    return 0;
}
```

**빌드:**
```bash
cmake -B build
cmake --build build
```

CURL의 헤더 경로, 링크 옵션이 모두 자동으로 설정됩니다!

---

## PUBLIC vs PRIVATE vs INTERFACE

이게 헷갈리는 포인트인데, 예제로 명확히 해봅시다.

### 시나리오

**libA** (라이브러리)
- `curl`을 사용 (헤더에서도 사용)
- `pthread`를 사용 (구현에서만 사용)

```cmake
add_library(libA src/a.cpp)

target_link_libraries(libA
    PUBLIC
        CURL::libcurl     # 헤더에 노출됨
    PRIVATE
        Threads::Threads  # 내부 구현에만 사용
)
```

**myapp** (실행 파일, libA 사용)
```cmake
add_executable(myapp main.cpp)
target_link_libraries(myapp libA)
```

### 결과

`myapp`는 자동으로:
- ✅ CURL을 링크 (PUBLIC이니까)
- ❌ pthread를 모름 (PRIVATE이니까)

### 언제 뭘 쓰나요?

**PUBLIC:**
- 헤더 파일에 포함되는 라이브러리
- 예: `class MyClass { std::shared_ptr<CurlHandle> handle_; };`

**PRIVATE:**
- `.cpp` 파일에서만 쓰는 라이브러리
- 예: 내부적으로만 멀티스레딩

**INTERFACE:**
- 헤더 온리 라이브러리
- 예: `nlohmann/json`, `Eigen`

---

## 5단계: 실전 프로젝트 구조

이제 진짜 프로젝트처럼 만들어봅시다.

```
awesome-http/
├── CMakeLists.txt          # 루트
├── include/
│   └── awesome/
│       ├── client.h
│       └── request.h
├── src/
│   ├── client.cpp
│   └── request.cpp
├── examples/
│   ├── CMakeLists.txt
│   └── simple_get.cpp
└── tests/
    ├── CMakeLists.txt
    └── test_client.cpp
```

### 루트 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(awesome_http VERSION 0.1.0 LANGUAGES CXX)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 의존성
find_package(CURL REQUIRED)

# 메인 라이브러리
add_library(awesome_http
    src/client.cpp
    src/request.cpp
)

# 네임스페이스를 가진 alias
add_library(awesome::http ALIAS awesome_http)

# 헤더 경로
target_include_directories(awesome_http
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 링크
target_link_libraries(awesome_http
    PUBLIC
        CURL::libcurl
)

# 컴파일러 경고
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(awesome_http PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()

# 서브 디렉토리
add_subdirectory(examples)
add_subdirectory(tests)
```

### examples/CMakeLists.txt

```cmake
add_executable(simple_get simple_get.cpp)
target_link_libraries(simple_get awesome::http)
```

### tests/CMakeLists.txt

```cmake
enable_testing()

add_executable(test_client test_client.cpp)
target_link_libraries(test_client awesome::http)

add_test(NAME ClientTest COMMAND test_client)
```

### 빌드 & 테스트

```bash
# 빌드
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8

# 테스트
ctest --test-dir build --output-on-failure

# 예제 실행
./build/examples/simple_get
```

---

## 유용한 팁들

### 1. 빌드 타입

```bash
# Debug (디버깅 심볼, 최적화 없음)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release (최적화, 디버깅 심볼 없음)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# RelWithDebInfo (최적화 + 디버깅 심볼)
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### 2. 병렬 빌드

```bash
# CPU 코어 수만큼 병렬 컴파일
cmake --build build -j8
```

### 3. Verbose 모드

```bash
# 실제 컴파일 명령어 보기
cmake --build build --verbose
```

### 4. 특정 타겟만 빌드

```bash
# myapp만 빌드
cmake --build build --target myapp
```

### 5. Clean

```bash
# 빌드 결과물 삭제
cmake --build build --target clean

# 또는 그냥 build 디렉토리 삭제
rm -rf build
```

---

## 자주 하는 실수들

### 1. file(GLOB) 사용

```cmake
# ❌ 나쁜 예
file(GLOB SOURCES src/*.cpp)
add_library(mylib ${SOURCES})
```

**문제점:**
- 새 파일을 추가해도 CMake가 인식 못 함
- 다시 `cmake` 명령을 실행해야 함

```cmake
# ✅ 좋은 예
add_library(mylib
    src/file1.cpp
    src/file2.cpp
    src/file3.cpp
)
```

### 2. include_directories 사용

```cmake
# ❌ 나쁜 예 (글로벌 설정)
include_directories(${PROJECT_SOURCE_DIR}/include)
```

**문제점:**
- 모든 타겟에 영향을 줌
- 의존성이 불명확

```cmake
# ✅ 좋은 예 (타겟별 설정)
target_include_directories(mylib PUBLIC include)
```

### 3. 소스 디렉토리에서 빌드

```bash
# ❌ 나쁜 예
cd myproject
cmake .
make
```

**문제점:**
- 소스 디렉토리가 빌드 파일로 오염됨
- 나중에 정리하기 어려움

```bash
# ✅ 좋은 예
cmake -B build
cmake --build build
```

---

## 실전 예제: Avalanche 프로젝트

실제로 제가 만든 [Avalanche](https://github.com/yhna941/avalanche) 프로젝트의 CMake 설정입니다.

```cmake
cmake_minimum_required(VERSION 3.15)
project(avalanche VERSION 0.1.0)

# C++ 표준
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 의존성
find_package(CURL REQUIRED)
find_package(pybind11 REQUIRED)

# 코어 라이브러리 (C++)
add_library(avalanche_core STATIC
    src/core/client.cpp
    src/core/stats.cpp
    src/core/loadtest.cpp
)

target_include_directories(avalanche_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(avalanche_core
    PUBLIC
        CURL::libcurl
)

# Python 바인딩
pybind11_add_module(_avalanche
    src/bindings/py_avalanche.cpp
)

target_link_libraries(_avalanche
    PRIVATE
        avalanche_core
)
```

핵심 포인트:
- 코어 라이브러리와 Python 바인딩 분리
- PUBLIC/PRIVATE 명확히 구분
- pybind11의 helper 함수 활용

---

## 마무리

Modern CMake의 핵심은:

1. **타겟 중심 사고**
   - 글로벌 변수 말고 `target_*` 명령어
   
2. **의존성 명확히**
   - PUBLIC/PRIVATE/INTERFACE 구분
   
3. **Generator Expression 활용**
   - 빌드/설치 시나리오 구분

4. **재사용 가능한 구조**
   - `find_package`로 찾을 수 있게 export

CMake는 처음엔 어렵지만, 이 원칙들만 알면 대부분의 프로젝트를 관리할 수 있습니다.

## 더 공부하려면

- [CMake 공식 문서](https://cmake.org/cmake/help/latest/)
- [Professional CMake](https://crascit.com/professional-cmake/) (유료지만 최고의 책)
- [More Modern CMake](https://github.com/Bagira80/More-Modern-CMake) (GitHub 레포, 무료)

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*

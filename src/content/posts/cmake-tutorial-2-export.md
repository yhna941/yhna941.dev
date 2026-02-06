---
title: "Modern CMake Tutorial 2편: 라이브러리 배포하기 (Export & Install)"
description: "내가 만든 C++ 라이브러리를 다른 사람들이 쉽게 사용할 수 있도록 export/install/packaging하는 방법을 알아봅니다."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["cmake", "cpp", "library", "packaging"]
draft: false
---

# Modern CMake Tutorial 2편: 라이브러리 배포하기

1편에서는 CMake로 프로젝트를 빌드하는 법을 배웠습니다. 이번에는 **내가 만든 라이브러리를 다른 사람들이 쉽게 쓸 수 있게** 만드는 방법을 알아봅시다.

## 목표

우리가 만들 것:
```cmake
# 다른 프로젝트에서 이렇게 쓸 수 있게!
find_package(MyAwesomeLib REQUIRED)
target_link_libraries(myapp MyAwesomeLib::core)
```

단 세 줄로 내 라이브러리를 사용할 수 있게 만들겠습니다.

---

## 시나리오: MathLib 만들기

간단한 수학 라이브러리를 만들고, 배포해봅시다.

**프로젝트 구조:**
```
mathlib/
├── CMakeLists.txt
├── include/
│   └── mathlib/
│       ├── basic.h
│       └── advanced.h
├── src/
│   ├── basic.cpp
│   └── advanced.cpp
└── cmake/
    └── MathLibConfig.cmake.in
```

### 헤더 파일

**include/mathlib/basic.h:**
```cpp
#pragma once

namespace mathlib {

int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b);
double divide(double a, double b);

}  // namespace mathlib
```

**include/mathlib/advanced.h:**
```cpp
#pragma once
#include <vector>

namespace mathlib {

double mean(const std::vector<double>& numbers);
double median(std::vector<double> numbers);
double stddev(const std::vector<double>& numbers);

}  // namespace mathlib
```

### 구현 파일

**src/basic.cpp:**
```cpp
#include "mathlib/basic.h"
#include <stdexcept>

namespace mathlib {

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

double divide(double a, double b) {
    if (b == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    return a / b;
}

}  // namespace mathlib
```

**src/advanced.cpp:**
```cpp
#include "mathlib/advanced.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace mathlib {

double mean(const std::vector<double>& numbers) {
    if (numbers.empty()) return 0.0;
    double sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
    return sum / numbers.size();
}

double median(std::vector<double> numbers) {
    if (numbers.empty()) return 0.0;
    
    std::sort(numbers.begin(), numbers.end());
    size_t n = numbers.size();
    
    if (n % 2 == 0) {
        return (numbers[n/2 - 1] + numbers[n/2]) / 2.0;
    } else {
        return numbers[n/2];
    }
}

double stddev(const std::vector<double>& numbers) {
    if (numbers.size() < 2) return 0.0;
    
    double avg = mean(numbers);
    double sq_sum = 0.0;
    
    for (double num : numbers) {
        sq_sum += (num - avg) * (num - avg);
    }
    
    return std::sqrt(sq_sum / (numbers.size() - 1));
}

}  // namespace mathlib
```

---

## CMakeLists.txt - 기본 구조

먼저 기본적인 라이브러리부터 만듭시다.

```cmake
cmake_minimum_required(VERSION 3.15)
project(MathLib VERSION 1.0.0 LANGUAGES CXX)

# C++ 표준
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 라이브러리 생성
add_library(mathlib
    src/basic.cpp
    src/advanced.cpp
)

# Alias (네임스페이스)
add_library(MathLib::core ALIAS mathlib)

# 헤더 경로
target_include_directories(mathlib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
```

여기까지는 1편에서 배운 내용입니다. 이제 **설치(install)**와 **내보내기(export)**를 추가합시다.

---

## Install: 파일 설치하기

### 라이브러리 파일 설치

```cmake
include(GNUInstallDirs)

install(TARGETS mathlib
    EXPORT MathLibTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

**하나씩 뜯어봅시다:**

**TARGETS mathlib:**
- `mathlib` 타겟을 설치한다

**EXPORT MathLibTargets:**
- 나중에 다른 프로젝트에서 import할 수 있도록 "MathLibTargets"라는 이름으로 export한다

**LIBRARY/ARCHIVE/RUNTIME:**
- `LIBRARY`: 공유 라이브러리 (`.so`, `.dylib`)
- `ARCHIVE`: 정적 라이브러리 (`.a`, `.lib`)
- `RUNTIME`: 실행 파일 또는 DLL (Windows)

**CMAKE_INSTALL_LIBDIR:**
- 시스템에 맞는 라이브러리 경로
- Linux: `/usr/local/lib`
- macOS: `/usr/local/lib`
- Windows: `C:/Program Files/MathLib/lib`

### 헤더 파일 설치

```cmake
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

`include/` 디렉토리 전체를 설치 경로에 복사합니다.

결과:
```
/usr/local/include/
└── mathlib/
    ├── basic.h
    └── advanced.h
```

---

## Export: CMake 설정 파일 생성

다른 프로젝트가 `find_package(MathLib)`로 찾을 수 있게 만들어봅시다.

### 1. Targets 파일 export

```cmake
install(EXPORT MathLibTargets
    FILE MathLibTargets.cmake
    NAMESPACE MathLib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathLib
)
```

**EXPORT MathLibTargets:**
- 아까 위에서 정의한 export 이름

**FILE MathLibTargets.cmake:**
- 생성될 파일 이름

**NAMESPACE MathLib::**
- 타겟 이름 앞에 붙는 네임스페이스
- `mathlib` → `MathLib::mathlib`

**DESTINATION:**
- 설치될 경로
- 예: `/usr/local/lib/cmake/MathLib/MathLibTargets.cmake`

### 2. Config 파일 생성

**cmake/MathLibConfig.cmake.in:**
```cmake
@PACKAGE_INIT@

include("${CMAKE_CURRENT_LIST_DIR}/MathLibTargets.cmake")

check_required_components(MathLib)
```

이 템플릿 파일을 CMake가 처리해서 실제 Config 파일을 만듭니다.

**CMakeLists.txt에 추가:**
```cmake
include(CMakePackageConfigHelpers)

# Config 파일 생성
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MathLibConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MathLibConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathLib
)

# Version 파일 생성
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MathLibConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Config 파일들 설치
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MathLibConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MathLibConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathLib
)
```

**configure_package_config_file:**
- 템플릿 파일(`.in`)을 실제 파일로 변환
- `@PACKAGE_INIT@` 같은 매크로를 치환

**write_basic_package_version_file:**
- 버전 체크 파일 생성
- `find_package(MathLib 1.0 REQUIRED)`처럼 버전 지정 가능

**COMPATIBILITY SameMajorVersion:**
- 같은 major 버전끼리만 호환
- 1.0, 1.1, 1.2는 호환 / 2.0은 비호환

---

## 전체 CMakeLists.txt

모든 걸 합치면:

```cmake
cmake_minimum_required(VERSION 3.15)
project(MathLib VERSION 1.0.0 LANGUAGES CXX)

# C++ 표준
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 라이브러리 생성
add_library(mathlib
    src/basic.cpp
    src/advanced.cpp
)

# Alias
add_library(MathLib::core ALIAS mathlib)

# 헤더 경로
target_include_directories(mathlib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 컴파일러 경고
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mathlib PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()

# 설치 관련
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# 라이브러리 설치
install(TARGETS mathlib
    EXPORT MathLibTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 헤더 설치
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Targets export
install(EXPORT MathLibTargets
    FILE MathLibTargets.cmake
    NAMESPACE MathLib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathLib
)

# Config 파일 생성
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MathLibConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MathLibConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathLib
)

# Version 파일 생성
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MathLibConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Config 파일 설치
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MathLibConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MathLibConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MathLib
)
```

---

## 빌드 & 설치

### 1. 빌드

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 2. 설치

```bash
# 시스템에 설치 (관리자 권한 필요)
sudo cmake --install build

# 또는 특정 경로에 설치
cmake --install build --prefix ~/.local
```

### 3. 설치된 파일 확인

```bash
# macOS/Linux
tree /usr/local/

/usr/local/
├── include/
│   └── mathlib/
│       ├── basic.h
│       └── advanced.h
├── lib/
│   ├── libmathlib.a
│   └── cmake/
│       └── MathLib/
│           ├── MathLibConfig.cmake
│           ├── MathLibConfigVersion.cmake
│           └── MathLibTargets.cmake
```

완벽! 이제 다른 프로젝트에서 사용할 수 있습니다.

---

## 사용하기: 다른 프로젝트에서

이제 누군가 내 라이브러리를 쓴다고 해봅시다.

**calculator/main.cpp:**
```cpp
#include <iostream>
#include <vector>
#include "mathlib/basic.h"
#include "mathlib/advanced.h"

int main() {
    // 기본 연산
    std::cout << "5 + 3 = " << mathlib::add(5, 3) << std::endl;
    std::cout << "10 / 2 = " << mathlib::divide(10, 2) << std::endl;
    
    // 통계
    std::vector<double> data = {1.5, 2.3, 4.1, 3.7, 5.2};
    std::cout << "Mean: " << mathlib::mean(data) << std::endl;
    std::cout << "Median: " << mathlib::median(data) << std::endl;
    std::cout << "StdDev: " << mathlib::stddev(data) << std::endl;
    
    return 0;
}
```

**calculator/CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(calculator)

set(CMAKE_CXX_STANDARD 17)

# MathLib 찾기 (우리가 만든 라이브러리!)
find_package(MathLib 1.0 REQUIRED)

# 실행 파일
add_executable(calculator main.cpp)

# 링크 (단 한 줄!)
target_link_libraries(calculator MathLib::core)
```

**빌드:**
```bash
cmake -B build
cmake --build build
./build/calculator
```

**결과:**
```
5 + 3 = 8
10 / 2 = 5
Mean: 3.36
Median: 3.7
StdDev: 1.46
```

완벽하게 작동합니다! 🎉

---

## find_package는 어떻게 찾나요?

CMake는 다음 경로들을 순서대로 검색합니다:

1. **CMAKE_PREFIX_PATH** 환경변수
2. **시스템 기본 경로:**
   - `/usr/local/lib/cmake/`
   - `/usr/lib/cmake/`
   - `C:/Program Files/`

3. **MathLib_DIR** 변수 (직접 지정)
   ```bash
   cmake -B build -DMathLib_DIR=/custom/path/lib/cmake/MathLib
   ```

### 커스텀 설치 경로 사용

라이브러리를 `/opt/mylibs`에 설치했다면:

```bash
# 설치
cmake --install build --prefix /opt/mylibs

# 사용
export CMAKE_PREFIX_PATH=/opt/mylibs:$CMAKE_PREFIX_PATH
cmake -B build
```

또는:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/opt/mylibs
```

---

## Export (build tree)

설치하지 않고도 빌드 디렉토리에서 바로 사용할 수 있게 만들 수 있습니다.

**CMakeLists.txt에 추가:**
```cmake
# Build tree export (설치 안 해도 됨)
export(EXPORT MathLibTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/MathLibTargets.cmake
    NAMESPACE MathLib::
)

# 패키지 레지스트리에 등록 (선택)
export(PACKAGE MathLib)
```

이렇게 하면:
```bash
# MathLib 프로젝트 빌드만 하고
cd mathlib
cmake -B build
cmake --build build

# 설치 없이 바로 사용 가능
cd ../calculator
cmake -B build -DMathLib_DIR=/path/to/mathlib/build
```

개발 중일 때 유용합니다!

---

## 실전 팁

### 1. Debug/Release 동시 설치

```cmake
# 설치 시 빌드 타입 포함
install(TARGETS mathlib
    EXPORT MathLibTargets-${CMAKE_BUILD_TYPE}
    # ...
)
```

이렇게 하면 Debug와 Release 버전을 동시에 설치할 수 있습니다.

### 2. 컴포넌트 분리

```cmake
# 여러 라이브러리
add_library(mathlib_basic src/basic.cpp)
add_library(mathlib_advanced src/advanced.cpp)

# 별도로 export
install(TARGETS mathlib_basic mathlib_advanced
    EXPORT MathLibTargets
    # ...
)
```

사용자가 원하는 것만 선택:
```cmake
find_package(MathLib COMPONENTS basic REQUIRED)
```

### 3. pkg-config 지원

`.pc` 파일도 생성하면 더 좋습니다:

**cmake/mathlib.pc.in:**
```
prefix=@CMAKE_INSTALL_PREFIX@
libdir=${prefix}/@CMAKE_INSTALL_LIBDIR@
includedir=${prefix}/@CMAKE_INSTALL_INCLUDEDIR@

Name: MathLib
Description: Simple math library
Version: @PROJECT_VERSION@
Libs: -L${libdir} -lmathlib
Cflags: -I${includedir}
```

**CMakeLists.txt:**
```cmake
configure_file(cmake/mathlib.pc.in mathlib.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mathlib.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
```

이제 `pkg-config`로도 사용 가능:
```bash
g++ main.cpp $(pkg-config --cflags --libs mathlib)
```

---

## 헤더 온리 라이브러리

만약 `.cpp` 파일이 없고 헤더만 있다면?

```cmake
add_library(mathlib INTERFACE)

target_include_directories(mathlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS mathlib
    EXPORT MathLibTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

**INTERFACE:**
- 컴파일할 게 없음
- 헤더만 제공
- 사용자가 링크하면 include 경로만 추가됨

예: `nlohmann/json`, `Eigen`, `range-v3`

---

## 실전 예제: CURL

유명한 libcurl의 설치 구조를 봅시다:

```bash
# macOS (Homebrew)
brew install curl

# 설치 경로
/opt/homebrew/
├── include/
│   └── curl/
│       └── curl.h
├── lib/
│   ├── libcurl.dylib
│   ├── cmake/
│   │   └── CURL/
│   │       ├── CURLConfig.cmake
│   │       └── CURLTargets.cmake
│   └── pkgconfig/
│       └── libcurl.pc
```

그래서 우리가 이렇게 쓸 수 있는 거죠:
```cmake
find_package(CURL REQUIRED)
target_link_libraries(myapp CURL::libcurl)
```

---

## 버전 체크

**Config 파일에서:**
```cmake
# cmake/MathLibConfig.cmake.in
@PACKAGE_INIT@

set(MathLib_VERSION @PROJECT_VERSION@)

include("${CMAKE_CURRENT_LIST_DIR}/MathLibTargets.cmake")

check_required_components(MathLib)

# 의존성 체크 (필요하면)
include(CMakeFindDependencyMacro)
# find_dependency(SomeDependency REQUIRED)
```

**사용 시:**
```cmake
# 정확한 버전
find_package(MathLib 1.0.0 EXACT REQUIRED)

# 최소 버전
find_package(MathLib 1.0 REQUIRED)

# 버전 범위 (CMake 3.19+)
find_package(MathLib 1.0...2.0 REQUIRED)
```

---

## 요약

라이브러리를 배포하려면:

1. **install(TARGETS)**: 라이브러리 파일 설치
2. **install(DIRECTORY)**: 헤더 파일 설치
3. **install(EXPORT)**: Targets 파일 생성
4. **configure_package_config_file**: Config 파일 생성
5. **write_basic_package_version_file**: Version 파일 생성

그러면 사용자는:
```cmake
find_package(YourLib REQUIRED)
target_link_libraries(app YourLib::core)
```

단 두 줄로 내 라이브러리를 쓸 수 있습니다!

---

## 다음 글 예고

3편에서는:
- **CPack**: 설치 패키지 만들기 (`.deb`, `.rpm`, `.dmg`)
- **FetchContent**: 의존성 자동 다운로드
- **ExternalProject**: 복잡한 외부 프로젝트 빌드

기대해주세요! 🚀

---

*질문이나 피드백은 [GitHub](https://github.com/yhna941)에서 환영합니다!*

---
title: "Modern CMake Tutorial: From Zero to Production"
description: "A comprehensive guide to modern CMake - targets, properties, and best practices for C++ projects."
pubDate: 2026-02-06
author: "Yh Na"
tags: ["cmake", "cpp", "build-systems", "tutorial"]
draft: false
---

# Modern CMake Tutorial: From Zero to Production

CMake gets a bad rap. But modern CMake (3.15+) is actually pretty nice. Let's build something real.

## What We're Building

A C++ library with:
- Multiple source files
- External dependencies
- Tests
- Installation support
- Export for downstream projects

## The Old Way (Bad)

```cmake
# DON'T DO THIS
include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(myapp main.cpp utils.cpp)
target_link_libraries(myapp pthread)
```

**Problems**:
- Global state (`include_directories`)
- No transitive dependencies
- Hard to maintain
- Doesn't scale

## The New Way (Good)

```cmake
# Modern CMake
add_library(mylib
    src/foo.cpp
    src/bar.cpp
)

target_include_directories(mylib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mylib
    PUBLIC
        fmt::fmt
    PRIVATE
        Threads::Threads
)
```

**Benefits**:
- Target-scoped properties
- Clear PUBLIC/PRIVATE/INTERFACE
- Generator expressions for different contexts
- Transitive dependencies just work

## Real Example: Building a Library

Let's build a simple HTTP client library.

### Project Structure

```
http-client/
├── CMakeLists.txt
├── include/
│   └── httpclient/
│       └── client.h
├── src/
│   ├── client.cpp
│   └── connection.cpp
└── tests/
    └── test_client.cpp
```

### Root CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(httpclient VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies
find_package(CURL REQUIRED)

# Library target
add_library(httpclient
    src/client.cpp
    src/connection.cpp
)

# Alias for consistent usage
add_library(httpclient::httpclient ALIAS httpclient)

# Include directories
target_include_directories(httpclient
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(httpclient
    PUBLIC
        CURL::libcurl
)

# Compiler warnings
target_compile_options(httpclient
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Tests
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS httpclient
    EXPORT httpclientTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Generate config files
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/httpclientConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/httpclientConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/httpclient
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/httpclientConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/httpclientConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/httpclientConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/httpclient
)

install(EXPORT httpclientTargets
    FILE httpclientTargets.cmake
    NAMESPACE httpclient::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/httpclient
)
```

## Key Concepts

### 1. Targets, Not Variables

Modern CMake is about **targets** and **properties**.

```cmake
# Bad
set(MYLIB_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
include_directories(${MYLIB_INCLUDE_DIRS})

# Good
target_include_directories(mylib PUBLIC include)
```

### 2. PUBLIC vs PRIVATE vs INTERFACE

- **PUBLIC**: Used by this target AND its consumers
- **PRIVATE**: Used only by this target
- **INTERFACE**: Used only by consumers (header-only libs)

```cmake
target_link_libraries(mylib
    PUBLIC
        fmt::fmt          # Exposed in headers
    PRIVATE
        Threads::Threads  # Implementation detail
)
```

### 3. Generator Expressions

Context-aware values:

```cmake
# Different paths for build vs install
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include>

# Compiler-specific flags
$<$<CXX_COMPILER_ID:GNU>:-march=native>

# Configuration-specific options
$<$<CONFIG:Debug>:-O0 -g>
```

### 4. Modern Package Finding

```cmake
# Old way
find_package(Boost REQUIRED COMPONENTS system)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(myapp ${Boost_LIBRARIES})

# New way
find_package(Boost REQUIRED COMPONENTS system)
target_link_libraries(myapp Boost::system)
```

## Building

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j8

# Test
ctest --test-dir build --output-on-failure

# Install
cmake --install build --prefix /usr/local
```

## Best Practices

### 1. Minimum Version

```cmake
cmake_minimum_required(VERSION 3.15)  # Be specific!
```

### 2. Out-of-Source Builds

```bash
cmake -B build   # Good
cmake .          # Bad (pollutes source)
```

### 3. Avoid `file(GLOB)`

```cmake
# Bad - doesn't track new files
file(GLOB SOURCES src/*.cpp)

# Good - explicit list
add_library(mylib
    src/foo.cpp
    src/bar.cpp
)
```

### 4. Use `target_*` Commands

All modern CMake commands are target-scoped:
- `target_include_directories()`
- `target_link_libraries()`
- `target_compile_options()`
- `target_compile_definitions()`

### 5. Export Your Targets

Make your library findable by downstream projects:

```cmake
install(EXPORT myTargets
    FILE myTargets.cmake
    NAMESPACE my::
    DESTINATION lib/cmake/my
)
```

## Real-World Example: Avalanche

Here's a snippet from [Avalanche](https://github.com/yhna941/avalanche), a C++ load testing framework:

```cmake
# Core library
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

# Python bindings
pybind11_add_module(_avalanche
    src/bindings/py_avalanche.cpp
)

target_link_libraries(_avalanche
    PRIVATE
        avalanche_core
)
```

Clean, modular, maintainable.

## Common Patterns

### Optional Features

```cmake
option(BUILD_TESTS "Build tests" ON)
option(BUILD_DOCS "Build documentation" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

### Conditional Compilation

```cmake
if(ENABLE_FEATURE_X)
    target_compile_definitions(mylib PRIVATE FEATURE_X_ENABLED)
    target_sources(mylib PRIVATE src/feature_x.cpp)
endif()
```

### Platform-Specific Code

```cmake
if(UNIX AND NOT APPLE)
    target_sources(mylib PRIVATE src/linux_impl.cpp)
elseif(APPLE)
    target_sources(mylib PRIVATE src/macos_impl.cpp)
elseif(WIN32)
    target_sources(mylib PRIVATE src/windows_impl.cpp)
endif()
```

## Conclusion

Modern CMake isn't scary. Key principles:

1. **Think in targets**, not variables
2. **Use PUBLIC/PRIVATE/INTERFACE** correctly
3. **Export targets** for downstream use
4. **Avoid global commands** (include_directories, link_libraries)
5. **Use generator expressions** for context-aware config

Your CMake should read like a dependency graph, not a bash script.

## Resources

- [Professional CMake](https://crascit.com/professional-cmake/) - The book
- [CMake docs](https://cmake.org/cmake/help/latest/) - Official reference
- [More Modern CMake](https://github.com/Bagira80/More-Modern-CMake) - Best practices

---

*Questions? Find me on [GitHub](https://github.com/yhna941).*

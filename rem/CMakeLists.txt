cmake_minimum_required(VERSION 3.24)
project(sparse_rem VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(REM_BUILD_TESTS "Build C++ tests" ON)
option(REM_BUILD_BENCHMARKS "Build benchmark tools" ON)
option(REM_BUILD_PYTHON "Build the pybind11 extension" ON)
option(REM_NATIVE_ARCH "Tune release builds for the local CPU" OFF)
option(REM_ENABLE_LTO "Enable link-time optimization" OFF)
option(REM_ENABLE_FAST_MATH "Benchmark-only unsafe floating point mode" OFF)
option(REM_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG 3234af2c03549bc85656bfd3a86993bf1cd8aef1
  GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(unordered_dense)

if(APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  message(FATAL_ERROR "The primary macOS build must be native ARM64, not ${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_library(rem_core
  src/api.cpp
  src/assignment_operator.cpp
  src/hamming_graph.cpp
  src/packed_states.cpp
  src/projection.cpp
  src/solvers.cpp
  src/thread_pool.cpp
)
target_include_directories(rem_core PUBLIC include)
target_include_directories(rem_core PRIVATE src)
target_link_libraries(rem_core PRIVATE unordered_dense::unordered_dense)
target_compile_features(rem_core PUBLIC cxx_std_20)
set_target_properties(rem_core PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE YES
)

if(MSVC)
  target_compile_options(rem_core PRIVATE /O2 /W4)
else()
  target_compile_options(rem_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    $<$<CONFIG:Release>:-O3>
  )
  if(REM_NATIVE_ARCH)
    if(APPLE)
      target_compile_options(rem_core PRIVATE -mcpu=native)
    else()
      target_compile_options(rem_core PRIVATE -march=native)
    endif()
  endif()
  if(REM_ENABLE_FAST_MATH)
    target_compile_options(rem_core PRIVATE -ffast-math)
    target_compile_definitions(rem_core PUBLIC REM_FAST_MATH=1)
  endif()
  if(REM_ENABLE_SANITIZERS)
    target_compile_options(rem_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(rem_core PRIVATE -fsanitize=address,undefined)
  endif()
endif()

if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(rem_core PUBLIC "${ACCELERATE_FRAMEWORK}")
  target_compile_definitions(
    rem_core PRIVATE REM_USE_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1)
else()
  find_package(LAPACK REQUIRED)
  target_link_libraries(rem_core PUBLIC LAPACK::LAPACK)
  target_compile_definitions(rem_core PRIVATE REM_USE_LAPACK=1)
endif()

if(REM_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT REM_IPO_SUPPORTED OUTPUT REM_IPO_ERROR)
  if(REM_IPO_SUPPORTED)
    set_property(TARGET rem_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(WARNING "LTO unavailable: ${REM_IPO_ERROR}")
  endif()
endif()

if(REM_BUILD_TESTS)
  enable_testing()
  add_executable(rem_tests tests/test_rem.cpp)
  target_link_libraries(rem_tests PRIVATE rem_core)
  if(REM_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(rem_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(rem_tests PRIVATE -fsanitize=address,undefined)
  endif()
  add_test(NAME rem_tests COMMAND rem_tests)
endif()

if(REM_BUILD_BENCHMARKS)
  add_executable(rem_benchmark tools/rem_benchmark.cpp)
  target_link_libraries(rem_benchmark PRIVATE rem_core)
  add_executable(rem_bench_elements benchmarks/bench_elements.cpp)
  target_link_libraries(rem_bench_elements PRIVATE rem_core unordered_dense::unordered_dense)
  target_include_directories(rem_bench_elements PRIVATE src)
endif()

if(REM_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON NEW)
  FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
    URL_HASH SHA256=741633da746b7c738bb71f1854f957b9da660bcd2dce68d71949037f0969d0ca
  )
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(rem_cpp python/bindings.cpp)
  target_link_libraries(rem_cpp PRIVATE rem_core)
  target_include_directories(rem_cpp PRIVATE src)
endif()

message(STATUS "REM target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "REM C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

install(
  TARGETS rem_core
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(REM_BUILD_PYTHON)
  install(TARGETS rem_cpp LIBRARY DESTINATION .)
endif()

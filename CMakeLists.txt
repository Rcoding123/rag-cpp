cmake_minimum_required(VERSION 3.20)
project(rag_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Kernels dispatch at compile time; default to the build host's ISA.
set(RAG_ARCH "native" CACHE STRING "Value for -march (native, x86-64-v3, x86-64-v4, ...)")
option(RAG_SANITIZE "Build with ASan+UBSan" OFF)
option(RAG_TSAN "Build with ThreadSanitizer" OFF)
option(RAG_HUGEPAGES "Back large arrays with transparent huge pages (faster scans, slower first touch)" ON)

add_library(rag INTERFACE)
target_include_directories(rag INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  # MSVC has no -march and can't target "the build host"; map the levels to
  # /arch. AVX2 implies FMA under MSVC (see RAG_AVX2_FMA in common.hpp).
  # "native" assumes an AVX2 CPU (Intel Haswell / AMD Excavator or newer);
  # use -DRAG_ARCH=x86-64 for a scalar build that runs anywhere.
  if(RAG_ARCH STREQUAL "x86-64-v4")
    set(RAG_MSVC_ARCH /arch:AVX512)
  elseif(RAG_ARCH STREQUAL "x86-64" OR RAG_ARCH STREQUAL "x86-64-v2")
    set(RAG_MSVC_ARCH "")
  else()
    set(RAG_MSVC_ARCH /arch:AVX2)
  endif()
  # /Zc:__cplusplus: report C++20 correctly; /bigobj: the test units are large;
  # /utf-8: sources and test strings are UTF-8; /Zc:preprocessor: standard preprocessor.
  target_compile_options(rag INTERFACE ${RAG_MSVC_ARCH} /W4 /permissive- /Zc:__cplusplus /Zc:preprocessor /utf-8 /bigobj /EHsc
                                       /wd4244 /wd4267 /wd4324)
  # C4244/C4267: integer and size_t narrowing (GCC's -Wconversion checks these on Linux);
  # C4324: structure padded because of alignas (intended: cache-line alignment).
  target_compile_definitions(rag INTERFACE NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(rag INTERFACE -march=${RAG_ARCH} -Wall -Wextra -Wpedantic -Wconversion -Wno-sign-conversion)
endif()
if(NOT RAG_HUGEPAGES)
  target_compile_definitions(rag INTERFACE RAG_NO_HUGEPAGES)
endif()
if(RAG_SANITIZE)
  if(MSVC)
    target_compile_options(rag INTERFACE /fsanitize=address)  # MSVC has ASan but no UBSan
  else()
    target_compile_options(rag INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(rag INTERFACE -fsanitize=address,undefined)
  endif()
endif()
if(RAG_TSAN)
  if(MSVC)
    message(FATAL_ERROR "RAG_TSAN needs GCC or Clang (ThreadSanitizer is not available with MSVC)")
  endif()
  target_compile_options(rag INTERFACE -fsanitize=thread)
  target_link_options(rag INTERFACE -fsanitize=thread)
endif()

find_package(Threads REQUIRED)
# GoogleTest: use an installed copy (Linux packages, vcpkg) or download it.
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  message(STATUS "rag-cpp: GoogleTest not found; downloading v1.15.2")
  include(FetchContent)
  FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
                       DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # match CMake's default MSVC runtime
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()
enable_testing()
# List each test binary's cases when ctest runs, not right after the build:
# a binary built for a newer ISA than the build machine's (e.g. the AVX-512
# preset on an AVX2 CI runner) then still builds.
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
add_executable(rag_tests tests/rag_tests.cpp tests/forgery_tests.cpp tests/reference_tests.cpp tests/text_clean_tests.cpp tests/agent_tests.cpp tests/api_examples_tests.cpp tests/model_based_tests.cpp tests/market_making_tests.cpp tests/mlp_tests.cpp tests/agent_edge_case_tests.cpp tests/agent_coverage_tests.cpp tests/rag_property_tests.cpp tests/policy_table_tests.cpp tests/orderflow_tests.cpp)
target_link_libraries(rag_tests PRIVATE rag GTest::gtest_main Threads::Threads)
include(GoogleTest)
gtest_discover_tests(rag_tests DISCOVERY_TIMEOUT 120)

find_package(benchmark QUIET)
if(benchmark_FOUND)
  add_executable(rag_bench bench/rag_bench.cpp)
  target_link_libraries(rag_bench PRIVATE rag benchmark::benchmark)
endif()
add_executable(rag_stress tests/stress.cpp)
target_link_libraries(rag_stress PRIVATE rag Threads::Threads)
add_test(NAME stress_all_modes COMMAND rag_stress 2)  # 2 s per dense mode; longer runs: rag_stress <seconds>

add_executable(rag_latency bench/latency.cpp)
target_link_libraries(rag_latency PRIVATE rag)

add_executable(rag_demo examples/demo.cpp)
target_link_libraries(rag_demo PRIVATE rag)

add_executable(agent_demo examples/agent_demo.cpp)
target_link_libraries(agent_demo PRIVATE rag)

# The recommended learned policy (E25, A48): N averaged networks trained in
# parallel, compiled to tables, median-voted. The smoke test also runs the
# threads under the sanitizer jobs.
add_executable(train_ensemble examples/train_ensemble.cpp)
target_link_libraries(train_ensemble PRIVATE rag Threads::Threads)
add_test(NAME train_ensemble_smoke COMMAND train_ensemble 2 300 2 ${CMAKE_CURRENT_BINARY_DIR}/smoke_ensemble.tbl)

# libFuzzer targets (Clang only): cmake -DCMAKE_CXX_COMPILER=clang++ -DRAG_FUZZ=ON
option(RAG_FUZZ "Build libFuzzer targets (requires Clang)" OFF)
if(RAG_FUZZ)
  foreach(t fuzz_load fuzz_text fuzz_policy_table)
    add_executable(${t} tests/fuzz/${t}.cpp)
    target_link_libraries(${t} PRIVATE rag)
    target_compile_options(${t} PRIVATE -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined -g)
    target_link_options(${t} PRIVATE -fsanitize=fuzzer,address,undefined)
  endforeach()
  add_executable(make_seeds tests/fuzz/make_seeds.cpp)
  target_link_libraries(make_seeds PRIVATE rag)
endif()

# Optimization (NLopt) and data analysis (Eigen) helpers: rag/agent/tuning.hpp,
# rag/agent/analysis.hpp, the tune_agent tool and their tests. Optional:
# Debian/Ubuntu libnlopt-cxx-dev and libeigen3-dev. The rest doesn't need them.
option(RAG_OPTLIBS "Build the NLopt/Eigen tuning and analysis targets when available" ON)
if(RAG_OPTLIBS)
  find_package(Eigen3 3.3 QUIET NO_MODULE)
  find_package(NLopt QUIET)
  if(Eigen3_FOUND AND NLopt_FOUND)
    message(STATUS "rag-cpp: tuning/analysis enabled (Eigen ${Eigen3_VERSION}, NLopt ${NLopt_VERSION})")
    add_executable(rag_optlib_tests tests/optional_libs_tests.cpp)
    target_link_libraries(rag_optlib_tests PRIVATE rag Eigen3::Eigen NLopt::nlopt GTest::gtest_main Threads::Threads)
    gtest_discover_tests(rag_optlib_tests)
    add_executable(tune_agent examples/tune_agent.cpp)
    target_link_libraries(tune_agent PRIVATE rag Eigen3::Eigen NLopt::nlopt)
  else()
    message(STATUS "rag-cpp: Eigen3/NLopt not found; tuning/analysis targets disabled")
  endif()
endif()

# PDF support (rag/pdf.hpp, the rag_pdf tool) needs poppler-cpp
# (Debian/Ubuntu: libpoppler-cpp-dev). The rest of the library does not.
option(RAG_PDF "Build PDF support when poppler-cpp is available" ON)
if(RAG_PDF)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(POPPLER_CPP QUIET IMPORTED_TARGET poppler-cpp)
  endif()
  if(POPPLER_CPP_FOUND)
    message(STATUS "rag-cpp: PDF support enabled (poppler-cpp ${POPPLER_CPP_VERSION})")
    add_library(rag_pdf_support INTERFACE)
    target_link_libraries(rag_pdf_support INTERFACE rag PkgConfig::POPPLER_CPP)
    add_executable(rag_pdf examples/rag_pdf.cpp)
    target_link_libraries(rag_pdf PRIVATE rag_pdf_support)
    # CLI smoke test: index this repository's docs and query them (POSIX shell).
    if(NOT WIN32)
    add_test(NAME rag_pdf_cli_smoke COMMAND sh -c
      "$<TARGET_FILE:rag_pdf> index \"$TMPDIR_RAG\"/cli.ragidx ${CMAKE_CURRENT_SOURCE_DIR}/docs && $<TARGET_FILE:rag_pdf> ask \"$TMPDIR_RAG\"/cli.ragidx 'champion challenger' -k 2 | grep -q 'chunk'")
    set_tests_properties(rag_pdf_cli_smoke PROPERTIES ENVIRONMENT "TMPDIR_RAG=${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    add_executable(rag_pdf_tests tests/pdf_tests.cpp)
    target_include_directories(rag_pdf_tests PRIVATE tests)
    target_link_libraries(rag_pdf_tests PRIVATE rag_pdf_support GTest::gtest_main Threads::Threads)
    gtest_discover_tests(rag_pdf_tests)
  else()
    message(STATUS "rag-cpp: poppler-cpp not found; PDF support disabled")
  endif()
endif()

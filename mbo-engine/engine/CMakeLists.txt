cmake_minimum_required(VERSION 3.20)
project(mbo_engine CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

# zstd: prefer real headers, else link the runtime lib directly (see dbn.hpp).
find_path(ZSTD_INCLUDE zstd.h)
find_library(ZSTD_LIB NAMES zstd libzstd.so.1)
if(NOT ZSTD_LIB)
  message(FATAL_ERROR "libzstd not found")
endif()

add_library(mbo INTERFACE)
target_include_directories(mbo INTERFACE include)
target_link_libraries(mbo INTERFACE ${ZSTD_LIB})

add_executable(dbn_digest tools/dbn_digest.cpp)
target_link_libraries(dbn_digest mbo)

enable_testing()
add_executable(book_test tests/book_test.cpp)
target_link_libraries(book_test mbo)
add_test(NAME book_test COMMAND book_test)

add_executable(book_check tools/book_check.cpp)
target_link_libraries(book_check mbo)

add_executable(bar_extract tools/bar_extract.cpp)
target_link_libraries(bar_extract mbo)
add_executable(bars_test tests/bars_test.cpp)
target_link_libraries(bars_test mbo)
add_test(NAME bars_test COMMAND bars_test)

add_executable(tob_dump tools/tob_dump.cpp)
target_link_libraries(tob_dump mbo)

add_executable(fifo_check tools/fifo_check.cpp)
target_link_libraries(fifo_check mbo)

add_executable(trades_extract tools/trades_extract.cpp)
target_link_libraries(trades_extract mbo)

# Parity reference: the original rag-cpp FeatureBuilder (header-only; needs the rag-cpp include dir two levels up).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/rag/orderflow/features.hpp)
  add_executable(feat_ref tools/feat_ref.cpp)
  target_include_directories(feat_ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
endif()

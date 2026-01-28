cmake_minimum_required(VERSION 3.16)
project(StringProject VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -march=x86-64-v2  -fno-plt)
add_compile_options(-fno-stack-protector -fomit-frame-pointer -flto=full)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-mllvm -inline-threshold=5000)
    add_link_options(-static) 
    add_link_options(-static-libgcc -static-libstdc++) 
endif()

#add_compile_options(-fsanitize=address -fsanitize=undefined)
#add_link_options(-fsanitize=address -fsanitize=undefined)

set(STRING_LIB_SOURCES String.hpp) 

#add_executable(tests_benchAVX512 benchmark.cpp ${STRING_LIB_SOURCES})
#target_compile_options(tests_benchAVX512 PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq -falign-loops=32)

#add_executable(tests_benchAVX2 benchmark.cpp ${STRING_LIB_SOURCES})
#target_compile_options(tests_benchAVX2 PRIVATE -mavx2 -mno-avx512f -mprefer-vector-width=256 -falign-loops=32)
#
#add_executable(tests_benchfallback benchmark.cpp ${STRING_LIB_SOURCES})
#
add_executable(tests_benchUniversal benchmark.cpp ${STRING_LIB_SOURCES})
target_compile_options(tests_benchUniversal PRIVATE -falign-loops=32)
add_executable(string_prog main.cpp ${STRING_LIB_SOURCES})
target_compile_options(tests_benchUniversal PRIVATE -falign-loops=32)
add_executable(tests_logic Tests.cpp ${STRING_LIB_SOURCES})
target_compile_options(tests_logic PRIVATE -falign-loops=32)
add_executable(tests_string TestsString.cpp ${STRING_LIB_SOURCES})
target_compile_options(tests_string PRIVATE -falign-loops=32)

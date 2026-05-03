cmake_minimum_required(VERSION 3.10)

add_library(regression SHARED regression_cpu.c)
target_include_directories(regression PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(regression PUBLIC m)

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(regression PUBLIC OpenMP::OpenMP_C)
endif()

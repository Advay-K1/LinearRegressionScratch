cmake_minimum_required(VERSION 3.14)
project(LinearRegLibrary)

find_package(xtensor REQUIRED)

add_executable(regression_lib src/LinearRegression.cpp includes/LinearRegression.hpp)

target_include_directories(regression_lib PRIVATE includes)

target_link_libraries(regression_lib PRIVATE xtensor)

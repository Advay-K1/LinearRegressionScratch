# cmake_minimum_required(VERSION 3.14)
# project(LinearRegLibrary)

# find_package(xtensor REQUIRED)

# add_executable(regression_lib src/LinearRegression.cpp includes/LinearRegression.hpp)

# target_include_directories(regression_lib PRIVATE includes)

# target_link_libraries(regression_lib PRIVATE xtensor)


# cmake_minimum_required(VERSION 3.14)
# project(LinearRegLibrary)

# # Find xtensor package
# find_package(xtensor REQUIRED)


# include_directories(/usr/local/include/)

# add_executable(regression_lib src/LinearRegression.cpp includes/LinearRegression.hpp)

# target_include_directories(regression_lib PRIVATE includes)

# target_link_libraries(regression_lib PRIVATE xtensor)



cmake_minimum_required(VERSION 3.14)
project(LinearRegression)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(xtensor REQUIRED)

include_directories("/usr/local/include")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

include_directories("${CMAKE_SOURCE_DIR}/include")

add_executable(LinearRegression src/main.cpp src/LinearRegression.cpp)

target_link_libraries(LinearRegression PRIVATE xtensor BLAS::BLAS LAPACK::LAPACK)

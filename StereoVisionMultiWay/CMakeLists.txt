cmake_minimum_required(VERSION 3.11)
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
project(StereoVisionMultiWay LANGUAGES C CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# This is set separately because nvcc doesn't understand these flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native -std=c++17")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -march=native")
set(OpenCV_DIR "/global/homes/z/zeid/opencv/build")


find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
# find_package(OpenCL REQUIRED)

#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_VERBOSE_MAKEFILE ON)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
)

add_executable(GenerateDisparityVisualization 
    src/GenerateDisparityVisualization.cpp
    src/CudaFunctions.cu
    src/CudaSimdFunctions.cu
    src/CudaDisparityMapGenerator.cpp
    src/CudaSimdDisparityMapGenerator.cpp
    src/DisparityMapGeneratorFactory.cpp
    src/OpenClDisparityMapGenerator.cpp
    src/OpenMpThreadedDisparityMapGenerator.cpp
    src/OpenMpThreadedSimdDisparityMapGenerator.cpp
    src/SingleThreadedDisparityMapGenerator.cpp
    src/SingleThreadedSimdDisparityMapGenerator.cpp)

target_link_libraries(GenerateDisparityVisualization
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARY_DIRS}
  ${OpenCL_LIBRARY}
)

add_custom_command(
    TARGET GenerateDisparityVisualization
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/src/OpenClFunctions.cl
    ${CMAKE_CURRENT_BINARY_DIR}/OpenClFunctions.cl)

add_executable(SpeedTest 
    src/SpeedTest.cpp
    src/CudaFunctions.cu
    src/CudaSimdFunctions.cu
    src/CudaDisparityMapGenerator.cpp
    src/CudaSimdDisparityMapGenerator.cpp
    src/DisparityMapGeneratorFactory.cpp
    src/OpenClDisparityMapGenerator.cpp
    src/OpenMpThreadedDisparityMapGenerator.cpp
    src/OpenMpThreadedSimdDisparityMapGenerator.cpp
    src/SingleThreadedDisparityMapGenerator.cpp
    src/SingleThreadedSimdDisparityMapGenerator.cpp)

target_link_libraries(SpeedTest
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARY_DIRS}
  ${OpenCL_LIBRARY}
)

add_custom_command(
    TARGET SpeedTest
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/src/OpenClFunctions.cl
    ${CMAKE_CURRENT_BINARY_DIR}/OpenClFunctions.cl)

add_executable(TestSadSimd
    src/TestSadSimd.cpp)

set_target_properties(TestSadSimd
    PROPERTIES
    COMPILE_FLAGS
    "-save-temps")

target_link_libraries(TestSadSimd
    ${OpenCV_LIBRARIES}
)
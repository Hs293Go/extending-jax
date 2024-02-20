cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)
message(STATUS "Using CMake version: " ${CMAKE_VERSION})

# Find pybind11
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

# CPU op library
pybind11_add_module(cpu_ops ${CMAKE_CURRENT_LIST_DIR}/src/cpu_ops.cc)
install(TARGETS cpu_ops LIBRARY DESTINATION .)

# Include the CUDA extensions if possible
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  pybind11_add_module(gpu_ops ${CMAKE_CURRENT_LIST_DIR}/src/kernels.cc.cu
                      ${CMAKE_CURRENT_LIST_DIR}/src/gpu_ops.cc)
  target_compile_features(gpu_ops PRIVATE cxx_std_17)
  target_link_libraries(gpu_ops PRIVATE Eigen3::Eigen)
  install(TARGETS gpu_ops LIBRARY DESTINATION .)
else()
  message(STATUS "Building without CUDA")
endif()

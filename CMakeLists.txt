cmake_minimum_required(VERSION 3.28)
project(DeepLOB)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)

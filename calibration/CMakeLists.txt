cmake_minimum_required(VERSION 3.5)
project(calibration)

find_package(OpenCV 4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
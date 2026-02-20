cmake_minimum_required (VERSION 2.6)
project (cmbp2024)
set (CMAKE_CXX_FLAGS "-Wall -std=c++11 -g -fopenmp -O3 -pthread")

include_directories(graphics)


add_executable(xor_test 
    main.cpp
)
add_executable(filetrain 
    filetrain.cpp
)

add_executable(graphics_demo 
    graphics_demo.cpp
)

add_executable(thread_test
    thread_test.cpp
)
add_executable(benchmark
    benchmark.cpp
)



target_link_libraries(graphics_demo ${CMAKE_SOURCE_DIR}/graphics/libgraphics.a SDL2 SDL2_ttf)

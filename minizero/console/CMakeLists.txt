file(GLOB SRCS *.cpp)

add_library(console ${SRCS})
target_include_directories(console PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    console
    actor
    config
    environment
    network
    iig_data_generator
    utils
    zero
    ${TORCH_LIBRARIES}
)
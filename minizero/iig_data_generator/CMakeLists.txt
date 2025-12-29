file(GLOB SRCS *.cpp)

add_library(iig_data_generator ${SRCS})
target_include_directories(iig_data_generator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    iig_data_generator
    config
    environment
    network
    utils
    ${TORCH_LIBRARIES}
)
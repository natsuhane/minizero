file(GLOB SRCS *.cpp)

add_library(iig ${SRCS})
target_include_directories(iig PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    iig
    actor
    config
    environment
    network
    utils
    ${TORCH_LIBRARIES}
)
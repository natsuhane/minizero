#pragma once

#include "alphazero_network.h"
#include "info_set_generator_network.h"
#include "muzero_network.h"
#include "network.h"
#include "siamese_network.h"
#include <memory>
#include <string>

namespace minizero::network {

inline std::shared_ptr<Network> createNetwork(const std::string& nn_file_name, const int gpu_id)
{
    // TODO: how to speed up?
    Network base_network;
    base_network.loadModel(nn_file_name, -1);

    std::shared_ptr<Network> network;
    if (base_network.getNetworkTypeName() == "alphazero") {
        network = std::make_shared<AlphaZeroNetwork>();
        std::dynamic_pointer_cast<AlphaZeroNetwork>(network)->loadModel(nn_file_name, gpu_id);
    } else if (base_network.getNetworkTypeName() == "muzero" || base_network.getNetworkTypeName() == "muzero_atari") {
        network = std::make_shared<MuZeroNetwork>();
        std::dynamic_pointer_cast<MuZeroNetwork>(network)->loadModel(nn_file_name, gpu_id);
    } else if (base_network.getNetworkTypeName() == "siamese") {
        network = std::make_shared<SiameseNetwork>();
        std::dynamic_pointer_cast<SiameseNetwork>(network)->loadModel(nn_file_name, gpu_id);
    } else if (base_network.getNetworkTypeName() == "info_set_generator") {
        network = std::make_shared<InfoSetGeneratorNetwork>();
        std::dynamic_pointer_cast<InfoSetGeneratorNetwork>(network)->loadModel(nn_file_name, gpu_id);
    } else {
        // should not be here
        assert(false);
    }

    return network;
}

} // namespace minizero::network

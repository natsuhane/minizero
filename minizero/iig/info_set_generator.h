#pragma once

#include "environment.h"
#include "info_set_generator_network.h"
#include "network.h"
#include <memory>
#include <vector>

namespace minizero::iig {

struct ISItem {
    Environment env_;
    float acc_prob_;
    std::vector<float> probs_;
    std::vector<float> feature_;
};

class InfoSetGenerator {
public:
    InfoSetGenerator(std::shared_ptr<minizero::network::Network> network)
        : network_(std::static_pointer_cast<minizero::network::InfoSetGeneratorNetwork>(network)) {}

    std::vector<ISItem> generate(const Environment& env, int max_info_set_size = minizero::config::iig_max_infoset_size);
    std::vector<std::vector<ISItem>> generate(const std::vector<Environment>& envs, int max_info_set_size = minizero::config::iig_max_infoset_size);

private:
    std::shared_ptr<minizero::network::InfoSetGeneratorNetwork> network_;
};

} // namespace minizero::iig

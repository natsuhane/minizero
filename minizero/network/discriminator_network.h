#pragma once

#include "configuration.h"
#include "network.h"
#include "utils.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace minizero::network {

class DiscriminatorNetworkOutput : public NetworkOutput {
public:
    float value_;

    DiscriminatorNetworkOutput()
    {
        value_ = 0.0f;
    }
};

class DiscriminatorNetwork : public Network {
public:
    DiscriminatorNetwork()
    {
        clear();
    }

    void loadModel(const std::string& nn_file_name, const int gpu_id) override
    {
        assert(batch_size_ == 0); // should avoid loading model when batch size is not 0
        Network::loadModel(nn_file_name, gpu_id);
        clear();
    }

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << Network::toString();
        return oss.str();
    }

    int pushBack(std::vector<float> features)
    {
        const int num_board_input_channels = config::iig_discriminator_feature_channels;
        assert(static_cast<int>(features.size()) == num_board_input_channels * getInputChannelHeight() * getInputChannelWidth());
        assert(batch_size_ < kReserved_batch_size);

        int index;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            index = batch_size_++;
            tensor_input_.resize(batch_size_);
        }
        tensor_input_[index] = torch::from_blob(features.data(), {1, num_board_input_channels, getInputChannelHeight(), getInputChannelWidth()}).clone();
        return index;
    }

    std::vector<std::shared_ptr<NetworkOutput>> forward()
    {
        assert(batch_size_ > 0);
        auto forward_result = network_.forward(std::vector<torch::jit::IValue>{torch::cat(tensor_input_).to(getDevice())}).toGenericDict();

        auto value_output = forward_result.at("value").toTensor().to(at::kCPU);
        assert(value_output.numel() == batch_size_);

        std::vector<std::shared_ptr<NetworkOutput>> network_outputs;
        for (int i = 0; i < batch_size_; ++i) {
            network_outputs.emplace_back(std::make_shared<DiscriminatorNetworkOutput>());
            auto discriminator_network_output = std::static_pointer_cast<DiscriminatorNetworkOutput>(network_outputs.back());
            discriminator_network_output->value_ = value_output[i].item<float>();
        }
        clear();
        return network_outputs;
    }

    inline int getBatchSize() const { return batch_size_; }

protected:
    inline void clear()
    {
        batch_size_ = 0;
        tensor_input_.clear();
        tensor_input_.reserve(kReserved_batch_size);
    }

    int batch_size_;
    std::mutex mutex_;
    std::vector<torch::Tensor> tensor_input_;

    const int kReserved_batch_size = 16384;
};

} // namespace minizero::network

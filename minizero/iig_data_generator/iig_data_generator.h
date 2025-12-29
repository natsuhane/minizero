#pragma once

#include "environment.h"
#include "go.h"
#include "network.h"
#include "paralleler.h"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace minizero::iig_data_generator {

class ThreadSharedData : public utils::BaseSharedData {
public:
    std::size_t getAvailableGameIndex();
    void outputGames(const std::string& sgf);
    std::vector<std::shared_ptr<minizero::network::NetworkOutput>> gpuForward(int gpu_id, const std::vector<std::vector<float>>& features);

    std::size_t game_index_;
    std::mutex mutex_;
    std::ofstream fout_;
    std::vector<std::string> sgfs_;
    std::vector<std::shared_ptr<network::Network>> networks_;
};

class SlaveThread : public utils::BaseSlaveThread {
public:
    SlaveThread(int id, std::shared_ptr<utils::BaseSharedData> shared_data)
        : BaseSlaveThread(id, shared_data) {}

    void initialize() override { is_done = false; }
    void runJob() override;
    bool isDone() override { return is_done; }

private:
    bool is_done;

    std::vector<minizero::env::GamePair<minizero::env::go::GoBitboard>> randomGenerateEnv(Environment& env, int K, int distance);
    std::vector<int> filterBoards(Environment& env, std::vector<minizero::env::GamePair<minizero::env::go::GoBitboard>>& negative_outputs, float threshold);
    std::string idtoString(const std::vector<int>& neg_ids);
    inline std::shared_ptr<ThreadSharedData> getSharedData() { return std::static_pointer_cast<ThreadSharedData>(shared_data_); }
};

class IIGDataGenerator : public utils::BaseParalleler {
public:
    IIGDataGenerator() {}

    void initialize() override;
    void summarize() override {}

private:
    virtual void createNeuralNetworks();

    void createSharedData() override { shared_data_ = std::make_shared<ThreadSharedData>(); }
    std::shared_ptr<utils::BaseSlaveThread> newSlaveThread(int id) override { return std::make_shared<SlaveThread>(id, shared_data_); }
    inline std::shared_ptr<ThreadSharedData> getSharedData() { return std::static_pointer_cast<ThreadSharedData>(shared_data_); }
};

} // namespace minizero::iig_data_generator

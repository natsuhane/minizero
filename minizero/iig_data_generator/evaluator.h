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

class EvaluatorSharedData : public utils::BaseSharedData {
public:
    std::size_t getAvailableGameIndex();
    std::vector<std::shared_ptr<minizero::network::NetworkOutput>> gpuForward(int nn_id, const std::vector<std::vector<float>>& features);

    std::size_t game_index_;
    std::mutex mutex_;
    std::vector<std::string> sgfs_;
    std::vector<std::shared_ptr<std::mutex>> nn_mutexs_;
    std::vector<std::shared_ptr<network::Network>> networks_;
};

class EvaluatorThread : public utils::BaseSlaveThread {
public:
    EvaluatorThread(int id, std::shared_ptr<utils::BaseSharedData> shared_data)
        : BaseSlaveThread(id, shared_data) {}

    void initialize() override { is_done_ = false; }
    void runJob() override;
    bool isDone() override { return is_done_; }

private:
    bool is_done_;

    void evaluateOneGame(const std::string& sgf);
    inline std::shared_ptr<EvaluatorSharedData> getSharedData() { return std::static_pointer_cast<EvaluatorSharedData>(shared_data_); }
};

class Evaluator : public utils::BaseParalleler {
public:
    Evaluator() {}

    void initialize() override;
    void summarize() override;

private:
    virtual void createNeuralNetworks();

    void createSharedData() override { shared_data_ = std::make_shared<EvaluatorSharedData>(); }
    std::shared_ptr<utils::BaseSlaveThread> newSlaveThread(int id) override { return std::make_shared<EvaluatorThread>(id, shared_data_); }
    inline std::shared_ptr<EvaluatorSharedData> getSharedData() { return std::static_pointer_cast<EvaluatorSharedData>(shared_data_); }
};

} // namespace minizero::iig_data_generator

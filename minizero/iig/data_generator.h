#pragma once

#include "environment.h"
#include "go.h"
#include "network.h"
#include "paralleler.h"
#include <array>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace minizero::iig {

class ThreadSharedData : public utils::BaseSharedData {
public:
    std::size_t getAvailableGameIndex();

    // for statistics
    void addNumNeg(int n);
    void addValueSample(float v, minizero::env::Player player, int threshold_idx);
    void writeValueHistogramCSV(const std::string& path);
    void addValueDiff(float v, minizero::env::Player player);
    void writeValueDiffHistogramCSV(const std::string& path);
    void addNumPos(float v, minizero::env::Player player);
    void writePosHistogramCSV(const std::string& path);

    void outputGames(const std::string& sgf);
    std::vector<std::shared_ptr<minizero::network::NetworkOutput>> gpuForward(int nn_id, const std::vector<std::vector<float>>& features);

    std::size_t game_index_;
    std::mutex mutex_;
    std::ofstream fout_;
    std::vector<std::string> sgfs_;
    std::vector<std::shared_ptr<std::mutex>> nn_mutexs_;
    std::vector<std::shared_ptr<network::Network>> networks_;

    // for statistics
    int total_steps_ = 0;
    int total_neg_ = 0;
    // histogram
    static constexpr float kValueMin = -1.0f;
    static constexpr float kValueMax = 1.0f;
    static constexpr float kBinWidth = 0.2f;
    static constexpr int kNumBins = 10;
    static constexpr float kPosBinWidth = 0.05f;
    static constexpr int kNumPosBins = 40;

    // value thresshold = 0.1, 0.2, ..., 1.0
    minizero::env::GamePair<std::array<std::array<std::int64_t, kNumBins>, 10>> value_hist_;
    minizero::env::GamePair<std::array<std::int64_t, 20>> value_diff_;
    minizero::env::GamePair<std::array<std::int64_t, kNumPosBins>> pos_hist_;
};

class SlaveThread : public utils::BaseSlaveThread {
public:
    SlaveThread(int id, std::shared_ptr<utils::BaseSharedData> shared_data)
        : BaseSlaveThread(id, shared_data) {}

    void initialize() override { is_done_ = false; }
    void runJob() override;
    bool isDone() override { return is_done_; }

private:
    class EnvWithLegalActions {
    public:
        bool is_valid;
        Environment env;
        std::vector<std::pair<Action, float>> legal_actions;

        EnvWithLegalActions() { is_valid = true; }
        EnvWithLegalActions(const Environment& e, const std::vector<std::pair<Action, float>>& la) : env(e), legal_actions(la) { is_valid = true; }
    };

    void statistic(const Environment& true_env, const std::vector<EnvWithLegalActions>& info_set_envs, env::Player turn);
    void genNegativeByPolicy(EnvironmentLoader& env_loader);
    void assignLegalActionProbabilities(std::vector<EnvWithLegalActions>& info_set_envs, const std::vector<std::shared_ptr<minizero::network::NetworkOutput>>& nn_outputs);
    void verification(const EnvironmentLoader& true_env_loader, const std::vector<std::vector<std::string>>& verification_sgfs);
    std::vector<int> filterBoards(const Environment& env, const EnvironmentLoader& env_loader, std::vector<minizero::env::GamePair<minizero::env::go::GoBitboard>>& negative_outputs, float threshold, int game_index, int game_step);
    std::string idtoString(const std::vector<int>& neg_ids);
    inline std::shared_ptr<ThreadSharedData> getSharedData() { return std::static_pointer_cast<ThreadSharedData>(shared_data_); }
    bool is_done_;
};

class DataGenerator : public utils::BaseParalleler {
public:
    DataGenerator() {}

    void initialize() override;
    void summarize() override;

private:
    virtual void createNeuralNetworks();

    void createSharedData() override { shared_data_ = std::make_shared<ThreadSharedData>(); }
    std::shared_ptr<utils::BaseSlaveThread> newSlaveThread(int id) override { return std::make_shared<SlaveThread>(id, shared_data_); }
    inline std::shared_ptr<ThreadSharedData> getSharedData() { return std::static_pointer_cast<ThreadSharedData>(shared_data_); }
};

} // namespace minizero::iig

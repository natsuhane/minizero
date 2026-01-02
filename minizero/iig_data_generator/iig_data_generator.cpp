#include "iig_data_generator.h"
#include "alphazero_network.h"
#include "configuration.h"
#include "create_network.h"
#include <algorithm>
#include <memory>
#include <random>
#include <torch/cuda.h>

namespace minizero::iig_data_generator {

using namespace network;

std::size_t ThreadSharedData::getAvailableGameIndex()
{
    std::lock_guard lock(mutex_);
    return (game_index_ < sgfs_.size() ? game_index_++ : sgfs_.size());
}

void ThreadSharedData::outputGames(const std::string& sgf)
{
    std::lock_guard lock(mutex_);
    fout_ << sgf << std::endl;
}

std::vector<std::shared_ptr<NetworkOutput>> ThreadSharedData::gpuForward(int nn_id, const std::vector<std::vector<float>>& features)
{
    std::lock_guard lock(*nn_mutexs_[nn_id]);
    std::shared_ptr<AlphaZeroNetwork> az_network = std::static_pointer_cast<AlphaZeroNetwork>(networks_[nn_id]);
    for (auto& feature : features) { az_network->pushBack(feature); }
    return az_network->forward();
}

void SlaveThread::runJob()
{
    while (true) {
        size_t game_index = getSharedData()->getAvailableGameIndex();
        if (game_index >= getSharedData()->sgfs_.size()) {
            is_done = true;
            return;
        }
            EnvironmentLoader env_loader;
            const std::string& sgf = getSharedData()->sgfs_[game_index];
            if (!env_loader.loadFromString(sgf)) { continue; }

            Environment env;
            std::vector<int> neg_ids;
            for (auto& action_pair : env_loader.getActionPairs()) {
                env.act(action_pair.first);

                auto negative_outputs = env_loader.generateNegativeBitboards(env, config::siamese_max_random_perturbations, true);
                neg_ids = filterBoards(env, env_loader, negative_outputs, config::siamese_value_threshold);
                action_pair.second["N"] = idtoString(neg_ids);
            }
            env_loader.addTag("I", std::to_string(game_index));
            getSharedData()->outputGames(env_loader.toString());
        }
}

std::vector<int> SlaveThread::filterBoards(
    const Environment& env,
    const EnvironmentLoader& env_loader,
    std::vector<env::GamePair<env::go::GoBitboard>>& negative_outputs,
    float threshold)
{
    std::vector<std::vector<float>> features;
    features.push_back(env.getFeatures()); // positive board
    for (auto& pair : negative_outputs) { features.push_back(env_loader.bitboardToFeature(pair, env.getTurn(), utils::Rotation::kRotationNone, true)); }

    auto network_output = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), features);
    std::shared_ptr<network::AlphaZeroNetworkOutput> output_pos = std::static_pointer_cast<network::AlphaZeroNetworkOutput>(network_output[0]);
    std::vector<int> neg_ids;
    for (size_t i = 1; i < network_output.size(); i++) {
        std::shared_ptr<network::AlphaZeroNetworkOutput> output_neg = std::static_pointer_cast<network::AlphaZeroNetworkOutput>(network_output[i]);
        if (abs(output_neg->value_ - output_pos->value_) < threshold) { neg_ids.push_back(i - 1); }
    }
    return neg_ids;
}

std::string SlaveThread::idtoString(const std::vector<int>& neg_ids)
{
    if (neg_ids.empty()) { return ""; }

    std::stringstream oss;
    oss << neg_ids[0];
    for (size_t i = 1; i < neg_ids.size(); ++i) {
        oss << "," << neg_ids[i];
    }
    return oss.str();
}

void IIGDataGenerator::initialize()
{
    int num_threads = std::max(static_cast<int>(torch::cuda::device_count()), config::zero_num_threads);
    std::cerr << "Create threads ..." << std::endl;
    createSlaveThreads(num_threads);
    std::cerr << "Create neural networks ... " << std::endl;
    createNeuralNetworks();

    // load games
    std::cerr << "Load games ..." << std::endl;
    std::ifstream fin(config::siamese_input_record_file_name);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << config::siamese_input_record_file_name << std::endl;
        exit(-1);
    }
    getSharedData()->sgfs_.clear();
    for (std::string sgf; std::getline(fin, sgf);) { getSharedData()->sgfs_.push_back(sgf); }

    // output file
    getSharedData()->fout_.open(config::siamese_output_record_file_name, std::ios::out);

    std::cerr << "Finish initializing" << std::endl;
}

void IIGDataGenerator::createNeuralNetworks()
{
    int num_networks = std::min(static_cast<int>(torch::cuda::device_count()), config::zero_num_parallel_games);
    const int num_networks_per_GPU = config::num_networks_per_GPU;
    assert(num_networks > 0);
    getSharedData()->networks_.resize(num_networks_per_GPU * num_networks);
    for (int gpu_id = 0; gpu_id < num_networks_per_GPU * num_networks; ++gpu_id) {
        getSharedData()->nn_mutexs_.push_back(std::make_shared<std::mutex>());
        getSharedData()->networks_[gpu_id] = createNetwork(config::nn_file_name, gpu_id % num_networks);
    }
}

} // namespace minizero::iig_data_generator

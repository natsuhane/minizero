#include "evaluator.h"
#include "configuration.h"
#include "create_network.h"
#include "go.h"
#include "random.h"
#include "siamese_network.h"
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <cmath>   // std::log10
#include <iomanip> // std::setprecision
#include <iostream>
#include <memory>
#include <random>
#include <torch/cuda.h>

namespace minizero::iig_data_generator {

using namespace network;
using namespace utils;

std::size_t EvaluatorSharedData::getAvailableGameIndex()
{
    std::lock_guard lock(mutex_);
    return (game_index_ < sgfs_.size() ? game_index_++ : sgfs_.size());
}

std::vector<std::shared_ptr<NetworkOutput>> EvaluatorSharedData::gpuForward(int nn_id, const std::vector<std::vector<float>>& features)
{
    std::lock_guard lock(*nn_mutexs_[nn_id]);

    if (networks_[nn_id]->getNetworkTypeName() == "siamese") {
        std::shared_ptr<SiameseNetwork> siamese_network = std::static_pointer_cast<SiameseNetwork>(networks_[nn_id]);
        for (auto& feature : features) {
            if (feature.size() > config::siamese_nn_feature_channels * config::env_board_size * config::env_board_size) {
                siamese_network->pushBackAnchor(feature);
            } else {
                siamese_network->pushBackBoard(feature);
            }
        }
        return siamese_network->forward();
    } else {
        std::cerr << "Unknown network type: " << networks_[nn_id]->getNetworkTypeName() << std::endl;
        return {};
    }
}

void EvaluatorSharedData::calcAvgRank(int rank, int num_negatives)
{
    std::lock_guard lock(mutex_);
    avg_rank_ += rank / num_negatives;
    total_steps_++;
}

void EvaluatorSharedData::addRank1()
{
    std::lock_guard lock(mutex_);
    rank_one_++;
}

void EvaluatorThread::runJob()
{
    while (true) {
        size_t game_index = getSharedData()->getAvailableGameIndex();
        if (game_index >= getSharedData()->sgfs_.size()) {
            is_done_ = true;
            return;
        }
        // print progress
        if (game_index % 500 == 0) { std::cerr << "Processing game " << game_index << std::endl; }

        // evaluate one game
        if (getSharedData()->networks_[0]->getNetworkTypeName() == "siamese") {
            evaluateSiamese(getSharedData()->sgfs_[game_index]);
        } else {
            std::cerr << "Unknown network type from siamese_nn_file_name: " << getSharedData()->networks_[0]->getNetworkTypeName() << std::endl;
            is_done_ = true;
            return;
        }
    }
}

void EvaluatorThread::evaluateSiamese(const std::string& sgf)
{
    EnvironmentLoader env_loader;
    if (!env_loader.loadFromString(sgf)) { return; }

    std::vector<std::vector<float>> anchors, positives;
    for (size_t pos = 0; pos < env_loader.getActionPairs().size(); ++pos) {
        // TODO: support random rotation?
        std::vector<float> anchor = env_loader.getAnchor(pos, utils::Rotation::kRotationNone);
        anchors.push_back(anchor);
        std::vector<float> positive = env_loader.getPositive(pos, utils::Rotation::kRotationNone);
        positives.push_back(positive);
    }
    auto anchor_res = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), anchors);
    auto positive_res = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), positives);

    for (size_t pos = 0; pos < env_loader.getActionPairs().size(); ++pos) {
        int num_negatives = std::stoi(env_loader.getActionPairs()[pos].second["N"]);
        if (num_negatives == 0) { continue; }

        std::vector<std::vector<float>> negatives;
        for (int neg_id = 0; neg_id < num_negatives; ++neg_id) {
            std::vector<float> negative = env_loader.getNegative(pos, utils::Rotation::kRotationNone, neg_id);
            negatives.push_back(negative);
        }

        // calculate distance
        auto negative_res = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), negatives);
        auto anchor_emb = std::static_pointer_cast<SiameseNetworkOutput>(anchor_res[pos]);
        auto pos_emb = std::static_pointer_cast<SiameseNetworkOutput>(positive_res[pos]);
        float pos_distace = utils::distance(pos_emb->embeddings_, anchor_emb->embeddings_);
        int rank = 1;
        float min = 99999, max = 0, sum = 0;
        for (int neg_id = 0; neg_id < num_negatives; ++neg_id) {
            auto neg_emb = std::static_pointer_cast<SiameseNetworkOutput>(negative_res[neg_id]);
            float dis = utils::distance(neg_emb->embeddings_, anchor_emb->embeddings_);
            if (dis <= pos_distace) { ++rank; }
            min = std::min(min, dis);
            max = std::max(max, dis);
            sum += dis;
        }
        std::cerr << "pos = " << std::setw(4) << pos
                  << ", # negs = " << std::setw(4) << num_negatives
                  << ", rank = " << std::setw(4) << rank
                  << ", pos_dis = " << std::setw(8) << pos_distace
                  << ", min = " << std::setw(8) << min
                  << ", max = " << std::setw(8) << max
                  << ", avg = " << std::setw(8) << sum / num_negatives << std::endl;
        getSharedData()->calcAvgRank(rank, num_negatives);
        if (rank == 1) { getSharedData()->addRank1(); }
    }
}

void Evaluator::initialize()
{
    int num_threads = std::max(static_cast<int>(torch::cuda::device_count()), config::zero_num_threads);
    std::cerr << "Create threads ..." << std::endl;
    createSlaveThreads(num_threads);
    std::cerr << "Create neural networks ... " << std::endl;
    createNeuralNetworks();

    // load games
    std::cerr << "Load games ..." << std::endl;
    std::ifstream fin(config::siamese_eval_sgf_file_name);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << config::siamese_eval_sgf_file_name << std::endl;
        exit(-1);
    }
    getSharedData()->sgfs_.clear();
    for (std::string sgf; std::getline(fin, sgf);) { getSharedData()->sgfs_.push_back(sgf); }

    std::cerr << "Finish initializing" << std::endl;
}

void Evaluator::summarize()
{
    std::cerr << "Average rank: " << static_cast<float>(getSharedData()->avg_rank_) / getSharedData()->total_steps_ << std::endl;
    std::cerr << "Possibility of rank 1: " << static_cast<float>(getSharedData()->rank_one_) / getSharedData()->total_steps_ << std::endl;
}

void Evaluator::createNeuralNetworks()
{
    int num_networks = std::min(static_cast<int>(torch::cuda::device_count()), config::zero_num_parallel_games);
    const int num_networks_per_GPU = config::num_networks_per_GPU;
    assert(num_networks > 0);
    getSharedData()->networks_.resize(num_networks_per_GPU * num_networks);
    for (int gpu_id = 0; gpu_id < num_networks_per_GPU * num_networks; ++gpu_id) {
        getSharedData()->nn_mutexs_.push_back(std::make_shared<std::mutex>());
        getSharedData()->networks_[gpu_id] = createNetwork(config::siamese_nn_file_name, gpu_id % num_networks);
    }
}

} // namespace minizero::iig_data_generator

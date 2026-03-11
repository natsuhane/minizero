#include "evaluator.h"
#include "configuration.h"
#include "create_network.h"
#include "go.h"
#include "info_set_generator_network.h"
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

namespace minizero::iig {

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
            if (static_cast<int>(feature.size()) > config::iig_nn_feature_channels * config::env_board_size * config::env_board_size) {
                siamese_network->pushBackAnchor(feature);
            } else {
                siamese_network->pushBackBoard(feature);
            }
        }
        return siamese_network->forward();
    } else if (networks_[nn_id]->getNetworkTypeName() == "info_set_generator") {
        std::shared_ptr<InfoSetGeneratorNetwork> info_set_generator_network = std::static_pointer_cast<InfoSetGeneratorNetwork>(networks_[nn_id]);
        for (auto& feature : features) { info_set_generator_network->pushBack(feature); }
        return info_set_generator_network->forward();
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
        } else if (getSharedData()->networks_[0]->getNetworkTypeName() == "info_set_generator") {
            evaluateInfoSetGenerator(getSharedData()->sgfs_[game_index]);
        } else {
            std::cerr << "Unknown network type from iig_nn_file_name: " << getSharedData()->networks_[0]->getNetworkTypeName() << std::endl;
            is_done_ = true;
            return;
        }
    }
}

void EvaluatorThread::evaluateSiamese(const std::string& sgf)
{
    // TODO: no use for now
    // EnvironmentLoader env_loader;
    // if (!env_loader.loadFromString(sgf)) { return; }

    // std::vector<std::vector<float>> anchors, positives;
    // for (size_t pos = 0; pos < env_loader.getActionPairs().size(); ++pos) {
    //     std::vector<float> anchor = env_loader.getAnchor(pos, utils::Rotation::kRotationNone);
    //     anchors.push_back(anchor);
    //     std::vector<float> positive = env_loader.getPositive(pos, utils::Rotation::kRotationNone);
    //     positives.push_back(positive);
    // }
    // auto anchor_res = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), anchors);
    // auto positive_res = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), positives);

    // for (size_t pos = 0; pos < env_loader.getActionPairs().size(); ++pos) {
    //     int num_negatives = std::stoi(env_loader.getActionPairs()[pos].second["N"]);
    //     if (num_negatives == 0) { continue; }
    //     std::vector<std::vector<float>> negatives;
    //     for (int neg_id = 0; neg_id < num_negatives; ++neg_id) {
    //         std::vector<float> negative = env_loader.getNegative(pos, utils::Rotation::kRotationNone, neg_id);
    //         negatives.push_back(negative);
    //     }

    //     // calculate distance
    //     auto negative_res = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), negatives);
    //     auto anchor_emb = std::static_pointer_cast<SiameseNetworkOutput>(anchor_res[pos]);
    //     auto pos_emb = std::static_pointer_cast<SiameseNetworkOutput>(positive_res[pos]);
    //     float pos_distace = utils::distance(pos_emb->embeddings_, anchor_emb->embeddings_);
    //     int rank = 1;
    //     float min = 99999, max = 0, sum = 0;
    //     for (int neg_id = 0; neg_id < num_negatives; ++neg_id) {
    //         auto neg_emb = std::static_pointer_cast<SiameseNetworkOutput>(negative_res[neg_id]);
    //         float dis = utils::distance(neg_emb->embeddings_, anchor_emb->embeddings_);
    //         if (dis <= pos_distace) { ++rank; }
    //         min = std::min(min, dis);
    //         max = std::max(max, dis);
    //         sum += dis;
    //     }
    //     std::cerr << "pos = " << std::setw(4) << pos
    //               << ", # negs = " << std::setw(4) << num_negatives
    //               << ", rank = " << std::setw(4) << rank
    //               << ", pos_dis = " << std::setw(8) << pos_distace
    //               << ", min = " << std::setw(8) << min
    //               << ", max = " << std::setw(8) << max
    //               << ", avg = " << std::setw(8) << sum / num_negatives << std::endl;
    //     getSharedData()->calcAvgRank(rank, num_negatives);
    //     if (rank == 1) { getSharedData()->addRank1(); }
    // }
}

void EvaluatorThread::evaluateInfoSetGenerator(const std::string& sgf)
{
    // TODO: no use for now
    // EnvironmentLoader env_loader;
    // if (!env_loader.loadFromString(sgf)) { return; }

    // // *** pos cannot be 0 ***
    // // for (int pos = config::iig_game_step; pos < config::iig_game_step + 1; ++pos) {
    // for (int pos = 1; pos < static_cast<int>(env_loader.getActionPairs().size()); ++pos) {
    //     if (pos >= static_cast<int>(env_loader.getActionPairs().size())) { return; }
    //     Rotation rotation = config::actor_use_random_rotation_features ? static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize)) : Rotation::kRotationNone;
    //     Environment env;
    //     for (int i = 0; i < pos; ++i) { env.act(env_loader.getActionPairs()[i].first); }
    //     // std::cerr << env.toString()
    //     //           << "current turn: " << env::playerToChar(env.getTurn())
    //     //           << ", pos = " << pos << std::endl;

    //     int move_number = (pos % 2 == 0 ? 1 : 0);
    //     float min_prob = 1.0f;
    //     std::vector<std::vector<float>> features;
    //     while (move_number < pos) {
    //         features.push_back(env.getInfoSetGeneratorFeatures(move_number, rotation));

    //         // ------- print features for debugging -------
    //         // {
    //         //     std::cerr << "move number: " << move_number << ", after rotation: " << getRotationString(rotation) << std::endl;
    //         //     for (int j = 0; j < 28; ++j) {
    //         //         std::cerr << "feature channel " << j << ": \n";
    //         //         for (int row = 8; row > -1; --row) {
    //         //             for (int col = 0; col < 9; ++col) {
    //         //                 std::cerr << std::lround(features.back()[j * 81 + row * 9 + col]) << " ";
    //         //             }
    //         //             std::cerr << std::endl;
    //         //         }
    //         //         std::cerr << std::endl;
    //         //     }
    //         //   std::cerr << std::endl;
    //         // }

    //         move_number += 2;
    //     }
    //     auto res = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), features);

    //     std::vector<uint64_t> possible(101, 0);
    //     for (int threshold = 0; threshold <= 100; ++threshold) {
    //         uint64_t p = 1;
    //         float t = threshold / 100.0f;

    //         move_number = (pos % 2 == 0 ? 1 : 0);
    //         int counter = 0;

    //         while (move_number < pos) {
    //             Environment temp_env;
    //             for (int i = 0; i < move_number; ++i) { temp_env.act(env_loader.getActionPairs()[i].first); }
    //             // std::cerr << temp_env.toString()
    //             //           << "move number = " << move_number << std::endl;

    //             int ans_grid = env_loader.getActionPairs()[move_number].first.getActionID();
    //             int rotated_ans_grid = env_loader.getRotateAction(ans_grid, rotation);

    //             uint64_t count = 0;
    //             for (int i = 0; i < env_loader.getPolicySize(); ++i) {
    //                 if (std::static_pointer_cast<InfoSetGeneratorNetworkOutput>(res[counter])->policy_[i] >= t) { ++count; }
    //             }
    //             p *= std::max(uint64_t(1), count);
    //             min_prob = std::min(min_prob, std::static_pointer_cast<InfoSetGeneratorNetworkOutput>(res[counter])->policy_[rotated_ans_grid]);

    //             // ------- print labels and predictions for debugging -------
    //             // {
    //             //     std::vector<float> labels(env_loader.getPolicySize(), 0.0f);
    //             //     labels[ans_grid] = 1.0f;
    //             //     std::cerr << "Info set generator training labels: " << std::endl;
    //             //     std::cerr << "    A     B     C     D     E     F     G     H     I" << std::endl;
    //             //     for (int row = 8; row > -1; --row) {
    //             //         std::cerr << row + 1 << " ";
    //             //         for (int col = 0; col < 9; ++col) {
    //             //             int i = row * 9 + col;
    //             //             if (labels[i] > 0.5f) {
    //             //                 std::cerr << "\033[31m";
    //             //             }
    //             //             std::cerr << std::fixed << std::setprecision(3) << labels[i] << " " << "\033[0m";
    //             //         }
    //             //         std::cerr << std::endl;
    //             //     }
    //             //     std::cerr << std::endl;
    //             //     std::cerr << "Info set generator network predictions: " << std::endl;
    //             //     for (int row = 8; row > -1; --row) {
    //             //         std::cerr << row + 1 << " ";
    //             //         for (int col = 0; col < 9; ++col) {
    //             //             int i = row * 9 + col;
    //             //             int rotated_i = env_loader.getRotatePosition(i, rotation);
    //             //             float pred = std::static_pointer_cast<InfoSetGeneratorNetworkOutput>(res[counter])->policy_[rotated_i];
    //             //             if (i == ans_grid) {
    //             //                 std::cerr << "\033[31m";
    //             //             } else if (pred >= 0.5f) {
    //             //                 std::cerr << "\033[32m";
    //             //             }
    //             //             std::cerr << std::fixed << std::setprecision(3) << pred << " " << "\033[0m";
    //             //         }
    //             //         std::cerr << std::endl;
    //             //     }
    //             //     std::cerr << std::endl;
    //             // }

    //             counter++;
    //             move_number += 2;
    //         }
    //         possible[threshold] = p;
    //     }

    //     {
    //         std::lock_guard lock(getSharedData()->mutex_);
    //         for (int i = 0; i < 101; ++i) {
    //             if (min_prob * 100 >= i) { getSharedData()->corrects_[i] += 1.0f; }
    //             getSharedData()->possibles_[i] += possible[i];
    //         }
    //         ++getSharedData()->totals_;
    //     }
    // }
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
    std::ifstream fin(config::iig_eval_sgf_file_name);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << config::iig_eval_sgf_file_name << std::endl;
        exit(-1);
    }
    getSharedData()->sgfs_.clear();
    for (std::string sgf; std::getline(fin, sgf);) { getSharedData()->sgfs_.push_back(sgf); }

    getSharedData()->corrects_.clear();
    getSharedData()->corrects_.resize(101, 0.0f);
    getSharedData()->possibles_.clear();
    getSharedData()->possibles_.resize(101, 0);
    getSharedData()->totals_ = 0;

    std::cerr << "Finish initializing" << std::endl;
}

void Evaluator::summarize()
{
    // ------- for Siamese -------
    std::cerr << "Average rank: " << static_cast<float>(getSharedData()->avg_rank_) / getSharedData()->total_steps_ << std::endl;
    std::cerr << "Possibility of rank 1: " << static_cast<float>(getSharedData()->rank_one_) / getSharedData()->total_steps_ << std::endl;

    // ------- for BoardPredictor & InfoSetGenerator -------
    float total = getSharedData()->totals_;
    std::cerr << "guess correct vs infoset size (for " << total << " samples)" << std::endl;
    std::ofstream outfile("evaluation_result.csv");
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not create evaluation_result.csv" << std::endl;
        return;
    }
    outfile << "threshold,accuracy,log_avg_possibility\n";
    for (int i = 0; i <= 100; ++i) {
        float threshold = i / 100.0f;
        float correct_rate = getSharedData()->corrects_[i] / total;
        double possible_rate = static_cast<double>(getSharedData()->possibles_[i]) / total;
        std::cerr << "  threshold " << threshold << ": " << correct_rate << " " << possible_rate << std::endl;
        double log_size = std::log10(possible_rate);
        outfile << threshold << ","
                << correct_rate << ","
                << log_size << "\n";
    }
    outfile.close();
    std::cerr << "Data saved to evaluation_result.csv" << std::endl;
}

void Evaluator::createNeuralNetworks()
{
    int num_networks = std::min(static_cast<int>(torch::cuda::device_count()), config::zero_num_parallel_games);
    const int num_networks_per_GPU = config::num_networks_per_GPU;
    assert(num_networks > 0);
    getSharedData()->networks_.resize(num_networks_per_GPU * num_networks);
    for (int gpu_id = 0; gpu_id < num_networks_per_GPU * num_networks; ++gpu_id) {
        getSharedData()->nn_mutexs_.push_back(std::make_shared<std::mutex>());
        getSharedData()->networks_[gpu_id] = createNetwork(config::iig_nn_file_name, gpu_id % num_networks);
    }
}

} // namespace minizero::iig

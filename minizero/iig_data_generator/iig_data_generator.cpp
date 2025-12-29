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

std::vector<std::shared_ptr<NetworkOutput>> ThreadSharedData::gpuForward(int gpu_id, const std::vector<std::vector<float>>& features)
{
    std::lock_guard lock(mutex_);
    std::shared_ptr<AlphaZeroNetwork> az_network = std::static_pointer_cast<AlphaZeroNetwork>(networks_[gpu_id]);
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
            auto negative_outputs = randomGenerateEnv(env, config::siamese_max_random_perturbations, config::siamese_max_move_distance);
            neg_ids = filterBoards(env, negative_outputs, config::siamese_value_threshold);
            action_pair.second["N"] = idtoString(neg_ids);
        }

        env_loader.addTag("I", std::to_string(game_index));
        getSharedData()->outputGames(env_loader.toString());
    }
}

std::vector<env::GamePair<env::go::GoBitboard>> SlaveThread::randomGenerateEnv(Environment& env, int K, int distance)
{
    std::vector<env::GamePair<env::go::GoBitboard>> outputs;
    env::GamePair<env::go::GoBitboard> stone_bitboard = env.getStoneBitboard();
    env::go::GoBitboard opp_bitboard = stone_bitboard.get(env::getNextPlayer(env.getTurn(), 2)); // 361 bit

    // collect opponent stone positions
    std::vector<int> pos_list;
    while (!opp_bitboard.none()) {
        int pos = opp_bitboard._Find_first();
        opp_bitboard.reset(pos);
        pos_list.push_back(pos);
    }

    std::mt19937 random_generator;
    random_generator.seed(config::program_seed);
    std::uniform_int_distribution<int> int_distribution;
    int warmup_times = config::siamese_perturbation_warmup;
    for (int k = 0; k < K + warmup_times; k++) {
        int index = int_distribution(random_generator) % pos_list.size();
        int pos = pos_list[index];
        std::vector<int> new_pos_list;
        for (int x = -distance; x <= distance; x++) {
            for (int y = -distance; y <= distance; y++) {
                if (x == 0 && y == 0) { continue; }
                if (std::abs(x) + std::abs(y) > distance) { continue; }
                int nx = pos % minizero::config::env_board_size + x;
                int ny = pos / minizero::config::env_board_size + y;
                if (nx < 0 || nx >= minizero::config::env_board_size || ny < 0 || ny >= minizero::config::env_board_size) { continue; }
                new_pos_list.push_back(ny * minizero::config::env_board_size + nx);
            }
        }

        std::shuffle(new_pos_list.begin(), new_pos_list.end(), random_generator);
        for (size_t i = 0; i < new_pos_list.size(); i++) {
            int new_pos = new_pos_list[i];
            if (stone_bitboard.get(env::Player::kPlayer1).test(new_pos) ||
                stone_bitboard.get(env::Player::kPlayer2).test(new_pos)) {
                continue;
            }
            pos_list[index] = new_pos;
            stone_bitboard.get(env::getNextPlayer(env.getTurn(), 2)).reset(pos);
            stone_bitboard.get(env::getNextPlayer(env.getTurn(), 2)).set(new_pos);

            Environment test;
            env::go::GoBitboard b = stone_bitboard.get(env::Player::kPlayer1);
            env::go::GoBitboard w = stone_bitboard.get(env::Player::kPlayer2);

            while (!b.none()) {
                int p = b._Find_first();
                b.reset(p);
                test.act(env::go::GoAction(p, env::Player::kPlayer1));
            }
            while (!w.none()) {
                int p = w._Find_first();
                w.reset(p);
                test.act(env::go::GoAction(p, env::Player::kPlayer2));
            }
            if (test.getStoneBitboard().get(env::Player::kPlayer1) != stone_bitboard.get(env::Player::kPlayer1) ||
                test.getStoneBitboard().get(env::Player::kPlayer2) != stone_bitboard.get(env::Player::kPlayer2)) {
                stone_bitboard.get(env::getNextPlayer(env.getTurn(), 2)).reset(new_pos);
                stone_bitboard.get(env::getNextPlayer(env.getTurn(), 2)).set(pos);
                continue;
            }

            if (k >= warmup_times) { outputs.emplace_back(stone_bitboard); }
            break;
        }
    }
    return outputs;
}

std::vector<int> SlaveThread::filterBoards(
    Environment& env,
    std::vector<env::GamePair<env::go::GoBitboard>>& negative_outputs,
    float threshold)
{
    std::vector<std::vector<float>> features;

    features.push_back(env.getFeatures()); // positive board

    for (auto& pair : negative_outputs) {
        std::vector<float> feature(env.getNumInputChannels() * env.getBoardSize() * env.getBoardSize());

        // Go related
        env::go::GoBitboard bits = pair.get(env::Player::kPlayer1); // 361 bits, use smallest 81 bits (9x9)
        for (size_t i = 0; i < 81; ++i) { feature[i] = (bits[i] ? 1.0f : 0.0f); }
        bits = pair.get(env::Player::kPlayer2);
        for (size_t i = 0; i < 81; ++i) { feature[i + 81] = (bits[i] ? 1.0f : 0.0f); }
        for (int i = 2 * 81; i < 16 * 81; ++i) { feature[i] = feature[i % (2 * 81)]; }

        std::fill_n(feature.begin() + 16 * 81, 81, env.getTurn() == env::Player::kPlayer1 ? 1.0f : 0.0f);
        std::fill_n(feature.begin() + 17 * 81, 81, env.getTurn() == env::Player::kPlayer2 ? 1.0f : 0.0f);

        features.push_back(feature);
    }

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
    assert(num_networks > 0);
    getSharedData()->networks_.resize(num_networks);
    for (int gpu_id = 0; gpu_id < num_networks; ++gpu_id) {
        getSharedData()->networks_[gpu_id] = createNetwork(config::nn_file_name, gpu_id);
    }
}

} // namespace minizero::iig_data_generator

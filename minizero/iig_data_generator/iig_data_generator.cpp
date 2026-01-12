#include "iig_data_generator.h"
#include "alphazero_network.h"
#include "configuration.h"
#include "create_network.h"
#include "go.h"
#include "random.h"
#include <algorithm>
#include <memory>
#include <random>
#include <torch/cuda.h>
#include <utility>

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
        // print progress
        if (game_index % 500 == 0) {
            std::cerr << "Processing game " << game_index << std::endl;
        }

        EnvironmentLoader env_loader;
        const std::string& sgf = getSharedData()->sgfs_[game_index];
        if (!env_loader.loadFromString(sgf)) { continue; }

        env_loader.addTag("I", std::to_string(game_index));
        genNegativeByPolicy(env_loader);
    }
}

void SlaveThread::genNegativeByPolicy(EnvironmentLoader& env_loader)
{
    // for first move
    env::GamePair<std::vector<EnvWithLegalActions>> info_set_envs;
    auto results = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), {Environment().getFeatures()});
    for (auto& turn : {env::Player::kPlayer1, env::Player::kPlayer2}) {
        info_set_envs.get(turn).push_back(EnvWithLegalActions(Environment(), {}));
        assignLegalActionProbabilities(info_set_envs.get(turn), results);
    }

    // for each move
    Environment true_env;
    for (size_t pos = 0; pos < env_loader.getActionPairs().size(); ++pos) {
        Action action = env_loader.getActionPairs()[pos].first;
        env::Player turn = action.getPlayer();
        env::Player next_turn = action.nextPlayer();
        true_env.act(action);

        // my turn, act true action + forward
        std::vector<std::vector<float>> features;
        for (auto& env_with_actions : info_set_envs.get(turn)) {
            if (!env_with_actions.env.isLegalAction(action)) { env_with_actions.is_valid = false; }
            env_with_actions.env.act(action);
            // TODO: feature rotation? (maple)
            features.push_back(env_with_actions.env.getFeatures());
        }
        auto results = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), features);
        assignLegalActionProbabilities(info_set_envs.get(turn), results);

        // generate next envs for next turn
        std::ostringstream oss;
        std::vector<EnvWithLegalActions> next_info_set_envs;
        for (size_t i = 0; i < info_set_envs.get(next_turn).size(); ++i) {
            EnvWithLegalActions& env_with_actions = info_set_envs.get(next_turn)[i];
            for (const auto& action_pair : env_with_actions.legal_actions) {
                Environment env_copy = env_with_actions.env;
                env_copy.act(action_pair.first);
                next_info_set_envs.push_back(EnvWithLegalActions(env_copy, {}));
                oss << utils::SGFLoader::actionIDToSGFString(action_pair.first.getActionID(), env_copy.getBoardSize());
                if (static_cast<int>(next_info_set_envs.size()) >= config::siamese_max_num_negatives) { break; }
            }
            if (static_cast<int>(next_info_set_envs.size()) >= config::siamese_max_num_negatives) { break; }
            if (i != info_set_envs.get(next_turn).size() - 1) { oss << ";"; }
        }
        env_loader.getActionPairs()[pos].second["A"] = oss.str();
        env_loader.getActionPairs()[pos].second["N"] = std::to_string(next_info_set_envs.size());
        if (next_info_set_envs.empty()) { next_info_set_envs.push_back(EnvWithLegalActions(true_env, {})); }
        info_set_envs.get(next_turn) = next_info_set_envs;

    }

    getSharedData()->outputGames(env_loader.toString());
}

void SlaveThread::assignLegalActionProbabilities(std::vector<EnvWithLegalActions>& info_set_envs, const std::vector<std::shared_ptr<NetworkOutput>>& nn_outputs)
{
    const float kPolicyThreshold = config::siamese_generator_policy_threshold;
    for (size_t i = 0; i < info_set_envs.size(); ++i) {
        EnvWithLegalActions& env_with_actions = info_set_envs[i];
        if (!env_with_actions.is_valid) { continue; }

        std::vector<std::pair<Action, float>>& legal_actions = env_with_actions.legal_actions;
        legal_actions.clear();
        std::shared_ptr<AlphaZeroNetworkOutput> output = std::static_pointer_cast<AlphaZeroNetworkOutput>(nn_outputs[i]);
        for (auto& action : env_with_actions.env.getLegalActions()) { legal_actions.push_back({action, output->policy_[action.getActionID()]}); }
        std::sort(legal_actions.begin(), legal_actions.end(), [](const std::pair<Action, float>& a, const std::pair<Action, float>& b) { return a.second > b.second; });
        for (size_t size_with_threshold = 0; size_with_threshold < legal_actions.size(); ++size_with_threshold) {
            if (legal_actions[size_with_threshold].second >= kPolicyThreshold) { continue; }
            legal_actions.resize(size_with_threshold + 1);
            break;
        }
    }
}

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
    std::ifstream fin(config::siamese_generator_input_sgf);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << config::siamese_generator_input_sgf << std::endl;
        exit(-1);
    }
    getSharedData()->sgfs_.clear();
    for (std::string sgf; std::getline(fin, sgf);) { getSharedData()->sgfs_.push_back(sgf); }

    // output file
    getSharedData()->fout_.open(config::siamese_generator_output_sgf, std::ios::out);

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

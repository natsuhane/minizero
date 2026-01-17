#include "iig_data_generator.h"
#include "alphazero_network.h"
#include "configuration.h"
#include "create_network.h"
#include "go.h"
#include "random.h"
#include <algorithm>
#include <filesystem>
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

void ThreadSharedData::addNumNeg(int n)
{
    std::lock_guard lock(mutex_);
    total_steps_++;
    total_neg_ += n;
}

// Num negatives vs true board value
void ThreadSharedData::addValueSample(float value, minizero::env::Player player, int threshold_idx)
{
    std::lock_guard lock(mutex_);
    int idx = static_cast<int>(std::floor((value - kValueMin) / kBinWidth));
    idx = std::max(0, std::min(idx, kNumBins - 1));
    value_hist_.get(player)[threshold_idx][idx]++;
}

void ThreadSharedData::writeValueHistogramCSV(const std::string& path)
{
    std::ofstream fout;
    for (auto& player : {env::Player::kPlayer1, env::Player::kPlayer2}) {
        fout.open(path.substr(0, path.find_last_of('.')) + "_" + std::string(1, env::playerToChar(player)) + ".csv", std::ios::out);
        if (!fout.is_open()) {
            std::cerr << "Failed to open " << path << std::endl;
            continue;
        }
        fout << "bin_left,bin_right,thresh01,thresh02,thresh03,thresh04,thresh05,thresh06,thresh07,thresh08,thresh09,thresh10\n";
        fout << std::fixed << std::setprecision(6);
        for (int i = 0; i < kNumBins; ++i) {
            float left = kValueMin + i * kBinWidth;
            float right = left + kBinWidth;
            fout << left << "," << right;
            for (int t = 0; t < 10; ++t) {
                int sum = 0;
                for (int c = 0; c <= t; ++c) { sum += value_hist_.get(player)[c][i]; }
                fout << "," << sum;
            }
            fout << "\n";
        }
        fout.close();
    }
}

// neg value - pos value
void ThreadSharedData::addValueDiff(float v_diff, minizero::env::Player player)
{
    std::lock_guard lock(mutex_);
    int idx = static_cast<int>(std::floor((v_diff + 2.0f) / 0.2f));
    idx = std::max(0, std::min(idx, 19));
    value_diff_.get(player)[idx]++;
}

void ThreadSharedData::writeValueDiffHistogramCSV(const std::string& path)
{
    std::lock_guard lock(mutex_);
    std::ofstream fout(path, std::ios::out);
    if (!fout.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        return;
    }
    fout << "bin_left,bin_right,black,white\n";
    fout << std::fixed << std::setprecision(6);

    for (int i = 0; i < 20; ++i) {
        float l = -2.0f + i * 0.2f;
        float r = l + 0.2f;
        fout << l << "," << r << "," << value_diff_.get(env::Player::kPlayer1)[i] << "," << value_diff_.get(env::Player::kPlayer2)[i] << "\n";
    }
    fout.close();
}

// calculate positive board count by value
void ThreadSharedData::addNumPos(float v, minizero::env::Player player)
{
    std::lock_guard lock(mutex_);
    int idx = static_cast<int>(std::floor((v - kValueMin) / kPosBinWidth));
    idx = std::max(0, std::min(idx, kNumPosBins - 1));
    pos_hist_.get(player)[idx]++;
}

void ThreadSharedData::writePosHistogramCSV(const std::string& path)
{
    std::lock_guard lock(mutex_);

    std::ofstream fout("statistic/pos_histogram.csv", std::ios::out);
    if (!fout.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        return;
    }
    fout << "bin_left,bin_right,black,white\n";
    fout << std::fixed << std::setprecision(6);

    for (int i = 0; i < kNumPosBins; ++i) {
        float l = kValueMin + i * kPosBinWidth;
        float r = l + kPosBinWidth;
        fout << l << "," << r << "," << pos_hist_.get(env::Player::kPlayer1)[i] << "," << pos_hist_.get(env::Player::kPlayer2)[i] << "\n";
    }
    fout.close();
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
            is_done_ = true;
            return;
        }
        // print progress
        if (game_index % 500 == 0) {
            std::cout << "Processing game " << game_index << std::endl;
        }

        EnvironmentLoader env_loader;
        const std::string& sgf = getSharedData()->sgfs_[game_index];
        if (!env_loader.loadFromString(sgf)) { continue; }

        if (env_loader.getTag("I").empty()) { std::cerr << "Source sgf doesn't include I tag." << std::endl; }
        genNegativeByPolicy(env_loader);
    }
void SlaveThread::statistic(const Environment& true_env, const std::vector<EnvWithLegalActions>& info_set_envs, env::Player turn)
{
    std::vector<std::vector<float>> features;
    features.push_back(true_env.getFeatures());
    for (const auto& env_with_actions : info_set_envs) {
        features.push_back(env_with_actions.env.getFeatures());
    }
    auto res = getSharedData()->gpuForward(id_ % getSharedData()->networks_.size(), features);
    std::shared_ptr<AlphaZeroNetworkOutput> output = std::static_pointer_cast<AlphaZeroNetworkOutput>(res[0]);
    float pos_value = output->value_;
    getSharedData()->addNumPos(pos_value, turn);

    for (size_t i = 1; i < features.size(); ++i) {
        std::shared_ptr<AlphaZeroNetworkOutput> output = std::static_pointer_cast<AlphaZeroNetworkOutput>(res[i]);
        float neg_value = output->value_;
        if (abs(neg_value - pos_value) < 1.0f) {
            getSharedData()->addValueSample(pos_value,
                                            turn,
                                            std::floor(abs(neg_value - pos_value) * 10)); // threshold = 0.1, 0.2, ..., 1.0
        }
        getSharedData()->addValueDiff(neg_value - pos_value, turn);
    }
    getSharedData()->addNumNeg(info_set_envs.size());
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
    std::vector<std::vector<std::string>> verification_sgfs(env_loader.getActionPairs().size());
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
                if (config::siamese_generator_verification) {
                    EnvironmentLoader verify_env_loader;
                    verify_env_loader.loadFromEnvironment(env_copy);
                    verification_sgfs[pos].push_back(verify_env_loader.toString());
                }
                oss << utils::SGFLoader::actionIDToSGFString(action_pair.first.getActionID(), env_copy.getBoardSize());
                if (static_cast<int>(next_info_set_envs.size()) >= config::siamese_max_num_negatives) { break; }
            }
            if (static_cast<int>(next_info_set_envs.size()) >= config::siamese_max_num_negatives) { break; }
            if (i != info_set_envs.get(next_turn).size() - 1) { oss << ";"; }
        }
        env_loader.getActionPairs()[pos].second["A"] = oss.str();
        env_loader.getActionPairs()[pos].second["N"] = std::to_string(next_info_set_envs.size());
        if (next_info_set_envs.empty()) { next_info_set_envs.push_back(EnvWithLegalActions(true_env, {})); }
        if (config::siamese_generator_statistic) { statistic(true_env, next_info_set_envs, next_turn); }
        info_set_envs.get(next_turn) = next_info_set_envs;
    }

    getSharedData()->outputGames(env_loader.toString());
    if (config::siamese_generator_verification) { verification(env_loader, verification_sgfs); }
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

void SlaveThread::verification(const EnvironmentLoader& true_env_loader, const std::vector<std::vector<std::string>>& verification_sgfs)
{
    for (size_t pos = 0; pos < true_env_loader.getActionPairs().size(); ++pos) {
        int num_negatives = std::stoi(true_env_loader.getActionPairs()[pos].second.at("N"));
        if (num_negatives != static_cast<int>(verification_sgfs[pos].size())) {
            std::cerr << "Verification failed at move " << pos << ": number of generated sgfs (" << verification_sgfs[pos].size()
                      << ") doesn't match the recorded number of negatives (" << num_negatives << ")." << std::endl;
            exit(-1);
        }

        for (int neg_id = 0; neg_id < num_negatives; ++neg_id) {
            std::vector<Action> action_history = true_env_loader.getNegativeActionHistory(pos, neg_id);
            Environment env;
            for (auto& a : action_history) { env.act(a); }
            EnvironmentLoader verify_env_loader;
            verify_env_loader.loadFromEnvironment(env);
            if (verify_env_loader.toString() != verification_sgfs[pos][neg_id]) {
                std::cerr << "Verification failed at move " << pos << ", negative id " << neg_id << ": generated sgf doesn't match the recorded sgf." << std::endl;
                exit(-1);
            }
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

void IIGDataGenerator::summarize()
{
    if (config::siamese_generator_statistic == false) { return; }
    const std::filesystem::path stat_path("statistic");
    if (!std::filesystem::exists(stat_path)) { std::filesystem::create_directory(stat_path); }

    getSharedData()->fout_.close();
    std::cerr << "avg negative boards per step = " << static_cast<float>(getSharedData()->total_neg_) / getSharedData()->total_steps_ << std::endl;

    const std::string csv_path = "statistic/value_histogram.csv";
    getSharedData()->writeValueHistogramCSV(csv_path);
    std::cerr << "value histogram csv saved to: " << csv_path << std::endl;

    const std::string diff_csv_path = "statistic/value_diff_histogram.csv";
    getSharedData()->writeValueDiffHistogramCSV(diff_csv_path);
    std::cerr << "value diff histogram csv saved to: " << diff_csv_path << std::endl;

    const std::string pos_csv_path = "statistic/pos_histogram.csv";
    getSharedData()->writePosHistogramCSV(pos_csv_path);
    std::cerr << "pos histogram csv saved to: " << pos_csv_path << std::endl;
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

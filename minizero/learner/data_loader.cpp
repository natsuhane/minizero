#include "data_loader.h"
#include "configuration.h"
#include "environment.h"
#include "random.h"
#include "rotation.h"
#include "go.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <unordered_set>
#include <set>
#include <numeric>
#include <deque>

namespace minizero::learner {

using namespace minizero;
using namespace minizero::utils;
using namespace minizero::env::go;
using minizero::env::Player;

// Helper Structures

struct MoveInfo {
    bool legal;
    std::vector<int> captured_stones;
};

struct MoveEvent {
    Player mover;
    int pos;
    std::vector<int> captured_stones;
};

struct SeqState {
    std::vector<GoAction> seq;
    GoHashKey hash;
};

// Core Helper Functions (for generating the infomation set)

GoEnv rebuildGoEnvToStep(const EnvironmentLoader& env_loader, int target_pos) {
    const int board_size = env_loader.getBoardSize();
    GoEnv env(board_size);
    
    const auto& action_pairs = env_loader.getActionPairs();
    int end_pos = std::min(target_pos, static_cast<int>(action_pairs.size()));
    
    for (int i = 0; i < end_pos; ++i) {
        if (!env.act(action_pairs[i].first)) {
            std::cerr << "Error: Failed to apply action at step " << i << std::endl;
            break;
        }
    }
    
    return env;
}

std::vector<GoEnv> rebuildFullHistory(const EnvironmentLoader& env_loader, int target_pos) {
    const int board_size = env_loader.getBoardSize();
    const auto& action_pairs = env_loader.getActionPairs();
    
    std::vector<GoEnv> history;
    history.reserve(target_pos + 1);
    
    GoEnv env(board_size);
    history.push_back(env);  // Initial state
    
    int end_pos = std::min(target_pos, static_cast<int>(action_pairs.size()));
    for (int i = 0; i < end_pos; ++i) {
        if (!env.act(action_pairs[i].first)) {
            std::cerr << "Error: Failed to apply action at step " << i << std::endl;
            break;
        }
        history.push_back(env);
    }
    
    return history;
}

MoveInfo analyzeMove(const GoEnv& before, const GoEnv& after, const GoAction& action) {
    MoveInfo info;
    info.legal = true;

    Player opponent = (action.getPlayer() == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1;
    const auto& old_stones = before.getStoneBitboard();
    const auto& new_stones = after.getStoneBitboard();

    // find captured stones
    GoBitboard removed = old_stones.get(opponent) & ~new_stones.get(opponent);
    while (!removed.none()) {
        int pos = removed._Find_first();
        removed.reset(pos);
        info.captured_stones.push_back(pos);
    }
    
    return info;
}

std::vector<MoveEvent> buildMoveEvents(
    const std::vector<GoEnv>& history,
    const EnvironmentLoader& env_loader)
{
    const auto& action_pairs = env_loader.getActionPairs();
    std::vector<MoveEvent> events;
    events.reserve(action_pairs.size());

    for (size_t k = 0; k < action_pairs.size() && k + 1 < history.size(); ++k) {
        const GoEnv& before = history[k];
        const GoEnv& after = history[k + 1];
        const GoAction& action = action_pairs[k].first;
        
        MoveInfo info = analyzeMove(before, after, action);
        MoveEvent ev{action.getPlayer(), action.getActionID(), std::move(info.captured_stones)};
        events.push_back(std::move(ev));
    }
    
    return events;
}

int countStonesOnBoard(const GoEnv& env, Player p) {
    const int N = env.getBoardSize();
    int cnt = 0;
    for (int i = 0; i < N * N; ++i) {
        if (env.getGrid(i).getPlayer() == p) ++cnt;
    }
    return cnt;
}

std::pair<std::unordered_set<int>, std::unordered_set<int>> computeMustSets(
    int move_number,
    Player perspective,
    const std::vector<MoveEvent>& events,
    const std::vector<GoEnv>& history)
{
    const int N = history[0].getBoardSize();
    const int PASS = N * N;
    auto other = [](Player p){ return (p == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1; };

    const Player MY = perspective;
    const Player OPP = other(MY);

    std::unordered_set<int> must_my, must_opp;

    for (int i = 0; i < move_number && i < static_cast<int>(events.size()); ++i) {
        const auto& ev = events[i];
        const GoEnv& before = history[i];

        if (ev.mover == MY) {
            if (ev.pos != PASS) {
                must_my.insert(ev.pos);
            }
            for (int c : ev.captured_stones) {
                must_opp.erase(c);
            }
        } else {
            if (!ev.captured_stones.empty()) {
                if (ev.pos != PASS) {
                    must_opp.insert(ev.pos);
                }
                for (int s : ev.captured_stones) {
                    for (int nb : before.getGrid(s).getNeighbors()) {
                        if (before.getGrid(nb).getPlayer() == OPP) {
                            must_opp.insert(nb);
                        }
                    }
                }
            }
            for (int c : ev.captured_stones) {
                must_my.erase(c);
            }
        }
    }

    std::unordered_set<int> must_black, must_white;
    if (MY == Player::kPlayer1) {
        must_black = std::move(must_my);
        must_white = std::move(must_opp);
    } else {
        must_white = std::move(must_my);
        must_black = std::move(must_opp);
    }
    return {std::move(must_black), std::move(must_white)};
}

bool breaksSatisfiedMust(
    const GoEnv& before, const GoEnv& after, const GoAction& a,
    const std::unordered_set<int>& satisfied_black,
    const std::unordered_set<int>& satisfied_white)
{
    MoveInfo info = analyzeMove(before, after, a);
    for (int c : info.captured_stones) {
        if (a.getPlayer() == Player::kPlayer1) {
            if (satisfied_white.count(c)) return true;
        } else {
            if (satisfied_black.count(c)) return true;
        }
    }
    return false;
}

std::vector<SeqState> sampleInfoSetAtMove(
    int board_size,
    int move_number,
    const std::unordered_set<int>& must_black,
    const std::unordered_set<int>& must_white,
    size_t target_samples,
    Player my_perspective,
    int target_black_count,
    int target_white_count,
    int max_total_attempts = 2000)
{
    const int PASS = board_size * board_size;
    std::vector<SeqState> out;
    out.reserve(target_samples);
    std::unordered_set<GoHashKey> seen;

    auto other = [](Player p){ return (p == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1; };
    Player opp = other(my_perspective);

    std::mt19937 rng{std::random_device{}()};

    auto try_place_specific = [&](GoEnv& env,
                                  int pos,
                                  Player p,
                                  std::unordered_set<int>& satisfied_black,
                                  std::unordered_set<int>& satisfied_white,
                                  std::vector<GoAction>& seq)->bool 
    {
        GoEnv test = env;
        GoAction a(pos, p);
        if (!test.act(a)) return false;
        if (breaksSatisfiedMust(env, test, a, satisfied_black, satisfied_white)) return false;
        env = std::move(test);
        seq.push_back(a);
        if (p == Player::kPlayer1) {
            satisfied_black.insert(pos);
        } else {
            satisfied_white.insert(pos);
        }
        return true;
    };

    for (int attempt = 0; attempt < max_total_attempts && out.size() < target_samples; ++attempt) {
        GoEnv env(board_size);
        std::vector<GoAction> seq;
        seq.reserve(128);

        std::unordered_set<int> pending_black = must_black;
        std::unordered_set<int> pending_white = must_white;
        std::unordered_set<int> satisfied_black, satisfied_white;

        bool fail = false;
        Player turn = Player::kPlayer1;

        // Step 1: Place all MUST stones
        while ((!pending_black.empty() || !pending_white.empty()) && !fail) {
            auto& pending_me = (turn == Player::kPlayer1) ? pending_black : pending_white;
            bool placed = false;

            if (!pending_me.empty()) {
                std::vector<int> cand(pending_me.begin(), pending_me.end());
                std::shuffle(cand.begin(), cand.end(), rng);
                for (int pos : cand) {
                    if (try_place_specific(env, pos, turn, satisfied_black, satisfied_white, seq)) {
                        pending_me.erase(pos);
                        placed = true;
                        break;
                    }
                }
            }

            if (!placed) {
                GoEnv test = env;
                GoAction pass(PASS, turn);
                if (!test.act(pass)) { fail = true; break; }
                env = std::move(test);
                seq.push_back(pass);
            }

            turn = other(turn);
        }
        
        if (fail || !pending_black.empty() || !pending_white.empty()) continue;

        // Step 2: Place opponent's stones until target count
        int cur_black = countStonesOnBoard(env, Player::kPlayer1);
        int cur_white = countStonesOnBoard(env, Player::kPlayer2);

        int need_opp = (opp == Player::kPlayer1) ? 
                       std::max(0, target_black_count - cur_black) : 
                       std::max(0, target_white_count - cur_white);

        int safety_steps = board_size * board_size * 2;
        while (!fail && safety_steps-- > 0 && need_opp > 0) {
            if (turn == my_perspective) {
                GoEnv test = env;
                GoAction pass(PASS, turn);
                if (!test.act(pass)) { fail = true; break; }
                env = std::move(test);
                seq.push_back(pass);
            } else {
                std::vector<GoAction> legal = env.getLegalActions();
                if (legal.empty()) { fail = true; break; }
                std::shuffle(legal.begin(), legal.end(), rng);

                bool moved = false;
                for (const auto& a : legal) {
                    int id = a.getActionID();
                    if (id == PASS) continue;
                    if (a.getPlayer() != opp) continue;

                    GoEnv test = env;
                    if (!test.act(a)) continue;
                    if (breaksSatisfiedMust(env, test, a, satisfied_black, satisfied_white)) continue;

                    env = std::move(test);
                    seq.push_back(a);
                    moved = true;
                    --need_opp;
                    break;
                }
                if (!moved) { fail = true; break; }
            }
            turn = other(turn);
        }

        if (fail) continue;

        int fin_black = countStonesOnBoard(env, Player::kPlayer1);
        int fin_white = countStonesOnBoard(env, Player::kPlayer2);
        if (fin_black != target_black_count || fin_white != target_white_count) continue;

        GoHashKey h = env.getHashKey();
        if (seen.insert(h).second) {
            out.push_back(SeqState{std::move(seq), h});
        }
    }

    return out;
}

// Feature Extraction Functions

std::vector<float> extractBoardState(const GoEnv& env, utils::Rotation rotation) {
    const int N = env.getBoardSize();
    std::vector<float> board_state;
    board_state.reserve(2 * N * N);
    
    // Channel 0: Black stones
    for (int pos = 0; pos < N * N; ++pos) {
        int rot_pos = env.getRotatePosition(pos, utils::reversed_rotation[static_cast<int>(rotation)]);
        board_state.push_back(env.getStoneBitboard().get(Player::kPlayer1).test(rot_pos) ? 1.0f : 0.0f);
    }
    
    // Channel 1: White stones
    for (int pos = 0; pos < N * N; ++pos) {
        int rot_pos = env.getRotatePosition(pos, utils::reversed_rotation[static_cast<int>(rotation)]);
        board_state.push_back(env.getStoneBitboard().get(Player::kPlayer2).test(rot_pos) ? 1.0f : 0.0f);
    }
    
    return board_state;  // 2 * N * N floats
}

std::vector<float> extractAnchorFeatures(
    const EnvironmentLoader& env_loader,
    int target_pos,
    utils::Rotation rotation)
{
    const int N = env_loader.getBoardSize();
    const int H = 12;  // history length
    const int C = 6;   // channels per timestep
    const int PASS = N * N;
    
    std::vector<float> anchor;
    anchor.reserve(H * C * N * N);
    
    std::vector<GoEnv> history = rebuildFullHistory(env_loader, target_pos);
    const auto& action_pairs = env_loader.getActionPairs();
    std::vector<MoveEvent> events = buildMoveEvents(history, env_loader);
    
    Player my_perspective = (target_pos > 0 && target_pos <= static_cast<int>(action_pairs.size())) ?
                            action_pairs[target_pos - 1].first.nextPlayer() :
                            Player::kPlayer1;
    
    for (int t = 0; t < H; ++t) {
        int step_idx = target_pos - H + 1 + t;
        
        if (step_idx < 0 || step_idx >= static_cast<int>(history.size())) {
            for (int c = 0; c < C; ++c) {
                for (int pos = 0; pos < N * N; ++pos) {
                    anchor.push_back(0.0f);
                }
            }
            continue;
        }
        
        const GoEnv& env_at_t = history[step_idx];
        const auto& my_stones = env_at_t.getStoneBitboard().get(my_perspective);
        const auto& opp_stones = env_at_t.getStoneBitboard().get(
            my_perspective == Player::kPlayer1 ? Player::kPlayer2 : Player::kPlayer1
        );
        
        // get action if it exists
        bool is_pass = false;
        std::vector<int> captured_this_turn;
        if (step_idx > 0 && step_idx - 1 < static_cast<int>(events.size())) {
            const auto& ev = events[step_idx - 1];
            is_pass = (ev.pos == PASS);
            captured_this_turn = ev.captured_stones;
        }
        
        // Channel 0: self_stones
        for (int pos = 0; pos < N * N; ++pos) {
            int rot_pos = env_at_t.getRotatePosition(pos, utils::reversed_rotation[static_cast<int>(rotation)]);
            anchor.push_back(my_stones.test(rot_pos) ? 1.0f : 0.0f);
        }
        
        // Channel 1: illegal_attempts（all 0s temporarily）
        for (int pos = 0; pos < N * N; ++pos) {
            anchor.push_back(0.0f);
        }
        
        // Channel 2: legal_move
        for (int pos = 0; pos < N * N; ++pos) {
            int rot_pos = env_at_t.getRotatePosition(pos, utils::reversed_rotation[static_cast<int>(rotation)]);
            bool was_last_move = false;
            if (step_idx > 0 && step_idx - 1 < static_cast<int>(action_pairs.size())) {
                int last_action_id = action_pairs[step_idx - 1].first.getActionID();
                was_last_move = (rot_pos == last_action_id && last_action_id != PASS);
            }
            anchor.push_back(was_last_move ? 1.0f : 0.0f);
        }
        
        // Channel 3: captured_black
        for (int pos = 0; pos < N * N; ++pos) {
            int rot_pos = env_at_t.getRotatePosition(pos, utils::reversed_rotation[static_cast<int>(rotation)]);
            bool was_captured_black = false;
            for (int c : captured_this_turn) {
                if (c == rot_pos && step_idx > 0) {
                    const GoEnv& prev_env = history[step_idx - 1];
                    was_captured_black = (prev_env.getGrid(c).getPlayer() == Player::kPlayer1);
                    break;
                }
            }
            anchor.push_back(was_captured_black ? 1.0f : 0.0f);
        }
        
        // Channel 4: captured_white
        for (int pos = 0; pos < N * N; ++pos) {
            int rot_pos = env_at_t.getRotatePosition(pos, utils::reversed_rotation[static_cast<int>(rotation)]);
            bool was_captured_white = false;
            for (int c : captured_this_turn) {
                if (c == rot_pos && step_idx > 0) {
                    const GoEnv& prev_env = history[step_idx - 1];
                    was_captured_white = (prev_env.getGrid(c).getPlayer() == Player::kPlayer2);
                    break;
                }
            }
            anchor.push_back(was_captured_white ? 1.0f : 0.0f);
        }
        
        // Channel 5: is_pass
        for (int pos = 0; pos < N * N; ++pos) {
            anchor.push_back(is_pass ? 1.0f : 0.0f);
        }
    }
    
    return anchor;  // H * C * N * N floats
}

ReplayBuffer::ReplayBuffer()
{
    num_data_ = 0;
    game_priority_sum_ = 0.0f;
    game_priorities_.clear();
    position_priorities_.clear();
    env_loaders_.clear();
}

void ReplayBuffer::addData(const EnvironmentLoader& env_loader)
{
    std::pair<int, int> data_range = env_loader.getDataRange();
    std::deque<float> position_priorities(data_range.second + 1, 0.0f);
    float game_priority = 0.0f;
    for (int i = data_range.first; i <= data_range.second; ++i) {
        position_priorities[i] = std::pow((config::learner_use_per ? env_loader.getPriority(i) : 1.0f), config::learner_per_alpha);
        game_priority += position_priorities[i];
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // add new data to replay buffer
    num_data_ += (data_range.second - data_range.first + 1);
    position_priorities_.push_back(position_priorities);
    game_priorities_.push_back(game_priority);
    env_loaders_.push_back(env_loader);

    // remove old data if replay buffer is full
    const size_t replay_buffer_max_size = config::zero_replay_buffer * config::zero_num_games_per_iteration;
    while (position_priorities_.size() > replay_buffer_max_size) {
        data_range = env_loaders_.front().getDataRange();
        num_data_ -= (data_range.second - data_range.first + 1);
        position_priorities_.pop_front();
        game_priorities_.pop_front();
        env_loaders_.pop_front();
    }
}

std::pair<int, int> ReplayBuffer::sampleEnvAndPos()
{
    int env_id = sampleIndex(game_priorities_);
    int pos_id = sampleIndex(position_priorities_[env_id]);
    return {env_id, pos_id};
}

int ReplayBuffer::sampleIndex(const std::deque<float>& weight)
{
    std::discrete_distribution<> dis(weight.begin(), weight.end());
    return dis(Random::generator_);
}

float ReplayBuffer::getLossScale(const std::pair<int, int>& p)
{
    if (!config::learner_use_per) { return 1.0f; }

    // calculate importance sampling ratio
    int env_id = p.first, pos = p.second;
    float prob = position_priorities_[env_id][pos] / game_priority_sum_;
    return std::pow((num_data_ * prob), (-config::learner_per_init_beta));
}

std::string DataLoaderSharedData::getNextEnvString()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string env_string = "";
    if (!env_strings_.empty()) {
        env_string = env_strings_.front();
        env_strings_.pop_front();
    }
    return env_string;
}

int DataLoaderSharedData::getNextBatchIndex()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return (batch_index_ < config::learner_batch_size ? batch_index_++ : config::learner_batch_size);
}

void DataLoaderThread::initialize()
{
    int seed = config::program_auto_seed ? std::random_device()() : config::program_seed + id_;
    Random::seed(seed);
}

void DataLoaderThread::runJob()
{
    if (!getSharedData()->env_strings_.empty()) {
        while (addEnvironmentLoader()) {}
    } else {
        while (sampleData()) {}
    }
}

bool DataLoaderThread::addEnvironmentLoader()
{
    std::string env_string = getSharedData()->getNextEnvString();
    if (env_string.empty()) { return false; }

    EnvironmentLoader env_loader;
    if (env_loader.loadFromString(env_string)) { getSharedData()->replay_buffer_.addData(env_loader); }
    return true;
}

bool DataLoaderThread::sampleData()
{
    int batch_index = getSharedData()->getNextBatchIndex();
    if (batch_index >= config::learner_batch_size) { return false; }

    if (config::nn_type_name == "siamese") {
        setIIGTrainingData(batch_index);
    } else if (config::nn_type_name == "alphazero") {
        setAlphaZeroTrainingData(batch_index);
    } else if (config::nn_type_name == "muzero") {
        setMuZeroTrainingData(batch_index);
    } else {
        return false; // should not be here
    }

    return true;
}

void DataLoaderThread::setIIGTrainingData(int batch_index)
{
    // random pickup one position
    std::pair<int, int> p = getSharedData()->replay_buffer_.sampleEnvAndPos();
    int env_id = p.first, pos = p.second;

    // IIG training data
    Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
    std::vector<float> anchor = getAnchor(env_id, pos, rotation);
    std::vector<float> positive = getPositive(env_id, pos, rotation);
    std::vector<float> negative = getNegative(env_id, pos, rotation);

    // write data to data_ptr
    std::copy(anchor.begin(), anchor.end(), getSharedData()->getDataPtr()->anchor_ + anchor.size() * batch_index);
    std::copy(positive.begin(), positive.end(), getSharedData()->getDataPtr()->positive_ + positive.size() * batch_index);
    std::copy(negative.begin(), negative.end(), getSharedData()->getDataPtr()->negative_ + negative.size() * batch_index);
}

void DataLoaderThread::setAlphaZeroTrainingData(int batch_index)
{
    // random pickup one position
    std::pair<int, int> p = getSharedData()->replay_buffer_.sampleEnvAndPos();
    int env_id = p.first, pos = p.second;

    // AlphaZero training data
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
    float loss_scale = getSharedData()->replay_buffer_.getLossScale(p);
    std::vector<float> features = env_loader.getFeatures(pos, rotation);
    std::vector<float> policy = env_loader.getPolicy(pos, rotation);
    std::vector<float> value = env_loader.getValue(pos);

    // write data to data_ptr
    getSharedData()->getDataPtr()->loss_scale_[batch_index] = loss_scale;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index] = p.first;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index + 1] = p.second;
    std::copy(features.begin(), features.end(), getSharedData()->getDataPtr()->features_ + features.size() * batch_index);
    std::copy(policy.begin(), policy.end(), getSharedData()->getDataPtr()->policy_ + policy.size() * batch_index);
    std::copy(value.begin(), value.end(), getSharedData()->getDataPtr()->value_ + value.size() * batch_index);
}

void DataLoaderThread::setMuZeroTrainingData(int batch_index)
{
    // random pickup one position
    std::pair<int, int> p = getSharedData()->replay_buffer_.sampleEnvAndPos();
    int env_id = p.first, pos = p.second;

    // MuZero training data
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
    float loss_scale = getSharedData()->replay_buffer_.getLossScale(p);
    std::vector<float> features = env_loader.getFeatures(pos, rotation);
    std::vector<float> action_features, policy, value, reward, tmp;
    for (int step = 0; step <= config::learner_muzero_unrolling_step; ++step) {
        // action features
        if (step < config::learner_muzero_unrolling_step) {
            tmp = env_loader.getActionFeatures(pos + step, rotation);
            action_features.insert(action_features.end(), tmp.begin(), tmp.end());
        }

        // policy
        tmp = env_loader.getPolicy(pos + step, rotation);
        policy.insert(policy.end(), tmp.begin(), tmp.end());

        // value
        tmp = env_loader.getValue(pos + step);
        value.insert(value.end(), tmp.begin(), tmp.end());

        // reward
        if (step < config::learner_muzero_unrolling_step) {
            tmp = env_loader.getReward(pos + step);
            reward.insert(reward.end(), tmp.begin(), tmp.end());
        }
    }

    // write data to data_ptr
    getSharedData()->getDataPtr()->loss_scale_[batch_index] = loss_scale;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index] = p.first;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index + 1] = p.second;
    std::copy(features.begin(), features.end(), getSharedData()->getDataPtr()->features_ + features.size() * batch_index);
    std::copy(action_features.begin(), action_features.end(), getSharedData()->getDataPtr()->action_features_ + action_features.size() * batch_index);
    std::copy(policy.begin(), policy.end(), getSharedData()->getDataPtr()->policy_ + policy.size() * batch_index);
    std::copy(value.begin(), value.end(), getSharedData()->getDataPtr()->value_ + value.size() * batch_index);
    std::copy(reward.begin(), reward.end(), getSharedData()->getDataPtr()->reward_ + reward.size() * batch_index);
}

std::vector<float> DataLoaderThread::getAnchor(int env_id, int pos, utils::Rotation rotation)
{
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    return extractAnchorFeatures(env_loader, pos, rotation);
}

std::vector<float> DataLoaderThread::getPositive(int env_id, int pos, utils::Rotation rotation)
{
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    GoEnv env = rebuildGoEnvToStep(env_loader, pos);
    return extractBoardState(env, rotation);
}

std::vector<float> DataLoaderThread::getNegative(int env_id, int pos, utils::Rotation rotation)
{
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    const int board_size = env_loader.getBoardSize();
    
    std::vector<GoEnv> history = rebuildFullHistory(env_loader, pos);
    
    const GoEnv& truth_env = history[std::min(pos, static_cast<int>(history.size()) - 1)];
    GoHashKey truth_hash = truth_env.getHashKey();
    
    std::vector<MoveEvent> events = buildMoveEvents(history, env_loader);
    
    Player my_perspective = (pos > 0 && pos <= static_cast<int>(env_loader.getActionPairs().size())) ?
                            env_loader.getActionPairs()[pos - 1].first.nextPlayer() :
                            Player::kPlayer1;
    
    auto [must_black, must_white] = computeMustSets(pos, my_perspective, events, history);
    
    int target_black = countStonesOnBoard(truth_env, Player::kPlayer1);
    int target_white = countStonesOnBoard(truth_env, Player::kPlayer2);
    
    const size_t NUM_CANDIDATES = 50;
    std::vector<SeqState> info_set = sampleInfoSetAtMove(
        board_size, pos, must_black, must_white, NUM_CANDIDATES,
        my_perspective, target_black, target_white, 500
    );
    
    // choose one (not ground truth) from the info set
    std::vector<SeqState> negatives;
    for (const auto& seq_state : info_set) {
        if (seq_state.hash != truth_hash) {
            negatives.push_back(seq_state);
        }
    }
    
    if (negatives.empty()) {
        std::cerr << "[WARNING] Info set only contains ground truth." << std::endl;
        return std::vector<float>();
    }
    
    static std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(0, negatives.size() - 1);
    const auto& selected = negatives[dist(rng)];
    
    GoEnv neg_env(board_size);
    for (const auto& action : selected.seq) {
        neg_env.act(action);
    }
    
    return extractBoardState(neg_env, rotation);
}

DataLoader::DataLoader(const std::string& conf_file_name)
{
    env::setUpEnv();
    config::ConfigureLoader cl;
    config::setConfiguration(cl);
    cl.loadFromFile(conf_file_name);
}

void DataLoader::initialize()
{
    createSlaveThreads(config::learner_num_thread);
    getSharedData()->createDataPtr();
}

void DataLoader::loadDataFromFile(const std::string& file_name)
{
    std::ifstream fin(file_name, std::ifstream::in);
    for (std::string content; std::getline(fin, content);) { getSharedData()->env_strings_.push_back(content); }

    for (auto& t : slave_threads_) { t->start(); }
    for (auto& t : slave_threads_) { t->finish(); }
    getSharedData()->replay_buffer_.game_priority_sum_ = std::accumulate(getSharedData()->replay_buffer_.game_priorities_.begin(), getSharedData()->replay_buffer_.game_priorities_.end(), 0.0f);
}

void DataLoader::sampleData()
{
    getSharedData()->batch_index_ = 0;
    for (auto& t : slave_threads_) { t->start(); }
    for (auto& t : slave_threads_) { t->finish(); }
}

void DataLoader::updatePriority(int* sampled_index, float* batch_values)
{
    // TODO: use multiple threads
    for (int batch_index = 0; batch_index < config::learner_batch_size; ++batch_index) {
        int env_id = sampled_index[2 * batch_index];
        int pos_id = sampled_index[2 * batch_index + 1];

        EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
        for (int step = 0; step <= config::learner_muzero_unrolling_step; ++step) {
            float new_value = utils::invertValue(batch_values[step * config::learner_batch_size + batch_index]);
            env_loader.setActionPairInfo(pos_id + step, "V", std::to_string(new_value));
        }
        getSharedData()->replay_buffer_.position_priorities_[env_id][pos_id] = std::pow(env_loader.getPriority(pos_id), config::learner_per_alpha);
    }

    // recalculate priority to correct floating number error (TODO: speedup this)
    for (size_t i = 0; i < getSharedData()->replay_buffer_.game_priorities_.size(); ++i) {
        getSharedData()->replay_buffer_.game_priorities_[i] = std::accumulate(getSharedData()->replay_buffer_.position_priorities_[i].begin(), getSharedData()->replay_buffer_.position_priorities_[i].end(), 0.0f);
    }
    getSharedData()->replay_buffer_.game_priority_sum_ = std::accumulate(getSharedData()->replay_buffer_.game_priorities_.begin(), getSharedData()->replay_buffer_.game_priorities_.end(), 0.0f);
}

} // namespace minizero::learner

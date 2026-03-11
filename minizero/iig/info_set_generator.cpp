#include "info_set_generator.h"
#include "environment.h"
#include "go.h"
#include "info_set_generator_network.h"
#include "rotation.h"
#include <functional>
#include <queue>
#include <unordered_set>
#include <vector>

namespace minizero::iig {

using namespace env;
using namespace network;
using namespace utils;

std::vector<ISItem> InfoSetGenerator::generate(const Environment& env, int max_info_set_size /*= config::iig_max_infoset_size*/)
{
    std::vector<ISItem> info_set;
    // TODO: make it general for all iig games
#if PHANTOMGO
    env::go::GoBitboard known_b_stone_bitboard = env.getImperfectEnv(env.getTurn()).getStoneBitboard().get(env::Player::kPlayer1);
    env::go::GoBitboard known_w_stone_bitboard = env.getImperfectEnv(env.getTurn()).getStoneBitboard().get(env::Player::kPlayer2);
    int pos;
    Environment init_env;
    while (!known_b_stone_bitboard.none() || !known_w_stone_bitboard.none()) {
        if (!known_b_stone_bitboard.none()) {
            pos = known_b_stone_bitboard._Find_first();
            known_b_stone_bitboard.reset(pos);
            init_env.act(Action(pos, env::Player::kPlayer1));
        }
        if (!known_w_stone_bitboard.none()) {
            pos = known_w_stone_bitboard._Find_first();
            known_w_stone_bitboard.reset(pos);
            init_env.act(Action(pos, env::Player::kPlayer2));
        }
    }
    known_b_stone_bitboard = env.getImperfectEnv(env.getTurn()).getStoneBitboard().get(env::Player::kPlayer1);
    known_w_stone_bitboard = env.getImperfectEnv(env.getTurn()).getStoneBitboard().get(env::Player::kPlayer2);

    std::unordered_set<env::go::GoBitboard> seen;
    std::priority_queue<ISItem, std::vector<ISItem>, std::function<bool(const ISItem&, const ISItem&)>>
        queue([](const ISItem& a, const ISItem& b) { return a.acc_prob_ < b.acc_prob_; });
    queue.push({init_env, 1.0f, {}, std::vector<float>(env.getBoardSize() * env.getBoardSize(), 0.0f)});
    std::vector<float> env_features = env.getFeatures(false, Rotation::kRotationNone);
    while (!queue.empty() && static_cast<int>(info_set.size()) < max_info_set_size) {
        auto item = queue.top();
        queue.pop();
        float acc_prob = item.acc_prob_;
        Environment current_env = item.env_;

        // is pass action => found one possible info state
        if (current_env.getActionHistory().size() != 0 &&
            current_env.getActionHistory().back().getActionID() == env.getBoardSize() * env.getBoardSize()) {
            info_set.push_back(item);
            continue;
        }

        std::vector<float> last_features = item.feature_;
        std::vector<float> features = env_features;
        features.insert(features.end(), last_features.begin(), last_features.end());

        network_->pushBack(features);
        auto res = network_->forward();
        auto policy_output = std::static_pointer_cast<InfoSetGeneratorNetworkOutput>(res[0]);

        // normallize policy probability
        float legal_policy_sum = 0.0f;
        for (size_t pos = 0; pos < policy_output->policy_.size(); ++pos) {
            Action action(pos, env::getNextPlayer(env.getTurn(), env.getNumPlayer()));
            if (!current_env.isLegalAction(action)) { continue; }
            legal_policy_sum += policy_output->policy_[pos];
        }
        for (size_t pos = 0; pos < policy_output->policy_.size(); ++pos) {
            Action action(pos, env::getNextPlayer(env.getTurn(), env.getNumPlayer()));
            if (!current_env.isLegalAction(action)) {
                policy_output->policy_[pos] = 0.0f;
                continue;
            }
            policy_output->policy_[pos] /= legal_policy_sum;
        }

        int max_index = std::max_element(policy_output->policy_.begin(), policy_output->policy_.end()) - policy_output->policy_.begin();
        for (size_t pos = 0; pos < policy_output->policy_.size(); ++pos) {
            Action action(pos, env::getNextPlayer(env.getTurn(), env.getNumPlayer()));
            if (!current_env.isLegalAction(action)) {
                if (static_cast<int>(pos) == max_index) { policy_output->policy_.back() = 1.0f; } // TODO: at least we have pass
                continue;
            }
            Environment next_env = current_env;
            next_env.act(action);

            env::go::GoBitboard new_b_stone_bitboard = next_env.getPerfectEnv().getStoneBitboard().get(env::Player::kPlayer1);
            env::go::GoBitboard new_w_stone_bitboard = next_env.getPerfectEnv().getStoneBitboard().get(env::Player::kPlayer2);
            if (!(known_b_stone_bitboard & ~new_b_stone_bitboard).none() || !(known_w_stone_bitboard & ~new_w_stone_bitboard).none()) {
                if (static_cast<int>(pos) == max_index) { policy_output->policy_.back() = 1.0f; } // TODO: at least we have pass
                continue;
            }
            env::go::GoBitboard put = (new_b_stone_bitboard & ~known_b_stone_bitboard) | (new_w_stone_bitboard & ~known_w_stone_bitboard);

            float p = policy_output->policy_[pos];
            if (p < config::iig_generator_policy_threshold) {
                if (static_cast<int>(pos) == env.getBoardSize() * env.getBoardSize() && static_cast<int>(pos) == max_index) {
                    p = 1.0f;
                } else {
                    if (static_cast<int>(pos) == max_index) { policy_output->policy_.back() = 1.0f; } // TODO: at least we have pass
                    continue;
                }
            }
            std::vector<float> new_probs = item.probs_;
            new_probs.push_back(p);
            std::vector<float> new_last_features = last_features;

            if (static_cast<int>(pos) != env.getBoardSize() * env.getBoardSize()) {
                new_last_features[static_cast<int>(pos)] = 1.0f;
                if (seen.count(put)) { continue; }
                seen.insert(put);
            }

            queue.push({next_env, acc_prob * p, new_probs, new_last_features});
        }
    }
#else
#endif
    return info_set;
}

std::vector<std::vector<ISItem>> InfoSetGenerator::generate(const std::vector<Environment>& envs, int max_info_set_size /*= config::iig_max_infoset_size*/)
{
    std::vector<std::vector<ISItem>> info_set(envs.size());
    std::vector<env::GamePair<go::GoBitboard>> known_bitboards;
    std::vector<Environment> init_envs;
    std::vector<std::vector<float>> features;
    std::vector<std::unordered_set<go::GoBitboard>> seens(envs.size());
    std::vector<std::priority_queue<ISItem, std::vector<ISItem>, std::function<bool(const ISItem&, const ISItem&)>>> queues;

#if PHANTOMGO
    // setup
    for (const auto& env : envs) {
        known_bitboards.push_back({env.getImperfectEnv(env.getTurn()).getStoneBitboard().get(env::Player::kPlayer1),
                                   env.getImperfectEnv(env.getTurn()).getStoneBitboard().get(env::Player::kPlayer2)});
        int pos;
        go::GoBitboard p1_bitboard = known_bitboards.back().get(env::Player::kPlayer1), p2_bitboard = known_bitboards.back().get(env::Player::kPlayer2);
        init_envs.push_back(Environment());
        while (!p1_bitboard.none() || !p2_bitboard.none()) {
            if (!p1_bitboard.none()) {
                pos = p1_bitboard._Find_first();
                p1_bitboard.reset(pos);
                init_envs.back().act(Action(pos, env::Player::kPlayer1));
            }
            if (!p2_bitboard.none()) {
                pos = p2_bitboard._Find_first();
                p2_bitboard.reset(pos);
                init_envs.back().act(Action(pos, env::Player::kPlayer2));
            }
        }
        features.push_back(env.getFeatures(false, Rotation::kRotationNone));
        queues.push_back(std::priority_queue<ISItem, std::vector<ISItem>, std::function<bool(const ISItem&, const ISItem&)>>(
            [](const ISItem& a, const ISItem& b) { return a.acc_prob_ < b.acc_prob_; }));
        queues.back().push({init_envs.back(), 1.0f, {}, std::vector<float>(env.getBoardSize() * env.getBoardSize(), 0.0f)});
    }

    while (true) {
        // before NN
        bool is_end = true;
        for (size_t i = 0; i < envs.size(); ++i) {
            if (queues[i].empty() || static_cast<int>(info_set[i].size()) >= max_info_set_size) { continue; }

            auto item = queues[i].top();
            if (item.env_.getActionHistory().size() != 0 &&
                item.env_.getActionHistory().back().getActionID() == envs[i].getBoardSize() * envs[i].getBoardSize()) {
                continue;
            }

            is_end = false;
            std::vector<float> last_features = item.feature_;
            std::vector<float> input_features = features[i];
            input_features.insert(input_features.end(), last_features.begin(), last_features.end());
            network_->pushBack(input_features);
        }
        if (is_end) {
            for (size_t i = 0; i < envs.size(); ++i) {
                if (queues[i].empty() || static_cast<int>(info_set[i].size()) >= max_info_set_size) { continue; }

                auto item = queues[i].top();
                if (item.env_.getActionHistory().size() != 0 &&
                    item.env_.getActionHistory().back().getActionID() == envs[i].getBoardSize() * envs[i].getBoardSize()) {
                    info_set[i].push_back(item);
                    queues[i].pop();
                    continue;
                }
            }
            break;
        }

        // forward NN
        auto res = network_->forward();

        // after NN
        int counter = 0;
        for (size_t i = 0; i < envs.size(); ++i) {
            if (queues[i].empty() || static_cast<int>(info_set[i].size()) >= max_info_set_size) { continue; }

            auto item = queues[i].top();
            if (item.env_.getActionHistory().size() != 0 &&
                item.env_.getActionHistory().back().getActionID() == envs[i].getBoardSize() * envs[i].getBoardSize()) {
                info_set[i].push_back(item);
                queues[i].pop();
                continue;
            }

            queues[i].pop();
            float acc_prob = item.acc_prob_;
            Environment current_env = item.env_;
            const Environment& env = envs[i];
            std::vector<float> last_features = item.feature_;

            auto policy_output = std::static_pointer_cast<InfoSetGeneratorNetworkOutput>(res[counter++]);

            // normallize policy probability
            float legal_policy_sum = 0.0f;
            for (size_t pos = 0; pos < policy_output->policy_.size(); ++pos) {
                Action action(pos, env::getNextPlayer(env.getTurn(), env.getNumPlayer()));
                if (!current_env.isLegalAction(action)) { continue; }
                legal_policy_sum += policy_output->policy_[pos];
            }
            for (size_t pos = 0; pos < policy_output->policy_.size(); ++pos) {
                Action action(pos, env::getNextPlayer(env.getTurn(), env.getNumPlayer()));
                if (!current_env.isLegalAction(action)) {
                    policy_output->policy_[pos] = 0.0f;
                    continue;
                }
                policy_output->policy_[pos] /= legal_policy_sum;
            }

            int max_index = std::max_element(policy_output->policy_.begin(), policy_output->policy_.end()) - policy_output->policy_.begin();
            for (size_t pos = 0; pos < policy_output->policy_.size(); ++pos) {
                Action action(pos, env::getNextPlayer(env.getTurn(), env.getNumPlayer()));
                if (!current_env.isLegalAction(action)) {
                    if (static_cast<int>(pos) == max_index) { policy_output->policy_.back() = 1.0f; } // TODO: at least we have pass
                    continue;
                }
                Environment next_env = current_env;
                next_env.act(action);

                env::go::GoBitboard new_b_stone_bitboard = next_env.getPerfectEnv().getStoneBitboard().get(env::Player::kPlayer1);
                env::go::GoBitboard new_w_stone_bitboard = next_env.getPerfectEnv().getStoneBitboard().get(env::Player::kPlayer2);
                if (!(known_bitboards[i].get(Player::kPlayer1) & ~new_b_stone_bitboard).none() || !(known_bitboards[i].get(Player::kPlayer2) & ~new_w_stone_bitboard).none()) {
                    if (static_cast<int>(pos) == max_index) { policy_output->policy_.back() = 1.0f; } // TODO: at least we have pass
                    continue;
                }
                env::go::GoBitboard put = (new_b_stone_bitboard & ~known_bitboards[i].get(Player::kPlayer1)) |
                                          (new_w_stone_bitboard & ~known_bitboards[i].get(Player::kPlayer2));

                float p = policy_output->policy_[pos];
                if (p < config::iig_generator_policy_threshold) {
                    if (static_cast<int>(pos) == env.getBoardSize() * env.getBoardSize() && static_cast<int>(pos) == max_index) {
                        p = 1.0f;
                    } else {
                        if (static_cast<int>(pos) == max_index) { policy_output->policy_.back() = 1.0f; } // TODO: at least we have pass
                        continue;
                    }
                }
                std::vector<float> new_probs = item.probs_;
                new_probs.push_back(p);
                std::vector<float> new_last_features = last_features;

                if (static_cast<int>(pos) != env.getBoardSize() * env.getBoardSize()) {
                    new_last_features[static_cast<int>(pos)] = 1.0f;
                    if (seens[i].count(put)) { continue; }
                    seens[i].insert(put);
                }

                queues[i].push({next_env, acc_prob * p, new_probs, new_last_features});
            }
        }
    }
#else
#endif
    return info_set;
}

}; // namespace minizero::iig

#include "data_loader.h"
#include "configuration.h"
#include "environment.h"
#include "go.h"
#include "random.h"
#include "rotation.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

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

struct NegativeBoard {
    GoBitboard black;
    GoBitboard white;
};

// Core Helper Functions (for generating the infomation set)

GoEnv rebuildGoEnvToStep(const EnvironmentLoader& env_loader, int target_pos)
{
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

std::vector<GoEnv> rebuildFullHistory(const EnvironmentLoader& env_loader, int target_pos)
{
    const int board_size = env_loader.getBoardSize();
    const auto& action_pairs = env_loader.getActionPairs();

    std::vector<GoEnv> history;
    history.reserve(target_pos + 1);

    GoEnv env(board_size);
    history.push_back(env); // Initial state

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

MoveInfo analyzeMove(const GoEnv& before, const GoEnv& after, const GoAction& action)
{
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

int countStonesOnBoard(const GoEnv& env, Player p)
{
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
    auto other = [](Player p) { return (p == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1; };

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

// ===== Move-Stone Sampling Helper Functions =====

std::vector<int> identifyMovableStones(
    const GoEnv& truth_env,
    const std::unordered_set<int>& must_black,
    const std::unordered_set<int>& must_white,
    Player my_perspective)
{
    std::vector<int> movable;
    const int board_size = truth_env.getBoardSize();

    // Get OPPONENT's stones
    Player opponent = (my_perspective == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1;
    const auto& opp_stones = truth_env.getStoneBitboard().get(opponent);

    // Determine which MUST set to use for OPPONENT
    const auto& must_set = (opponent == Player::kPlayer1) ? must_black : must_white;

    // Find all OPPONENT's stones that are not in MUST set
    for (int pos = 0; pos < board_size * board_size; ++pos) {
        if (!opp_stones.test(pos)) continue; // Not opponent's stone
        if (must_set.count(pos)) continue;   // Is MUST stone, cannot move

        movable.push_back(pos);
    }

    return movable;
}

std::vector<int> findCandidateTargets(
    const GoEnv& truth_env,
    int source_pos,
    Player perspective,
    int max_distance)
{
    std::vector<int> candidates;
    const int board_size = truth_env.getBoardSize();
    const int board_area = board_size * board_size;

    // color of moving stone
    Player stone_color = truth_env.getGrid(source_pos).getPlayer();
    if (stone_color == Player::kPlayerNone) {
        return candidates; // skip for empty position
    }

    Player opponent = (stone_color == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1;

    // original coordinate
    int src_row = source_pos / board_size;
    int src_col = source_pos % board_size;

    auto try_add_pos = [&](int pos) {
        const GoGrid& grid = truth_env.getGrid(pos);
        if (grid.getPlayer() != Player::kPlayerNone) return;

        // check if it captures stones
        bool would_capture = false;
        const std::vector<int>& neighbors = grid.getNeighbors();
        for (int nb_pos : neighbors) {
            const GoGrid& nb_grid = truth_env.getGrid(nb_pos);
            if (nb_grid.getPlayer() != opponent) continue;

            const GoBlock* nb_block = nb_grid.getBlock();
            if (!nb_block) continue;

            if (nb_block->getNumLiberty() == 1) {
                const GoBitboard& liberty_bb = nb_block->getLibertyBitboard();
                if (liberty_bb.test(pos)) {
                    // one liberty
                    would_capture = true;
                    break;
                }
            }
        }
        if (would_capture) return;

        // legality
        GoAction a(pos, stone_color);
        if (!truth_env.isLegalAction(a)) return;

        candidates.push_back(pos);
    };

    // manhattan distance limitation
    if (max_distance > 0 && max_distance < board_size) {
        for (int dr = -max_distance; dr <= max_distance; ++dr) {
            int row = src_row + dr;
            if (row < 0 || row >= board_size) continue;

            int max_dc = max_distance - std::abs(dr);
            for (int dc = -max_dc; dc <= max_dc; ++dc) {
                int col = src_col + dc;
                if (col < 0 || col >= board_size) continue;

                int pos = row * board_size + col;
                if (pos == source_pos) continue;

                try_add_pos(pos);
            }
        }
    } else { // no distance limitation (whole board)
        for (int pos = 0; pos < board_area; ++pos) {
            if (pos == source_pos) continue;
            try_add_pos(pos);
        }
    }

    return candidates;
}

std::vector<NegativeBoard> sampleMoveStoneNegativesBitboard(
    const GoEnv& truth_env,
    const std::unordered_set<int>& must_black,
    const std::unordered_set<int>& must_white,
    Player my_perspective,
    size_t target_samples,
    int max_move_distance)
{
    const int board_size = truth_env.getBoardSize();
    const int board_area = board_size * board_size;

    const auto& black_truth = truth_env.getStoneBitboard().get(Player::kPlayer1);
    const auto& white_truth = truth_env.getStoneBitboard().get(Player::kPlayer2);

    std::vector<NegativeBoard> negatives;
    negatives.reserve(target_samples);

    // Find all movable stones
    auto movable = identifyMovableStones(truth_env, must_black, must_white, my_perspective);
    if (movable.empty() || target_samples == 0) {
        return negatives;
    }

    // cache candidate targets for every source_pos
    std::vector<std::vector<int>> target_cache(board_area);
    std::vector<char> has_cache(board_area, 0);

    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<> stone_dis(0, static_cast<int>(movable.size()) - 1);

    const int attempts_limit = static_cast<int>(target_samples) * 20;
    int attempts = 0;

    while (negatives.size() < target_samples && attempts < attempts_limit) {
        ++attempts;

        int source_pos = movable[stone_dis(rng)];

        if (!has_cache[source_pos]) {
            target_cache[source_pos] =
                findCandidateTargets(truth_env, source_pos, my_perspective, max_move_distance);
            has_cache[source_pos] = 1;
        }

        const auto& candidates = target_cache[source_pos];
        if (candidates.empty()) {
            continue;
        }

        std::uniform_int_distribution<> target_dis(0, static_cast<int>(candidates.size()) - 1);
        int target_pos = candidates[target_dis(rng)];
        if (target_pos == source_pos) continue;

        Player stone_color = truth_env.getGrid(source_pos).getPlayer();
        if (stone_color == Player::kPlayerNone) continue;

        GoBitboard black_bb = black_truth;
        GoBitboard white_bb = white_truth;

        if (stone_color == Player::kPlayer1) {
            black_bb.reset(source_pos);
            black_bb.set(target_pos);
        } else {
            white_bb.reset(source_pos);
            white_bb.set(target_pos);
        }

        negatives.push_back(NegativeBoard{black_bb, white_bb});
    }

    return negatives;
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
    int max_total_attempts = 100)
{
    const int PASS = board_size * board_size;
    std::vector<SeqState> out;
    out.reserve(target_samples);
    std::unordered_set<GoHashKey> seen;

    auto other = [](Player p) { return (p == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1; };
    Player opp = other(my_perspective);

    std::mt19937 rng{std::random_device{}()};

    auto try_place_specific = [&](GoEnv& env,
                                  int pos,
                                  Player p,
                                  std::unordered_set<int>& satisfied_black,
                                  std::unordered_set<int>& satisfied_white,
                                  std::vector<GoAction>& seq) -> bool {
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
                if (!test.act(pass)) {
                    fail = true;
                    break;
                }
                env = std::move(test);
                seq.push_back(pass);
            }

            turn = other(turn);
        }

        if (fail || !pending_black.empty() || !pending_white.empty()) continue;

        // Step 2: Place opponent's stones until target count
        int cur_black = countStonesOnBoard(env, Player::kPlayer1);
        int cur_white = countStonesOnBoard(env, Player::kPlayer2);

        int need_opp = (opp == Player::kPlayer1) ? std::max(0, target_black_count - cur_black) : std::max(0, target_white_count - cur_white);

        int safety_steps = board_size * board_size * 2;
        while (!fail && safety_steps-- > 0 && need_opp > 0) {
            if (turn == my_perspective) {
                GoEnv test = env;
                GoAction pass(PASS, turn);
                if (!test.act(pass)) {
                    fail = true;
                    break;
                }
                env = std::move(test);
                seq.push_back(pass);
            } else {
                std::vector<GoAction> legal = env.getLegalActions();
                if (legal.empty()) {
                    fail = true;
                    break;
                }
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
                if (!moved) {
                    fail = true;
                    break;
                }
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

std::size_t computeBoardHash(const GoBitboard& black_bb,
                             const GoBitboard& white_bb,
                             int board_area)
{
    std::size_t h = 1469598103934665603ull; // FNV offset
    for (int pos = 0; pos < board_area; ++pos) {
        unsigned char v = 0;
        if (black_bb.test(pos)) v |= 1;
        if (white_bb.test(pos)) v |= 2;
        h ^= static_cast<std::size_t>(v);
        h *= 1099511628211ull; // FNV prime
    }
    return h;
}

// Seqstae to bitboard
std::vector<NegativeBoard> seqStatesToNegativesBitboard(
    const std::vector<SeqState>& info_set,
    int board_size,
    size_t max_num)
{
    std::vector<NegativeBoard> out;
    out.reserve(std::min(max_num, info_set.size()));

    for (const auto& s : info_set) {
        if (out.size() >= max_num) break;

        GoEnv env(board_size);
        for (const auto& a : s.seq) {
            env.act(a);
        }

        const auto& black_bb = env.getStoneBitboard().get(Player::kPlayer1);
        const auto& white_bb = env.getStoneBitboard().get(Player::kPlayer2);
        out.push_back(NegativeBoard{black_bb, white_bb});
    }

    return out;
}

std::vector<float> extractBoardStateFromBitboard(
    const GoEnv& ref_env,
    const GoBitboard& black_bb,
    const GoBitboard& white_bb,
    utils::Rotation rotation)
{
    const int N = ref_env.getBoardSize();
    std::vector<float> board_state;
    board_state.reserve(2 * N * N);

    // Channel 0: Black stones
    for (int pos = 0; pos < N * N; ++pos) {
        int rot_pos = ref_env.getRotatePosition(
            pos, utils::reversed_rotation[static_cast<int>(rotation)]);
        board_state.push_back(black_bb.test(rot_pos) ? 1.0f : 0.0f);
    }

    // Channel 1: White stones
    for (int pos = 0; pos < N * N; ++pos) {
        int rot_pos = ref_env.getRotatePosition(
            pos, utils::reversed_rotation[static_cast<int>(rotation)]);
        board_state.push_back(white_bb.test(rot_pos) ? 1.0f : 0.0f);
    }

    return board_state;
}

std::vector<float> extractBoardState(const GoEnv& env, utils::Rotation rotation)
{
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

    return board_state; // 2 * N * N floats
}

std::vector<float> extractAnchorFeatures(
    const EnvironmentLoader& env_loader,
    int target_pos,
    utils::Rotation rotation)
{
    const int N = env_loader.getBoardSize();
    const int H = 12; // history length
    const int C = 6;  // channels per timestep
    const int PASS = N * N;

    std::vector<float> anchor;
    anchor.reserve(H * C * N * N);

    std::vector<GoEnv> history = rebuildFullHistory(env_loader, target_pos);
    const auto& action_pairs = env_loader.getActionPairs();
    std::vector<MoveEvent> events = buildMoveEvents(history, env_loader);

    Player my_perspective = (target_pos > 0 && target_pos <= static_cast<int>(action_pairs.size())) ? action_pairs[target_pos - 1].first.nextPlayer() : Player::kPlayer1;

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

    return anchor; // H * C * N * N floats
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

void ReplayBuffer::addTestingData(const EnvironmentLoader& env_loader)
{
    std::lock_guard<std::mutex> lock(mutex_);
    env_loaders_.push_back(env_loader);
}

std::pair<int, int> ReplayBuffer::sampleEnvAndPos()
{
    if (env_loaders_.empty()) {
        std::cerr << "FATAL: replay_buffer is empty! No data loaded." << std::endl;
        throw std::runtime_error("Empty replay buffer");
    }

    if (game_priorities_.empty() || position_priorities_.empty()) {
        std::cerr << "FATAL: priorities not initialized!" << std::endl;
        throw std::runtime_error("Uninitialized priorities");
    }

    int env_id = sampleIndex(game_priorities_);

    if (env_id < 0 || env_id >= static_cast<int>(position_priorities_.size())) {
        std::cerr << "FATAL: env_id " << env_id << " out of bounds [0, "
                  << position_priorities_.size() << ")" << std::endl;
        throw std::runtime_error("Invalid env_id");
    }

    const auto& pos_prio = position_priorities_[env_id];
    if (pos_prio.size() > 1000) {
        std::cerr << "FATAL: position_priorities_[" << env_id << "].size() = "
                  << pos_prio.size() << " is unreasonable!" << std::endl;
        throw std::runtime_error("Corrupted position_priorities");
    }

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

std::pair<int, int> DataLoaderSharedData::getNextEnvPosIndex()
{
    std::lock_guard<std::mutex> lock(mutex_);
    EnvironmentLoader& env_loader = replay_buffer_.env_loaders_[env_index_];
    const auto& action_pairs = env_loader.getActionPairs();
    if (pos_index_ >= static_cast<int>(action_pairs.size())) {
        // move to next environment
        env_index_++;
        pos_index_ = 0;
    }
    return {env_index_, pos_index_++};
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
    if (env_loader.loadFromString(env_string)) {
        if (config::siamese_mode == "training") {
            getSharedData()->replay_buffer_.addData(env_loader);
        } else {
            getSharedData()->replay_buffer_.addTestingData(env_loader);
        }
    }
    return true;
}

bool DataLoaderThread::sampleData()
{
    int batch_index = getSharedData()->getNextBatchIndex();
    if (batch_index >= config::learner_batch_size) { return false; }

    if (config::nn_type_name == "siamese") {
        if (config::siamese_mode == "training") {
            setIIGTrainingData(batch_index);
        } else if (config::siamese_mode == "testing") {
            std::pair<int, int> p = getSharedData()->getNextEnvPosIndex();
            setIIGTestingData(batch_index, p.first, p.second);
        } else {
            return false;
        }
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

    int train_candidates = std::max(config::siamese_num_negatives, 1);
    std::vector<float> negative = getNegative(
        env_id,
        pos,
        rotation,
        /*num_outputs=*/1,
        /*num_candidates=*/train_candidates);

    // write data to data_ptr
    std::copy(anchor.begin(), anchor.end(), getSharedData()->getDataPtr()->anchor_ + anchor.size() * batch_index);
    std::copy(positive.begin(), positive.end(), getSharedData()->getDataPtr()->positive_ + positive.size() * batch_index);
    std::copy(negative.begin(), negative.end(), getSharedData()->getDataPtr()->negative_ + negative.size() * batch_index);
}

void DataLoaderThread::setIIGTestingData(int batch_index, int env_id, int pos)
{
    // get next position
    Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
    std::vector<float> anchor = getAnchor(env_id, pos, rotation);
    std::vector<float> positive = getPositive(env_id, pos, rotation);

    int eval_outputs = std::max(config::siamese_eval_num_negatives, 1);
    int eval_candidates = eval_outputs;
    std::vector<float> negative = getNegative(
        env_id,
        pos,
        rotation,
        /*num_outputs=*/eval_outputs,
        /*num_candidates=*/eval_candidates);

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

std::vector<float> DataLoaderThread::getNegative(
    int env_id,
    int pos,
    utils::Rotation rotation,
    int num_outputs,
    int num_candidates)
{
    assert(num_candidates >= num_outputs && "num_candidates must be >= num_outputs");

    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    const int board_size = env_loader.getBoardSize();
    const int board_area = board_size * board_size;

    // Rebuild history & ground truth
    std::vector<GoEnv> history = rebuildFullHistory(env_loader, pos);
    const GoEnv& truth_env = history[std::min(pos, static_cast<int>(history.size()) - 1)];

    const auto& truth_black_bb = truth_env.getStoneBitboard().get(Player::kPlayer1);
    const auto& truth_white_bb = truth_env.getStoneBitboard().get(Player::kPlayer2);
    std::size_t truth_board_hash = computeBoardHash(truth_black_bb, truth_white_bb, board_area);

    // calculate MUST
    std::vector<MoveEvent> events = buildMoveEvents(history, env_loader);

    Player my_perspective =
        (pos > 0 && pos <= static_cast<int>(env_loader.getActionPairs().size()))
            ? env_loader.getActionPairs()[pos - 1].first.nextPlayer()
            : Player::kPlayer1;

    auto [must_black, must_white] =
        computeMustSets(pos, my_perspective, events, history);

    int target_black = countStonesOnBoard(truth_env, Player::kPlayer1);
    int target_white = countStonesOnBoard(truth_env, Player::kPlayer2);

    // generate candidate negatives（in bitboard)
    std::vector<NegativeBoard> candidate_negatives;

    if (config::siamese_sampling_strategy == "move_stone") {
        candidate_negatives = sampleMoveStoneNegativesBitboard(
            truth_env,
            must_black,
            must_white,
            my_perspective,
            static_cast<size_t>(num_candidates),
            config::siamese_max_move_distance);

    } else if (config::siamese_sampling_strategy == "random") {
        auto info_set = sampleInfoSetAtMove(
            board_size,
            pos,
            must_black,
            must_white,
            num_candidates,
            my_perspective,
            target_black,
            target_white);
        candidate_negatives = seqStatesToNegativesBitboard(
            info_set,
            board_size,
            static_cast<size_t>(num_candidates));

    } else if (config::siamese_sampling_strategy == "hybrid") {
        size_t move_stone_count = static_cast<size_t>(
            num_candidates *
            config::siamese_move_stone_ratio);
        size_t random_count =
            num_candidates - move_stone_count;

        auto move_samples = sampleMoveStoneNegativesBitboard(
            truth_env,
            must_black,
            must_white,
            my_perspective,
            move_stone_count,
            config::siamese_max_move_distance);

        auto random_info_set = sampleInfoSetAtMove(
            board_size,
            pos,
            must_black,
            must_white,
            static_cast<size_t>(random_count),
            my_perspective,
            target_black,
            target_white);

        auto random_samples = seqStatesToNegativesBitboard(
            random_info_set,
            board_size,
            random_count);

        candidate_negatives.reserve(move_samples.size() + random_samples.size());
        candidate_negatives.insert(candidate_negatives.end(),
                                   move_samples.begin(),
                                   move_samples.end());
        candidate_negatives.insert(candidate_negatives.end(),
                                   random_samples.begin(),
                                   random_samples.end());

    } else {
        std::cerr << "[WARNING] Unknown sampling strategy: "
                  << config::siamese_sampling_strategy
                  << ". Falling back to random." << std::endl;
        auto info_set = sampleInfoSetAtMove(
            board_size,
            pos,
            must_black,
            must_white,
            num_candidates,
            my_perspective,
            target_black,
            target_white);
        candidate_negatives = seqStatesToNegativesBitboard(
            info_set,
            board_size,
            static_cast<size_t>(num_candidates));
    }

    if (candidate_negatives.empty()) {
        return std::vector<float>();
    }

    // hash
    std::unordered_set<std::size_t> seen;
    seen.insert(truth_board_hash);

    std::vector<NegativeBoard> negatives;
    negatives.reserve(std::min<int>(num_outputs,
                                    static_cast<int>(candidate_negatives.size())));

    for (const auto& nb : candidate_negatives) {
        if (static_cast<int>(negatives.size()) >= num_outputs) break;
        std::size_t h = computeBoardHash(nb.black, nb.white, board_area);
        if (!seen.insert(h).second) continue;
        negatives.push_back(nb);
    }

    if (negatives.empty()) {
        return std::vector<float>();
    }

    // Debug
    static std::atomic<int> sample_count{0};
    int current_sample = ++sample_count;

    if (config::siamese_debug_output &&
        (current_sample <= 3 || current_sample % 100 == 0)) {
        std::cerr << "\n=== Negative Sampling (Sample #"
                  << current_sample << ") ===\n";

        std::cerr << "\n[POSITIVE] Ground Truth Board:\n";
        std::cerr << truth_env.toString() << std::endl;

        const auto& nb = negatives.front();
        std::cerr << "\n[NEGATIVE] Sampled Board (bitboard view):\n";

        for (int row = board_size - 1; row >= 0; --row) {
            for (int col = 0; col < board_size; ++col) {
                int p = row * board_size + col;

                char ch = '.';
                if (nb.black.test(p))
                    ch = 'X';
                else if (nb.white.test(p))
                    ch = 'O';

                std::cerr << ch << ' ';
            }
            std::cerr << '\n';
        }
        std::cerr << std::endl;

        std::cerr << "=====================================\n\n";
    }

    // Construct the tensor for network
    std::vector<float> result;
    result.reserve(num_outputs * 2 * board_area);

    for (const auto& nb : negatives) {
        std::vector<float> board_state =
            extractBoardStateFromBitboard(truth_env, nb.black, nb.white, rotation);
        result.insert(result.end(), board_state.begin(), board_state.end());
    }

    // fill up to num_outputs
    int produced = static_cast<int>(negatives.size());
    while (produced < num_outputs) {
        result.insert(result.end(), 2 * board_area, 0.0f);
        ++produced;
    }

    return result;
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

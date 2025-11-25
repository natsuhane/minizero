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

    // 1. Get the color of the stone to move
    Player stone_color = truth_env.getGrid(source_pos).getPlayer();
    if (stone_color == Player::kPlayerNone) {
        return candidates; // source_pos is empty, error
    }

    // 2. Define opponent color (the side that might get captured)
    Player opponent = (stone_color == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1;

    // 3. Calculate source coordinates (for distance filter)
    int source_row = source_pos / board_size;
    int source_col = source_pos % board_size;

    // 4. Scan all positions
    for (int pos = 0; pos < board_size * board_size; ++pos) {
        // 4.1 Check if position is empty
        if (truth_env.getGrid(pos).getPlayer() != Player::kPlayerNone) {
            continue; // Not empty, skip
        }

        // 4.2 Distance filter
        if (max_distance > 0) {
            int row = pos / board_size;
            int col = pos % board_size;
            int manhattan = std::abs(row - source_row) + std::abs(col - source_col);

            if (manhattan > max_distance) {
                continue; // Too far, skip
            }
        }

        // 4.3 Check legality (suicide, ko)
        GoAction test_action(pos, stone_color);
        if (!truth_env.isLegalAction(test_action)) {
            continue; // Illegal (suicide or ko), skip
        }

        // 4.4 Check if placing here would capture opponent stones
        // Capture condition: neighbor has opponent block with only 1 liberty, and that liberty is pos
        bool would_capture = false;

        const std::vector<int>& neighbors = truth_env.getGrid(pos).getNeighbors();
        for (int nb_pos : neighbors) {
            const GoGrid& nb_grid = truth_env.getGrid(nb_pos);

            // Check if neighbor is opponent stone
            if (nb_grid.getPlayer() != opponent) {
                continue;
            }

            // Get the block this stone belongs to
            const GoBlock* nb_block = nb_grid.getBlock();
            if (!nb_block) continue;

            // Check if block has only 1 liberty
            if (nb_block->getNumLiberty() == 1) {
                // Check if that single liberty is exactly pos
                const GoBitboard& liberty_bb = nb_block->getLibertyBitboard();
                if (liberty_bb.test(pos)) {
                    // This block's only liberty is pos!
                    // Placing at pos would capture this block
                    would_capture = true;
                    break;
                }
            }
        }

        if (would_capture) {
            continue; // Would capture, skip
        }

        // 4.5 Passed all checks, add to candidates
        candidates.push_back(pos);
    }

    return candidates;
}

std::optional<SeqState> reconstructBoardWithStoneConfig(
    int board_size,
    int move_number,
    const std::unordered_set<int>& must_black,
    const std::unordered_set<int>& must_white,
    const std::map<int, Player>& desired_stones,
    Player my_perspective)
{
    const int PASS = board_size * board_size;
    std::vector<GoAction> seq;
    seq.reserve(128);

    auto other = [](Player p) { return (p == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1; };
    Player opponent = other(my_perspective);

    GoEnv env(board_size);
    std::unordered_set<int> satisfied_black, satisfied_white;
    Player turn = Player::kPlayer1;

    auto try_place_specific = [&](GoEnv& env, int pos, Player p,
                                  std::unordered_set<int>& satisfied_black,
                                  std::unordered_set<int>& satisfied_white,
                                  std::vector<GoAction>& seq) -> bool {
        GoEnv test = env;
        GoAction a(pos, p);
        if (!test.act(a)) {
            return false;
        }
        if (breaksSatisfiedMust(env, test, a, satisfied_black, satisfied_white)) {
            return false;
        }
        env = std::move(test);
        seq.push_back(a);
        if (p == Player::kPlayer1) {
            satisfied_black.insert(pos);
        } else {
            satisfied_white.insert(pos);
        }
        return true;
    };

    // Helper to pass until it's a specific player's turn
    auto pass_until_turn = [&](Player target_player) -> bool {
        while (turn != target_player) {
            GoEnv test = env;
            GoAction pass(PASS, turn);
            if (!test.act(pass)) {
                return false;
            }
            env = std::move(test);
            seq.push_back(pass);
            turn = other(turn);
        }
        return true;
    };

    // Helper to place stones in a fixed order
    auto place_stones_fixed_order = [&](const std::vector<int>& positions, Player player) -> bool {
        for (int pos : positions) {
            // Pass until it's the right player's turn
            if (!pass_until_turn(player)) {
                return false;
            }
            // Try to place the stone
            if (try_place_specific(env, pos, turn, satisfied_black, satisfied_white, seq)) {
                turn = other(turn);
            } else {
                return false;
            }
        }
        return true;
    };

    // Count expected stones
    int expected_black = 0, expected_white = 0;
    for (const auto& [pos, player] : desired_stones) {
        if (player == Player::kPlayer1)
            expected_black++;
        else
            expected_white++;
    }

    // Phase 1: Place opponent's MUST stones
    const auto& must_opp = (opponent == Player::kPlayer1) ? must_black : must_white;
    std::vector<int> opp_must_positions(must_opp.begin(), must_opp.end());

    if (!place_stones_fixed_order(opp_must_positions, opponent)) {
        return std::nullopt;
    }

    // Phase 2: Place own MUST stones
    const auto& must_my = (my_perspective == Player::kPlayer1) ? must_black : must_white;
    std::vector<int> my_must_positions(must_my.begin(), must_my.end());

    if (!place_stones_fixed_order(my_must_positions, my_perspective)) {
        return std::nullopt;
    }

    // Phase 3: Place all other desired stones

    // Separate non-MUST stones by color
    std::vector<int> other_black, other_white;
    for (const auto& [pos, player] : desired_stones) {
        // Skip if already placed as MUST
        if (player == Player::kPlayer1) {
            if (!must_black.count(pos)) {
                other_black.push_back(pos);
            }
        } else {
            if (!must_white.count(pos)) {
                other_white.push_back(pos);
            }
        }
    }

    // Place opponent's other stones first
    if (opponent == Player::kPlayer1) {
        if (!place_stones_fixed_order(other_black, Player::kPlayer1)) {
            return std::nullopt;
        }
    } else {
        if (!place_stones_fixed_order(other_white, Player::kPlayer2)) {
            return std::nullopt;
        }
    }

    // Place own other stones
    if (my_perspective == Player::kPlayer1) {
        if (!place_stones_fixed_order(other_black, Player::kPlayer1)) {
            return std::nullopt;
        }
    } else {
        if (!place_stones_fixed_order(other_white, Player::kPlayer2)) {
            return std::nullopt;
        }
    }

    int final_black = env.getStoneBitboard().get(Player::kPlayer1).count();
    int final_white = env.getStoneBitboard().get(Player::kPlayer2).count();

    if (final_black != expected_black || final_white != expected_white) {
        return std::nullopt;
    }

    GoHashKey h = env.getHashKey();
    return SeqState{std::move(seq), h};
}

std::vector<SeqState> sampleInfoSetByMovingStones(
    const GoEnv& truth_env,
    const std::unordered_set<int>& must_black,
    const std::unordered_set<int>& must_white,
    Player my_perspective,
    size_t target_samples,
    int max_move_distance,
    int move_number = -1,
    int total_moves = -1,
    int must_black_size = -1,
    int must_white_size = -1)
{
    const int board_size = truth_env.getBoardSize();
    const GoHashKey truth_hash = truth_env.getHashKey();

    // Ground truth stone counts
    int truth_black = truth_env.getStoneBitboard().get(Player::kPlayer1).count();
    int truth_white = truth_env.getStoneBitboard().get(Player::kPlayer2).count();

    std::vector<SeqState> out;
    out.reserve(target_samples);
    std::unordered_set<GoHashKey> seen;

    // Identify movable stones
    auto movable = identifyMovableStones(truth_env, must_black, must_white, my_perspective);

    if (movable.empty()) {
        // std::cerr << "[WARNING] No movable stones! Cannot use Move-Stone sampling." << std::endl;
        return out;
    }

    std::mt19937 rng{std::random_device{}()};

    int success_count = 0;
    int attempt_count = 0;
    int consecutive_failures = 0;
    const int max_consecutive_failures = 10;

    while (success_count < static_cast<int>(target_samples)) {
        attempt_count++;

        std::uniform_int_distribution<> stone_dis(0, static_cast<int>(movable.size()) - 1);
        int selected_stone = movable[stone_dis(rng)];

        std::map<int, Player> desired_stones;
        std::map<int, Player> original_stones;

        // Add all stones from truth_env
        const auto& black_stones = truth_env.getStoneBitboard().get(Player::kPlayer1);
        const auto& white_stones = truth_env.getStoneBitboard().get(Player::kPlayer2);

        for (int pos = 0; pos < board_size * board_size; ++pos) {
            if (black_stones.test(pos)) {
                desired_stones[pos] = Player::kPlayer1;
                original_stones[pos] = Player::kPlayer1;
            } else if (white_stones.test(pos)) {
                desired_stones[pos] = Player::kPlayer2;
                original_stones[pos] = Player::kPlayer2;
            }
        }

        // Remove selected stone from desired configuration
        desired_stones.erase(selected_stone);

        // Find new target position for the selected stone
        auto candidates = findCandidateTargets(truth_env, selected_stone, my_perspective, max_move_distance);

        if (candidates.empty()) {
            consecutive_failures++;
            if (consecutive_failures >= max_consecutive_failures) {
                break;
            }
            continue;
        }

        // Randomly select a target position
        std::uniform_int_distribution<> target_dis(0, static_cast<int>(candidates.size()) - 1);
        int target_pos = candidates[target_dis(rng)];

        // Add stone at new position (preserving original player)
        Player original_player = original_stones[selected_stone];
        desired_stones[target_pos] = original_player;

        // Reconstruct board with the new configuration
        auto result = reconstructBoardWithStoneConfig(
            board_size, 0, must_black, must_white, desired_stones, my_perspective);

        if (!result.has_value()) {
            consecutive_failures++;
            if (consecutive_failures >= max_consecutive_failures) {
                break;
            }
            continue;
        }

        // Reconstruct the negative board to verify
        GoEnv neg_env(board_size);
        for (const auto& action : result->seq) {
            neg_env.act(action);
        }

        int neg_black = neg_env.getStoneBitboard().get(Player::kPlayer1).count();
        int neg_white = neg_env.getStoneBitboard().get(Player::kPlayer2).count();

        // Verify it's different from the positive
        if (result->hash == truth_hash) {
            consecutive_failures++;
            if (consecutive_failures >= max_consecutive_failures) {
                break;
            }
            continue;
        }

        // Check for duplicates
        if (seen.count(result->hash)) {
            consecutive_failures++;
            if (consecutive_failures >= max_consecutive_failures) {
                break;
            }
            continue;
        }

        // Verify stone counts match
        if (neg_black != truth_black || neg_white != truth_white) {
            consecutive_failures++;
            if (consecutive_failures >= max_consecutive_failures) {
                break;
            }
            continue;
        }

        // Success! Add to output
        seen.insert(result->hash);
        out.push_back(std::move(*result));
        success_count++;
        consecutive_failures = 0; // Reset failure counter on success
    }

    return out;
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
    std::vector<float> negative = getNegative(env_id, pos, rotation, 1);

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
    std::vector<float> negative = getNegative(env_id, pos, rotation, config::siamese_num_negatives);

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

std::vector<float> DataLoaderThread::getNegative(int env_id, int pos, utils::Rotation rotation, int num_negatives)
{
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    const int board_size = env_loader.getBoardSize();

    std::vector<GoEnv> history = rebuildFullHistory(env_loader, pos);

    const GoEnv& truth_env = history[std::min(pos, static_cast<int>(history.size()) - 1)];
    GoHashKey truth_hash = truth_env.getHashKey();

    std::vector<MoveEvent> events = buildMoveEvents(history, env_loader);

    Player my_perspective = (pos > 0 && pos <= static_cast<int>(env_loader.getActionPairs().size())) ? env_loader.getActionPairs()[pos - 1].first.nextPlayer() : Player::kPlayer1;

    auto [must_black, must_white] = computeMustSets(pos, my_perspective, events, history);

    int target_black = countStonesOnBoard(truth_env, Player::kPlayer1);
    int target_white = countStonesOnBoard(truth_env, Player::kPlayer2);

    // Prepare information for passing to sampling functions
    int total_moves = static_cast<int>(env_loader.getActionPairs().size());

    // Select sampling strategy based on configuration
    std::vector<SeqState> info_set;

    if (config::siamese_sampling_strategy == "move_stone") {
        // std::cerr << "[INFO] Using Move-Stone sampling strategy" << std::endl;
        info_set = sampleInfoSetByMovingStones(
            truth_env, must_black, must_white, my_perspective,
            config::siamese_max_sample_negatives,
            config::siamese_max_move_distance,    // max_move_distance from config
            pos,                                  // move_number
            total_moves,                          // total_moves
            static_cast<int>(must_black.size()),  // must_black_size
            static_cast<int>(must_white.size())); // must_white_size

    } else if (config::siamese_sampling_strategy == "random") {
        // std::cerr << "[INFO] Using Random sampling strategy" << std::endl;
        info_set = sampleInfoSetAtMove(
            board_size, pos, must_black, must_white,
            config::siamese_max_sample_negatives,
            my_perspective, target_black, target_white);

    } else if (config::siamese_sampling_strategy == "hybrid") {
        // Calculate number of samples for each strategy
        size_t move_stone_count = static_cast<size_t>(
            config::siamese_max_sample_negatives * config::siamese_move_stone_ratio);
        size_t random_count = config::siamese_max_sample_negatives - move_stone_count;

        // Generate move_stone samples
        auto move_stone_samples = sampleInfoSetByMovingStones(
            truth_env, must_black, must_white, my_perspective,
            move_stone_count,
            config::siamese_max_move_distance,    // max_move_distance from config
            pos,                                  // move_number
            total_moves,                          // total_moves
            static_cast<int>(must_black.size()),  // must_black_size
            static_cast<int>(must_white.size())); // must_white_size

        // Generate random samples
        auto random_samples = sampleInfoSetAtMove(
            board_size, pos, must_black, must_white, random_count,
            my_perspective, target_black, target_white);

        // Combine both
        info_set.insert(info_set.end(), move_stone_samples.begin(), move_stone_samples.end());
        info_set.insert(info_set.end(), random_samples.begin(), random_samples.end());

    } else {
        std::cerr << "[WARNING] Unknown sampling strategy: " << config::siamese_sampling_strategy
                  << ". Falling back to random." << std::endl;
        info_set = sampleInfoSetAtMove(
            board_size, pos, must_black, must_white,
            config::siamese_max_sample_negatives,
            my_perspective, target_black, target_white);
    }

    // Visual comparison output (for debugging)
    static std::atomic<int> sample_count{0};
    int current_sample = ++sample_count;

    if (config::siamese_debug_output && (current_sample <= 3 || current_sample % 100 == 0)) { // Show first 3 and every 100th
        std::cerr << "\n=== Negative Sampling Comparison (Sample #" << current_sample << ") ===" << std::endl;

        // Print Positive (Ground Truth) board
        std::cerr << "\n[POSITIVE] Ground Truth Board:" << std::endl;
        std::cout << truth_env.toString() << std::endl;
        std::cerr << "Hash: " << truth_hash << std::endl;
        std::cerr << "Black: " << target_black << ", White: " << target_white << std::endl;

        // Print first Negative board (if available)
        if (!info_set.empty()) {
            // Find first negative that is different from positive
            const SeqState* first_neg = nullptr;
            for (const auto& seq_state : info_set) {
                if (seq_state.hash != truth_hash) {
                    first_neg = &seq_state;
                    break;
                }
            }

            if (first_neg) {
                GoEnv neg_env(board_size);
                for (const auto& action : first_neg->seq) {
                    neg_env.act(action);
                }

                std::cerr << "\n[NEGATIVE] Sampled Board (sample 1):" << std::endl;
                std::cout << neg_env.toString() << std::endl;
                std::cerr << "Hash: " << neg_env.getHashKey() << std::endl;
                int neg_black = neg_env.getStoneBitboard().get(Player::kPlayer1).count();
                int neg_white = neg_env.getStoneBitboard().get(Player::kPlayer2).count();
                std::cerr << "Black: " << neg_black << ", White: " << neg_white << std::endl;
            } else {
                std::cerr << "\n[WARNING] All sampled boards are same as positive!" << std::endl;
            }
        } else {
            std::cerr << "\n[WARNING] No negatives sampled!" << std::endl;
        }

        std::cerr << "=====================================\n"
                  << std::endl;
    }

    // choose num_negatives boards (not ground truth) from the info set
    std::vector<SeqState> negatives;
    for (const auto& seq_state : info_set) {
        if (seq_state.hash != truth_hash) {
            negatives.push_back(seq_state);
        }
        if (negatives.size() >= static_cast<size_t>(num_negatives)) {
            break;
        }
    }

    if (negatives.empty()) {
        static std::atomic<int> empty_count{0};
        int current_count = ++empty_count;

        if (current_count % 500 == 0) {
            std::cerr << "[WARNING] Empty negatives: " << current_count << std::endl;
        }
        return std::vector<float>();
    }

    std::vector<float> result;
    for (const auto& neg : negatives) {
        GoEnv neg_env(board_size);
        for (const auto& action : neg.seq) {
            neg_env.act(action);
        }

        std::vector<float> board_state = extractBoardState(neg_env, rotation);
        result.insert(result.end(), board_state.begin(), board_state.end());
    }
    // fill up to num_negatives
    while (negatives.size() < static_cast<size_t>(num_negatives)) {
        result.insert(result.end(), board_size * board_size * 2, 0.0f);
        negatives.push_back(SeqState{});
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

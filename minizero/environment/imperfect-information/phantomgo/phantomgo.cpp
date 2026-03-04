#include "phantomgo.h"
#include "color_message.h"
#include "random.h"
#include "utils.h"
#include <utility>

namespace minizero::env::phantomgo {

using namespace minizero::env::go;
using namespace minizero::utils;

std::string bitboardToBoardString(const go::GoBitboard& bitboard, int board_size)
{
    std::ostringstream oss;
    for (int row = board_size - 1; row >= 0; --row) {
        for (int col = 0; col < board_size; ++col) {
            oss << bitboard.test(row * board_size + col) << " ";
        }
        oss << std::endl;
    }
    return oss.str();
}

std::string featuresToBoardString(const std::vector<float>& features, int board_size, int num_columns, const std::vector<std::string> feature_headers /*= {}*/)
{
    std::ostringstream oss;
    int num_features = features.size() / (board_size * board_size);
    for (int i = 0; i < num_features; i += num_columns) {
        if (i / num_columns < static_cast<int>(feature_headers.size())) { oss << feature_headers[i / num_columns] << ":\n"; }
        for (int row = board_size - 1; row >= 0; --row) {
            for (int j = i; j < i + num_columns && j < num_features; ++j) {
                for (int col = 0; col < board_size; ++col) {
                    int f = std::lround(features[j * board_size * board_size + row * board_size + col]);
                    std::string str = std::to_string(f);
                    oss << (f > 0 ? getColorText(str, TextType::kBold, TextColor::kRed, TextColor::kBlack) : str) << " ";
                }
                oss << "    ";
            }
            oss << std::endl;
        }
        oss << std::endl;
    }
    return oss.str();
}

ImperfectGoEnv& ImperfectGoEnv::operator=(const ImperfectGoEnv& env)
{
    go::GoEnv::operator=(env);
    view_player_ = env.view_player_;
    num_opp_stones_ = env.num_opp_stones_;
    tried_pos_ = env.tried_pos_;
    tried_pos_history_ = env.tried_pos_history_;
    capture_opp_history_ = env.capture_opp_history_;
    known_stone_bitboard_history_ = env.known_stone_bitboard_history_;
    return *this;
}

void ImperfectGoEnv::reset()
{
    go::GoEnv::reset();
    num_opp_stones_ = 0;
    tried_pos_.reset();
    tried_pos_history_.clear();
    capture_opp_history_.clear();
    known_stone_bitboard_history_.clear();
}

bool ImperfectGoEnv::act(const PhantomGoAction& action)
{
    if (!go::GoEnv::isLegalAction(action)) { return false; }
    bool success = go::GoEnv::act(action);
    if (!success) { return false; }
    tried_pos_history_.push_back(tried_pos_);
    tried_pos_.reset();
    return true;
}

bool ImperfectGoEnv::isLegalAction(const PhantomGoAction& action) const
{
    /*
        we only need to consider the position that we have tried and the position that already has stones
        illegal action (suicide) are allowed to try since it might be legal in perfect information environment
        e.g., C4 is illegal from the point of our view, but it is actually legal action (capturing) in perfect env
              PERFECT         IMPERFECT
              A B C D          A B C D
            4 X X . X 4      4 X X . X 4
            3 X X X X 3      3 . X X . 3
            2 O O O O 2      2 O O O O 2
            1 . O>O O 1      1 . O>O>O 1
              A B C D          A B C D
    */
    GoBitboard bitboard = tried_pos_ | stone_bitboard_.get(Player::kPlayer1) | stone_bitboard_.get(Player::kPlayer2);
    if (bitboard.test(action.getActionID())) { return false; }
    return true;
}

std::vector<float> ImperfectGoEnv::getFeatures(utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    /*  34 channels:
        0~7. our position for last 8 turns
        8~15. known opponent position for last 8 turns
        16~23. our tried position for last 8 turns
        24~31. our capture position for last 8 turns
        32. black turn
        33. white turn
    */
    int num_grids = board_size_ * board_size_;
    Player next_turn = getNextPlayer(view_player_, kPhantomGoNumPlayer);
    std::vector<float> features(getNumInputChannels() * num_grids, 0.0f);
    for (int i = 0; i < 8; ++i) { // 0~15 channels
        int index = static_cast<int>(known_stone_bitboard_history_.size()) - 1 - i;
        if (index < 0 || index >= static_cast<int>(known_stone_bitboard_history_.size())) { break; }
        const GamePair<GoBitboard>& past_stone_bitboard = known_stone_bitboard_history_[index];
        for (int pos = 0; pos < num_grids; ++pos) {
            int rotated_pos = getRotatePosition(pos, rotation);
            if (past_stone_bitboard.get(view_player_).test(pos)) { features[i * num_grids + rotated_pos] = 1.0f; }
            if (past_stone_bitboard.get(next_turn).test(pos)) { features[(i + 8) * num_grids + rotated_pos] = 1.0f; }
        }
    }
    for (int i = 0; i < 8; ++i) { // 16~23 channels
        int index = static_cast<int>(tried_pos_history_.size()) - i;
        if (index < 0 || index >= static_cast<int>(tried_pos_history_.size())) { break; }
        const GoBitboard& past_tried_pos = (i == 0 ? tried_pos_ : tried_pos_history_[index]);
        for (int pos = 0; pos < num_grids; ++pos) {
            int rotated_pos = getRotatePosition(pos, rotation);
            if (past_tried_pos.test(pos)) { features[(i + 16) * num_grids + rotated_pos] = 1.0f; }
        }
    }
    for (int i = 0; i < 8; ++i) { // 24~31 channels
        int index = static_cast<int>(capture_opp_history_.size()) - 1 - i;
        if (index < 0 || index >= static_cast<int>(capture_opp_history_.size())) { break; }
        const GoBitboard& past_capture_opp = capture_opp_history_[index];
        for (int pos = 0; pos < num_grids; ++pos) {
            int rotated_pos = getRotatePosition(pos, rotation);
            if (past_capture_opp.test(pos)) { features[(i + 24) * num_grids + rotated_pos] = 1.0f; }
        }
    }
    std::fill(features.begin() + (view_player_ == Player::kPlayer1 ? 32 : 33) * num_grids,
              features.begin() + (view_player_ == Player::kPlayer1 ? 33 : 34) * num_grids, 1.0f);
    return features;
}

void ImperfectGoEnv::update(go::GoBitboard captured_stone, Player captured_player)
{
    updateBoard(captured_stone, captured_player);
    if (view_player_ == captured_player) {
        // after opp action, we should update the number of opp stones
        ++num_opp_stones_;
        known_stone_bitboard_history_.push_back(stone_bitboard_);
    } else {
        // after our action, we should update captured history
        capture_opp_history_.push_back(captured_stone);
        num_opp_stones_ -= static_cast<int>(captured_stone.count());
    }
}

std::string ImperfectGoEnv::infoString() const
{
    std::ostringstream oss;
    oss << "ImperfectGoEnv info:" << std::endl;
    oss << "action history:";
    for (auto& a : getActionHistory()) { oss << " " << a.getActionID(); }
    oss << std::endl;
    oss << "view player: " << playerToChar(view_player_) << std::endl;
    oss << "num opponent stones: " << num_opp_stones_ << std::endl;
    oss << "tried positions:\n"
        << bitboardToBoardString(tried_pos_, board_size_);
    oss << "tried positions history size: " << tried_pos_history_.size() << std::endl;
    oss << "tried positions back:\n"
        << (tried_pos_history_.empty() ? "" : bitboardToBoardString(tried_pos_history_.back(), board_size_));
    oss << "capture opponent positions history size: " << capture_opp_history_.size() << std::endl;
    oss << "capture opponent positions back:\n"
        << (capture_opp_history_.empty() ? "" : bitboardToBoardString(capture_opp_history_.back(), board_size_));
    oss << "features:\n";
    oss << featuresToBoardString(getFeatures(utils::Rotation::kRotationNone), board_size_, 8,
                                 {"Our previous 8 boards",
                                  "Opponent previous 8 boards",
                                  "Our tried position for last 8 turns",
                                  "Our capture position for last 8 turns"});
    return oss.str();
}

void ImperfectGoEnv::updateBoard(go::GoBitboard captured_stone, Player captured_player)
{
    if (captured_stone.none()) { return; }
    go::GoBitboard surrounding_bitboard = (dilateBitboard(captured_stone) & ~captured_stone);
    std::vector<PhantomGoAction> actions;
    for (auto& action : actions_) {
        if (captured_stone.test(action.getActionID())) { continue; }
        if (surrounding_bitboard.test(action.getActionID())) { surrounding_bitboard.reset(action.getActionID()); }
        actions.push_back(action);
    }
    int pos;
    while (!surrounding_bitboard.none()) {
        pos = surrounding_bitboard._Find_first();
        surrounding_bitboard.reset(pos);
        actions.push_back(PhantomGoAction(pos, getNextPlayer(captured_player, kPhantomGoNumPlayer)));
    }
    replayGoEnv(actions);
}

void ImperfectGoEnv::replayGoEnv(const std::vector<PhantomGoAction>& actions)
{
    // replay the go environment but also maintain the history variables in imperfect env, so we only reset the go environment
    go::GoEnv::reset();
    for (const auto& action : actions) { go::GoEnv::act(action); }
}

bool PhantomGoEnv::act(const PhantomGoAction& action)
{
    bool success = ImperfectInformationEnv<PhantomGoAction, go::GoEnv, ImperfectGoEnv>::act(action);
    if (!success) {
        if (imperfect_env_.get(getTurn()).isLegalAction(action)) {
            // if illegal hint rule enabled, we should update known stone before update tried pos since it will place the stone on imperfect env
            if (config::env_phantomgo_has_illegal_hint_rule && perfect_env_.getGrid(action.getActionID()).getPlayer() != Player::kPlayerNone) {
                imperfect_env_.get(getTurn()).updateKnownStone(PhantomGoAction(action.getActionID(), action.nextPlayer()));
            }
            imperfect_env_.get(getTurn()).updateTriedPos(action);
        }
        return false;
    }

    imperfect_env_.get(Player::kPlayer1).update(perfect_env_.getCapturedStoneBitBoard(), getTurn());
    imperfect_env_.get(Player::kPlayer2).update(perfect_env_.getCapturedStoneBitBoard(), getTurn());
    return success;
}

void PhantomGoEnv::sampleOneInformationSet(int seed /*= utils::Random::randInt()*/)
{
    assert(!isTerminal());
    PhantomGoEnv sampled_env = createSampledEnvironment(seed);
    *this = sampled_env;
}

std::vector<float> PhantomGoEnv::getPlayerFeatures(utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    /* 6 channels:
        0. our all stone
        1. opponent all stone
        2-3. turn
        4. we known opp's stone
        5. opp knowns our stone
    */
    std::vector<float> features = perfect_env_.getFeatures(rotation);
    std::vector<float> known_stone_features(2 * getBoardSize() * getBoardSize(), 0.0f);
    env::Player our_player = getTurn();
    env::Player opp_player = getNextPlayer(getTurn(), kPhantomGoNumPlayer);
    auto our_know_stone_bitboard = imperfect_env_.get(our_player).getStoneBitboard().get(opp_player);
    while (!our_know_stone_bitboard.none()) {
        int pos = our_know_stone_bitboard._Find_first();
        our_know_stone_bitboard.reset(pos);
        int rotated_pos = getRotatePosition(pos, rotation);
        known_stone_features[rotated_pos] = 1.0f;
    }
    auto opp_know_stone_bitboard = imperfect_env_.get(opp_player).getStoneBitboard().get(our_player);
    while (!opp_know_stone_bitboard.none()) {
        int pos = opp_know_stone_bitboard._Find_first();
        opp_know_stone_bitboard.reset(pos);
        int rotated_pos = getRotatePosition(pos, rotation);
        known_stone_features[getBoardSize() * getBoardSize() + rotated_pos] = 1.0f;
    }
    features.insert(features.end(), known_stone_features.begin(), known_stone_features.end());
    return features;
}

std::string PhantomGoEnv::toSGFString(bool with_tried /*= true*/) const
{
    std::ostringstream oss;
    oss << "(;FF[4]GM[1]SZ[" << perfect_env_.getBoardSize() << "]KM[" << perfect_env_.getKomi() << "]";
    const auto& action_history = (with_tried ? getActionHistory() : perfect_env_.getActionHistory());
    for (const auto& action : action_history) {
        oss << ";" << std::string(1, env::playerToChar(action.getPlayer()))
            << "[" << SGFLoader::actionIDToSGFString(action.getActionID(), perfect_env_.getBoardSize()) + "]";
    }
    oss << ")";
    return oss.str();
}

std::string PhantomGoEnv::infoString() const
{
    std::ostringstream oss;
    oss << "PhantomGo (sgf):";
    for (auto& a : getActionHistory()) { oss << " " << a.getActionID(); }
    oss << std::endl;
    oss << "Perfect Env:    ";
    for (auto& a : perfect_env_.getActionHistory()) { oss << " " << a.getActionID(); }
    oss << std::endl;
    oss << std::endl;

    oss << imperfect_env_.get(getTurn()).infoString() << std::endl;
    return oss.str();
}

PhantomGoEnv PhantomGoEnv::createSampledEnvironment(int seed) const
{
    std::mt19937 generator(seed);
    const ImperfectGoEnv& imperfect_env = imperfect_env_.get(getTurn());

    // play our actions
    Player turn = getTurn();
    Player next_turn = getNextPlayer(turn, kPhantomGoNumPlayer);
    PhantomGoEnv sampled_env;
    for (size_t i = 0; i < imperfect_env.getActionHistory().size(); ++i) {
        const PhantomGoAction& action = imperfect_env.getActionHistory()[i];
        if (action.getPlayer() != turn) { continue; }
        if (imperfect_env.isPassAction(action)) { continue; }
        sampled_env.act(action);
        sampled_env.act(PhantomGoAction(action.getActionID(), getNextPlayer(turn, kPhantomGoNumPlayer)));
    }

    // play known opponent stones
    GoBitboard known_stone = imperfect_env.getStoneBitboard().get(next_turn);
    int remaining_opp_stones = imperfect_env.getNumOpponentStones() - known_stone.count();
    while (!known_stone.none()) {
        int pos = known_stone._Find_first();
        known_stone.reset(pos);
        sampled_env.act(PhantomGoAction(pos, next_turn));
        sampled_env.act(PhantomGoAction(pos, turn));
    }

    // play unknown opponent stones
    std::uniform_int_distribution<int> int_distribution;
    std::vector<PhantomGoAction> opp_legal_actions;
    for (int pos = 0; pos < imperfect_env.getBoardSize() * imperfect_env.getBoardSize(); ++pos) {
        PhantomGoAction action(pos, next_turn);
        if (!imperfect_env.isLegalAction(action)) { continue; }
        opp_legal_actions.push_back(action);
    }
    for (int i = 0; i < remaining_opp_stones && !opp_legal_actions.empty(); ++i) {
        int selected_index = int_distribution(generator) % opp_legal_actions.size();
        const PhantomGoAction& action = opp_legal_actions[selected_index];
        if (sampled_env.getPerfectEnv().isLegalAction(action) && !sampled_env.getPerfectEnv().isCaptureMove(action)) {
            sampled_env.act(action);
            sampled_env.act(PhantomGoAction(action.getActionID(), turn));
        } else {
            --i;
        }
        opp_legal_actions[selected_index] = opp_legal_actions.back();
        opp_legal_actions.pop_back();
    }
    sampled_env.setTurn(turn);

    assert(sampled_env.getImperfectEnv(turn).getStoneBitboard().get(turn) == imperfect_env_.get(turn).getStoneBitboard().get(turn));
    assert(!sampled_env.isTerminal());
    return sampled_env;
}

std::pair<std::vector<float>, std::vector<float>> PhantomGoEnvLoader::getISGeneratorFeaturesAndLabel(const int pos, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    PhantomGoEnv env;
    for (int i = 0; i < pos; ++i) { env.act(getActionPairs()[i].first); }

    GoBitboard known_stone_bitboard;
    Player next_turn = getNextPlayer(env.getTurn(), kPhantomGoNumPlayer);
    if (!env.getImperfectEnv(env.getTurn()).getKnownStoneBitboardHistory().empty()) {
        known_stone_bitboard = env.getImperfectEnv(env.getTurn()).getKnownStoneBitboardHistory().back().get(next_turn);
    }
    GoBitboard unknown_stone_bitboard = (env.getPerfectEnv().getStoneBitboard().get(next_turn) & ~known_stone_bitboard);
    std::vector<int> unknown_stone_pos;
    while (!unknown_stone_bitboard.none()) {
        unknown_stone_pos.push_back(unknown_stone_bitboard._Find_first());
        unknown_stone_bitboard.reset(unknown_stone_pos.back());
    }
    std::shuffle(unknown_stone_pos.begin(), unknown_stone_pos.end(), Random::generator_);

    std::vector<float> labels(getPolicySize(), 0.0f);
    int num_predicted = Random::randInt() % (unknown_stone_pos.size() + 1);
    for (size_t i = num_predicted; i < unknown_stone_pos.size(); ++i) { labels[getRotateAction(unknown_stone_pos[i], rotation)] = 1.0f / (unknown_stone_pos.size() - num_predicted); }
    if (num_predicted == static_cast<int>(unknown_stone_pos.size())) {
        labels.back() = 1.0f; // PASS for termination
    }

    std::vector<float> features = env.getFeatures(false, rotation);
    std::vector<float> predicted_feature(board_size_ * board_size_, 0.0f);
    for (int i = 0; i < num_predicted; ++i) { predicted_feature[getRotateAction(unknown_stone_pos[i], rotation)] = 1.0f; }
    features.insert(features.end(), predicted_feature.begin(), predicted_feature.end());

    return {features, labels};
}

} // namespace minizero::env::phantomgo

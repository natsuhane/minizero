#include "darkhex.h"
#include "color_message.h"
#include "random.h"
#include "utils.h"
#include <utility>

#include <iostream>

/*
If a player wants to place a piece on a square that has already been captured:
- In classic DarkHex, player receive the information that the square is not available, and allows to choose another square.
- In abrupt DarkHex, player lose their turn. (Don't implement this rule for now)

For more DarkHex variants:
- https://webdocs.cs.ualberta.ca/~hayward/theses/bedir.pdf
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10213697/

Remove the swap rule, because it is confusing to decide.
*/

namespace minizero::env::darkhex {

using namespace minizero::env::hex;
using namespace minizero::utils;

void ImperfectHexEnv::reset()
{
    hex::HexEnv::reset();
    num_stones_ = 0;
}

bool ImperfectHexEnv::act(const DarkHexAction& action)
{
    if (!isLegalAction(action)) { return false; }
    bool success = hex::HexEnv::act(action);
    if (!success) { return false; }
    if (action.getPlayer() == view_player_) { ++num_stones_; }
    return true;
}

std::vector<float> ImperfectHexEnv::getFeatures(utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    /*  4 channels:
        0~1. our/opponent's stones in my imperfect view
        2. black turn
        3. white turn
    */
    int num_grids = board_size_ * board_size_;
    std::vector<float> features(getNumInputChannels() * num_grids, 0.0f);
    for (int pos = 0; pos < num_grids; ++pos) {
        features[pos] = (getBoard()[pos].player == turn_ ? 1.0f : 0.0f);
        features[num_grids + pos] = (getBoard()[pos].player == getNextPlayer(turn_, kDarkHexNumPlayer) ? 1.0f : 0.0f);
    }
    std::fill(features.begin() + (view_player_ == Player::kPlayer1 ? 2 : 3) * num_grids,
              features.begin() + (view_player_ == Player::kPlayer1 ? 3 : 4) * num_grids, 1.0f);

    return features;
}

bool DarkHexEnv::act(const DarkHexAction& action)
{
    // successfully act this action on perfect board
    bool success = ImperfectInformationEnv<DarkHexAction, hex::HexEnv, ImperfectHexEnv>::act(action);
    if (!success) {
        // update opponent's occupied grid to my imperfect board
        if (imperfect_env_.get(getTurn()).isLegalAction(action)) { imperfect_env_.get(getTurn()).act(DarkHexAction(action.getActionID(), action.nextPlayer())); }
        return false;
    }
    return success;
}

void DarkHexEnv::sampleOneInformationSet(int seed /*= utils::Random::randInt()*/)
{
    assert(!isTerminal());

    std::mt19937 generator(seed);

    Player turn = getTurn();
    Player next_turn = getNextPlayer(turn, kDarkHexNumPlayer);
    DarkHexEnv sampled_env;
    const ImperfectHexEnv& imperfect_env = imperfect_env_.get(turn);
    int remaining_opp_stones = imperfect_env_.get(next_turn).getNumStones();

    // play our actions & known opponent stones
    for (size_t i = 0; i < imperfect_env.getActionHistory().size(); ++i) {
        const DarkHexAction& action = imperfect_env.getActionHistory()[i];
        sampled_env.act(action);
        sampled_env.act(DarkHexAction(action.getActionID(), getNextPlayer(action.getPlayer(), kDarkHexNumPlayer)));
        if (action.getPlayer() == next_turn) { --remaining_opp_stones; }
    }
    // play unknown opponent stones
    std::vector<DarkHexAction> opp_legal_actions;
    for (int pos = 0; pos < imperfect_env.getBoardSize() * imperfect_env.getBoardSize(); ++pos) {
        DarkHexAction action(pos, next_turn);
        if (!imperfect_env.isLegalAction(action)) { continue; }
        opp_legal_actions.push_back(action);
    }
    std::uniform_int_distribution<int> int_distribution;
    for (int i = 0; i < remaining_opp_stones && !opp_legal_actions.empty(); ++i) {
        int selected_index = int_distribution(generator) % opp_legal_actions.size();
        const DarkHexAction& action = opp_legal_actions[selected_index];
        if (!sampled_env.getPerfectEnv().isWinningMove(action)) {
            sampled_env.act(action);
            sampled_env.act(DarkHexAction(action.getActionID(), turn));
        } else {
            --i;
        }
        opp_legal_actions[selected_index] = opp_legal_actions.back();
        opp_legal_actions.pop_back();
    }
    sampled_env.setTurn(turn);

    assert(!sampled_env.isTerminal());
    *this = sampled_env;
}

std::vector<float> DarkHexEnv::getPlayerFeatures(utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
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
    env::Player opp_player = getNextPlayer(getTurn(), kDarkHexNumPlayer);
    int num_grids = perfect_env_.getBoardSize() * perfect_env_.getBoardSize();
    for (int pos = 0; pos < num_grids; ++pos) {
        known_stone_features[pos] = (imperfect_env_.get(our_player).getBoard()[pos].player == opp_player ? 1.0f : 0.0f);
        known_stone_features[num_grids + pos] = (imperfect_env_.get(opp_player).getBoard()[pos].player == our_player ? 1.0f : 0.0f);
    }
    features.insert(features.end(), known_stone_features.begin(), known_stone_features.end());
    return features;
}

// TODO
std::vector<float> DarkHexEnv::getDiscriminatorFeatures(utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    std::vector<float> dummy;
    return dummy;
}

std::pair<std::vector<float>, std::vector<float>> DarkHexEnvLoader::getISGeneratorFeaturesAndLabel(const int pos, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    std::pair<std::vector<float>, std::vector<float>> dummy;
    return dummy;
}

} // namespace minizero::env::darkhex

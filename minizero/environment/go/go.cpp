#include "go.h"
#include "color_message.h"
#include "configuration.h"
#include "random.h"
#include "rotation.h"
#include "sgf_loader.h"
#include <algorithm>
#include <atomic>
#include <deque>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>

namespace minizero::env::go {

using namespace minizero::utils;

GoHashKey turn_hash_key;
std::vector<GoHashKey> empty_hash_key;
std::vector<GamePair<GoHashKey>> grids_hash_key;
std::vector<std::vector<GamePair<GoHashKey>>> sequence_hash_key;

void initialize()
{
    std::mt19937_64 generator;
    generator.seed(0);
    turn_hash_key = generator();

    // empty & grid hash key
    empty_hash_key.resize(kMaxGoBoardSize * kMaxGoBoardSize);
    grids_hash_key.resize(kMaxGoBoardSize * kMaxGoBoardSize);
    for (int pos = 0; pos < kMaxGoBoardSize * kMaxGoBoardSize; ++pos) {
        empty_hash_key[pos] = generator();
        grids_hash_key[pos].get(Player::kPlayer1) = generator();
        grids_hash_key[pos].get(Player::kPlayer2) = generator();
    }

    // sequence hash key
    sequence_hash_key.resize(2 * kMaxGoBoardSize * kMaxGoBoardSize);
    for (int move = 0; move < 2 * kMaxGoBoardSize * kMaxGoBoardSize; ++move) {
        sequence_hash_key[move].resize(kMaxGoBoardSize * kMaxGoBoardSize + 1);
        for (int pos = 0; pos < kMaxGoBoardSize * kMaxGoBoardSize + 1; ++pos) {
            sequence_hash_key[move][pos].get(Player::kPlayer1) = generator();
            sequence_hash_key[move][pos].get(Player::kPlayer2) = generator();
        }
    }
}

GoHashKey getGoTurnHashKey()
{
    assert(config::env_go_ko_rule == "positional" || config::env_go_ko_rule == "situational");
    return (config::env_go_ko_rule == "positional" ? 0 : turn_hash_key);
}

GoHashKey getGoEmptyHashKey(int position)
{
    assert(position >= 0 && position < kMaxGoBoardSize * kMaxGoBoardSize);
    return empty_hash_key[position];
}

GoHashKey getGoGridHashKey(int position, Player p)
{
    assert(position >= 0 && position < kMaxGoBoardSize * kMaxGoBoardSize);
    assert(p == Player::kPlayer1 || p == Player::kPlayer2);
    return grids_hash_key[position].get(p);
}

GoHashKey getGoSequenceHashKey(int move, int position, Player p)
{
    assert(move >= 0 && move < 2 * kMaxGoBoardSize * kMaxGoBoardSize);
    assert(position >= 0 && position <= kMaxGoBoardSize * kMaxGoBoardSize);
    assert(p == Player::kPlayer1 || p == Player::kPlayer2);
    return sequence_hash_key[move][position].get(p);
}

GoEnv& GoEnv::operator=(const GoEnv& env)
{
    board_size_ = env.board_size_;
    komi_ = env.komi_;
    turn_ = env.turn_;
    hash_key_ = env.hash_key_;
    board_mask_bitboard_ = env.board_mask_bitboard_;
    board_left_boundary_bitboard_ = env.board_left_boundary_bitboard_;
    board_right_boundary_bitboard_ = env.board_right_boundary_bitboard_;
    free_area_id_bitboard_ = env.free_area_id_bitboard_;
    free_block_id_bitboard_ = env.free_block_id_bitboard_;
    stone_bitboard_ = env.stone_bitboard_;
    benson_bitboard_ = env.benson_bitboard_;
    grids_ = env.grids_;
    areas_ = env.areas_;
    blocks_ = env.blocks_;
    actions_ = env.actions_;
    stone_bitboard_history_ = env.stone_bitboard_history_;
    hashkey_history_ = env.hashkey_history_;
    hash_table_ = env.hash_table_;
    captured_stone_bitboard_ = env.captured_stone_bitboard_;
    num_captured_stones_ = env.num_captured_stones_;

    // reset grid's block and area pointer
    for (auto& grid : grids_) {
        if (grid.getBlock()) { grid.setBlock(&blocks_[grid.getBlock()->getID()]); }
        if (grid.getArea(Player::kPlayer1)) { grid.setArea(Player::kPlayer1, &areas_[grid.getArea(Player::kPlayer1)->getID()]); }
        if (grid.getArea(Player::kPlayer2)) { grid.setArea(Player::kPlayer2, &areas_[grid.getArea(Player::kPlayer2)->getID()]); }
    }
    return *this;
}

void GoEnv::reset()
{
    komi_ = minizero::config::env_go_komi;
    turn_ = Player::kPlayer1;
    hash_key_ = 0;
    stone_bitboard_.reset();
    benson_bitboard_.reset();
    board_mask_bitboard_.reset();
    for (int i = 0; i < board_size_ * board_size_; ++i) {
        grids_[i].reset(board_size_);
        areas_[i].reset();
        blocks_[i].reset();
        board_mask_bitboard_.set(i);
    }
    free_area_id_bitboard_.reset();
    free_area_id_bitboard_ = ~free_area_id_bitboard_ & board_mask_bitboard_;
    free_block_id_bitboard_.reset();
    free_block_id_bitboard_ = ~free_block_id_bitboard_ & board_mask_bitboard_;
    board_left_boundary_bitboard_.reset();
    board_right_boundary_bitboard_.reset();
    for (int i = 0; i < board_size_; ++i) {
        board_left_boundary_bitboard_.set(i * board_size_);
        board_right_boundary_bitboard_.set(i * board_size_ + (board_size_ - 1));
    }
    actions_.clear();
    stone_bitboard_history_.clear();
    hashkey_history_.clear();
    hash_table_.clear();
    captured_stone_bitboard_.reset();
    num_captured_stones_.reset();
    num_captured_stones_.get(Player::kPlayer1).resize(board_size_ * board_size_, 0.0f);
    num_captured_stones_.get(Player::kPlayer2).resize(board_size_ * board_size_, 0.0f);
}

bool GoEnv::act(const GoAction& action)
{
    if (!isLegalAction(action)) { return false; }

    const int position = action.getActionID();
    const Player player = action.getPlayer();

    // handle global status
    turn_ = action.nextPlayer();
    hash_key_ ^= getGoTurnHashKey();
    actions_.push_back(action);
    captured_stone_bitboard_.reset();

    if (isPassAction(action)) {
        stone_bitboard_history_.push_back(stone_bitboard_);
        hashkey_history_.push_back(hash_key_);
        hash_table_.insert(hash_key_);
        return true;
    }

    // set grid color
    GoGrid& grid = grids_[position];
    grid.setPlayer(player);
    hash_key_ ^= getGoGridHashKey(position, player);

    // create new block
    GoBlock* new_block = newBlock();
    grid.setBlock(new_block);
    new_block->setPlayer(player);
    new_block->addGrid(position);
    new_block->addHashKey(getGoGridHashKey(position, player));

    // combine with neighbor own blocks and capture neighbor opponent blocks
    for (const auto& neighbor_pos : grid.getNeighbors()) {
        GoGrid& neighbor_grid = grids_[neighbor_pos];
        if (neighbor_grid.getPlayer() == Player::kPlayerNone) {
            new_block->addLiberty(neighbor_pos);
        } else {
            GoBlock* neighbor_block = neighbor_grid.getBlock();
            neighbor_block->removeLiberty(position);
            if (neighbor_block->getPlayer() == player) {
                new_block = combineBlocks(new_block, neighbor_block);
            } else {
                if (neighbor_block->getNumLiberty() == 0) {
                    captured_stone_bitboard_ |= neighbor_block->getGridBitboard();

                    GoBitboard grid_bitboard = neighbor_block->getGridBitboard();
                    while (!grid_bitboard.none()) {
                        int pos = grid_bitboard._Find_first();
                        grid_bitboard.reset(pos);
                        ++num_captured_stones_.get(neighbor_block->getPlayer())[pos];
                    }

                    removeBlockFromBoard(neighbor_block);
                }
            }
        }
    }

    stone_bitboard_.get(player) |= new_block->getGridBitboard();
    stone_bitboard_history_.push_back(stone_bitboard_);
    hashkey_history_.push_back(hash_key_);
    hash_table_.insert(hash_key_);

    // update area & benson
    updateArea(action);
    updateBenson(action);

    assert(checkDataStructure());
    return true;
}

bool GoEnv::act(const std::vector<std::string>& action_string_args)
{
    return act(GoAction(action_string_args, board_size_));
}

std::vector<GoAction> GoEnv::getLegalActions() const
{
    std::vector<GoAction> actions;
    for (int pos = 0; pos <= board_size_ * board_size_; ++pos) {
        GoAction action(pos, turn_);
        if (!isLegalAction(action)) { continue; }
        actions.push_back(action);
    }
    return actions;
}

bool GoEnv::isLegalAction(const GoAction& action) const
{
    assert(action.getActionID() >= 0 && action.getActionID() <= board_size_ * board_size_);
    assert(action.getPlayer() == Player::kPlayer1 || action.getPlayer() == Player::kPlayer2);

    if (isPassAction(action)) { return true; }

    const int position = action.getActionID();
    const Player player = action.getPlayer();
    const GoGrid& grid = grids_[position];
    if (grid.getPlayer() != Player::kPlayerNone) { return false; }

    bool is_legal = false;
    GoBitboard check_neighbor_block_bitboard;
    GoHashKey new_hash_key = hash_key_ ^ getGoTurnHashKey() ^ getGoGridHashKey(position, player);
    for (const auto& neighbor_pos : grid.getNeighbors()) {
        const GoGrid& neighbor_grid = grids_[neighbor_pos];
        if (neighbor_grid.getPlayer() == Player::kPlayerNone) {
            is_legal = true;
        } else {
            const GoBlock* neighbor_block = neighbor_grid.getBlock();
            if (check_neighbor_block_bitboard.test(neighbor_block->getID())) { continue; }

            check_neighbor_block_bitboard.set(neighbor_block->getID());
            if (neighbor_block->getPlayer() == player) {
                if (neighbor_block->getNumLiberty() > 1) { is_legal = true; }
            } else {
                if (neighbor_block->getNumLiberty() == 1) {
                    new_hash_key ^= neighbor_block->getHashKey();
                    is_legal = true;
                }
            }
        }
    }

    return (is_legal && hash_table_.count(new_hash_key) == 0);
}

bool GoEnv::isTerminal() const
{
    // two consecutive passes
    if (actions_.size() >= 2 &&
        isPassAction(actions_.back()) &&
        isPassAction(actions_[actions_.size() - 2])) { return true; }

    // game length exceeds 2 * boardsize * boardsize
    if (static_cast<int>(actions_.size()) > 2 * board_size_ * board_size_) { return true; }

    return false;
}

float GoEnv::getEvalScore(bool is_resign /*= false*/) const
{
    Player eval;
    if (is_resign) {
        eval = getNextPlayer(turn_, kGoNumPlayer);
    } else {
        GamePair<float> territory = calculateTrompTaylorTerritory();
        eval = (territory.get(Player::kPlayer1) > territory.get(Player::kPlayer2))
                   ? Player::kPlayer1
                   : ((territory.get(Player::kPlayer1) < territory.get(Player::kPlayer2))
                          ? Player::kPlayer2
                          : Player::kPlayerNone);
    }

    switch (eval) {
        case Player::kPlayer1: return 1.0f;
        case Player::kPlayer2: return -1.0f;
        default: return 0.0f;
    }
}

std::vector<float> GoEnv::getFeatures(utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    /* 18 channels:
        0~15. own/opponent position for last 8 turns
        16. black turn
        17. white turn
    */
    std::vector<float> features;
    for (int channel = 0; channel < 18; ++channel) {
        for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
            int rotation_pos = getRotatePosition(pos, utils::reversed_rotation[static_cast<int>(rotation)]);
            if (channel < 16) {
                int last_n_turn = stone_bitboard_history_.size() - 1 - channel / 2;
                if (last_n_turn < 0) {
                    features.push_back(0.0f);
                } else {
                    const GamePair<GoBitboard>& last_n_trun_stone_bitboard = stone_bitboard_history_[last_n_turn];
                    Player player = (channel % 2 == 0 ? turn_ : getNextPlayer(turn_, kGoNumPlayer));
                    features.push_back(last_n_trun_stone_bitboard.get(player).test(rotation_pos) ? 1.0f : 0.0f);
                }
            } else if (channel == 16) {
                features.push_back((turn_ == Player::kPlayer1 ? 1.0f : 0.0f));
            } else if (channel == 17) {
                features.push_back((turn_ == Player::kPlayer2 ? 1.0f : 0.0f));
            }
        }
    }
    return features;
}

std::vector<float> GoEnv::getSiameseFeatures(utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    /* 4 channels:
            0~1. own/opponent position for last turns
            2. black turn
            3. white turn
    */
    std::vector<float> features;
    for (int channel = 0; channel < 4; ++channel) {
        for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
            int rotation_pos = getRotatePosition(pos, utils::reversed_rotation[static_cast<int>(rotation)]);
            if (channel < 2) {
                if (stone_bitboard_history_.empty()) {
                    features.push_back(0.0f);
                } else {
                    Player player = (channel % 2 == 0 ? turn_ : getNextPlayer(turn_, kGoNumPlayer));
                    features.push_back(stone_bitboard_history_.back().get(player).test(rotation_pos) ? 1.0f : 0.0f);
                }
            } else if (channel == 2) {
                features.push_back((turn_ == Player::kPlayer1 ? 1.0f : 0.0f));
            } else if (channel == 3) {
                features.push_back((turn_ == Player::kPlayer2 ? 1.0f : 0.0f));
            }
        }
    }
    return features;
}

std::vector<float> GoEnv::getInfoSetGeneratorFeatures(int move_number, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    /* 28 channels:
        0~7. our previous 8 boards
        8~15. opponent previous 8 boards
        16~23. our next 8 moves position (16 moves ahead)
        24. our current board
        25. opponent current board
        26. our color is black
        27. our color is white
    */
    Player next_turn = getNextPlayer(turn_, kGoNumPlayer);
    const int num_grids = board_size_ * board_size_;
    std::vector<float> features(28 * num_grids, 0.0f);
    for (int i = 0; i < 8; ++i) { // 0~15 channels
        if (move_number - i - 1 < 0) { break; }
        const GamePair<GoBitboard>& past_stone_bitboard = stone_bitboard_history_[move_number - i - 1];
        for (int pos = 0; pos < num_grids; ++pos) {
            int rotated_pos = getRotatePosition(pos, rotation);
            if (past_stone_bitboard.get(turn_).test(pos)) { features[i * num_grids + rotated_pos] = 1.0f; }
            if (past_stone_bitboard.get(next_turn).test(pos)) { features[(i + 8) * num_grids + rotated_pos] = 1.0f; }
        }
    }
    for (int i = 0; i < 8; ++i) { // 16~23 channels
        if (move_number + i * 2 + 1 >= static_cast<int>(actions_.size())) { break; }
        const GoAction& future_action = actions_[move_number + i * 2 + 1];
        features[(i + 16) * num_grids + getRotatePosition(future_action.getActionID(), rotation)] = 1.0f;
    }
    for (int pos = 0; pos < num_grids; ++pos) { // 24~25 channels
        int rotated_pos = getRotatePosition(pos, rotation);
        if (stone_bitboard_history_.size() > 0 && stone_bitboard_history_.back().get(turn_).test(pos)) { features[24 * num_grids + rotated_pos] = 1.0f; }
        if (move_number - 1 > 0 && move_number - 1 < static_cast<int>(stone_bitboard_history_.size()) && stone_bitboard_history_[move_number - 1].get(next_turn).test(pos)) { features[25 * num_grids + rotated_pos] = 1.0f; }
    }
    std::fill(features.begin() + (turn_ == Player::kPlayer1 ? 26 : 27) * num_grids,
              features.begin() + (turn_ == Player::kPlayer1 ? 27 : 28) * num_grids, 1.0f);
    return features;
}

std::vector<float> GoEnv::getActionFeatures(const GoAction& action, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    std::vector<float> action_features(board_size_ * board_size_, 0.0f);
    if (!isPassAction(action)) { action_features[getRotateAction(action.getActionID(), rotation)] = 1.0f; }
    return action_features;
}

std::string GoEnv::toString() const
{
    int last_move_pos = -1, last2_move_pos = -1;
    if (actions_.size() >= 1) { last_move_pos = actions_.back().getActionID(); }
    if (actions_.size() >= 2) { last2_move_pos = actions_[actions_.size() - 2].getActionID(); }

    std::unordered_map<Player, std::pair<std::string, TextColor>> player_to_text_color({{Player::kPlayerNone, {".", TextColor::kBlack}},
                                                                                        {Player::kPlayer1, {"O", TextColor::kBlack}},
                                                                                        {Player::kPlayer2, {"O", TextColor::kWhite}}});
    std::ostringstream oss;
    oss << getCoordinateString() << std::endl;
    for (int row = board_size_ - 1; row >= 0; --row) {
        oss << getColorText((row + 1 < 10 ? " " : "") + std::to_string(row + 1), TextType::kBold, TextColor::kBlack, TextColor::kYellow);
        for (int col = 0; col < board_size_; ++col) {
            int pos = row * board_size_ + col;
            const GoGrid& grid = grids_[pos];
            const std::pair<std::string, TextColor> text_pair = player_to_text_color[grid.getPlayer()];
            if (pos == last_move_pos) {
                oss << getColorText(">", TextType::kBold, TextColor::kRed, TextColor::kYellow);
                oss << getColorText(text_pair.first, TextType::kBold, text_pair.second, TextColor::kYellow);
            } else if (pos == last2_move_pos) {
                oss << getColorText(">", TextType::kNormal, TextColor::kRed, TextColor::kYellow);
                oss << getColorText(text_pair.first, TextType::kBold, text_pair.second, TextColor::kYellow);
            } else {
                oss << getColorText(" " + text_pair.first, TextType::kBold, text_pair.second, TextColor::kYellow);
            }
        }
        oss << getColorText(" " + std::to_string(row + 1) + (row + 1 < 10 ? " " : ""), TextType::kBold, TextColor::kBlack, TextColor::kYellow);
        oss << std::endl;
    }
    oss << getCoordinateString() << std::endl;
    return oss.str();
}

std::string GoEnv::toSGFString() const
{
    std::ostringstream oss;
    oss << "(;FF[4]GM[1]SZ[" << board_size_ << "]KM[" << komi_ << "]";
    for (const auto& action : actions_) {
        oss << ";" << std::string(1, env::playerToChar(action.getPlayer()))
            << "[" << SGFLoader::actionIDToSGFString(action.getActionID(), board_size_) + "]";
    }
    oss << ")";
    return oss.str();
}

GoBitboard GoEnv::dilateBitboard(const GoBitboard& bitboard) const
{
    return ((bitboard << board_size_) |                           // move up
            (bitboard >> board_size_) |                           // move down
            ((bitboard & ~board_left_boundary_bitboard_) >> 1) |  // move left
            ((bitboard & ~board_right_boundary_bitboard_) << 1) | // move right
            bitboard) &
           board_mask_bitboard_;
}

bool GoEnv::isCaptureMove(const GoAction& action)
{
    if (isPassAction(action)) { return false; }

    const Player player = action.getPlayer();
    const GoGrid& grid = getGrid(action.getActionID());
    assert(grid.getPlayer() == Player::kPlayerNone);

    for (const auto& neighbor_pos : grid.getNeighbors()) {
        const GoGrid& neighbor_grid = getGrid(neighbor_pos);
        const GoBlock* neighbor_block = neighbor_grid.getBlock();
        if (neighbor_block == nullptr) { continue; }
        if (neighbor_block->getPlayer() == player) { continue; }
        if (neighbor_block->getNumLiberty() == 1) { return true; }
    }

    return false;
}

void GoEnv::initialize()
{
    grids_.clear();
    areas_.clear();
    blocks_.clear();
    for (int pos = 0; pos < board_size_ * board_size_; ++pos) {
        grids_.emplace_back(GoGrid(pos, board_size_));
        areas_.emplace_back(GoArea(pos));
        blocks_.emplace_back(GoBlock(pos));
    }
}

GoBlock* GoEnv::newBlock()
{
    assert(!free_block_id_bitboard_.none());
    int id = free_block_id_bitboard_._Find_first();
    free_block_id_bitboard_.reset(id);
    return &blocks_[id];
}

void GoEnv::removeBlock(GoBlock* block)
{
    assert(block && !free_block_id_bitboard_.test(block->getID()));
    free_block_id_bitboard_.set(block->getID());
    block->reset();
}

void GoEnv::removeBlockFromBoard(GoBlock* block)
{
    assert(block);

    // update area
    GoArea* area = nullptr;
    GoBitboard area_id = block->getNeighborAreaIDBitboard();
    while (!area_id.none()) {
        int id = area_id._Find_first();
        area_id.reset(id);
        if (!area) {
            area = &areas_[id];
        } else {
            area = mergeArea(area, &areas_[id]);
        }
    }
    assert(area);
    area->setNumGrid(area->getNumGrid() + block->getNumGrid());
    area->setAreaBitBoard(area->getAreaBitboard() | block->getGridBitboard());
    area->removeNeighborBlockIDBitboard(block->getID());
    if (area->getNumGrid() == board_size_ * board_size_) {
        // remove area when area include whole board
        removeArea(area);
        area = nullptr;
    }

    // remove block
    GoBitboard grid_bitboard = block->getGridBitboard();
    while (!grid_bitboard.none()) {
        int pos = grid_bitboard._Find_first();
        grid_bitboard.reset(pos);

        GoGrid& grid = grids_[pos];
        grid.setPlayer(Player::kPlayerNone);
        grid.setBlock(nullptr);
        grid.setArea(block->getPlayer(), area);
        for (const auto& neighbor_pos : grid.getNeighbors()) {
            GoGrid& neighbor_grid = grids_[neighbor_pos];
            if (neighbor_grid.getPlayer() != getNextPlayer(block->getPlayer(), kGoNumPlayer)) { continue; }
            neighbor_grid.getBlock()->addLiberty(pos);
        }
    }
    hash_key_ ^= block->getHashKey();
    stone_bitboard_.get(block->getPlayer()) &= ~block->getGridBitboard();
    removeBlock(block);
}

GoBlock* GoEnv::combineBlocks(GoBlock* block1, GoBlock* block2)
{
    assert(block1 && block2);

    if (block1 == block2) { return block1; }
    if (block1->getNumGrid() < block2->getNumGrid()) { return combineBlocks(block2, block1); }

    // link grid to new block
    GoBitboard grid_bitboard = block2->getGridBitboard();
    while (!grid_bitboard.none()) {
        int pos = grid_bitboard._Find_first();
        grid_bitboard.reset(pos);
        grids_[pos].setBlock(block1);
    }

    // link area to new block
    GoBitboard new_area_id_bitboard = block2->getNeighborAreaIDBitboard();
    while (!new_area_id_bitboard.none()) {
        int id = new_area_id_bitboard._Find_first();
        new_area_id_bitboard.reset(id);
        areas_[id].removeNeighborBlockIDBitboard(block2->getID());
        areas_[id].addNeighborBlockIDBitboard(block1->getID());
    }

    block1->combineWithBlock(block2);
    removeBlock(block2);
    return block1;
}

void GoEnv::updateArea(const GoAction& action)
{
    if (isPassAction(action)) { return; }
    assert(grids_[action.getActionID()].getPlayer() != Player::kPlayerNone);

    // use last move to update area:
    //    1. last move in own area:
    //        a. last move splits area => remove current area and find area
    //        b. last move didn't split area => remove current move from area
    //    2. last move not in own area => find area
    GoGrid& grid = grids_[action.getActionID()];
    Player own_player = grid.getPlayer();
    GoArea* own_area = grid.getArea(own_player);
    std::vector<GoBitboard> areas_bitboard = findAreas(action);
    if (own_area && areas_bitboard.size() == 1) {
        if (own_area->getNumGrid() == 1) {
            removeArea(own_area);
        } else {
            grid.setArea(action.getPlayer(), nullptr);
            grid.getBlock()->addNeighborAreaIDBitboard(own_area->getID());
            own_area->setNumGrid(own_area->getNumGrid() - 1);
            own_area->getAreaBitboard().reset(grid.getPosition());
            own_area->getNeighborBlockIDBitboard().set(grid.getBlock()->getID());
        }
    } else {
        if (own_area) { removeArea(own_area); }
        for (const auto& area_bitboard : areas_bitboard) { addArea(own_player, area_bitboard); }
    }
}

void GoEnv::addArea(Player player, const GoBitboard& area_bitboard)
{
    assert(!free_area_id_bitboard_.none());

    // get available area id
    int area_id = free_area_id_bitboard_._Find_first();
    free_area_id_bitboard_.reset(area_id);

    GoArea* area = &areas_[area_id];
    area->setNumGrid(area_bitboard.count());
    area->setPlayer(player);
    area->setAreaBitBoard(area_bitboard);

    // link grids pointer
    GoBitboard grid_bitboard = area_bitboard;
    while (!grid_bitboard.none()) {
        int pos = grid_bitboard._Find_first();
        grid_bitboard.reset(pos);
        grids_[pos].setArea(player, area);
    }

    // link blocks pointer
    GoBitboard neighbor_block_bitboard = dilateBitboard(area_bitboard) & stone_bitboard_.get(player);
    while (!neighbor_block_bitboard.none()) {
        int pos = neighbor_block_bitboard._Find_first();
        GoBlock* block = grids_[pos].getBlock();
        block->addNeighborAreaIDBitboard(area->getID());
        area->addNeighborBlockIDBitboard(block->getID());
        neighbor_block_bitboard &= ~block->getGridBitboard();
    }
}

void GoEnv::removeArea(GoArea* area)
{
    assert(area && !free_area_id_bitboard_.test(area->getID()));

    // remove grids pointer
    GoBitboard area_bitboard = area->getAreaBitboard();
    while (!area_bitboard.none()) {
        int pos = area_bitboard._Find_first();
        area_bitboard.reset(pos);
        grids_[pos].setArea(area->getPlayer(), nullptr);
    }

    // remove blocks pointer
    GoBitboard neighbor_block_id = area->getNeighborBlockIDBitboard();
    while (!neighbor_block_id.none()) {
        int block_id = neighbor_block_id._Find_first();
        neighbor_block_id.reset(block_id);
        blocks_[block_id].removeNeighborAreaIDBitboard(area->getID());
    }

    // remove area
    free_area_id_bitboard_.set(area->getID());
    area->reset();
}

GoArea* GoEnv::mergeArea(GoArea* area1, GoArea* area2)
{
    assert(area1 && area2);
    if (area1->getNumGrid() < area2->getNumGrid()) { return mergeArea(area2, area1); }

    GoBitboard area2_bitboard = area2->getAreaBitboard(); // save area2 bitboard before removing area2
    GoBitboard area2_nbr_block_id = area2->getNeighborBlockIDBitboard();
    area1->combineWithArea(area2);
    removeArea(area2);
    while (!area2_bitboard.none()) { // link grid to area
        int pos = area2_bitboard._Find_first();
        area2_bitboard.reset(pos);
        grids_[pos].setArea(area1->getPlayer(), area1);
    }
    while (!area2_nbr_block_id.none()) { // link block to area
        int id = area2_nbr_block_id._Find_first();
        area2_nbr_block_id.reset(id);
        blocks_[id].addNeighborAreaIDBitboard(area1->getID());
    }
    return area1;
}

std::vector<GoBitboard> GoEnv::findAreas(const GoAction& action)
{
    const GoGrid& grid = grids_[action.getActionID()];
    const std::vector<int>& neighbors = grid.getNeighbors();
    std::vector<GoBitboard> areas;
    GoBitboard checked_area;
    GoBitboard boundary_bitboard = ~stone_bitboard_.get(action.getPlayer()) & board_mask_bitboard_;
    for (const auto& pos : neighbors) {
        const GoGrid& neighbor_grid = grids_[pos];
        if (neighbor_grid.getPlayer() == action.getPlayer()) { continue; }
        if (checked_area.test(pos)) { continue; }
        GoBitboard area_bitboard = floodFillBitBoard(pos, boundary_bitboard);
        checked_area |= area_bitboard;
        areas.push_back(area_bitboard);
    }
    return areas;
}

void GoEnv::updateBenson(const GoAction& action)
{
    if (isPassAction(action)) { return; }
    assert(grids_[action.getActionID()].getPlayer() != Player::kPlayerNone);

    const GoGrid& grid = grids_[action.getActionID()];
    const GoBlock* block = grid.getBlock();

    // update own benson
    GoBitboard& own_benson_bitboard = benson_bitboard_.get(action.getPlayer());
    if (own_benson_bitboard.test(action.getActionID()) || block->getNeighborAreaIDBitboard().count() > 1) {
        own_benson_bitboard = findBensonBitboard(stone_bitboard_.get(block->getPlayer()));
    }

    // update opponent benson
    Player next_player = action.nextPlayer();
    const GoArea* opponent_area = grid.getArea(next_player);
    if (opponent_area && !benson_bitboard_.get(next_player).test(action.getActionID()) &&
        (opponent_area->getAreaBitboard() & ~dilateBitboard(stone_bitboard_.get(next_player)) & ~stone_bitboard_.get(action.getPlayer())).none()) {
        benson_bitboard_.get(next_player) |= findBensonBitboard(stone_bitboard_.get(next_player));
    }
}

GoBitboard GoEnv::findBensonBitboard(GoBitboard block_bitboard) const
{
    // construct vital areas for each block
    GoBitboard benson_area_id, benson_block_id;
    std::vector<GoBitboard> block_neighbor_vital_areas(board_size_ * board_size_, GoBitboard());
    GoBitboard stone_bitboard = stone_bitboard_.get(Player::kPlayer1) | stone_bitboard_.get(Player::kPlayer2);
    while (!block_bitboard.none()) {
        int pos = block_bitboard._Find_first();
        const GoBlock* block = grids_[pos].getBlock();
        block_bitboard &= ~block->getGridBitboard();

        GoBitboard block_neighbor_area_id = block->getNeighborAreaIDBitboard();
        while (!block_neighbor_area_id.none()) {
            int area_id = block_neighbor_area_id._Find_first();
            block_neighbor_area_id.reset(area_id);

            const GoArea* area = &areas_[area_id];
            if (!(area->getAreaBitboard() & ~block->getLibertyBitboard() & ~stone_bitboard).none()) { continue; }
            block_neighbor_vital_areas[block->getID()].set(area_id);
            benson_block_id.set(block->getID());
            benson_area_id.set(area_id);
        }
    }

    // Benson's algorithm
    bool is_over = false;
    GoBitboard benson_bitboard;
    while (!is_over && !benson_block_id.none()) {
        is_over = true;
        benson_bitboard.reset();

        // 1. Remove from X all Black chains with less than two vital Black-enclosed regions in R
        GoBitboard next_benson_block_id;
        while (!benson_block_id.none()) {
            int block_id = benson_block_id._Find_first();
            benson_block_id.reset(block_id);

            if ((block_neighbor_vital_areas[block_id] & benson_area_id).count() < 2) {
                is_over = false;
                continue;
            }
            next_benson_block_id.set(block_id);
            benson_bitboard |= blocks_[block_id].getGridBitboard();
        }
        benson_block_id = next_benson_block_id;

        // 2. Remove from R all Black - enclosed regions with a surrounding stone in a chain not in X
        GoBitboard next_benson_area_id;
        while (!benson_area_id.none()) {
            int area_id = benson_area_id._Find_first();
            benson_area_id.reset(area_id);

            if (!(areas_[area_id].getNeighborBlockIDBitboard() & ~benson_block_id).none()) {
                is_over = false;
                continue;
            }
            next_benson_area_id.set(area_id);
            benson_bitboard |= areas_[area_id].getAreaBitboard();
        }
        benson_area_id = next_benson_area_id;
    }
    return benson_bitboard;
}

std::string GoEnv::getCoordinateString() const
{
    std::ostringstream oss;
    oss << "  ";
    for (int i = 0; i < board_size_; ++i) {
        char c = 'A' + i + ('A' + i >= 'I' ? 1 : 0);
        oss << " " + std::string(1, c);
    }
    oss << "   ";
    return getColorText(oss.str(), TextType::kBold, TextColor::kBlack, TextColor::kYellow);
}

GoBitboard GoEnv::floodFillBitBoard(int start_position, const GoBitboard& boundary_bitboard) const
{
    GoBitboard flood_fill_bitboard;
    flood_fill_bitboard.set(start_position);
    bool need_dilate = true;
    while (need_dilate) {
        GoBitboard dilate_bitboard = dilateBitboard(flood_fill_bitboard) & boundary_bitboard;
        need_dilate = (flood_fill_bitboard != dilate_bitboard);
        flood_fill_bitboard = dilate_bitboard;
    }
    return flood_fill_bitboard;
}

GamePair<float> GoEnv::calculateTrompTaylorTerritory() const
{
    GamePair<float> territory(stone_bitboard_.get(Player::kPlayer1).count(), stone_bitboard_.get(Player::kPlayer2).count() + komi_);
    GoBitboard empty_stone_bitboard = ~(stone_bitboard_.get(Player::kPlayer1) | stone_bitboard_.get(Player::kPlayer2)) & board_mask_bitboard_;
    while (!empty_stone_bitboard.none()) {
        int pos = empty_stone_bitboard._Find_first();

        // check is surrounded by only one's color
        GoBitboard flood_fill_bitboard = floodFillBitBoard(pos, empty_stone_bitboard);
        GoBitboard surrounding_bitboard = dilateBitboard(flood_fill_bitboard) & ~flood_fill_bitboard;
        if ((surrounding_bitboard & ~stone_bitboard_.get(Player::kPlayer1)).none()) {
            territory.get(Player::kPlayer1) += flood_fill_bitboard.count();
        } else if ((surrounding_bitboard & ~stone_bitboard_.get(Player::kPlayer2)).none()) {
            territory.get(Player::kPlayer2) += flood_fill_bitboard.count();
        }

        empty_stone_bitboard &= ~flood_fill_bitboard;
    }

    return territory;
}

std::vector<float> GoEnvLoader::getActionFeatures(const int pos, utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const
{
    const GoAction& action = action_pairs_[pos].first;
    std::vector<float> action_features(getBoardSize() * getBoardSize(), 0.0f);
    if (pos < static_cast<int>(action_pairs_.size())) {
        if (!isPassAction(action)) { action_features[getRotateAction(action.getActionID(), rotation)] = 1.0f; }
    } else {
        int action_id = utils::Random::randInt() % (action_features.size() + 1);
        if (action_id < static_cast<int>(action_pairs_.size())) { action_features[action_id] = 1.0f; }
    }
    return action_features;
}

// ========== GoEnv Phantom Go Helper Functions ==========

int GoEnv::countStones(Player p) const
{
    const int N = getBoardSize();
    int cnt = 0;
    for (int i = 0; i < N * N; ++i) {
        if (getGrid(i).getPlayer() == p) ++cnt;
    }
    return cnt;
}

GoEnv::MoveInfo GoEnv::analyzeMove(const GoEnv& before, const GoEnv& after, const GoAction& action)
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

// ========== GoEnvLoader Phantom Go Helper Functions ==========

GoEnv GoEnvLoader::rebuildToStep(int target_pos) const
{
    const int board_size = getBoardSize();
    GoEnv env(board_size);

    const auto& action_pairs = getActionPairs();
    int end_pos = std::min(target_pos, static_cast<int>(action_pairs.size()));

    for (int i = 0; i < end_pos; ++i) {
        if (!env.act(action_pairs[i].first)) {
            std::cerr << "Error: Failed to apply action at step " << i << std::endl;
            break;
        }
    }

    return env;
}

std::vector<GoEnv> GoEnvLoader::rebuildFullHistory(int target_pos) const
{
    const int board_size = getBoardSize();
    const auto& action_pairs = getActionPairs();

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

std::vector<GoEnvLoader::MoveEvent> GoEnvLoader::buildMoveEvents(const std::vector<GoEnv>& history) const
{
    const auto& action_pairs = getActionPairs();
    std::vector<MoveEvent> events;
    events.reserve(action_pairs.size());

    for (size_t k = 0; k < action_pairs.size() && k + 1 < history.size(); ++k) {
        const GoEnv& before = history[k];
        const GoEnv& after = history[k + 1];
        const GoAction& action = action_pairs[k].first;

        GoEnv::MoveInfo info = GoEnv::analyzeMove(before, after, action);
        MoveEvent ev{action.getPlayer(), action.getActionID(), std::move(info.captured_stones)};
        events.push_back(std::move(ev));
    }

    return events;
}

std::pair<std::unordered_set<int>, std::unordered_set<int>> GoEnvLoader::computeMustSets(
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

std::unordered_set<int> GoEnvLoader::getUnmovableOpponentPositions(int pos) const
{
    // Rebuild history up to the given position
    std::vector<GoEnv> history = rebuildFullHistory(pos);
    std::vector<MoveEvent> events = buildMoveEvents(history);

    // Determine perspective (the player who is about to move)
    const auto& action_pairs = getActionPairs();
    Player my_perspective = (pos > 0 && pos <= static_cast<int>(action_pairs.size()))
                                ? action_pairs[pos - 1].first.nextPlayer()
                                : Player::kPlayer1;

    // Compute must sets
    auto [must_black, must_white] = computeMustSets(pos, my_perspective, events, history);

    // Return opponent's must set (stones that cannot be moved)
    Player opponent = (my_perspective == Player::kPlayer1) ? Player::kPlayer2 : Player::kPlayer1;
    return (opponent == Player::kPlayer1) ? must_black : must_white;
}

// ========== GoEnvLoader Siamese Learning Methods ==========

std::vector<float> GoEnvLoader::getAnchor(int target_pos, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    ++target_pos; // consistent with negative and positive
    const int N = getBoardSize();
    const int H = 12; // history length
    const int C = 6;  // channels per timestep
    const int PASS = N * N;

    std::vector<float> anchor;
    anchor.reserve(H * C * N * N);

    std::vector<GoEnv> history = rebuildFullHistory(target_pos);
    const auto& action_pairs = getActionPairs();
    std::vector<MoveEvent> events = buildMoveEvents(history);

    Player my_perspective = (target_pos > 0 && target_pos <= static_cast<int>(action_pairs.size()))
                                ? action_pairs[target_pos - 1].first.nextPlayer()
                                : Player::kPlayer1;

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

        // Channel 1: illegal_attempts (all 0s temporarily)
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

std::vector<float> GoEnvLoader::getPositive(int pos, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    GoEnv env;
    const auto& action_pairs = getActionPairs();
    for (int i = 0; i <= pos; ++i) { env.act(action_pairs[i].first); }
    return env.getSiameseFeatures(rotation);
}

std::vector<float> GoEnvLoader::getNegative(int pos, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/, int index /* = -1*/) const
{
    if (config::iig_sampling_strategy == "random_move_piece" || config::iig_sampling_strategy == "filter_by_value") {
        GoEnv env;
        const auto& action_pairs = getActionPairs();
        for (int i = 0; i < pos; ++i) { env.act(action_pairs[i].first); }

        // TODO: neg_bitboards can be empty? how to handle?
        auto neg_bitboards = generateNegativeBitboards(env, (index > 0 ? index : (utils::Random::randInt() % config::iig_max_infoset_size)), false);
        if (neg_bitboards.empty()) {
            return {};
        }
        return bitboardToFeature(neg_bitboards[0], env.getTurn(), rotation, false);
    } else if (config::iig_sampling_strategy == "move_by_policy") {
        int num_negatives = std::stoi(getActionPairs()[pos].second["N"]);
        int negative_id = (num_negatives == 0
                               ? 0
                               : (index == -1 ? Random::randInt() % std::stoi(getActionPairs()[pos].second["N"]) : index));
        GoEnv env;
        for (const auto& a : getNegativeActionHistory(pos, negative_id)) { env.act(a); }
        return env.getSiameseFeatures(rotation);
    } else {
        return {};
    }
}

std::vector<GoAction> GoEnvLoader::getNegativeActionHistory(int pos, int negative_id) const
{
    int id = negative_id;
    bool follow_true_board = false;
    env::Player turn = getActionPairs()[pos].first.getPlayer();
    std::deque<std::string> actions_str;
    for (int i = pos; i >= 0; --i) {
        if (getActionPairs()[i].first.getPlayer() == turn && std::stoi(getActionPairs()[i].second["N"]) == 0) { follow_true_board = true; }
        if (getActionPairs()[i].first.getPlayer() == turn && !follow_true_board) {
            const std::string& action_str = getActionPairs()[i].second["A"];
            std::vector<std::string> negatives_str = utils::stringToVector(action_str, ";", false);
            int index = 0;
            for (size_t j = 0; j < negatives_str.size(); ++j) {
                if (index + static_cast<int>(negatives_str[j].size()) / 2 > id) {
                    actions_str.push_front(negatives_str[j].substr(2 * (id - index), 2));
                    id = j;
                    break;
                }
                index += static_cast<int>(negatives_str[j].size()) / 2;
            }
        } else {
            actions_str.push_front(utils::SGFLoader::actionIDToSGFString(getActionPairs()[i].first.getActionID(), getBoardSize()));
        }
    }
    std::vector<GoAction> actions;
    for (int i = 0; i <= pos; ++i) { actions.push_back(GoAction(utils::SGFLoader::sgfStringToActionID(actions_str[i], getBoardSize()), getActionPairs()[i].first.getPlayer())); }

    return actions;
}

std::vector<env::GamePair<env::go::GoBitboard>> GoEnvLoader::generateNegativeBitboards(const GoEnv& env, int index, bool save_all /* = false*/) const
{
    // We now skip checking legality by GoEnv after moving stone (because it is only 0.032%)
    std::vector<env::GamePair<env::go::GoBitboard>> outputs;
    env::GamePair<env::go::GoBitboard> stone_bitboard = env.getStoneBitboard();
    Player opponent = env::getNextPlayer(env.getTurn(), 2);
    env::go::GoBitboard opp_bitboard = stone_bitboard.get(opponent);

    // [Optional] Get unmovable opponent positions based on capture history (time-consuming, comment out if not needed)
    int move_pos = static_cast<int>(env.getActionHistory().size());
    std::unordered_set<int> unmovable_positions = getUnmovableOpponentPositions(move_pos);
    // std::unordered_set<int> unmovable_positions; // empty set if disabled

    // collect opponent stone positions (excluding unmovable positions)
    std::vector<int> pos_list;
    while (!opp_bitboard.none()) {
        int pos = opp_bitboard._Find_first();
        opp_bitboard.reset(pos);
        // Skip unmovable positions (must-exist stones)
        if (!unmovable_positions.empty() && unmovable_positions.count(pos)) { continue; }
        pos_list.push_back(pos);
    }

    if (pos_list.empty()) { return outputs; }

    // Hash set for deduplication using incremental Zobrist hash
    std::unordered_set<GoHashKey> seen_hashes;
    GoHashKey current_hash = env.getHashKey();

    std::mt19937 random_generator;
    random_generator.seed(config::program_seed);
    std::uniform_int_distribution<int> int_distribution(0, pos_list.size() - 1);
    const int warmup_times = 100;
    const int distance = config::iig_max_move_distance;
    for (int k = 0; k < index + warmup_times; k++) {
        int idx = int_distribution(random_generator);
        int pos = pos_list[idx];
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
            pos_list[idx] = new_pos;
            stone_bitboard.get(opponent).reset(pos);
            stone_bitboard.get(opponent).set(new_pos);

            // Incremental hash update: XOR out old position, XOR in new position
            current_hash ^= getGoGridHashKey(pos, opponent);
            current_hash ^= getGoGridHashKey(new_pos, opponent);

            if (k < warmup_times) { break; }

            // Check for duplicate using Zobrist hash
            if (save_all && seen_hashes.count(current_hash)) {
                break; // Skip duplicate
            }
            seen_hashes.insert(current_hash);

            if (save_all || outputs.empty()) {
                outputs.emplace_back(stone_bitboard);
            } else {
                outputs[0] = stone_bitboard;
            }
            break;
        }
    }
    return outputs;
}

std::vector<float> GoEnvLoader::bitboardToFeature(const GamePair<GoBitboard>& bitboard, Player turn, utils::Rotation rotation, bool include_history) const
{
    int num_channels = (include_history ? 18 : 4);
    const int num_grids = getBoardSize() * getBoardSize();
    std::vector<float> feature(num_channels * num_grids, 0.0f);

    std::vector<env::Player> players = {env::Player::kPlayer1, env::Player::kPlayer2};
    for (const auto& player : players) {
        GoBitboard tmp = bitboard.get(player);
        while (!tmp.none()) {
            int pos = tmp._Find_first();
            if (pos < 0 || pos >= num_grids) { break; }
            tmp.reset(pos);
            feature[getRotatePosition(pos, rotation) + (player == turn ? 0 : num_grids)] = 1.0f;
        }
    }

    if (include_history) {
        for (int i = 2 * num_grids; i < 16 * num_grids; ++i) { feature[i] = feature[i % (2 * num_grids)]; }
        std::fill_n(feature.begin() + 16 * num_grids, num_grids, turn == Player::kPlayer1 ? 1.0f : 0.0f);
        std::fill_n(feature.begin() + 17 * num_grids, num_grids, turn == Player::kPlayer2 ? 1.0f : 0.0f);
    } else {
        std::fill_n(feature.begin() + 2 * num_grids, num_grids, turn == Player::kPlayer1 ? 1.0f : 0.0f);
        std::fill_n(feature.begin() + 3 * num_grids, num_grids, turn == Player::kPlayer2 ? 1.0f : 0.0f);
    }
    return feature;
}

} // namespace minizero::env::go

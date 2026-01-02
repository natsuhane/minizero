#include "go.h"
#include "color_message.h"
#include "configuration.h"
#include "random.h"
#include "rotation.h"
#include "sgf_loader.h"
#include <algorithm>
#include <atomic>
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
                if (neighbor_block->getNumLiberty() == 0) { removeBlockFromBoard(neighbor_block); }
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

GoBitboard GoEnv::dilateBitboard(const GoBitboard& bitboard) const
{
    return ((bitboard << board_size_) |                           // move up
            (bitboard >> board_size_) |                           // move down
            ((bitboard & ~board_left_boundary_bitboard_) >> 1) |  // move left
            ((bitboard & ~board_right_boundary_bitboard_) << 1) | // move right
            bitboard) &
           board_mask_bitboard_;
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

// ========== Phantom Go Helper Functions ==========

GoEnv rebuildGoEnvToStep(const GoEnvLoader& env_loader, int target_pos)
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

std::vector<GoEnv> rebuildFullHistory(const GoEnvLoader& env_loader, int target_pos)
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
    const GoEnvLoader& env_loader)
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

// ===== Internal Helper Functions for Move-Stone Sampling =====

namespace {

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

} // anonymous namespace

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
    int max_total_attempts)
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

// ========== Feature Extraction Functions ==========

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

    // Channel 2: Black's turn ?
    const float black_turn = (env.getTurn() == Player::kPlayer1) ? 1.0f : 0.0f;
    board_state.insert(board_state.end(), N * N, black_turn);

    // Channel 3: White's turn ?
    const float white_turn = (env.getTurn() == Player::kPlayer2) ? 1.0f : 0.0f;
    board_state.insert(board_state.end(), N * N, white_turn);

    return board_state; // 4 * N * N floats
}

// ========== GoEnvLoader Siamese Learning Methods ==========

std::vector<float> GoEnvLoader::getAnchor(int target_pos, utils::Rotation rotation) const
{
    const int N = getBoardSize();
    const int H = 12; // history length
    const int C = 6;  // channels per timestep
    const int PASS = N * N;

    std::vector<float> anchor;
    anchor.reserve(H * C * N * N);

    std::vector<GoEnv> history = rebuildFullHistory(*this, target_pos);
    const auto& action_pairs = getActionPairs();
    std::vector<MoveEvent> events = buildMoveEvents(history, *this);

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

std::vector<float> GoEnvLoader::getPositive(int pos, utils::Rotation rotation) const
{
    GoEnv env = rebuildGoEnvToStep(*this, pos);
    return extractBoardState(env, rotation);
}

std::vector<float> GoEnvLoader::getNegative(int pos, utils::Rotation rotation, int index /* = -1*/) const
{
    GoEnv env;
    const auto& action_pairs = getActionPairs();
    for (int i = 0; i < pos; ++i) { env.act(action_pairs[i].first); }

    auto neg_bitboards = generateNegativeBitboards(env, (index > 0 ? index : (utils::Random::randInt() % config::siamese_max_random_perturbations)), false);
    if (neg_bitboards.empty()) {
        return {};
    }
    return bitboardToFeature(neg_bitboards[0], env.getTurn(), rotation, false);
}

std::vector<env::GamePair<env::go::GoBitboard>> GoEnvLoader::generateNegativeBitboards(const GoEnv& env, int index, bool save_all /* = false*/) const
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

    if (pos_list.empty()) { return outputs; }

    std::mt19937 random_generator;
    random_generator.seed(config::program_seed);
    std::uniform_int_distribution<int> int_distribution(0, pos_list.size() - 1);
    const int warmup_times = 100;
    const int distance = config::siamese_max_move_distance;
    for (int k = 0; k < index + warmup_times; k++) {
        int index = int_distribution(random_generator);
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

            if (k < warmup_times) { break; }
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

    GoBitboard tmp = bitboard.get(Player::kPlayer1);
    while (!tmp.none()) {
        int pos = tmp._Find_first();
        tmp.reset(pos);
        feature[getRotatePosition(pos, rotation)] = 1.0f;
    }
    tmp = bitboard.get(Player::kPlayer2);
    while (!tmp.none()) {
        int pos = tmp._Find_first();
        tmp.reset(pos);
        feature[getRotatePosition(pos, rotation) + num_grids] = 1.0f;
    }

    if (include_history) {
        for (int i = 2 * num_grids; i < 16 * num_grids; ++i) { feature[i] = feature[i % (2 * num_grids)]; }
        std::fill_n(feature.begin() + 16 * 81, 81, turn == Player::kPlayer1 ? 1.0f : 0.0f);
        std::fill_n(feature.begin() + 17 * 81, 81, turn == Player::kPlayer2 ? 1.0f : 0.0f);
    } else {
        std::fill_n(feature.begin() + 2 * 81, 81, turn == Player::kPlayer1 ? 1.0f : 0.0f);
        std::fill_n(feature.begin() + 3 * 81, 81, turn == Player::kPlayer2 ? 1.0f : 0.0f);
    }
    return feature;
}

} // namespace minizero::env::go

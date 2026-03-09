#pragma once

#include "hex.h"
#include "imperfect_information_env.h"
#include "random.h"
#include <string>
#include <utility>
#include <vector>

namespace minizero::env::darkhex {

const std::string kDarkHexName = "darkhex";
const int kDarkHexNumPlayer = 2;
const int kDarkHexMaxBoardSize = 19;

typedef hex::HexAction DarkHexAction;

using Cell = minizero::env::hex::Cell;
using Flag = minizero::env::hex::Flag;

class ImperfectHexEnv : public hex::HexEnv {
public:
    ImperfectHexEnv(Player view_player, int board_size = config::env_board_size)
        : hex::HexEnv(), view_player_(view_player)
    {
        assert(getBoardSize() <= kDarkHexMaxBoardSize);
        reset();
    }

    void reset() override;
    bool act(const DarkHexAction& action) override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    int getNumStones() const { return num_stones_; }

    inline int getNumInputChannels() const override { return 4; }

private:
    Player view_player_;
    int num_stones_;
};

class DarkHexEnv : public ImperfectInformationEnv<DarkHexAction, hex::HexEnv, ImperfectHexEnv> {
public:
    DarkHexEnv(int board_size = config::env_board_size)
        : ImperfectInformationEnv<DarkHexAction, hex::HexEnv, ImperfectHexEnv>(
              hex::HexEnv(), GamePair<ImperfectHexEnv>(ImperfectHexEnv(Player::kPlayer1, board_size),
                                                       ImperfectHexEnv(Player::kPlayer2, board_size)))
    {
        assert(perfect_env_.getBoardSize() <= kDarkHexMaxBoardSize);
    }

    bool act(const DarkHexAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override { return act(DarkHexAction(action_string_args, getBoardSize())); }
    void sampleOneInformationSet(int seed = utils::Random::randInt()) override;
    std::vector<float> getPlayerFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getDiscriminatorFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;

    int getNumPlayerInputChannels() const override { return 6; }
    int getNumDiscriminatorInputChannels() const override { return imperfect_env_.get(getTurn()).getNumInputChannels(); }
    inline std::string name() const override { return kDarkHexName + "_" + std::to_string(getBoardSize()) + "x" + std::to_string(getBoardSize()); }
    inline int getNumPlayer() const override { return kDarkHexNumPlayer; }
    inline int getBoardSize() const { return perfect_env_.getBoardSize(); }
};

class DarkHexEnvLoader : public ImperfectInformationEnvLoader<DarkHexAction, DarkHexEnv> {
public:
    DarkHexEnvLoader()
        : ImperfectInformationEnvLoader<DarkHexAction, DarkHexEnv>(),
          board_size_(minizero::config::env_board_size) {}

    void loadFromEnvironment(const DarkHexEnv& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override
    {
        ImperfectInformationEnvLoader<DarkHexAction, DarkHexEnv>::loadFromEnvironment(env, action_info_history);
        addTag("SZ", std::to_string(env.getBoardSize()));
    }

    inline void addTag(const std::string& key, const std::string& value) override
    {
        ImperfectInformationEnvLoader<DarkHexAction, DarkHexEnv>::addTag(key, value);
        if (key == "SZ") { board_size_ = std::stoi(value); }
    }

    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override { return {}; }
    std::pair<std::vector<float>, std::vector<float>> getISGeneratorFeaturesAndLabel(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;

    inline std::vector<float> getValue(const int pos) const { return {getReturn()}; }
    inline std::string name() const override { return kDarkHexName; }
    inline int getPolicySize() const override { return getBoardSize() * getBoardSize(); }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
    inline int getBoardSize() const { return board_size_; }

private:
    int board_size_;
};

} // namespace minizero::env::darkhex

#pragma once

#include "go.h"
#include "imperfect_information_env.h"
#include "random.h"
#include <string>
#include <utility>
#include <vector>

namespace minizero::env::phantomgo {

const std::string kPhantomGoName = "phantomgo";
const int kPhantomGoNumPlayer = 2;
const int kPhantomGoMaxBoardSize = 19;

typedef go::GoAction PhantomGoAction;

inline void initialize() { go::initialize(); }
std::string bitboardToBoardString(const go::GoBitboard& bitboard, int board_size);
std::string featuresToBoardString(const std::vector<float>& features, int board_size, int num_columns, const std::vector<std::string> feature_headers = {});

class ImperfectGoEnv : public go::GoEnv {
public:
    ImperfectGoEnv(Player view_player, int board_size = config::env_board_size)
        : go::GoEnv(board_size), view_player_(view_player)
    {
        assert(getBoardSize() <= kPhantomGoMaxBoardSize);
        reset();
    }

    ImperfectGoEnv& operator=(const ImperfectGoEnv& env);

    void reset() override;
    bool act(const PhantomGoAction& action) override;
    bool isLegalAction(const PhantomGoAction& action) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<PhantomGoAction> sampledPerfectEnvActionHistory(int seed = utils::Random::randInt()) const;
    void update(go::GoBitboard captured_stone, Player captured_player);
    std::string infoString() const;

    inline int getNumInputChannels() const override { return 34; }
    inline int getNumOpponentStones() const { return num_opp_stones_; }
    inline const go::GoBitboard& getTriedPos() const { return tried_pos_; }
    inline const std::vector<go::GoBitboard>& getTriedPosHistory() const { return tried_pos_history_; }
    inline const std::vector<go::GoBitboard>& getCaptureOppHistory() const { return capture_opp_history_; }
    inline std::vector<GamePair<go::GoBitboard>>& getKnownStoneBitboardHistory() { return known_stone_bitboard_history_; }
    inline const std::vector<GamePair<go::GoBitboard>>& getKnownStoneBitboardHistory() const { return known_stone_bitboard_history_; }

    void updateBoard(go::GoBitboard captured_stone, Player captured_player);
    void updateTriedPos(const PhantomGoAction& action) { tried_pos_.set(action.getActionID()); }
    void updateKnownStone(const PhantomGoAction& action) { go::GoEnv::act(action); }

private:
    void replayGoEnv(const std::vector<PhantomGoAction>& actions);

    Player view_player_;
    int num_opp_stones_;
    go::GoBitboard tried_pos_;
    std::vector<go::GoBitboard> tried_pos_history_;
    std::vector<go::GoBitboard> capture_opp_history_;
    std::vector<GamePair<go::GoBitboard>> known_stone_bitboard_history_;
};

class PhantomGoEnv : public ImperfectInformationEnv<PhantomGoAction, go::GoEnv, ImperfectGoEnv> {
public:
    PhantomGoEnv(int board_size = config::env_board_size)
        : ImperfectInformationEnv<PhantomGoAction, go::GoEnv, ImperfectGoEnv>(
              go::GoEnv(board_size), GamePair<ImperfectGoEnv>(ImperfectGoEnv(Player::kPlayer1, board_size),
                                                              ImperfectGoEnv(Player::kPlayer2, board_size)))
    {
        assert(perfect_env_.getBoardSize() <= kPhantomGoMaxBoardSize);
    }

    bool act(const PhantomGoAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override { return act(PhantomGoAction(action_string_args, getBoardSize())); }
    void sampleOneInformationSet(int seed = utils::Random::randInt()) override;
    std::string toString() const override;
    std::string infoString() const;

    inline std::string name() const override { return kPhantomGoName + "_" + std::to_string(getBoardSize()) + "x" + std::to_string(getBoardSize()); }
    inline int getNumPlayer() const override { return kPhantomGoNumPlayer; }
    inline int getBoardSize() const { return perfect_env_.getBoardSize(); }
    inline float getKomi() const { return perfect_env_.getKomi(); }

private:
    PhantomGoEnv createEnvByActions(const std::vector<PhantomGoAction>& actions) const;
};

class PhantomGoEnvLoader : public ImperfectInformationEnvLoader<PhantomGoAction, PhantomGoEnv> {
public:
    PhantomGoEnvLoader()
        : ImperfectInformationEnvLoader<PhantomGoAction, PhantomGoEnv>(),
          board_size_(minizero::config::env_board_size) {}

    void loadFromEnvironment(const PhantomGoEnv& env, const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override
    {
        ImperfectInformationEnvLoader<PhantomGoAction, PhantomGoEnv>::loadFromEnvironment(env, action_info_history);
        addTag("SZ", std::to_string(env.getBoardSize()));
        addTag("KM", std::to_string(env.getKomi()));
    }

    inline void addTag(const std::string& key, const std::string& value) override
    {
        ImperfectInformationEnvLoader<PhantomGoAction, PhantomGoEnv>::addTag(key, value);
        if (key == "SZ") { board_size_ = std::stoi(value); }
    }

    std::pair<std::vector<float>, std::vector<float>> getISGeneratorFeaturesAndLabel(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override { return {}; } // TODO: fix this
    inline std::vector<float> getValue(const int pos) const { return {getReturn()}; }
    inline std::string name() const override { return kPhantomGoName; }
    inline int getPolicySize() const override { return getBoardSize() * getBoardSize() + 1; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return utils::getPositionByRotating(rotation, position, getBoardSize()); }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return getRotatePosition(action_id, rotation); }
    inline int getBoardSize() const { return board_size_; }

private:
    int board_size_;
};

} // namespace minizero::env::phantomgo

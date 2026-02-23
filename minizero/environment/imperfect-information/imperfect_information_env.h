#pragma once

#include "base_env.h"
#include "configuration.h"
#include "random.h"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace minizero::env {

template <class Action, class PerfectEnv, class ImperfectEnv>
class ImperfectInformationEnv : public BaseEnv<Action> {
public:
    ImperfectInformationEnv(PerfectEnv perfect_env, GamePair<ImperfectEnv> imperfect_env)
        : BaseEnv<Action>(),
          perfect_env_(perfect_env),
          imperfect_env_(imperfect_env)
    {
    }
    virtual ~ImperfectInformationEnv() = default;

    void reset() override
    {
        BaseEnv<Action>::actions_.clear();
        perfect_env_.reset();
        imperfect_env_.get(Player::kPlayer1).reset();
        imperfect_env_.get(Player::kPlayer2).reset();
    }

    bool act(const Action& action) override
    {
        if (!isLegalAction(action, true)) {
            if (isLegalAction(action, false)) { BaseEnv<Action>::actions_.push_back(action); }
            return false;
        }
        perfect_env_.act(action);
        imperfect_env_.get(action.getPlayer()).act(action);
        imperfect_env_.get(action.nextPlayer()).setTurn(getTurn());
        BaseEnv<Action>::actions_.push_back(action);
        return true;
    }

    std::vector<Action> getLegalActions() const override { return getLegalActions(true); }
    std::vector<Action> getLegalActions(bool is_perfect) const { return (is_perfect ? perfect_env_.getLegalActions() : imperfect_env_.get(getTurn()).getLegalActions()); }
    bool isLegalAction(const Action& action) const override { return isLegalAction(action, true); }
    bool isLegalAction(const Action& action, bool is_perfect) const { return (is_perfect ? perfect_env_.isLegalAction(action) : imperfect_env_.get(getTurn()).isLegalAction(action)); }
    bool isTerminal() const override { return perfect_env_.isTerminal(); }
    float getReward() const override { return perfect_env_.getReward(); }
    float getEvalScore(bool is_resign = false) const override { return perfect_env_.getEvalScore(is_resign); }
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override { return getFeatures(true, rotation); }
    std::vector<float> getFeatures(bool is_perfect, utils::Rotation rotation = utils::Rotation::kRotationNone) const { return (is_perfect ? perfect_env_.getFeatures(rotation) : imperfect_env_.get(getTurn()).getFeatures(rotation)); }
    std::vector<float> getActionFeatures(const Action& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override { return getActionFeatures(action, true, rotation); }
    std::vector<float> getActionFeatures(const Action& action, bool is_perfect, utils::Rotation rotation = utils::Rotation::kRotationNone) const { return (is_perfect ? perfect_env_.getActionFeatures(action, rotation) : imperfect_env_.get(getTurn()).getActionFeatures(action, rotation)); }
    int getNumInputChannels() const override { return getNumInputChannels(true); }
    int getNumInputChannels(bool is_perfect) const { return (is_perfect ? perfect_env_.getNumInputChannels() : imperfect_env_.get(getTurn()).getNumInputChannels()); }
    int getNumActionFeatureChannels() const override { return getNumActionFeatureChannels(true); }
    int getNumActionFeatureChannels(bool is_perfect) const { return (is_perfect ? perfect_env_.getNumActionFeatureChannels() : imperfect_env_.get(getTurn()).getNumActionFeatureChannels()); }
    int getInputChannelHeight() const override { return perfect_env_.getInputChannelHeight(); }
    int getInputChannelWidth() const override { return perfect_env_.getInputChannelWidth(); }
    int getHiddenChannelHeight() const override { return perfect_env_.getHiddenChannelHeight(); }
    int getHiddenChannelWidth() const override { return perfect_env_.getHiddenChannelWidth(); }
    int getPolicySize() const override { return perfect_env_.getPolicySize(); }
    int getDiscreteValueSize() const override { return perfect_env_.getDiscreteValueSize(); }
    int getRotatePosition(int position, utils::Rotation rotation) const override { return perfect_env_.getRotatePosition(position, rotation); }
    int getRotateAction(int action_id, utils::Rotation rotation) const override { return perfect_env_.getRotateAction(action_id, rotation); }
    std::string toString() const override
    {
        std::vector<std::vector<std::string>> board_str;
        if (config::env_iig_display_perfect_board) { board_str.push_back(utils::stringToVector(perfect_env_.toString(), "\n")); }
        if (config::env_iig_display_imperfect_p1_board) { board_str.push_back(utils::stringToVector(imperfect_env_.get(Player::kPlayer1).toString(), "\n")); }
        if (config::env_iig_display_imperfect_p2_board) { board_str.push_back(utils::stringToVector(imperfect_env_.get(Player::kPlayer2).toString(), "\n")); }

        std::ostringstream oss;
        for (size_t i = 0; i < board_str[0].size(); ++i) {
            for (auto& str : board_str) { oss << str[i] << "    "; }
            oss << std::endl;
        }
        return oss.str();
    }

    void setTurn(Player p)
    {
        perfect_env_.setTurn(p);
        imperfect_env_.get(Player::kPlayer1).setTurn(p);
        imperfect_env_.get(Player::kPlayer2).setTurn(p);
    }

    inline PerfectEnv& getPerfectEnv() { return perfect_env_; }
    inline const PerfectEnv& getPerfectEnv() const { return perfect_env_; }
    inline ImperfectEnv& getImperfectEnv(Player p) { return imperfect_env_.get(p); }
    inline const ImperfectEnv& getImperfectEnv(Player p) const { return imperfect_env_.get(p); }
    inline Player getTurn() const { return perfect_env_.getTurn(); }
    inline const std::vector<std::string>& getObservationHistory() const { return perfect_env_.getObservationHistory(); }

    virtual void sampleOneInformationSet(int seed = utils::Random::randInt()) = 0;

protected:
    PerfectEnv perfect_env_;
    GamePair<ImperfectEnv> imperfect_env_; // currently only support 2-player games
};

template <class Action, class Env>
class ImperfectInformationEnvLoader : public BaseEnvLoader<Action, Env> {
public:
    ImperfectInformationEnvLoader() : BaseEnvLoader<Action, Env>() {}
    virtual ~ImperfectInformationEnvLoader() = default;

    std::vector<float> getFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override { return getFeatures(pos, true, rotation); }
    std::vector<float> getFeatures(const int pos, bool is_perfect, utils::Rotation rotation = utils::Rotation::kRotationNone) const
    {
        // a slow but naive method which simply replays the game again to get features
        Env env;
        const auto& action_pairs_ = BaseEnvLoader<Action, Env>::action_pairs_;
        for (int i = 0; i < std::min(pos, static_cast<int>(action_pairs_.size())); ++i) { env.act(action_pairs_[i].first); }
        return env.getFeatures(is_perfect, rotation);
    }

    virtual std::pair<std::vector<float>, std::vector<float>> getISGeneratorFeaturesAndLabel(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const = 0;
};

} // namespace minizero::env

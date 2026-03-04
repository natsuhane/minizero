#include "zero_actor.h"
#include "random.h"
#include "time_system.h"
#include <algorithm>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minizero::actor {

using namespace minizero;
using namespace network;
using namespace utils;

void MCTSSearchData::clear()
{
    search_info_ = "";
    selected_node_ = nullptr;
    node_path_.clear();
}

void ZeroActor::reset()
{
    BaseActor::reset();
    enable_resign_ = (utils::Random::randReal() < config::zero_disable_resign_ratio ? false : true);
}

void ZeroActor::resetSearch()
{
    BaseActor::resetSearch();
    mcts_search_data_.node_path_.clear();
    getMCTS()->getRootNode()->setAction(Action(-1, env::getPreviousPlayer(env_.getTurn(), env_.getNumPlayer())));
    is_valid_states_.clear();
    nn_evaluated_batch_ids_.clear();
    anchor_embeddings_.clear();
    informative_states_.clear();
}

Action ZeroActor::think(bool with_play /*= false*/, bool display_board /*= false*/)
{
    resetSearch();
    boost::posix_time::ptime start_ptime = utils::TimeSystem::getLocalTime();
    while (!isSearchDone()) {
        step();
        int spent_million_second = (utils::TimeSystem::getLocalTime() - start_ptime).total_milliseconds();
        if (config::actor_mcts_think_time_limit > 0 && spent_million_second >= config::actor_mcts_think_time_limit * 1000) { break; }
    }
    if (!isSearchDone()) { handleSearchDone(); }
    if (with_play) { act(getSearchAction()); }
    if (display_board) { std::cerr << env_.toString() << mcts_search_data_.search_info_ << std::endl; }
    return getSearchAction();
}

void ZeroActor::beforeNNEvaluation()
{
    nn_evaluated_batch_ids_.clear();
    if (getMCTS()->getNumSimulation() == 0 && informative_states_.empty()) {
        if (config::iig_use_discriminator) { beforeDiscriminatorNNEvaluation(); }
        return;
    }
    mcts_search_data_.node_path_ = selection();
    if (alphazero_network_) {
        feature_rotation_ = config::actor_use_random_rotation_features ? static_cast<utils::Rotation>(utils::Random::randInt() % static_cast<int>(utils::Rotation::kRotateSize)) : utils::Rotation::kRotationNone;
        is_valid_states_.clear();
        for (int i = 0; i < pimc_repeat_; ++i) {
            Environment env_transition = informative_states_[i];
            is_valid_states_.push_back(calculateEnvironmentTransition(mcts_search_data_.node_path_, env_transition));
            if (is_valid_states_.back()) {
                nn_evaluated_batch_ids_.push_back(alphazero_network_->pushBack(env_transition.getFeatures(feature_rotation_)));
            } else {
                nn_evaluated_batch_ids_.push_back(-1);
            }
        }
    } else if (muzero_network_) {
        if (getMCTS()->getNumSimulation() == 0) { // initial inference for root node
            nn_evaluation_batch_id_ = muzero_network_->pushBackInitialData(env_.getFeatures());
        } else { // for non-root nodes
            const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
            MCTSNode* leaf_node = node_path.back();
            MCTSNode* parent_node = node_path[node_path.size() - 2];
            assert(parent_node && parent_node->getHiddenStateDataIndex() != -1);
            const std::vector<float>& hidden_state = getMCTS()->getTreeHiddenStateData().getData(parent_node->getHiddenStateDataIndex()).hidden_state_;
            nn_evaluation_batch_id_ = muzero_network_->pushBackRecurrentData(hidden_state, env_.getActionFeatures(leaf_node->getAction()));
        }
    } else {
        assert(false);
    }
}

void ZeroActor::afterNNEvaluation(const std::vector<std::shared_ptr<network::NetworkOutput>>& network_outputs)
{
    if (getMCTS()->getNumSimulation() == 0 && informative_states_.empty()) {
        if (config::iig_use_discriminator) {
            afterDiscriminatorNNEvaluation(network_outputs);
        } else {
            for (int i = 0; i < pimc_repeat_; ++i) {
                informative_states_.push_back(env_);
                informative_states_.back().sampleOneInformationSet(i);
            }
        }
        return;
    }

    const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
    MCTSNode* leaf_node = node_path.back();
    if (alphazero_network_) {
        std::vector<int> counts;
        std::vector<MCTS::ActionCandidate> action_candidates;
        for (int action_id = 0; action_id < env_.getPolicySize(); ++action_id) {
            action_candidates.push_back(MCTS::ActionCandidate(Action(action_id, env_.getTurn()), 0.0f, 0.0f));
            counts.push_back(0);
        }
        float value_sum = 0.0f, value_count = 0.0f;
        for (int i = 0; i < pimc_repeat_; ++i) {
            if (!is_valid_states_[i]) { continue; }
            Environment env_transition = informative_states_[i];
            calculateEnvironmentTransition(node_path, env_transition);
            std::shared_ptr<AlphaZeroNetworkOutput> alphazero_output = std::static_pointer_cast<AlphaZeroNetworkOutput>(network_outputs[nn_evaluated_batch_ids_[i]]);
            if (!env_transition.isTerminal()) { calculatePIMCActionPolicy(leaf_node, env_transition, alphazero_output, feature_rotation_, counts, action_candidates); }
            value_sum += (!env_transition.isTerminal() ? alphazero_output->value_ : env_transition.getEvalScore());
            value_count += 1.0f;
        }
        assert(value_count > 0);
        std::vector<MCTS::ActionCandidate> tmp;
        for (size_t i = 0; i < action_candidates.size(); ++i) {
            if (counts[i] == 0) { continue; }
            action_candidates[i].policy_ /= counts[i];
            action_candidates[i].policy_logit_ = std::log(action_candidates[i].policy_ + 1e-8f); // TODO: is 1e-8f ok? (5d)
            tmp.push_back(action_candidates[i]);
        }
        sort(tmp.begin(), tmp.end(), [](const MCTS::ActionCandidate& a, const MCTS::ActionCandidate& b) { return a.policy_ > b.policy_; });
        action_candidates = tmp;
        if (!action_candidates.empty()) { getMCTS()->expand(leaf_node, action_candidates); }
        getMCTS()->backup(node_path, value_sum / value_count, 0.0f);
    } else if (muzero_network_) {
        std::shared_ptr<MuZeroNetworkOutput> muzero_output = std::static_pointer_cast<MuZeroNetworkOutput>(network_outputs[nn_evaluated_batch_ids_[0]]);
        getMCTS()->expand(leaf_node, calculateMuZeroActionPolicy(leaf_node, muzero_output));
        getMCTS()->backup(node_path, muzero_output->value_, muzero_output->reward_);
        leaf_node->setHiddenStateDataIndex(getMCTS()->getTreeHiddenStateData().store(HiddenStateData(muzero_output->hidden_state_)));
    } else {
        assert(false);
    }
    if (leaf_node == getMCTS()->getRootNode()) { addNoiseToNodeChildren(leaf_node); }
    if (isSearchDone()) { handleSearchDone(); }
    if (config::actor_use_gumbel) { gumbel_zero_.sequentialHalving(getMCTS()); }
}

void ZeroActor::setNetwork(const std::shared_ptr<network::Network>& network)
{
    assert(network);
    if (network->getNetworkTypeName() == "alphazero") {
        alphazero_network_ = std::static_pointer_cast<AlphaZeroNetwork>(network);
    } else if (network->getNetworkTypeName() == "muzero" || network->getNetworkTypeName() == "muzero_atari") {
        muzero_network_ = std::static_pointer_cast<MuZeroNetwork>(network);
    } else if (network->getNetworkTypeName() == "discriminator") {
        discriminator_network_ = std::static_pointer_cast<DiscriminatorNetwork>(network);
    } else if (network->getNetworkTypeName() == "siamese") {
        siamese_network_ = std::static_pointer_cast<SiameseNetwork>(network);
    } else {
        assert(false);
    }
}

void ZeroActor::beforeDiscriminatorNNEvaluation()
{
    if (anchor_embeddings_.empty()) {
        nn_evaluated_batch_ids_.push_back(siamese_network_->pushBackAnchor(env_.getDiscriminatorFeatures()));
    } else {
        for (int i = 0; i < config::iig_max_infoset_size; ++i) {
            Environment env = env_;
            env.sampleOneInformationSet(i);
            nn_evaluated_batch_ids_.push_back(siamese_network_->pushBackBoard(env.getPlayerFeatures()));
        }
    }
}

void ZeroActor::afterDiscriminatorNNEvaluation(const std::vector<std::shared_ptr<network::NetworkOutput>>& network_outputs)
{
    if (nn_evaluated_batch_ids_.empty()) { return; }
    if (anchor_embeddings_.empty()) {
        anchor_embeddings_ = std::static_pointer_cast<SiameseNetworkOutput>(network_outputs[nn_evaluated_batch_ids_[0]])->embeddings_;
    } else {
        std::vector<std::pair<int, float>> distances;
        for (size_t i = 0; i < nn_evaluated_batch_ids_.size(); ++i) {
            auto board_embeddings = std::static_pointer_cast<SiameseNetworkOutput>(network_outputs[nn_evaluated_batch_ids_[i]])->embeddings_;
            float dist = utils::distance(anchor_embeddings_, board_embeddings);
            distances.push_back(std::make_pair(i, dist));
        }
        std::sort(distances.begin(), distances.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second < b.second; });
        for (int i = 0; i < pimc_repeat_ && i < static_cast<int>(distances.size()); ++i) {
            informative_states_.push_back(env_);
            informative_states_.back().sampleOneInformationSet(distances[i].first);
        }
    }
}

std::vector<std::pair<std::string, std::string>> ZeroActor::getActionInfo() const
{
    // ignore recording mcts action info if there is no search
    if (getMCTS()->getRootNode()->getCount() > 0) { return BaseActor::getActionInfo(); }
    return {};
}

std::string ZeroActor::getEnvReward() const
{
    std::ostringstream oss;
    oss << env_.getReward();
    return oss.str();
}

void ZeroActor::step()
{
    assert(alphazero_network_ || muzero_network_ || siamese_network_ || discriminator_network_);
    int num_simulation = getMCTS()->getNumSimulation();

    beforeNNEvaluation();
    auto network_output = siamese_network_ && siamese_network_->getBatchSize() > 0
                              ? siamese_network_->forward()
                              : (alphazero_network_ ? alphazero_network_->forward()
                                                    : (num_simulation == 0 ? muzero_network_->initialInference() : muzero_network_->recurrentInference()));
    afterNNEvaluation(network_output);
}

void ZeroActor::handleSearchDone()
{
    mcts_search_data_.selected_node_ = decideActionNode();
    const Action action = getSearchAction();
    std::ostringstream oss;
    oss << "model file name: " << getModelFileName() << std::endl
        << utils::TimeSystem::getTimeString("[Y/m/d H:i:s.f] ")
        << "move number: " << env_.getActionHistory().size()
        << ", action: " << action.toConsoleString()
        << " (" << action.getActionID() << ")"
        << ", reward: " << env_.getReward()
        << ", player: " << env::playerToChar(action.getPlayer());
    if (config::actor_mcts_value_rescale) { oss << ", value bound: (" << getMCTS()->getTreeValueBound().begin()->first << ", " << getMCTS()->getTreeValueBound().rbegin()->first << ")"; }
    oss << std::endl
        << "  root node info: " << getMCTS()->getRootNode()->toString() << std::endl
        << "action node info: " << mcts_search_data_.selected_node_->toString() << std::endl;
    mcts_search_data_.search_info_ = oss.str();
}

MCTSNode* ZeroActor::decideActionNode()
{
    if (config::actor_use_gumbel) {
        return gumbel_zero_.decideActionNode(getMCTS());
    } else {
        if (config::actor_select_action_by_count) {
            return getMCTS()->selectChildByMaxCount(getMCTS()->getRootNode());
        } else if (config::actor_select_action_by_softmax_count) {
            return getMCTS()->selectChildBySoftmaxCount(getMCTS()->getRootNode(), config::actor_select_action_softmax_temperature);
        }

        assert(false);
        return nullptr;
    }
}

void ZeroActor::addNoiseToNodeChildren(MCTSNode* node)
{
    assert(node && node->getNumChildren() > 0);
    if (config::actor_use_dirichlet_noise) {
        const float epsilon = config::actor_dirichlet_noise_epsilon;
        std::vector<float> dirichlet_noise = utils::Random::randDirichlet(config::actor_dirichlet_noise_alpha, node->getNumChildren());
        for (int i = 0; i < node->getNumChildren(); ++i) {
            MCTSNode* child = node->getChild(i);
            child->setPolicyNoise(dirichlet_noise[i]);
            child->setPolicy((1 - epsilon) * child->getPolicy() + epsilon * dirichlet_noise[i]);
        }
    } else if (config::actor_use_gumbel_noise) {
        std::vector<float> gumbel_noise = utils::Random::randGumbel(node->getNumChildren());
        for (int i = 0; i < node->getNumChildren(); ++i) {
            MCTSNode* child = node->getChild(i);
            child->setPolicyNoise(gumbel_noise[i]);
            child->setPolicyLogit(child->getPolicyLogit() + gumbel_noise[i]);
        }
    }
}

std::vector<MCTS::ActionCandidate> ZeroActor::calculateAlphaZeroActionPolicy(const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation)
{
    assert(alphazero_network_);
    std::vector<MCTS::ActionCandidate> action_candidates;
    for (size_t action_id = 0; action_id < alphazero_output->policy_.size(); ++action_id) {
        Action action(action_id, env_transition.getTurn());
        if (!env_transition.isLegalAction(action)) { continue; }
        int rotated_id = env_transition.getRotateAction(action_id, rotation);
        action_candidates.push_back(MCTS::ActionCandidate(action, alphazero_output->policy_[rotated_id], alphazero_output->policy_logits_[rotated_id]));
    }
    sort(action_candidates.begin(), action_candidates.end(), [](const MCTS::ActionCandidate& lhs, const MCTS::ActionCandidate& rhs) {
        return lhs.policy_ > rhs.policy_;
    });
    return action_candidates;
}

void ZeroActor::calculatePIMCActionPolicy(MCTSNode* leaf_node, const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation, std::vector<int>& counts, std::vector<MCTS::ActionCandidate>& action_candidates)
{
    assert(alphazero_network_);
    for (size_t action_id = 0; action_id < alphazero_output->policy_.size(); ++action_id) {
        Action action(action_id, env_transition.getTurn());
        if (!env_transition.isLegalAction(action, true)) { continue; }
        if (leaf_node == getMCTS()->getRootNode() && !env_.isLegalAction(action, false)) { continue; }
        int rotated_id = env_transition.getRotateAction(action_id, rotation);
        action_candidates[action_id].action_ = action;
        action_candidates[action_id].policy_ += alphazero_output->policy_[rotated_id];
        counts[action_id] += 1;
    }
}

std::vector<MCTS::ActionCandidate> ZeroActor::calculateMuZeroActionPolicy(MCTSNode* leaf_node, const std::shared_ptr<network::MuZeroNetworkOutput>& muzero_output)
{
    assert(muzero_network_);
    std::vector<MCTS::ActionCandidate> action_candidates;
    env::Player turn = leaf_node->getAction().nextPlayer();
    for (size_t action_id = 0; action_id < muzero_output->policy_.size(); ++action_id) {
        const Action action(action_id, turn);
        if (leaf_node == getMCTS()->getRootNode() && !env_.isLegalAction(action)) { continue; }
        action_candidates.push_back(MCTS::ActionCandidate(action, muzero_output->policy_[action_id], muzero_output->policy_logits_[action_id]));
    }
    sort(action_candidates.begin(), action_candidates.end(), [](const MCTS::ActionCandidate& lhs, const MCTS::ActionCandidate& rhs) {
        return lhs.policy_ > rhs.policy_;
    });
    return action_candidates;
}

Environment ZeroActor::getEnvironmentTransition(const std::vector<MCTSNode*>& node_path)
{
    Environment env = env_;
    for (size_t i = 1; i < node_path.size(); ++i) { env.act(node_path[i]->getAction()); }
    return env;
}

bool ZeroActor::calculateEnvironmentTransition(const std::vector<MCTSNode*>& node_path, Environment& env_transition)
{
    for (size_t j = 1; j < node_path.size(); ++j) {
        const Action& action = node_path[j]->getAction();
        if (env_transition.isLegalAction(action)) {
            env_transition.act(action);
            env::Player turn = env_transition.getTurn();
            if (!env_transition.getPerfectEnv().isPassAction(action)) { env_transition.act(Action(action.getActionID(), action.nextPlayer())); } // TODO: add a new getFeature function for reveal known bitboard?? (maple)
            env_transition.setTurn(turn);
        } else {
            return false;
        }
    }
    return true;
}

} // namespace minizero::actor

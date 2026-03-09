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
    if (!config::iig_use_merge_pimc) { resetPIMCSearch(); }
    use_tsl_ = (config::iig_use_tsl &&
                (config::zero_current_iteration < config::iig_tsl_transition_iteration ||
                 (config::zero_current_iteration >= config::iig_tsl_transition_iteration && config::zero_current_iteration < config::iig_tsl_end_iteration && utils::Random::randReal() < 0.5f)))
                   ? true
                   : false;
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
    informative_state_weights_.clear();
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
        if (config::iig_use_merge_pimc) {
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
        } else {
            if (getMCTS()->getNumSimulation() == 0) {
                if (pimc_count_ == 0) {
                    pimc_rotation_ = feature_rotation_;
                } else {
                    feature_rotation_ = pimc_rotation_;
                }
            }
            Environment env_transition = informative_states_[pimc_count_];
            calculateEnvironmentTransition(mcts_search_data_.node_path_, env_transition);
            nn_evaluated_batch_ids_.push_back(alphazero_network_->pushBack(env_transition.getPlayerFeatures(feature_rotation_)));
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
                informative_state_weights_.push_back(1.0f);
            }
        }
        return;
    }

    const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
    MCTSNode* leaf_node = node_path.back();
    if (alphazero_network_) {
        if (config::iig_use_merge_pimc) {
            std::vector<float> counts;
            std::vector<MCTS::ActionCandidate> action_candidates;
            for (int action_id = 0; action_id < env_.getPolicySize(); ++action_id) {
                action_candidates.push_back(MCTS::ActionCandidate(Action(action_id, env_.getTurn()), 0.0f, 0.0f));
                counts.push_back(0.0f);
            }
            float value_sum = 0.0f, value_count = 0.0f;
            for (int i = 0; i < pimc_repeat_; ++i) {
                if (!is_valid_states_[i]) { continue; }
                Environment env_transition = informative_states_[i];
                calculateEnvironmentTransition(node_path, env_transition);
                std::shared_ptr<AlphaZeroNetworkOutput> alphazero_output = std::static_pointer_cast<AlphaZeroNetworkOutput>(network_outputs[nn_evaluated_batch_ids_[i]]);
                if (!env_transition.isTerminal()) { calculatePIMCActionPolicy(leaf_node, env_transition, alphazero_output, feature_rotation_, counts, action_candidates, informative_state_weights_[i]); }
                value_sum += (!env_transition.isTerminal() ? alphazero_output->value_ : env_transition.getEvalScore());
                value_count += 1.0f;
            }
            assert(value_count > 0);
            std::vector<MCTS::ActionCandidate> tmp;
            for (size_t i = 0; i < action_candidates.size(); ++i) {
                if (counts[i] == 0.0f) { continue; }
                action_candidates[i].policy_ /= counts[i];
                action_candidates[i].policy_logit_ = std::log(action_candidates[i].policy_ + 1e-8f); // TODO: is 1e-8f ok? (5d)
                tmp.push_back(action_candidates[i]);
            }
            sort(tmp.begin(), tmp.end(), [](const MCTS::ActionCandidate& a, const MCTS::ActionCandidate& b) { return a.policy_ > b.policy_; });
            action_candidates = tmp;
            if (!action_candidates.empty()) { getMCTS()->expand(leaf_node, action_candidates); }
            getMCTS()->backup(node_path, value_sum / value_count, 0.0f);
        } else {
            Environment env_transition = informative_states_[pimc_count_];
            calculateEnvironmentTransition(mcts_search_data_.node_path_, env_transition);
            if (!env_transition.isTerminal()) {
                std::shared_ptr<AlphaZeroNetworkOutput> alphazero_output = std::static_pointer_cast<AlphaZeroNetworkOutput>(network_outputs[nn_evaluated_batch_ids_[0]]);
                getMCTS()->expand(leaf_node, calculateAlphaZeroActionPolicy(leaf_node, env_transition, alphazero_output, feature_rotation_));
                getMCTS()->backup(node_path, alphazero_output->value_, 0.0f);
            } else {
                getMCTS()->backup(node_path, env_transition.getEvalScore(), 0.0f);
            }
        }
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

void ZeroActor::resetPIMCSearch()
{
    pimc_count_ = 0;
    mcts_policy_strings_.clear();
    pimc_tree_nodes_.clear();
    pimc_tree_nodes_.resize(pimc_repeat_, std::vector<MCTSNode>(env_.getPolicySize() + 1, MCTSNode()));
}

void ZeroActor::beforeDiscriminatorNNEvaluation()
{
    if (anchor_embeddings_.empty()) {
        nn_evaluated_batch_ids_.push_back(siamese_network_->pushBackAnchor(env_.getDiscriminatorFeatures()));
    } else {
        for (int i = 0; i < config::iig_max_infoset_size; ++i) {
            Environment env = env_;
            if (!use_tsl_) { env.sampleOneInformationSet(i); }
            auto features = env.getPlayerFeatures();
            features.resize(config::iig_siamese_board_feature_channels * config::env_board_size * config::env_board_size);
            nn_evaluated_batch_ids_.push_back(siamese_network_->pushBackBoard(features));
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
            if (!use_tsl_) { informative_states_.back().sampleOneInformationSet(distances[i].first); }
            informative_state_weights_.push_back(1.0f);
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

    beforeNNEvaluation();
    if (siamese_network_ && siamese_network_->getBatchSize() > 0) {
        afterNNEvaluation(siamese_network_->forward());
    } else if (alphazero_network_ && alphazero_network_->getBatchSize() > 0) {
        afterNNEvaluation(alphazero_network_->forward());
    } else {
        afterNNEvaluation({});
    }
}

void ZeroActor::handleSearchDone()
{
    if (!config::iig_use_merge_pimc) {
        if (config::actor_use_gumbel) { mcts_policy_strings_.push_back(getMCTSPolicy()); }
        MCTSNode* root = getMCTS()->getRootNode();
        pimc_tree_nodes_[pimc_count_][0] = *root;
        pimc_tree_nodes_[pimc_count_][0].setFirstChild(pimc_tree_nodes_.back().data() + 1);
        for (int i = 0; i < root->getNumChildren(); ++i) {
            MCTSNode* child = root->getChild(i);
            MCTSNode& pimc_child = pimc_tree_nodes_[pimc_count_][child->getAction().getActionID() + 1];
            pimc_child = *child;
        }

        resetSearch();
        ++pimc_count_;
        if (pimc_count_ < pimc_repeat_) { return; }

        // at the final stage, we need to expand root's children again since we need to use current imperfect board to exapnd legal actions
        std::vector<MCTS::ActionCandidate> action_candidates;
        for (int action_id = 0; action_id < env_.getPolicySize(); ++action_id) {
            Action action(action_id, env_.getTurn());
            if (!env_.isLegalAction(action, false)) { continue; }
            action_candidates.push_back(MCTS::ActionCandidate(action, 0.0f, 0.0f));
        }
        getMCTS()->expand(root, action_candidates);
        for (size_t i = 0; i < pimc_tree_nodes_.size(); ++i) { root->add(pimc_tree_nodes_[i][0].getMean(), pimc_tree_nodes_[i][0].getCount()); }
        root->setCount(config::actor_num_simulation + 1);

        float policy_sum = 0.0f, value_sum = 0.0f;
        for (int i = 0; i < root->getNumChildren(); ++i) {
            MCTSNode* child = root->getChild(i);
            for (size_t j = 0; j < pimc_tree_nodes_.size(); ++j) {
                MCTSNode& pimc_child = pimc_tree_nodes_[j][child->getAction().getActionID() + 1];
                child->add(pimc_child.getMean(), pimc_child.getCount());
                policy_sum += pimc_child.getPolicy();
                value_sum += pimc_child.getValue();
                child->setPolicy(policy_sum / config::actor_pimc_repeat);
                child->setValue(value_sum / config::actor_pimc_repeat);
            }
        }
        if (config::actor_use_gumbel) { setMCTSPolicyString(); }
    }

    mcts_search_data_.selected_node_ = decideActionNode();
    const Action action = getSearchAction();
    assert(env_.isLegalAction(action, false));
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
    if (!config::iig_use_merge_pimc) {
        for (size_t i = 0; i < pimc_tree_nodes_.size(); ++i) {
            int index = mcts_search_data_.selected_node_ - getMCTS()->getRootNode()->getChild(0) + 1;
            oss << "pimc node info (" << i << "): " << pimc_tree_nodes_[i][index].toString() << std::endl;
        }
    }
    mcts_search_data_.search_info_ = oss.str();
    if (!config::iig_use_merge_pimc) { resetPIMCSearch(); }
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

void ZeroActor::setMCTSPolicyString()
{
    std::vector<float> pimc_policy(env_.getPolicySize(), 0.0f);
    for (auto& str : mcts_policy_strings_) {
        auto token = utils::stringToVector(str, ",");
        for (auto& t : token) {
            int action_id = std::stoi(t.substr(0, t.find(':')));
            float prob = std::stof(t.substr(t.find(':') + 1));
            assert(action_id >= 0 && action_id < static_cast<int>(pimc_policy.size()));
            pimc_policy[action_id] += prob;
        }
    }

    mcts_policy_string_ = "";
    for (size_t i = 0; i < pimc_policy.size(); ++i) {
        if (pimc_policy[i] == 0.0f) { continue; }
        if (!mcts_policy_string_.empty()) { mcts_policy_string_ += ","; }
        mcts_policy_string_ += std::to_string(i) + ":" + std::to_string(pimc_policy[i]);
    }
}

std::vector<MCTS::ActionCandidate> ZeroActor::calculateAlphaZeroActionPolicy(MCTSNode* leaf_node, const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation)
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

void ZeroActor::calculatePIMCActionPolicy(MCTSNode* leaf_node, const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation, std::vector<float>& counts, std::vector<MCTS::ActionCandidate>& action_candidates, const float policy_weight)
{
    assert(alphazero_network_);
    for (size_t action_id = 0; action_id < alphazero_output->policy_.size(); ++action_id) {
        Action action(action_id, env_transition.getTurn());
        if (!env_transition.isLegalAction(action, true)) { continue; }
        if (leaf_node == getMCTS()->getRootNode() && !env_.isLegalAction(action, false)) { continue; }
        int rotated_id = env_transition.getRotateAction(action_id, rotation);
        action_candidates[action_id].action_ = action;
        action_candidates[action_id].policy_ += alphazero_output->policy_[rotated_id] * policy_weight;
        counts[action_id] += policy_weight;
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
            if (config::iig_use_merge_pimc) {
                env::Player turn = env_transition.getTurn();
#if PHANTOMGO
                if (!env_transition.getPerfectEnv().isPassAction(action)) { env_transition.act(Action(action.getActionID(), action.nextPlayer())); }
#else
                env_transition.act(Action(action.getActionID(), action.nextPlayer()));
#endif
                env_transition.setTurn(turn);
            }
        } else {
            return false;
        }
    }
    return true;
}

} // namespace minizero::actor

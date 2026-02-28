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
    resetPIMCSearch();
}

void ZeroActor::resetSearch()
{
    BaseActor::resetSearch();
    mcts_search_data_.node_path_.clear();
    getMCTS()->getRootNode()->setAction(Action(-1, env::getPreviousPlayer(env_.getTurn(), env_.getNumPlayer())));
    info_set_count_ = -1;
}

void ZeroActor::resetPIMCSearch()
{
    pimc_count_ = 0;
    pimc_roots_.clear();
    pimc_roots_.resize(config::actor_pimc_repeat, MCTSNode());
    pimc_children_nodes_.clear();
    pimc_children_nodes_.resize(config::actor_pimc_repeat, std::vector<MCTSNode>(env_.getPolicySize(), MCTSNode()));
    pimc_envs_.clear();
    pimc_policy_.clear();
    pimc_policy_.resize(env_.getPolicySize(), 0.0f);
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
    mcts_search_data_.node_path_ = selection();
    if (alphazero_network_) {
        if (getMCTS()->getNumSimulation() == 0) {
            if (config::iig_use_discriminator && pimc_count_ == 0 && info_set_count_ < config::iig_max_infoset_size) {
                beforeDiscriminatorNNEvaluation();
                return;
            }
            if (pimc_count_ == 0) { env_backup_ = env_; }
            env_ = env_backup_;
            env_.sampleOneInformationSet(config::iig_use_discriminator ? informative_state_ids_[pimc_count_] : pimc_count_);
        }
        Environment env_transition = getEnvironmentTransition(mcts_search_data_.node_path_);
        feature_rotation_ = config::actor_use_random_rotation_features ? static_cast<utils::Rotation>(utils::Random::randInt() % static_cast<int>(utils::Rotation::kRotateSize)) : utils::Rotation::kRotationNone;
        nn_evaluation_batch_id_ = alphazero_network_->pushBack(env_transition.getFeatures(feature_rotation_));
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

void ZeroActor::afterNNEvaluation(const std::shared_ptr<NetworkOutput>& network_output)
{
    if (config::iig_use_discriminator && pimc_count_ == 0 && info_set_count_ < config::iig_max_infoset_size) {
        afterDiscriminatorNNEvaluation(network_output);
        return;
    }

    const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
    MCTSNode* leaf_node = node_path.back();
    if (alphazero_network_) {
        Environment env_transition = getEnvironmentTransition(node_path);
        if (!env_transition.isTerminal()) {
            std::shared_ptr<AlphaZeroNetworkOutput> alphazero_output = std::static_pointer_cast<AlphaZeroNetworkOutput>(network_output);
            getMCTS()->expand(leaf_node, calculateAlphaZeroActionPolicy(env_transition, alphazero_output, feature_rotation_));
            getMCTS()->backup(node_path, alphazero_output->value_, env_transition.getReward());
        } else {
            getMCTS()->backup(node_path, env_transition.getEvalScore(), env_transition.getReward());
        }
    } else if (muzero_network_) {
        std::shared_ptr<MuZeroNetworkOutput> muzero_output = std::static_pointer_cast<MuZeroNetworkOutput>(network_output);
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
    if (config::iig_discriminator_nn_type_name == "siamese") {
        if (info_set_count_ == -1) {
            nn_evaluation_batch_id_ = siamese_network_->pushBackAnchor(env_.getFeatures(false));
        } else {
            Environment env = env_;
            env.sampleOneInformationSet(info_set_count_);
            nn_evaluation_batch_id_ = siamese_network_->pushBackBoard(env.getFeatures(true));
        }
    }
}

void ZeroActor::afterDiscriminatorNNEvaluation(const std::shared_ptr<network::NetworkOutput>& network_output)
{
    auto embs = std::static_pointer_cast<SiameseNetworkOutput>(network_output)->embeddings_;

    if (info_set_count_ == -1) {
        anchor_embeddings_ = embs;
        info_set_distances_.clear();
    } else {
        float dist = utils::distance(anchor_embeddings_, embs);
        info_set_distances_.push_back(std::make_pair(info_set_count_, dist));
    }
    ++info_set_count_;

    if (info_set_distances_.size() == config::iig_max_infoset_size) {
        std::sort(info_set_distances_.begin(), info_set_distances_.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second < b.second;
        });

        informative_state_ids_.clear();
        for (int i = 0; i < config::actor_pimc_repeat && i < static_cast<int>(info_set_distances_.size()); ++i) {
            informative_state_ids_.push_back(info_set_distances_[i].first);
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
    int num_simulation_left = config::actor_num_simulation + 1 - num_simulation;
    int batch_size = std::min(config::actor_mcts_think_batch_size,
                              (alphazero_network_ || num_simulation > 0) ? num_simulation_left : 1 /* initial inference for root node */);
    assert(batch_size > 0);

    std::vector<std::tuple<int, utils::Rotation, decltype(mcts_search_data_.node_path_)>> batch_queries; // batch id, rotation, search path
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        beforeNNEvaluation();
        assert(nn_evaluation_batch_id_ == batch_id);
        if (mcts_search_data_.node_path_.back()->getVirtualLoss() == 0) {
            batch_queries.emplace_back(nn_evaluation_batch_id_, feature_rotation_, mcts_search_data_.node_path_);
        }
        for (auto node : mcts_search_data_.node_path_) { node->addVirtualLoss(); }
    }
    auto network_output = siamese_network_ && siamese_network_->getBatchSize() > 0
                              ? siamese_network_->forward()
                              : (alphazero_network_ ? alphazero_network_->forward()
                                                    : (num_simulation == 0 ? muzero_network_->initialInference() : muzero_network_->recurrentInference()));
    for (auto& query : batch_queries) {
        nn_evaluation_batch_id_ = std::get<0>(query);
        feature_rotation_ = std::get<1>(query);
        mcts_search_data_.node_path_ = std::get<2>(query);
        afterNNEvaluation(network_output[nn_evaluation_batch_id_]);
        if (mcts_search_data_.node_path_.empty()) { return; }
        auto virtual_loss = mcts_search_data_.node_path_.back()->getVirtualLoss();
        for (auto node : mcts_search_data_.node_path_) { node->removeVirtualLoss(virtual_loss); }
    }
}

void ZeroActor::accumulateMCTSPolicy(const std::string& policy_str, std::vector<float>& pimc_policy)
{
    std::istringstream ss(policy_str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        size_t delim_pos = token.find(':');
        if (delim_pos == std::string::npos) { continue; }

        int action_id = std::stoi(token.substr(0, delim_pos));
        float prob = std::stof(token.substr(delim_pos + 1));

        if (action_id >= 0 && action_id < static_cast<int>(pimc_policy.size())) {
            pimc_policy[action_id] += prob;
        }
    }
}

void ZeroActor::setAccumulateMCTSPolicy(std::vector<float>& pimc_policy)
{
    accumulate_mcts_policy_.clear();
    for (size_t i = 0; i < pimc_policy.size(); ++i) {
        if (pimc_policy[i] > 0.0f) {
            if (!accumulate_mcts_policy_.empty()) { accumulate_mcts_policy_ += ","; }
            accumulate_mcts_policy_ += std::to_string(i) + ":" + std::to_string(pimc_policy[i]);
        }
    }
}

void ZeroActor::handleSearchDone()
{
    env_ = env_backup_;
    MCTSNode* root = getMCTS()->getRootNode();
    if (config::actor_use_gumbel) { accumulateMCTSPolicy(gumbel_zero_.getMCTSPolicy(getMCTS()), pimc_policy_); }

    pimc_roots_[pimc_count_].add(root->getMean(), root->getCount());
    for (int i = 0; i < root->getNumChildren(); ++i) {
        MCTSNode* child = root->getChild(i);
        MCTSNode& pimc_child = pimc_children_nodes_[pimc_count_][child->getAction().getActionID()];
        pimc_child.setAction(child->getAction());
        pimc_child.add(child->getMean(), child->getCount());
        pimc_child.setPolicy(child->getPolicy());
        pimc_child.setValue(child->getValue());
    }
    resetSearch();
    ++pimc_count_;
    if (pimc_count_ < config::actor_pimc_repeat) { return; }

    // at the final stage, we need to expand root's children again since we need to use current imperfect board to exapnd legal actions
    std::vector<MCTS::ActionCandidate> action_candidates;
    for (int action_id = 0; action_id < env_.getPolicySize(); ++action_id) {
        Action action(action_id, env_.getTurn());
        if (!env_.isLegalAction(action, false)) { continue; }
        action_candidates.push_back(MCTS::ActionCandidate(action, 0.0f, 0.0f));
    }
    getMCTS()->expand(root, action_candidates);

    float policy_sum = 0.0f, value_sum = 0.0f;
    for (size_t i = 0; i < pimc_roots_.size(); ++i) {
        root->add(pimc_roots_[i].getMean(), pimc_roots_[i].getCount());
        for (int j = 0; j < root->getNumChildren(); ++j) {
            MCTSNode* child = root->getChild(j);
            MCTSNode& pimc_child = pimc_children_nodes_[i][child->getAction().getActionID()];
            child->add(pimc_child.getMean(), pimc_child.getCount());
            policy_sum += child->getPolicy();
            value_sum += child->getValue();
            child->setPolicy(policy_sum / config::actor_pimc_repeat);
            child->setValue(value_sum / config::actor_pimc_repeat);
        }
    }
    root->setCount(config::actor_num_simulation + 1);

    if (config::actor_use_gumbel) {
        for (size_t i = 0; i < pimc_policy_.size(); ++i) { pimc_policy_[i] /= config::actor_pimc_repeat; }
    }
    setAccumulateMCTSPolicy(pimc_policy_);

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
    for (size_t i = 0; i < pimc_children_nodes_.size(); ++i) {
        oss << "pimc node info (" << i << "): " << pimc_children_nodes_[i][mcts_search_data_.selected_node_->getAction().getActionID()].toString() << std::endl;
    }
    mcts_search_data_.search_info_ = oss.str();
    resetPIMCSearch();
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

} // namespace minizero::actor

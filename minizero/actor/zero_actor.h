#pragma once

#include "alphazero_network.h"
#include "base_actor.h"
#include "discriminator_network.h"
#include "gumbel_zero.h"
#include "mcts.h"
#include "muzero_network.h"
#include "siamese_network.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minizero::actor {

class MCTSSearchData {
public:
    std::string search_info_;
    MCTSNode* selected_node_;
    std::vector<MCTSNode*> node_path_;
    void clear();
};

class ZeroActor : public BaseActor {
public:
    ZeroActor(uint64_t tree_node_size)
        : tree_node_size_(tree_node_size)
    {
        alphazero_network_ = nullptr;
        muzero_network_ = nullptr;
        discriminator_network_ = nullptr;
        siamese_network_ = nullptr;
    }

    void reset() override;
    void resetSearch() override;
    Action think(bool with_play = false, bool display_board = false) override;
    void beforeNNEvaluation() override;
    void afterNNEvaluation(const std::vector<std::shared_ptr<network::NetworkOutput>>& network_outputs) override;
    bool isSearchDone() const override { return getMCTS()->reachMaximumSimulation(); }
    Action getSearchAction() const override { return mcts_search_data_.selected_node_->getAction(); }
    bool isResign() const override { return enable_resign_ && getMCTS()->isResign(mcts_search_data_.selected_node_); }
    std::string getSearchInfo() const override { return mcts_search_data_.search_info_; }
    void setNetwork(const std::shared_ptr<network::Network>& network) override;
    std::shared_ptr<Search> createSearch() override { return std::make_shared<MCTS>(tree_node_size_); }
    std::shared_ptr<MCTS> getMCTS() { return std::static_pointer_cast<MCTS>(search_); }
    const std::shared_ptr<MCTS> getMCTS() const { return std::static_pointer_cast<MCTS>(search_); }
    const std::vector<Environment>& getInformativeStates() const { return informative_states_; }

protected:
    void beforeDiscriminatorNNEvaluation();
    void afterDiscriminatorNNEvaluation(const std::vector<std::shared_ptr<network::NetworkOutput>>& network_outputs);
    std::vector<std::pair<std::string, std::string>> getActionInfo() const override;
    std::string getMCTSPolicy() const override { return (config::actor_use_gumbel ? gumbel_zero_.getMCTSPolicy(getMCTS()) : getMCTS()->getSearchDistributionString()); }
    std::string getMCTSValue() const override { return std::to_string(getMCTS()->getRootNode()->getMean()); }
    std::string getEnvReward() const override;
    std::string getModelFileName() const { return (alphazero_network_ ? alphazero_network_->getNetworkFileName() : (muzero_network_ ? muzero_network_->getNetworkFileName() : "")); }

    virtual void step();
    virtual void handleSearchDone();
    virtual MCTSNode* decideActionNode();
    virtual void addNoiseToNodeChildren(MCTSNode* node);
    virtual std::vector<MCTSNode*> selection() { return (config::actor_use_gumbel ? gumbel_zero_.selection(getMCTS()) : getMCTS()->select()); }

    std::vector<MCTS::ActionCandidate> calculateAlphaZeroActionPolicy(const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation);
    void calculatePIMCActionPolicy(MCTSNode* leaf_node, const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation, std::vector<int>& counts, std::vector<MCTS::ActionCandidate>& action_candidates);
    std::vector<MCTS::ActionCandidate> calculateMuZeroActionPolicy(MCTSNode* leaf_node, const std::shared_ptr<network::MuZeroNetworkOutput>& muzero_output);
    virtual Environment getEnvironmentTransition(const std::vector<MCTSNode*>& node_path);
    bool calculateEnvironmentTransition(const std::vector<MCTSNode*>& node_path, Environment& env_transition);

    bool enable_resign_;
    GumbelZero gumbel_zero_;
    uint64_t tree_node_size_;
    MCTSSearchData mcts_search_data_;
    utils::Rotation feature_rotation_;
    std::shared_ptr<network::AlphaZeroNetwork> alphazero_network_;
    std::shared_ptr<network::MuZeroNetwork> muzero_network_;
    std::shared_ptr<network::DiscriminatorNetwork> discriminator_network_;
    std::shared_ptr<network::SiameseNetwork> siamese_network_;

    // for pimc
    std::vector<int> is_valid_states_;
    std::vector<int> nn_evaluated_batch_ids_;
    std::vector<float> anchor_embeddings_;
    std::vector<Environment> informative_states_;
};

} // namespace minizero::actor

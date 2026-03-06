#include "arena.h"
#include "configuration.h"
#include "create_actor.h"
#include "create_network.h"
#include "random.h"
#include "time_system.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <torch/cuda.h>
#include <utility>

namespace minizero::iig {

using namespace actor;
using namespace network;
using namespace utils;

int ArenaSharedData::getAvailableActorIndex()
{
    std::lock_guard lock(mutex_);
    return (actor_index_ < static_cast<int>(actors_.size()) ? actor_index_++ : actors_.size());
}

void ArenaSharedData::outputGame(int actor_id)
{
    std::ostringstream oss;
    const std::shared_ptr<BaseActor>& actor = actors_[actor_id];

    actor->getActionInfoHistory().clear();
    std::string p1_nn_name = networks_.get(actor_id % 2 == 0 ? env::Player::kPlayer1 : env::Player::kPlayer2)[0]->getNetworkFileName();
    std::string p2_nn_name = networks_.get(actor_id % 2 == 1 ? env::Player::kPlayer1 : env::Player::kPlayer2)[0]->getNetworkFileName();
    std::string p1_dnn_name = discriminator_networks_.get(actor_id % 2 == 0 ? env::Player::kPlayer1 : env::Player::kPlayer2)[0]->getNetworkFileName();
    std::string p2_dnn_name = discriminator_networks_.get(actor_id % 2 == 1 ? env::Player::kPlayer1 : env::Player::kPlayer2)[0]->getNetworkFileName();
    std::string p1_pimc_repeat = std::to_string(actor_id % 2 == 0 ? config::iig_evaluation_player1_pimc_repeat : config::iig_evaluation_player2_pimc_repeat);
    std::string p2_pimc_repeat = std::to_string(actor_id % 2 == 1 ? config::iig_evaluation_player1_pimc_repeat : config::iig_evaluation_player2_pimc_repeat);

    oss << actor->getRecord({{"P1", p1_nn_name},
                             {"P2", p2_nn_name},
                             {"P1_DNN", p1_dnn_name},
                             {"P2_DNN", p2_dnn_name},
                             {"P1_PIMC_REPEAT", p1_pimc_repeat},
                             {"P2_PIMC_REPEAT", p2_pimc_repeat},
                             {"T", utils::TimeSystem::getTimeString("Y/m/d H:i:s.f")}});

    std::lock_guard lock(mutex_);
    std::cout << oss.str() << std::endl;
}

void ArenaThread::initialize()
{
    int seed = config::program_auto_seed ? std::random_device()() : config::program_seed + id_;
    Random::seed(seed);
}

void ArenaThread::runJob()
{
    if (getSharedData()->mcts_phase_ == MCTSPhase::kSelection) {
        while (selection()) {}
    } else if (getSharedData()->mcts_phase_ == MCTSPhase::kEvaluation) {
        evaluation();
    } else if (getSharedData()->mcts_phase_ == MCTSPhase::kBackup) {
        while (backup()) {}
    }
}

bool ArenaThread::selection()
{
    size_t actor_id = getSharedData()->getAvailableActorIndex();
    if (actor_id >= getSharedData()->actors_.size()) { return false; }

    std::shared_ptr<ZeroActor> actor = std::static_pointer_cast<ZeroActor>(getSharedData()->actors_[actor_id]);
    actor->beforeNNEvaluation();
    getSharedData()->actor_stages_[actor_id] = (actor->getMCTS()->getNumSimulation() == 0 && actor->getInformativeStates().empty());
    return true;
}

void ArenaThread::evaluation()
{
    if (id_ >= static_cast<int>(getSharedData()->networks_.get(env::Player::kPlayer1).size())) { return; }

    std::shared_ptr<Network>& network1 = getSharedData()->networks_.get(env::Player::kPlayer1)[id_];
    std::shared_ptr<AlphaZeroNetwork> az_network1 = std::static_pointer_cast<AlphaZeroNetwork>(network1);
    if (az_network1->getBatchSize() > 0) { getSharedData()->network_outputs_.get(env::Player::kPlayer1)[id_] = az_network1->forward(); }
    std::shared_ptr<Network>& network2 = getSharedData()->networks_.get(env::Player::kPlayer2)[id_];
    std::shared_ptr<AlphaZeroNetwork> az_network2 = std::static_pointer_cast<AlphaZeroNetwork>(network2);
    if (az_network2->getBatchSize() > 0) { getSharedData()->network_outputs_.get(env::Player::kPlayer2)[id_] = az_network2->forward(); }

    std::shared_ptr<Network>& discriminator_network1 = getSharedData()->discriminator_networks_.get(env::Player::kPlayer1)[id_];
    std::shared_ptr<SiameseNetwork> d_network1 = std::static_pointer_cast<SiameseNetwork>(discriminator_network1);
    if (d_network1->getBatchSize() > 0) { getSharedData()->discriminator_network_outputs_.get(env::Player::kPlayer1)[id_] = d_network1->forward(); }
    std::shared_ptr<Network>& discriminator_network2 = getSharedData()->discriminator_networks_.get(env::Player::kPlayer2)[id_];
    std::shared_ptr<SiameseNetwork> d_network2 = std::static_pointer_cast<SiameseNetwork>(discriminator_network2);
    if (d_network2->getBatchSize() > 0) { getSharedData()->discriminator_network_outputs_.get(env::Player::kPlayer2)[id_] = d_network2->forward(); }
}

bool ArenaThread::backup()
{
    size_t actor_id = getSharedData()->getAvailableActorIndex();
    if (actor_id >= getSharedData()->actors_.size()) { return false; }

    std::shared_ptr<BaseActor>& actor = getSharedData()->actors_[actor_id];
    int network_id = actor_id % getSharedData()->networks_.get(env::Player::kPlayer1).size();
    env::Player nn_player = actor->getEnvironment().getTurn();
    if (actor_id % 2 == 1) { nn_player = (nn_player == env::Player::kPlayer1 ? env::Player::kPlayer2 : env::Player::kPlayer1); }
    int stage = getSharedData()->actor_stages_[actor_id];
    if (stage == 0) {
        actor->afterNNEvaluation(getSharedData()->network_outputs_.get(nn_player)[network_id]);
    } else {
        actor->afterNNEvaluation(getSharedData()->discriminator_network_outputs_.get(nn_player)[network_id]);
    }
    if (actor->isSearchDone()) { handleSearchDone(actor_id); }
    return true;
}

void ArenaThread::handleSearchDone(int actor_id)
{
    assert(actor_id >= 0 && actor_id < static_cast<int>(getSharedData()->actors_.size()) && getSharedData()->actors_[actor_id]->isSearchDone());

    std::shared_ptr<BaseActor>& actor = getSharedData()->actors_[actor_id];
    if (!actor->isResign()) { actor->act(actor->getSearchAction()); }
    bool is_endgame = (actor->isResign() || actor->isEnvTerminal());
    bool display_game = (actor_id == 0);
    if (display_game) { std::cerr << actor->getEnvironment().toString() << actor->getSearchInfo() << std::endl; }
    if (is_endgame) {
        getSharedData()->outputGame(actor_id);
        actor->reset();
    } else {
        int game_length = actor->getEnvironment().getActionHistory().size();
        int sequence_length = config::zero_actor_intermediate_sequence_length;
        if (sequence_length > 0 && game_length >= sequence_length && (game_length - config::learner_n_step_return - config::learner_muzero_unrolling_step) % sequence_length == 0) { getSharedData()->outputGame(actor_id); }
        actor->resetSearch();
    }
}

void Arena::run()
{
    initialize();
    while (true) {
        getSharedData()->actor_index_ = 0;
        for (size_t i = 0; i < getSharedData()->actors_.size(); ++i) {
            auto& actor = getSharedData()->actors_[i];
            env::Player nn_player = actor->getEnvironment().getTurn();
            if (i % 2 == 1) { nn_player = (nn_player == env::Player::kPlayer1 ? env::Player::kPlayer2 : env::Player::kPlayer1); }
            int pimc_count = ((i % 2 == 0 && nn_player == env::Player::kPlayer1 || i % 2 == 1 && nn_player == env::Player::kPlayer2) ? config::iig_evaluation_player1_pimc_repeat : config::iig_evaluation_player2_pimc_repeat);
            std::static_pointer_cast<ZeroActor>(actor)->setPIMCRepeat(pimc_count);
            actor->setNetwork(getSharedData()->networks_.get(nn_player)[i % getSharedData()->networks_.get(nn_player).size()]);
            actor->setNetwork(getSharedData()->discriminator_networks_.get(nn_player)[i % getSharedData()->discriminator_networks_.get(nn_player).size()]);
        }
        for (auto& t : slave_threads_) { t->start(); }
        for (auto& t : slave_threads_) { t->finish(); }
        getSharedData()->mcts_phase_ = static_cast<MCTSPhase>((static_cast<int>(getSharedData()->mcts_phase_) + 1) % static_cast<int>(MCTSPhase::kSize));
    }
}

void Arena::initialize()
{
    int num_threads = std::max(static_cast<int>(torch::cuda::device_count()), config::zero_num_threads);
    createSlaveThreads(num_threads);
    createNeuralNetworks();
    createActors();
    getSharedData()->mcts_phase_ = MCTSPhase::kSelection;
}

void Arena::createNeuralNetworks()
{
    int num_networks = std::min(static_cast<int>(torch::cuda::device_count()), config::zero_num_parallel_games);
    assert(num_networks > 0);
    getSharedData()->networks_.get(env::Player::kPlayer1).resize(num_networks);
    getSharedData()->networks_.get(env::Player::kPlayer2).resize(num_networks);
    getSharedData()->discriminator_networks_.get(env::Player::kPlayer1).resize(num_networks, nullptr);
    getSharedData()->discriminator_networks_.get(env::Player::kPlayer2).resize(num_networks, nullptr);
    getSharedData()->network_outputs_.get(env::Player::kPlayer1).resize(num_networks);
    getSharedData()->network_outputs_.get(env::Player::kPlayer2).resize(num_networks);
    getSharedData()->discriminator_network_outputs_.get(env::Player::kPlayer1).resize(num_networks);
    getSharedData()->discriminator_network_outputs_.get(env::Player::kPlayer2).resize(num_networks);
    for (int gpu_id = 0; gpu_id < num_networks; ++gpu_id) {
        getSharedData()->networks_.get(env::Player::kPlayer1)[gpu_id] = createNetwork(config::iig_evaluation_player1_file_name, gpu_id);
        getSharedData()->networks_.get(env::Player::kPlayer2)[gpu_id] = createNetwork(config::iig_evaluation_player2_file_name, gpu_id);
        if (!config::iig_evaluation_discriminator1_file_name.empty()) { getSharedData()->discriminator_networks_.get(env::Player::kPlayer1)[gpu_id] = createNetwork(config::iig_evaluation_discriminator1_file_name, gpu_id); }
        if (!config::iig_evaluation_discriminator2_file_name.empty()) { getSharedData()->discriminator_networks_.get(env::Player::kPlayer2)[gpu_id] = createNetwork(config::iig_evaluation_discriminator2_file_name, gpu_id); }
    }
}

void Arena::createActors()
{
    assert(getSharedData()->networks_.get(env::Player::kPlayer1).size() > 0);
    std::shared_ptr<Network>& network = getSharedData()->networks_.get(env::Player::kPlayer1)[0];
    uint64_t tree_node_size = static_cast<uint64_t>(config::actor_num_simulation + 1) * network->getActionSize();
    for (int i = 0; i < config::zero_num_parallel_games; ++i) {
        getSharedData()->actors_.emplace_back(createActor(tree_node_size, getSharedData()->networks_.get(env::Player::kPlayer1)[i % getSharedData()->networks_.get(env::Player::kPlayer1).size()]));
        getSharedData()->actor_stages_.push_back(0);
    }
}

} // namespace minizero::iig

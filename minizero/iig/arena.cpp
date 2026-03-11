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
    if (getSharedData()->run_mcts_[actor_id] == 0) { return true; }

    std::shared_ptr<ZeroActor> actor = std::static_pointer_cast<ZeroActor>(getSharedData()->actors_[actor_id]);
    actor->beforeNNEvaluation();
    return true;
}

void ArenaThread::evaluation()
{
    if (id_ >= static_cast<int>(getSharedData()->networks_.size())) { return; }

    std::shared_ptr<AlphaZeroNetwork> az_network = std::static_pointer_cast<AlphaZeroNetwork>(getSharedData()->networks_[id_]);
    if (az_network->getBatchSize() > 0) { getSharedData()->network_outputs_[id_] = az_network->forward(); }

    if (config::iig_use_discriminator) {
        std::shared_ptr<SiameseNetwork> d_network = std::static_pointer_cast<SiameseNetwork>(getSharedData()->discriminator_networks_[id_]);
        if (d_network->getBatchSize() > 0) { getSharedData()->network_outputs_[id_] = d_network->forward(); }
    }
}

bool ArenaThread::backup()
{
    size_t actor_id = getSharedData()->getAvailableActorIndex();
    if (actor_id >= getSharedData()->actors_.size()) { return false; }
    if (getSharedData()->run_mcts_[actor_id] == 0) { return true; }

    std::shared_ptr<BaseActor>& actor = getSharedData()->actors_[actor_id];
    int network_id = actor_id % getSharedData()->networks_.size();
    actor->afterNNEvaluation(getSharedData()->network_outputs_[network_id]);
    if (actor->isSearchDone()) { handleSearchDone(actor_id); }
    return true;
}

void ArenaThread::handleSearchDone(int actor_id)
{
    assert(actor_id >= 0 && actor_id < static_cast<int>(getSharedData()->actors_.size()) && getSharedData()->actors_[actor_id]->isSearchDone());

    std::shared_ptr<BaseActor>& actor = getSharedData()->actors_[actor_id];
    if (!actor->isResign()) {
        actor->act(actor->getSearchAction());
        std::lock_guard lock(getSharedData()->mutex_);
        std::cout << "play " << actor_id << " " << env::playerToChar(actor->getSearchAction().getPlayer()) << " " << actor->getSearchAction().getActionID() << std::endl;
    }
    bool is_endgame = (actor->isResign() || actor->isEnvTerminal());
    if (actor_id == 0) { std::cerr << actor->getEnvironment().toString() << actor->getSearchInfo() << std::endl; }
    if (is_endgame) {
        std::lock_guard lock(getSharedData()->mutex_);
        std::cout << "game_over " << actor_id << " "
                  << env::playerToChar(actor->getSearchAction().getPlayer()) << " "
                  << config::iig_player_file_name << " "
                  << "\"" << config::iig_discriminator_file_name << "\" "
                  << config::actor_pimc_repeat << " "
                  << "\"" << config::iig_arena_tag << "\" " << std::endl;
        std::cout << "clear_board " << actor_id << std::endl;
        if (actor->getSearchAction().getPlayer() == env::Player::kPlayer1) {
            getSharedData()->set_genmove_ids_.push_back(actor_id);
            getSharedData()->run_mcts_[actor_id] = 0;
        } else {
            getSharedData()->run_mcts_[actor_id] = 1;
        }
        actor->reset();
    } else {
        getSharedData()->run_mcts_[actor_id] = (actor->getEnvironment().getTurn() == actor->getSearchAction().getPlayer());
        if (actor->getEnvironment().getTurn() != actor->getSearchAction().getPlayer()) { getSharedData()->set_genmove_ids_.push_back(actor_id); }
        actor->resetSearch();
    }
}

void Arena::run()
{
    initialize();
    std::string command;
    while (std::getline(std::cin, command)) {
        std::vector<std::string> commands = utils::stringToVector(command);
        if (commands[0] == "quit") {
            exit(0);
        } else if (commands[0] == "clear_board") {
            // format: clear_board <actor_id>
            int actor_id = std::stoi(commands[1]);
            std::shared_ptr<BaseActor>& actor = getSharedData()->actors_[actor_id];
            actor->reset();
            std::cout << "= " << std::endl;
        } else if (commands[0] == "play") {
            // format: play <actor_id> <player> <action_id>
            int actor_id = std::stoi(commands[1]);
            env::Player player = env::charToPlayer(commands[2][0]);
            int action_id = std::stoi(commands[3]);
            std::shared_ptr<BaseActor>& actor = getSharedData()->actors_[actor_id];
            actor->act(Action(action_id, player));
        } else if (commands[0] == "set_genmove") {
            // format: genmove <actor_id1> <actor_id2> ...
            for (size_t i = 1; i < commands.size(); ++i) { getSharedData()->run_mcts_[std::stoi(commands[i])] = 1; }
        } else if (commands[0] == "genmove") {
            // run MCTS
            bool is_end = false;
            getSharedData()->mcts_phase_ = MCTSPhase::kSelection;
            getSharedData()->set_genmove_ids_.clear();
            while (!is_end) {
                getSharedData()->actor_index_ = 0;
                for (auto& t : slave_threads_) { t->start(); }
                for (auto& t : slave_threads_) { t->finish(); }
                getSharedData()->mcts_phase_ = static_cast<MCTSPhase>((static_cast<int>(getSharedData()->mcts_phase_) + 1) % static_cast<int>(MCTSPhase::kSize));

                is_end = true;
                for (size_t i = 0; i < getSharedData()->actors_.size(); ++i) {
                    if (!getSharedData()->run_mcts_[i]) { continue; }
                    is_end = false;
                    break;
                }
            }
            std::cout << "set_genmove";
            for (auto& id : getSharedData()->set_genmove_ids_) { std::cout << " " << id; }
            std::cout << std::endl;
            std::cout << "genmove" << std::endl;
        } else if (commands[0] == "game_over") {
            // format: game_over <actor_id> <opp_player> <player_nn_file_name> <discriminator_nn_file_name> <pimc_repeat> <tag>
            int actor_id = std::stoi(commands[1]);
            env::Player opp_player = env::charToPlayer(commands[2][0]);
            std::string opp_player_nn_file_name = commands[3];
            std::string opp_discriminator_nn_file_name = commands[4].substr(1, commands[4].size() - 2);
            std::string opp_pimc_repeat = commands[5];
            std::string opp_tag = commands[6].substr(1, commands[6].size() - 2);
            std::shared_ptr<BaseActor>& actor = getSharedData()->actors_[actor_id];
            std::cerr << "[GAME_LOG] "
                      << actor->getRecord({{"B_NN", opp_player == env::Player::kPlayer1 ? opp_player_nn_file_name : config::iig_player_file_name},
                                           {"W_NN", opp_player == env::Player::kPlayer1 ? config::iig_player_file_name : opp_player_nn_file_name},
                                           {"B_DNN", opp_player == env::Player::kPlayer1 ? opp_discriminator_nn_file_name : config::iig_discriminator_file_name},
                                           {"W_DNN", opp_player == env::Player::kPlayer1 ? config::iig_discriminator_file_name : opp_discriminator_nn_file_name},
                                           {"B_PIMC_REPEAT", opp_player == env::Player::kPlayer1 ? opp_pimc_repeat : std::to_string(config::actor_pimc_repeat)},
                                           {"W_PIMC_REPEAT", opp_player == env::Player::kPlayer1 ? std::to_string(config::actor_pimc_repeat) : opp_pimc_repeat},
                                           {"B_TAG", opp_player == env::Player::kPlayer1 ? opp_tag : config::iig_arena_tag},
                                           {"W_TAG", opp_player == env::Player::kPlayer1 ? config::iig_arena_tag : opp_tag}})
                      << std::endl;
        }
    }
}

void Arena::initialize()
{
    int num_threads = std::max(static_cast<int>(torch::cuda::device_count()), config::zero_num_threads);
    createSlaveThreads(num_threads);
    createNeuralNetworks();
    createActors();
}

void Arena::createNeuralNetworks()
{
    int num_networks = std::min(static_cast<int>(torch::cuda::device_count()), config::zero_num_parallel_games);
    assert(num_networks > 0);
    getSharedData()->networks_.resize(num_networks);
    getSharedData()->discriminator_networks_.resize(num_networks, nullptr);
    getSharedData()->network_outputs_.resize(num_networks);
    for (int gpu_id = 0; gpu_id < num_networks; ++gpu_id) {
        getSharedData()->networks_[gpu_id] = createNetwork(config::iig_player_file_name, gpu_id);
        if (config::iig_use_discriminator) { getSharedData()->discriminator_networks_[gpu_id] = createNetwork(config::iig_discriminator_file_name, gpu_id); }
    }
}

void Arena::createActors()
{
    assert(getSharedData()->networks_.size() > 0);
    std::shared_ptr<Network>& network = getSharedData()->networks_[0];
    uint64_t tree_node_size = static_cast<uint64_t>(config::actor_num_simulation + 1) * network->getActionSize();
    for (int i = 0; i < config::zero_num_parallel_games; ++i) {
        getSharedData()->actors_.emplace_back(createActor(tree_node_size, getSharedData()->networks_[i % getSharedData()->networks_.size()]));
        if (config::iig_use_discriminator) { getSharedData()->actors_.back()->setNetwork(getSharedData()->discriminator_networks_[i % getSharedData()->discriminator_networks_.size()]); }
        getSharedData()->run_mcts_.push_back(0);
    }
}

} // namespace minizero::iig

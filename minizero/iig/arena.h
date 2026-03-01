#pragma once

#include "base_actor.h"
#include "network.h"
#include "paralleler.h"
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace minizero::iig {

enum class MCTSPhase {
    kSelection = 0,
    kEvaluation = 1,
    kBackup = 2,
    kSize = 3
};

class ArenaSharedData : public utils::BaseSharedData {
public:
    int getAvailableActorIndex();
    void outputGame(int actor_id);

    minizero::env::Player turn_;
    MCTSPhase mcts_phase_;
    int actor_index_;
    std::mutex mutex_;
    std::vector<std::shared_ptr<minizero::actor::BaseActor>> actors_;
    minizero::env::GamePair<std::vector<std::shared_ptr<network::Network>>> networks_;
    minizero::env::GamePair<std::vector<std::shared_ptr<network::Network>>> discriminator_networks_;
    std::vector<std::vector<std::shared_ptr<network::NetworkOutput>>> network_outputs_;
};

class ArenaThread : public utils::BaseSlaveThread {
public:
    ArenaThread(int id, std::shared_ptr<utils::BaseSharedData> shared_data)
        : BaseSlaveThread(id, shared_data) {}

    void initialize() override;
    void runJob() override;
    bool isDone() override { return false; }

protected:
    virtual bool selection();
    virtual void evaluation();
    virtual bool backup();
    virtual void handleSearchDone(int actor_id);
    inline std::shared_ptr<ArenaSharedData> getSharedData() { return std::static_pointer_cast<ArenaSharedData>(shared_data_); }
};

class Arena : public utils::BaseParalleler {
public:
    Arena() {}

    void run();
    void initialize() override;
    void summarize() override {}

protected:
    virtual void createNeuralNetworks();
    virtual void createActors();

    void createSharedData() override { shared_data_ = std::make_shared<ArenaSharedData>(); }
    std::shared_ptr<utils::BaseSlaveThread> newSlaveThread(int id) override { return std::make_shared<ArenaThread>(id, shared_data_); }
    inline std::shared_ptr<ArenaSharedData> getSharedData() { return std::static_pointer_cast<ArenaSharedData>(shared_data_); }
};

} // namespace minizero::iig

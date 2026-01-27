#include "info_set_generator.h"
#include "environment.h"
#include "info_set_generator_network.h"
#include "rotation.h"
#include <functional>
#include <queue>
#include <vector>

namespace minizero::iig {

using namespace network;
using namespace utils;

std::vector<ISItem> InfoSetGenerator::generate(const Environment& env, int max_info_set_size /*= config::iig_max_infoset_size*/)
{
    std::vector<ISItem> info_set;
    std::priority_queue<ISItem, std::vector<ISItem>, std::function<bool(const ISItem&, const ISItem&)>>
        queue([](const ISItem& a, const ISItem& b) { return a.acc_prob_ < b.acc_prob_; });
    queue.push({Environment(), 1.0f, {}});
    int counter = 0;
    while (!queue.empty() && static_cast<int>(info_set.size()) < max_info_set_size) {
        auto item = queue.top();
        queue.pop();
        float acc_prob = item.acc_prob_;

        // found one possible info state
        if (item.env_.getActionHistory().size() == env.getActionHistory().size()) {
            if (item.env_.getStoneBitboard().get(env.getTurn()) != env.getStoneBitboard().get(env.getTurn())) { continue; }
            info_set.push_back(item);
            continue;
        }

        // if our turn, play true action until reach opponent turn
        while (item.env_.getTurn() == env.getTurn()) {
            const Action& action = env.getActionHistory()[item.env_.getActionHistory().size()];
            if (!item.env_.isLegalAction(action)) { break; }
            item.env_.act(action);
        }
        if (item.env_.getTurn() == env.getTurn()) { continue; } // if still our turn, skip this case (only happens if true action is illegal)

        // create fake environment by playing pass moves for opponent to get features
        Environment fake_env = item.env_;
        for (size_t move = item.env_.getActionHistory().size(); move < env.getActionHistory().size(); ++move) {
            Action action = ((fake_env.getTurn() == env.getTurn())
                                 ? env.getActionHistory()[move]
                                 : Action(env.getBoardSize() * env.getBoardSize(), fake_env.getTurn())); // if not our turn, play pass
            fake_env.act(action);
        }

        std::vector<float> features = fake_env.getInfoSetGeneratorFeatures(item.env_.getActionHistory().size(), Rotation::kRotationNone);
        network_->pushBack(features);
        auto res = network_->forward();
        auto policy_output = std::static_pointer_cast<InfoSetGeneratorNetworkOutput>(res[0]);

        for (size_t pos = 0; pos < policy_output->policy_.size(); ++pos) {
            Action action(pos, item.env_.getTurn());
            if (!item.env_.isLegalAction(action)) { continue; }
            Environment next_env = item.env_;
            next_env.act(action);
            float p = policy_output->policy_[pos];
            if (p < config::iig_generator_policy_threshold) { continue; }
            std::vector<float> new_probs = item.probs_;
            new_probs.push_back(p);
            queue.push({next_env, acc_prob * p, new_probs});
        }
    }

    return info_set;
}
}; // namespace minizero::iig

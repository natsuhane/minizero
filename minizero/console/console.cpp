#include "console.h"
#include "configuration.h"
#include "create_actor.h"
#include "create_network.h"
#include "discriminator_network.h"
#include "sgf_loader.h"
#include "time_system.h"
#include <algorithm>
#include <climits>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

namespace minizero::console {

using namespace network;

Console::Console()
    : network_(nullptr),
      actor_(nullptr)
{
    RegisterFunction("gogui-analyze_commands", this, &Console::cmdGoguiAnalyzeCommands);
    RegisterFunction("list_commands", this, &Console::cmdListCommands);
    RegisterFunction("name", this, &Console::cmdName);
    RegisterFunction("version", this, &Console::cmdVersion);
    RegisterFunction("protocol_version", this, &Console::cmdProtocalVersion);
    RegisterFunction("clear_board", this, &Console::cmdClearBoard);
    RegisterFunction("showboard", this, &Console::cmdShowBoard);
    RegisterFunction("play", this, &Console::cmdPlay);
    RegisterFunction("boardsize", this, &Console::cmdBoardSize);
    RegisterFunction("genmove", this, &Console::cmdGenmove);
    RegisterFunction("reg_genmove", this, &Console::cmdGenmove);
    RegisterFunction("final_score", this, &Console::cmdFinalScore);
    RegisterFunction("pv", this, &Console::cmdPV);
    RegisterFunction("pv_string", this, &Console::cmdPVString);
    RegisterFunction("game_string", this, &Console::cmdGameString);
    RegisterFunction("load_model", this, &Console::cmdLoadModel);
    RegisterFunction("load_game_string", this, &Console::cmdLoadGameString);
    RegisterFunction("s_value", this, &Console::cmdSiameseValue);
    RegisterFunction("get_conf_str", this, &Console::cmdGetConfigString);
}

void Console::initialize()
{
    if (!network_) {
        network_ = createNetwork(config::nn_file_name, 0);
        if (config::iig_use_discriminator) { discriminator_network_ = createNetwork(config::iig_discriminator_file_name, 0); }
    }
    if (!actor_) {
        uint64_t tree_node_size = static_cast<uint64_t>(config::actor_num_simulation + 1) * network_->getActionSize();
        actor_ = actor::createActor(tree_node_size, network_);
    }
    actor_->setNetwork(network_);
    if (config::iig_use_discriminator) { actor_->setNetwork(discriminator_network_); }

    // forward the network several times to warmup since the first few forwards requires some initialization time
    const int num_warmup_forward = 3;
    if (network_->getNetworkTypeName() == "alphazero") {
        std::shared_ptr<network::AlphaZeroNetwork> alphazero_network = std::static_pointer_cast<network::AlphaZeroNetwork>(network_);
        for (int i = 0; i < num_warmup_forward; ++i) {
            for (int j = 0; j < config::actor_mcts_think_batch_size; ++j) { alphazero_network->pushBack(actor_->getEnvironment().getFeatures()); }
            alphazero_network->forward();
        }
    } else if (network_->getNetworkTypeName() == "muzero" || network_->getNetworkTypeName() == "muzero_atari") {
        std::shared_ptr<network::MuZeroNetwork> muzero_network = std::static_pointer_cast<network::MuZeroNetwork>(network_);
        for (int i = 0; i < num_warmup_forward; ++i) {
            for (int j = 0; j < config::actor_mcts_think_batch_size; ++j) { muzero_network->pushBackInitialData(actor_->getEnvironment().getFeatures()); }
            muzero_network->initialInference();
        }
    } else {
        assert(false); // should not be here
    }
}

void Console::executeCommand(std::string command)
{
    if (!network_ || !actor_) { initialize(); }
    if (command.back() == '\r') { command.pop_back(); }
    if (command.empty()) { return; }

    // parse command to args
    std::stringstream ss(command);
    std::string tmp;
    std::vector<std::string> args;
    while (std::getline(ss, tmp, ' ')) { args.push_back(tmp); }

    // save command id if first argument is a number
    command_id_ = "";
    if (!args[0].empty() && args[0].find_first_not_of("0123456789") == std::string::npos) {
        command_id_ = args[0];
        args.erase(args.begin());
    }

    // execute function
    if (function_map_.count(args[0]) == 0) { return reply(ConsoleResponse::kFail, "Unknown command: " + command); }
    (*function_map_[args[0]])(args);
}

void Console::cmdGoguiAnalyzeCommands(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }
    std::string registered_cmd = "sboard/policy_value/pv\n";
    reply(console::ConsoleResponse::kSuccess, registered_cmd);
}

void Console::cmdListCommands(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }
    std::ostringstream oss;
    for (const auto& command : function_map_) { oss << command.first << std::endl; }
    reply(ConsoleResponse::kSuccess, oss.str());
}

void Console::cmdName(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }
    reply(ConsoleResponse::kSuccess, "minizero");
}

void Console::cmdVersion(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }
    reply(ConsoleResponse::kSuccess, "1.0");
}

void Console::cmdProtocalVersion(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }
    reply(ConsoleResponse::kSuccess, "2");
}

void Console::cmdClearBoard(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }
    actor_->reset();
    reply(ConsoleResponse::kSuccess, "");
}

void Console::cmdShowBoard(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }
    reply(ConsoleResponse::kSuccess, "\n" + actor_->getEnvironment().toString());
}

void Console::cmdPlay(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 3, INT_MAX)) { return; }
    std::string action_string = args[2];
    std::vector<std::string> act_args;
    for (unsigned int i = 1; i < args.size(); i++) { act_args.push_back(args[i]); }
    if (!actor_->act(act_args) && !actor_->isEnvTerminal()) { return reply(ConsoleResponse::kFail, "Invalid action: \"" + action_string + "\""); }
    std::cerr << actor_->getEnvironment().toString() << std::endl;
    reply(ConsoleResponse::kSuccess, "");
}

void Console::cmdBoardSize(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 2, 2)) { return; }
    minizero::config::env_board_size = stoi(args[1]);
    initialize();
    reply(ConsoleResponse::kSuccess, "\n" + actor_->getEnvironment().toString());
}

void Console::cmdGenmove(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 2, 2)) { return; }

    if (actor_->isEnvTerminal()) { return reply(ConsoleResponse::kSuccess, "PASS"); }
    actor_->getEnvironment().setTurn(minizero::env::charToPlayer(args[1].c_str()[0]));
    boost::posix_time::ptime start_ptime = utils::TimeSystem::getLocalTime();
    const Action action = actor_->think((args[0] == "genmove" ? true : false), true);
    if (actor_->isEnvTerminal()) { std::cerr << "Game Over!" << std::endl; }
    std::cerr << "Spent Time = " << (utils::TimeSystem::getLocalTime() - start_ptime).total_milliseconds() / 1000.0f << " (s)" << std::endl;
    if (actor_->isResign()) { return reply(ConsoleResponse::kSuccess, "Resign"); }
    std::cerr << actor_->getEnvironment().toString() << std::endl;

    reply(ConsoleResponse::kSuccess, action.toConsoleString());
}

void Console::cmdFinalScore(const std::vector<std::string>& args)
{
    reply(ConsoleResponse::kSuccess, std::to_string(actor_->getEvalScore()));
}

void Console::cmdPV(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }

    float value;
    std::vector<float> policy;
    utils::Rotation rotation = config::actor_use_random_rotation_features ? static_cast<utils::Rotation>(utils::Random::randInt() % static_cast<int>(utils::Rotation::kRotateSize)) : utils::Rotation::kRotationNone;
    calculatePolicyValue(policy, value, rotation);

    const Environment& env_transition = actor_->getEnvironment();
    std::vector<std::pair<std::string, float>> sorted_policy;
    for (size_t action_id = 0; action_id < policy.size(); ++action_id) {
        Action action(action_id, env_transition.getTurn());
        if (!env_transition.isLegalAction(action)) { continue; }
        sorted_policy.push_back(make_pair(action.toConsoleString(), policy[action_id]));
    }

    std::ostringstream oss;
    std::sort(sorted_policy.begin(), sorted_policy.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) { return (a.second > b.second); });
    oss << "[rotation] " << utils::getRotationString(rotation) << std::endl;
    oss << "[policy] ";
    for (size_t i = 0; i < sorted_policy.size(); i++) { oss << sorted_policy[i].first << ": " << std::fixed << std::setprecision(3) << sorted_policy[i].second << " "; }
    oss << std::endl;
    oss << "[value] " << value << std::endl;
    std::cerr << oss.str() << std::endl;

    // for GUI
    oss.str("");
    oss.clear();
    int board_size = minizero::config::env_board_size;
    oss << std::endl;
    for (int row = board_size - 1; row >= 0; row--) {
        for (int col = 0; col < board_size; col++) {
            int action_id = row * board_size + col;
            oss << (env_transition.isLegalAction(Action(action_id, env_transition.getTurn())) ? std::to_string(policy[action_id] * 100).substr(0, 4) + "%" : "\"\"") << " ";
        }
        oss << std::endl;
    }

    reply(ConsoleResponse::kSuccess, oss.str());
}

void Console::cmdPVString(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }

    float value;
    std::vector<float> policy;
    utils::Rotation rotation = config::actor_use_random_rotation_features ? static_cast<utils::Rotation>(utils::Random::randInt() % static_cast<int>(utils::Rotation::kRotateSize)) : utils::Rotation::kRotationNone;
    calculatePolicyValue(policy, value, rotation);

    std::ostringstream oss;
    oss << std::endl;
    oss << "[value] " << value << std::endl;
    const Environment& env_transition = actor_->getEnvironment();
    for (size_t action_id = 0; action_id < policy.size(); ++action_id) {
        Action action(action_id, env_transition.getTurn());
        if (!env_transition.isLegalAction(action)) { continue; }
        oss << action.toConsoleString() << " " << std::to_string(policy[action_id] * 100).substr(0, 4) << " ";
    }
    reply(ConsoleResponse::kSuccess, oss.str());
}

void Console::cmdGameString(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }
    EnvironmentLoader env_loader;
    const Environment& env_transition = actor_->getEnvironment();
    env_loader.loadFromEnvironment(env_transition);
    reply(ConsoleResponse::kSuccess, env_loader.toString());
}

void Console::cmdLoadModel(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 2, 2)) { return; }
    minizero::config::nn_file_name = args[1];
    network_ = nullptr;
    initialize();
    reply(ConsoleResponse::kSuccess, "");
}

void Console::cmdLoadGameString(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 2, 2)) { return; }
    EnvironmentLoader env_loader;
    if (!env_loader.loadFromString(args[1])) { return reply(ConsoleResponse::kFail, "Failed to load game string"); }
    actor_->reset();
    for (const auto& action : env_loader.getActionPairs()) {
        actor_->act(action.first);
        std::cerr << "Played: " << env::playerToChar(action.first.getPlayer()) << " " << action.first.toConsoleString() << std::endl;
        std::cerr << actor_->getEnvironment().toString() << std::endl;
        std::cerr << std::endl;
    }
    reply(ConsoleResponse::kSuccess, "\n" + actor_->getEnvironment().toString());
}

void Console::cmdSiameseValue(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 1, 1)) { return; }

    std::shared_ptr<SiameseNetwork> network = std::static_pointer_cast<SiameseNetwork>(discriminator_network_);
    utils::Rotation rotation = config::actor_use_random_rotation_features ? static_cast<utils::Rotation>(utils::Random::randInt() % static_cast<int>(utils::Rotation::kRotateSize)) : utils::Rotation::kRotationNone;
    std::vector<float> anchor = actor_->getEnvironment().getFeatures(false, rotation);  // 34
    std::vector<float> positive = actor_->getEnvironment().getFeatures(true, rotation); // 4
    network->pushBackAnchor(anchor);
    auto res = network->forward();
    auto anchor_emb = std::static_pointer_cast<SiameseNetworkOutput>(res[0])->embeddings_;

    network->pushBackBoard(positive);
    for (int i = 0; i < 100; ++i) {
        Environment env_copy = actor_->getEnvironment();
        env_copy.sampleOneInformationSet(i);
        std::vector<float> negative = env_copy.getFeatures(true, rotation);
        network->pushBackBoard(negative);
    }
    res = network->forward();
    auto positive_emb = std::static_pointer_cast<SiameseNetworkOutput>(res[0])->embeddings_;
    std::cerr << "Positive Distance: " << utils::distance(anchor_emb, positive_emb) << std::endl
              << std::endl;

    // index, distance
    std::vector<std::pair<int, float>> all_index;
    for (int i = 0; i < 100; ++i) {
        Environment env_copy = actor_->getEnvironment();
        env_copy.sampleOneInformationSet(i);
        auto negative_emb = std::static_pointer_cast<SiameseNetworkOutput>(res[i + 1])->embeddings_;
        float dist = utils::distance(anchor_emb, negative_emb);
        all_index.push_back(std::make_pair(i, dist));
    }
    std::vector<int> top_n = {5, 10, 20, 50, 100};
    for (auto n : top_n) {
        std::cerr << "Top " << n << " Negative Distance: " << std::endl;
        float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::lowest(), sum = 0.0f;
        std::vector<float> tmp;
        for (int i = 0; i < n && i < static_cast<int>(all_index.size()); ++i) {
            float dist = all_index[i].second;
            min = std::min(min, dist);
            max = std::max(max, dist);
            sum += dist;
            tmp.push_back(dist);
        }
        std::cerr << "\tMin: " << min << "\tMax: " << max << "\tAvg: " << sum / n << "\tDev: " << utils::stddev(tmp) << std::endl;
    }

    std::sort(all_index.begin(), all_index.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second < b.second;
    });

    std::vector<std::vector<std::string>> board_str;
    std::vector<std::string> dist_str;
    for (size_t i = 0; i < 5 && i < all_index.size(); ++i) {
        auto negative_emb = std::static_pointer_cast<SiameseNetworkOutput>(res[all_index[i].first + 1])->embeddings_;
        Environment env_copy = actor_->getEnvironment();
        env_copy.sampleOneInformationSet(all_index[i].first);
        board_str.push_back(utils::stringToVector(env_copy.toString(), "\n"));
        dist_str.push_back(std::to_string(utils::distance(anchor_emb, negative_emb)));
    }
    std::ostringstream oss;
    for (size_t i = 0; i < board_str[0].size(); ++i) {
        for (auto& str : board_str) { oss << str[i] << "    "; }
        oss << std::endl;
    }
    oss << "Negative Distance: \n";
    for (auto& str : dist_str) { oss << str << "\t\t\t"; }
    std::cerr << oss.str() << std::endl;

    board_str.clear();
    dist_str.clear();
    for (size_t i = 0; i < 5 && i < all_index.size(); ++i) {
        auto negative_emb = std::static_pointer_cast<SiameseNetworkOutput>(res[all_index[all_index.size() - 1 - i].first + 1])->embeddings_;
        Environment env_copy = actor_->getEnvironment();
        env_copy.sampleOneInformationSet(all_index[all_index.size() - 1 - i].first);
        board_str.push_back(utils::stringToVector(env_copy.toString(), "\n"));
        dist_str.push_back(std::to_string(utils::distance(anchor_emb, negative_emb)));
    }
    oss.str("");
    for (size_t i = 0; i < board_str[0].size(); ++i) {
        for (auto& str : board_str) { oss << str[i] << "    "; }
        oss << std::endl;
    }
    oss << "Negative Distance: \n";
    for (auto& str : dist_str) { oss << str << "\t\t\t"; }
    std::cerr << oss.str() << std::endl;
    reply(ConsoleResponse::kSuccess, "");
}

void Console::cmdGetConfigString(const std::vector<std::string>& args)
{
    if (!checkArgument(args, 2, 2)) { return; }
    std::ostringstream oss;
    config::ConfigureLoader cl;
    config::setConfiguration(cl);
    oss << std::endl;
    for (auto& conf_key : utils::stringToVector(args[1], ":")) { oss << cl.getConfig(conf_key); }
    reply(ConsoleResponse::kSuccess, oss.str());
}

void Console::calculatePolicyValue(std::vector<float>& policy, float& value, utils::Rotation rotation /* = utils::Rotation::kRotationNone */)
{
    if (network_->getNetworkTypeName() == "alphazero") {
        std::shared_ptr<network::AlphaZeroNetwork> alphazero_network = std::static_pointer_cast<network::AlphaZeroNetwork>(network_);
        int index = alphazero_network->pushBack(actor_->getEnvironment().getFeatures(rotation));
        std::shared_ptr<NetworkOutput> network_output = alphazero_network->forward()[index];
        std::shared_ptr<minizero::network::AlphaZeroNetworkOutput> zero_output = std::static_pointer_cast<minizero::network::AlphaZeroNetworkOutput>(network_output);
        value = zero_output->value_;
        policy.clear();
        for (size_t action_id = 0; action_id < zero_output->policy_.size(); ++action_id) {
            int rotated_id = actor_->getEnvironment().getRotateAction(action_id, rotation);
            policy.push_back(zero_output->policy_[rotated_id]);
        }
    } else if (network_->getNetworkTypeName() == "muzero" || network_->getNetworkTypeName() == "muzero_atari") {
        std::shared_ptr<network::MuZeroNetwork> muzero_network = std::static_pointer_cast<network::MuZeroNetwork>(network_);
        int index = muzero_network->pushBackInitialData(actor_->getEnvironment().getFeatures());
        std::shared_ptr<NetworkOutput> network_output = muzero_network->initialInference()[index];
        std::shared_ptr<minizero::network::MuZeroNetworkOutput> zero_output = std::static_pointer_cast<minizero::network::MuZeroNetworkOutput>(network_output);
        policy = zero_output->policy_;
        value = zero_output->value_;
    } else {
        assert(false); // should not be here
    }
}

bool Console::checkArgument(const std::vector<std::string>& args, int min_argc, int max_argc)
{
    int size = args.size();
    if (size >= min_argc && size <= max_argc) { return true; }

    std::ostringstream oss;
    oss << "command requires ";
    if (min_argc == max_argc) {
        oss << "exactly " << min_argc << " argument" << (min_argc == 1 ? "" : "s");
    } else {
        oss << min_argc << " to " << max_argc << " arguments";
    }

    reply(ConsoleResponse::kFail, oss.str());
    return false;
}

void Console::reply(ConsoleResponse response, const std::string& reply)
{
    std::cout << static_cast<char>(response) << command_id_ << " " << reply << "\n\n";
}

} // namespace minizero::console

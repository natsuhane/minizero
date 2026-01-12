#include "mode_handler.h"
#include "actor_group.h"
#include "alphazero_network.h"
#include "console.h"
#include "create_network.h"
#include "git_info.h"
#include "iig_data_generator.h"
#include "obs_recover.h"
#include "obs_remover.h"
#include "ostream_redirector.h"
#include "random.h"
#include "sgf_loader.h"
#include "time_system.h"
#include "utils.h"
#include "zero_server.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace minizero::console {

using namespace minizero::utils;
using namespace minizero::network;

ModeHandler::ModeHandler()
{
    RegisterFunction("console", this, &ModeHandler::runConsole);
    RegisterFunction("sp", this, &ModeHandler::runSelfPlay);
    RegisterFunction("zero_server", this, &ModeHandler::runZeroServer);
    RegisterFunction("zero_training_name", this, &ModeHandler::runZeroTrainingName);
    RegisterFunction("env_test", this, &ModeHandler::runEnvTest);
    RegisterFunction("remove_obs", this, &ModeHandler::runRemoveObs);
    RegisterFunction("recover_obs", this, &ModeHandler::runRecoverObs);
    RegisterFunction("run", this, &ModeHandler::runDataSet);
    RegisterFunction("visualize_sgf", this, &ModeHandler::runVisualizeSgf);
}

void ModeHandler::run(int argc, char* argv[])
{
    if (argc % 2 != 1) { usage(); }

    env::setUpEnv();

    std::string mode_string = "console";
    std::string config_file = "";
    std::string config_string = "";
    config::ConfigureLoader cl;
    setDefaultConfiguration(cl);

    std::string gen_config = "";
    for (int i = 1; i < argc; i += 2) {
        std::string sCommand = std::string(argv[i]);

        if (sCommand == "-mode") {
            mode_string = argv[i + 1];
        } else if (sCommand == "-gen") {
            gen_config = argv[i + 1];
        } else if (sCommand == "-conf_file") {
            config_file = argv[i + 1];
        } else if (sCommand == "-conf_str") {
            config_string = argv[i + 1];
        } else {
            std::cerr << "unknown argument: " << sCommand << std::endl;
            usage();
        }
    }

    if (!readConfiguration(cl, config_file, config_string)) { exit(-1); }
    utils::OstreamRedirector::silence(std::cerr, config::program_quiet);                                  // silence std::cerr if program_quiet
    utils::Random::seed(config::program_auto_seed ? static_cast<int>(time(NULL)) : config::program_seed); // setup random seed

    if (!gen_config.empty()) {
        // generate configuration file after reading cfg file
        genConfiguration(cl, gen_config);
        exit(0);
    } else {
        std::cerr << "(Version: " << GIT_SHORT_HASH << ")" << std::endl;
        // run mode
        if (!function_map_.count(mode_string)) { usage(); }
        (*function_map_[mode_string])();
    }
}

void ModeHandler::usage()
{
    std::cout << "./minizero [arguments]" << std::endl;
    std::cout << "arguments:" << std::endl;
    std::cout << "\t-mode [" << getAllModesString() << "]" << std::endl;
    std::cout << "\t-gen configuration_file" << std::endl;
    std::cout << "\t-conf_file configuration_file" << std::endl;
    std::cout << "\t-conf_str configuration_string" << std::endl;
    exit(-1);
}

std::string ModeHandler::getAllModesString()
{
    std::string mode_string;
    for (const auto& m : function_map_) { mode_string += (mode_string.empty() ? "" : "|") + m.first; }
    return mode_string;
}

void ModeHandler::genConfiguration(config::ConfigureLoader& cl, const std::string& config_file)
{
    // check configure file is exist
    std::ifstream f(config_file);
    if (f.good()) {
        char ans = ' ';
        while (ans != 'y' && ans != 'n') {
            std::cerr << config_file << " already exist, do you want to overwrite it? [y/n]" << std::endl;
            std::cin >> ans;
        }
        if (ans == 'y') { std::cerr << "overwrite " << config_file << std::endl; }
        if (ans == 'n') {
            std::cerr << "didn't overwrite " << config_file << std::endl;
            f.close();
            return;
        }
    }
    f.close();

    std::ofstream fout(config_file);
    fout << cl.toString();
    fout.close();
}

bool ModeHandler::readConfiguration(config::ConfigureLoader& cl, const std::string& config_file, const std::string& config_string)
{
    if (!config_file.empty() && !cl.loadFromFile(config_file)) {
        std::cerr << "Failed to load configuration file." << std::endl;
        return false;
    }
    if (!config_string.empty() && !cl.loadFromString(config_string)) {
        std::cerr << "Failed to load configuration string." << std::endl;
        return false;
    }

    if (!config::program_quiet) { std::cerr << cl.toString(); }
    return true;
}

void ModeHandler::runConsole()
{
    console::Console console;
    std::string command;
    console.initialize();
    std::cerr << "Successfully started console mode" << std::endl;
    while (getline(std::cin, command)) {
        if (command == "quit") { break; }
        console.executeCommand(command);
    }
}

void ModeHandler::runSelfPlay()
{
    actor::ActorGroup ag;
    ag.run();
}

void ModeHandler::runZeroServer()
{
    zero::ZeroServer server;
    server.run();
}

void ModeHandler::runZeroTrainingName()
{
    std::cout << Environment().name()                      // name for environment
              << "_" << config::nn_type_name               // network & training algorithm
              << "_" << config::nn_num_blocks << "b"       // number of blocks
              << "x" << config::nn_num_hidden_channels     // number of hidden channels
              << "_k" << config::siamese_max_num_negatives // siamese max random perturbations
              << "-" << GIT_SHORT_HASH << std::endl;       // git hash info
}

void ModeHandler::runEnvTest()
{
    Environment env;
    env.reset();
    while (!env.isTerminal()) {
        std::vector<Action> legal_actions = env.getLegalActions();
        int index = utils::Random::randInt() % legal_actions.size();
        env.act(legal_actions[index]);
    }
    std::cout << env.toString() << std::endl;

    EnvironmentLoader env_loader;
    env_loader.loadFromEnvironment(env);
    std::cout << env_loader.toString() << std::endl;
}

void ModeHandler::runRemoveObs()
{
    std::string obs_file_path;
    std::cin >> obs_file_path;

    minizero::env::atari::ObsRemover ob;
    ob.initialize();
    ob.run(obs_file_path);
}

void ModeHandler::runRecoverObs()
{
    std::string obs_file_path;
    std::cin >> obs_file_path;

#if ATARI
    minizero::env::atari::ObsRecover ob;
    ob.initialize();
    ob.run(obs_file_path);
#else
    std::cout << "Currently, only support recover observation for atari games" << std::endl;
#endif
}

void ModeHandler::runDataSet()
{
    iig_data_generator::IIGDataGenerator data_generator;
    data_generator.run();
    return;
}

// visualize sgf with positive and negative samples
void ModeHandler::runVisualizeSgf()
{
    // find target game by id
    EnvironmentLoader env_loader;
    int target_game_id = config::siamese_game_id;
    std::ifstream fin(config::siamese_visualizer_input_sgf);
    std::cerr << "Searching for game id " << target_game_id << std::endl;
    for (std::string sgf; std::getline(fin, sgf);) {
        if (!env_loader.loadFromString(sgf) || std::stoi(env_loader.getTag("I")) != target_game_id) { continue; }
        break;
    }

    // positive sgf
    Environment env;
    std::string positive_sgf;
    std::vector<std::string> sgf_outputs;
    int target_game_step = config::siamese_game_step;
    const std::string sgf_prefix = "<div data-wgo=\"(;FF[4]GM[1]SZ[9]KM[7.000000]";
    const std::string sgf_suffix = ")\" data-wgo-layout=\"\"  data-wgo-move=\"100\" style=\"width: 10%; margin: 0\"></div>";
    std::shared_ptr<AlphaZeroNetwork> az_network = std::static_pointer_cast<AlphaZeroNetwork>(createNetwork(config::nn_file_name, 0));
    std::cerr << "Loaded environment up to step " << target_game_step << std::endl;
    for (int pos = 0; pos < target_game_step; ++pos) {
        env.act(env_loader.getActionPairs()[pos].first);
        positive_sgf += ";" +
                        std::string(1, env::playerToChar(env_loader.getActionPairs()[pos].first.getPlayer())) +
                        "[" +
                        utils::SGFLoader::actionIDToSGFString(env_loader.getActionPairs()[pos].first.getActionID(), env_loader.getBoardSize()) +
                        "]";
    }
    az_network->pushBack(env.getFeatures());
    sgf_outputs.push_back(sgf_prefix + positive_sgf + sgf_suffix);

    // negative sgfs
    std::vector<std::string> ids_str_01 = utils::stringToVector(env_loader.getActionPairs()[target_game_step - 1].second["N1"], ",");
    std::vector<std::string> ids_str_02 = utils::stringToVector(env_loader.getActionPairs()[target_game_step - 1].second["N2"], ",");
    ids_str_01.insert(ids_str_01.end(), ids_str_02.begin(), ids_str_02.end());
    auto neg_bitboards = env_loader.generateNegativeBitboards(env, config::siamese_max_random_perturbations, true);
    for (const auto& id_str : ids_str_01) {
        int id = std::stoi(id_str);
        std::string negative_sgf;
        std::vector<env::Player> players = {env::Player::kPlayer1, env::Player::kPlayer2};
        for (const auto& player : players) {
            env::go::GoBitboard bitboard = neg_bitboards[id].get(player);
            if (bitboard.none()) { continue; }
            negative_sgf += "A" + std::string(1, env::playerToChar(player));
            while (!bitboard.none()) {
                int p = bitboard._Find_first();
                bitboard.reset(p);
                negative_sgf += "[" + utils::SGFLoader::actionIDToSGFString(p, env_loader.getBoardSize()) + "]";
            }
        }
        az_network->pushBack(env_loader.bitboardToFeature(neg_bitboards[id], env.getTurn(), utils::Rotation::kRotationNone, true));
        sgf_outputs.push_back(sgf_prefix + negative_sgf + sgf_suffix);
    }

    // get values & output all sgfs
    auto network_output = az_network->forward();
    std::ofstream fout("visualizer/index.html");
    fout << "<!DOCTYPE HTML><html><head><meta charset=\"utf-8\"><title>WGo</title><script type=\"text/javascript\" src=\"wgo.js/wgo/wgo.js\"></script><script type=\"text/javascript\" src=\"wgo.js/wgo/kifu.js\"></script><script type=\"text/javascript\" src=\"wgo.js/wgo/sgfparser.js\"></script><script type=\"text/javascript\" src=\"wgo.js/wgo/player.js\"></script><script type=\"text/javascript\" src=\"wgo.js/wgo/basicplayer.js\"></script><script type=\"text/javascript\" src=\"wgo.js/wgo/basicplayer.component.js\"></script><script type=\"text/javascript\" src=\"wgo.js/wgo/basicplayer.infobox.js\"></script><script type=\"text/javascript\" src=\"wgo.js/wgo/basicplayer.commentbox.js\"></script><script type=\"text/javascript\" src=\"wgo.js/wgo/basicplayer.control.js\"></script><link rel=\"stylesheet\" type=\"text/css\" href=\"wgo.js/wgo/wgo.player.css\" /></head><body><div style=\"display: flex; flex-wrap: wrap; width: 100%;\">";
    float postive_value = std::static_pointer_cast<AlphaZeroNetworkOutput>(network_output[0])->value_;
    for (size_t i = 0; i < sgf_outputs.size(); ++i) {
        fout << sgf_outputs[i] << std::endl;

        // output values
        if (i % 10 != 9 && i != sgf_outputs.size() - 1) { continue; }
        for (size_t j = i - i % 10; j <= i && j < sgf_outputs.size(); ++j) {
            fout << "<div style=\"width: 10%; margin: 0; text-align: center;\">"
                 << std::fixed << std::setprecision(3)
                 << std::static_pointer_cast<AlphaZeroNetworkOutput>(network_output[j])->value_
                 << "(" << std::static_pointer_cast<AlphaZeroNetworkOutput>(network_output[j])->value_ - postive_value << ")"
                 << "</div>" << std::endl;
        }
    }
    fout << "</body></html>";
    fout.close();
}

} // namespace minizero::console

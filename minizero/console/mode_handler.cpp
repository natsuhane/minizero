#include "mode_handler.h"
#include "actor_group.h"
#include "alphazero_network.h"
#include "console.h"
#include "create_network.h"
#include "data_generator.h"
#include "evaluator.h"
#include "git_info.h"
#include "data_generator.h"
#include "info_set_generator.h"
#include "info_set_generator_network.h"
#include "obs_recover.h"
#include "obs_remover.h"
#include "ostream_redirector.h"
#include "random.h"
#include "sgf_loader.h"
#include "siamese_network.h"
#include "time_system.h"
#include "utils.h"
#include "zero_server.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
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
    RegisterFunction("evaluator", this, &ModeHandler::runEvaluator);
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
    std::cout << Environment().name()                  // name for environment
              << "_" << config::nn_type_name           // network & training algorithm
              << "_" << config::nn_num_blocks << "b"   // number of blocks
              << "x" << config::nn_num_hidden_channels // number of hidden channels
              << "_k" << config::iig_max_infoset_size  // siamese max random perturbations
              << "-" << GIT_SHORT_HASH << std::endl;   // git hash info
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
    iig::DataGenerator data_generator;
    data_generator.run();
    return;
}

struct ISQueueItem {
    Environment env_;
    float acc_prob_;
    std::vector<float> probs_;
};

// for info set generator
void ModeHandler::runVisualizeSgf()
{
    // find target game by id
    EnvironmentLoader env_loader;
    int target_game_id = config::iig_game_id;
    std::ifstream fin(config::iig_visualizer_input_sgf);
    std::cerr << "Searching for game id " << target_game_id << std::endl;
    for (std::string sgf; std::getline(fin, sgf);) {
        if (!env_loader.loadFromString(sgf) || std::stoi(env_loader.getTag("I")) != target_game_id) { continue; }
        break;
    }

    Environment true_env;
    std::vector<std::string> sgf_outputs;
    std::vector<float> prob_outputs;
    std::vector<std::vector<float>> prob_outputs_per_step;

    // true board
    std::cerr << "Loaded environment up to step " << config::iig_game_step << std::endl;
    for (int pos = 0; pos < config::iig_game_step; ++pos) { true_env.act(env_loader.getActionPairs()[pos].first); }
    sgf_outputs.push_back(true_env.toSGFString());
    prob_outputs.push_back(0.0f);
    prob_outputs_per_step.push_back(std::vector<float>());

    // info set
    iig::InfoSetGenerator is_generator(createNetwork(config::iig_nn_file_name, 0));
    std::vector<iig::ISItem> info_set = is_generator.generate(true_env);
    int true_board_id = 0;
    for (const auto& item : info_set) {
        sgf_outputs.push_back(item.env_.toSGFString());
            if (sgf_outputs.back() == sgf_outputs[0]) { true_board_id = sgf_outputs.size() - 1; }
        prob_outputs.push_back(item.acc_prob_);
        prob_outputs_per_step.push_back(item.probs_);
    }

    // get values & output all sgfs
    std::ostringstream board_oss, value_oss;
    for (size_t i = 0; i < sgf_outputs.size(); ++i) {
        std::string prob_per_steps;
        for (size_t j = 0; j < prob_outputs_per_step[i].size(); ++j) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(3) << prob_outputs_per_step[i][j];
            prob_per_steps += (j == 0 ? "" : ", ") + stream.str();
        }
        board_oss << "\"" << sgf_outputs[i] << "\"," << std::endl;
        value_oss << "\"" << std::fixed << std::setprecision(6)
                  << i << ": "
                  << prob_outputs[i]
                  << " (per step: [" << prob_per_steps << "])"
                  << "\"," << std::endl;
    }

    // read template html
    std::ifstream html_fin("visualizer/template.html");
    std::string line, template_html;
    while (std::getline(html_fin, line)) { template_html += line + "\n"; }
    html_fin.close();

    // replace template strings and output index.html
    template_html = template_html.replace(template_html.find("GAME_ID"), std::string("GAME_ID").length(), std::to_string(config::iig_game_id));
    template_html = template_html.replace(template_html.find("STEP_ID"), std::string("STEP_ID").length(), std::to_string(config::iig_game_step));
    template_html = template_html.replace(template_html.find("TOTAL_GAMES"), std::string("TOTAL_GAMES").length(), std::to_string(std::max(0, static_cast<int>(sgf_outputs.size()) - 1)));
    template_html = template_html.replace(template_html.find("COMMENT"), std::string("COMMENT").length(), (true_board_id == 0 ? "" : "True board at #" + std::to_string(true_board_id)));
    template_html = template_html.replace(template_html.find("BOARD_STR"), std::string("BOARD_STR").length(), board_oss.str());
    template_html = template_html.replace(template_html.find("VALUE_STR"), std::string("VALUE_STR").length(), value_oss.str());
    std::ofstream fout("visualizer/index.html");
    fout << template_html;
    fout.close();
}

void ModeHandler::runEvaluator()
{
    iig::Evaluator evaluator;
    evaluator.run();
}

} // namespace minizero::console

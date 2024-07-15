/*
  HypnoS, a UCI chess playing engine derived from Stockfish
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

  HypnoS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  HypnoS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "engine.h"

#include <cassert>
#include <deque>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "evaluate.h"
#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_common.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "syzygy/tbprobe.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"
#include "learn/learn.h"
#include "book/book.h"

namespace Hypnos {

namespace NN = Eval::NNUE;

constexpr auto StartFEN  = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr int  MaxHashMB = Is64Bit ? 33554432 : 2048;

Engine::Engine(std::string path) :
    binaryDirectory(
      CommandLine::get_binary_directory(path, CommandLine::get_working_directory())),
    numaContext(NumaConfig::from_system()),
    states(new std::deque<StateInfo>(1)),
    threads(),
    networks(
      numaContext,
      NN::Networks(
        NN::NetworkBig({EvalFileDefaultNameBig, "None", ""}, NN::EmbeddedNNUEType::BIG),
        NN::NetworkSmall({EvalFileDefaultNameSmall, "None", ""}, NN::EmbeddedNNUEType::SMALL))) {
    pos.set(StartFEN, false, &states->back());
    capSq = SQ_NONE;

    options["Debug Log File"] << Option("", [](const Option& o) -> std::optional<std::string> {
        start_logger(o);
        return std::nullopt;
    });

    options["NumaPolicy"] << Option("auto", [this](const Option& o) -> std::optional<std::string> {
        set_numa_config_from_option(o);
        return numa_config_information_as_string() + "\n" + thread_binding_information_as_string();
    });

    options["Threads"] << Option(1, 1, 1024, [this](const Option&) -> std::optional<std::string> {
        resize_threads();
        return thread_binding_information_as_string();
    });

    options["Hash"] << Option(16, 1, MaxHashMB, [this](const Option& o) -> std::optional<std::string> {
        set_tt_size(o);
        return std::nullopt;
    });

    options["Clear Hash"] << Option([this](const Option&) -> std::optional<std::string> {
        search_clear();
        return std::nullopt;
    });
    options["Ponder"] << Option(false);
    options["MultiPV"] << Option(1, 1, MAX_MOVES);
    options["Skill Level"] << Option(20, 0, 20);
    options["MoveOverhead"] << Option(10, 0, 5000);
    options["Minimum Thinking Time"] << Option(100, 0, 5000);
    options["Slow Mover"] << Option(100, 10, 1000);
    options["nodestime"] << Option(0, 0, 10000);
    options["UCI_Chess960"] << Option(false);
    options["UCI_LimitStrength"] << Option(false);
    options["UCI_Elo"] << Option(1320, 1320, 3190);
    options["UCI_ShowWDL"] << Option(true);

    for (int i = 0; i < BookManager::NumberOfBooks; ++i)
    {
        options[Util::format_string("CTG/BIN Book %d File", i + 1)]
          << Option(EMPTY, [this, i](const Option&) -> std::optional<std::string> {
                 init_bookMan(i);
                 return std::nullopt;
             });
        options[Util::format_string("Book %d Width", i + 1)] << Option(1, 1, 20);
        options[Util::format_string("Book %d Depth", i + 1)] << Option(255, 1, 255);
        options[Util::format_string("(CTG) Book %d Only Green", i + 1)] << Option(true);
    }
    options["SyzygyPath"] << Option("", [](const Option& o) -> std::optional<std::string> {
        Tablebases::init(o);
        return std::nullopt;
    });
    options["SyzygyProbeDepth"] << Option(1, 1, 100);
    options["Syzygy50MoveRule"] << Option(true);
    options["SyzygyProbeLimit"] << Option(7, 0, 7);
    options["EvalFile"] << Option(EvalFileDefaultNameBig, [this](const Option& o) -> std::optional<std::string> {
        load_big_network(o);
        return std::nullopt;
    });
    options["EvalFileSmall"] << Option(EvalFileDefaultNameSmall, [this](const Option& o) -> std::optional<std::string> {
        load_small_network(o);
        return std::nullopt;
    });
    options["Read only learning"] << Option(false, [this](const Option& o) -> std::optional<std::string> {
        LD.set_readonly(o);
        return std::nullopt;
    });
    options["Learning"] << Option("Off var Off var Standard var Self", "Off",
                                            [this](const Option& o) -> std::optional<std::string> {
                                                if (!(o == "Off"))
                                                {
                                                    LD.set_learning_mode(options, o);
                                                }
                                                return std::nullopt;
                                            });

    options["SmartMultiPVMode"] << Option(false);	
    options["Materialistic Evaluation Strategy"] << Option(0, -12, 12, [this](const Option& o) -> std::optional<std::string> {
        Eval::NNUE::MaterialisticEvaluationStrategy = 10 * (int) o;
        return std::nullopt;
    });
    options["Positional Evaluation Strategy"] << Option(0, -12, 12, [this](const Option& o) -> std::optional<std::string> {
        Eval::NNUE::PositionalEvaluationStrategy = 10 * (int) o;
        return std::nullopt;
    });

    options["Variety"] << Option("Off var Off var Standard var Aggressiveness", "Off",
                                 [this](const Option& o) -> std::optional<std::string> {
                                     Search::set_variety(o);
                                     return std::optional<std::string>{};
                                 });
    options["Concurrent Experience"]
      << Option(false);  //for a same experience file on a same folder
    load_networks();
    resize_threads();
}

std::uint64_t Engine::perft(const std::string& fen, Depth depth, bool isChess960) {
    verify_networks();

    return Benchmark::perft(fen, depth, isChess960);
}

void Engine::go(Search::LimitsType& limits) {
    assert(limits.perft == 0);
    verify_networks();
    limits.capSq = capSq;

    threads.start_thinking(options, pos, states, limits);
}
void Engine::stop() { threads.stop = true; }

void Engine::search_clear() {
    wait_for_search_finished();

    tt.clear(threads);
    threads.clear();

    // @TODO wont work with multiple instances
    Tablebases::init(options["SyzygyPath"]);  // Free mapped files
}

void Engine::set_on_update_no_moves(std::function<void(const Engine::InfoShort&)>&& f) {
    updateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_update_full(std::function<void(const Engine::InfoFull&)>&& f) {
    updateContext.onUpdateFull = std::move(f);
}

void Engine::set_on_iter(std::function<void(const Engine::InfoIter&)>&& f) {
    updateContext.onIter = std::move(f);
}

void Engine::set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) {
    updateContext.onBestmove = std::move(f);
}

void Engine::wait_for_search_finished() { threads.main_thread()->wait_for_search_finished(); }

void Engine::set_position(const std::string& fen, const std::vector<std::string>& moves) {
    // Drop the old state and create a new one
    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(fen, options["UCI_Chess960"], &states->back());

    capSq = SQ_NONE;
    for (const auto& move : moves)
    {
        auto m = UCIEngine::to_move(pos, move);

        if (m == Move::none())
            break;
        if (LD.is_enabled() && LD.learning_mode() != LearningMode::Self && !LD.is_paused())
        {
            PersistedLearningMove persistedLearningMove;

            persistedLearningMove.key                      = pos.key();
            persistedLearningMove.learningMove.depth       = 0;
            persistedLearningMove.learningMove.move        = m;
            persistedLearningMove.learningMove.score       = VALUE_NONE;
            persistedLearningMove.learningMove.performance = 100;

            LD.add_new_learning(persistedLearningMove.key, persistedLearningMove.learningMove);
        }
        states->emplace_back();
        pos.do_move(m, states->back());

        capSq          = SQ_NONE;
        DirtyPiece& dp = states->back().dirtyPiece;
        if (dp.dirty_num > 1 && dp.to[1] == SQ_NONE)
            capSq = m.to_sq();
    }
}

// modifiers

void Engine::set_numa_config_from_option(const std::string& o) {
    if (o == "auto" || o == "system")
    {
        numaContext.set_numa_config(NumaConfig::from_system());
    }
    else if (o == "hardware")
    {
        // Don't respect affinity set in the system.
        numaContext.set_numa_config(NumaConfig::from_system(false));
    }
    else if (o == "none")
    {
        numaContext.set_numa_config(NumaConfig{});
    }
    else
    {
        numaContext.set_numa_config(NumaConfig::from_string(o));
    }

    // Force reallocation of threads in case affinities need to change.
    resize_threads();
}

void Engine::resize_threads() {
    threads.wait_for_search_finished();
    threads.set(numaContext.get_numa_config(), {bookMan, options, threads, tt, networks},
                updateContext);

    // Reallocate the hash with the new threadpool size
    set_tt_size(options["Hash"]);
}
void Engine::init_bookMan(int bookIndex) { bookMan.init(bookIndex, options); }

void Engine::set_tt_size(size_t mb) {
    wait_for_search_finished();
    tt.resize(mb, threads);
}

void Engine::set_ponderhit(bool b) { threads.main_manager()->ponder = b; }

// network related

void Engine::verify_networks() const {
    networks->big.verify(options["EvalFile"]);
    networks->small.verify(options["EvalFileSmall"]);
}

void Engine::load_networks() {
    networks.modify_and_replicate([this](NN::Networks& networks_) {
        networks_.big.load(binaryDirectory, options["EvalFile"]);
        networks_.small.load(binaryDirectory, options["EvalFileSmall"]);
    });
    threads.clear();
}

void Engine::load_big_network(const std::string& file) {
    networks.modify_and_replicate(
      [this, &file](NN::Networks& networks_) { networks_.big.load(binaryDirectory, file); });
    threads.clear();
}

void Engine::load_small_network(const std::string& file) {
    networks.modify_and_replicate(
      [this, &file](NN::Networks& networks_) { networks_.small.load(binaryDirectory, file); });
    threads.clear();
}

void Engine::save_network(const std::pair<std::optional<std::string>, std::string> files[2]) {
    networks.modify_and_replicate([&files](NN::Networks& networks_) {
        networks_.big.save(files[0].first);
        networks_.small.save(files[1].first);
    });
}

// utility functions

void Engine::trace_eval() const {
    StateListPtr trace_states(new std::deque<StateInfo>(1));
    Position     p;
    p.set(pos.fen(), options["UCI_Chess960"], &trace_states->back());

    verify_networks();

    sync_cout << "\n" << Eval::trace(p, *networks) << sync_endl;
}

const OptionsMap& Engine::get_options() const { return options; }
OptionsMap&       Engine::get_options() { return options; }

std::string Engine::fen() const { return pos.fen(); }

void Engine::flip() { pos.flip(); }
void Engine::show_moves_bookMan(const Position& position) {
    bookMan.show_moves(position, options);
}
std::string Engine::visualize() const {
    std::stringstream ss;
    ss << pos;
    return ss.str();
}

std::vector<std::pair<size_t, size_t>> Engine::get_bound_thread_count_by_numa_node() const {
    auto                                   counts = threads.get_bound_thread_count_by_numa_node();
    const NumaConfig&                      cfg    = numaContext.get_numa_config();
    std::vector<std::pair<size_t, size_t>> ratios;
    NumaIndex                              n = 0;
    for (; n < counts.size(); ++n)
        ratios.emplace_back(counts[n], cfg.num_cpus_in_numa_node(n));
    if (!counts.empty())
        for (; n < cfg.num_numa_nodes(); ++n)
            ratios.emplace_back(0, cfg.num_cpus_in_numa_node(n));
    return ratios;
}

std::string Engine::get_numa_config_as_string() const {
    return numaContext.get_numa_config().to_string();
}

std::string Engine::numa_config_information_as_string() const {
    auto cfgStr = get_numa_config_as_string();
    return "Available processors: " + cfgStr;
}

std::string Engine::thread_binding_information_as_string() const {
    auto              boundThreadsByNode = get_bound_thread_count_by_numa_node();
    std::stringstream ss;

    size_t threadsSize = threads.size();
    ss << "Using " << threadsSize << (threadsSize > 1 ? " threads" : " thread");

    if (boundThreadsByNode.empty())
        return ss.str();

    ss << " with NUMA node thread binding: ";

    bool isFirst = true;

    for (auto&& [current, total] : boundThreadsByNode)
    {
        if (!isFirst)
            ss << ":";
        ss << current << "/" << total;
        isFirst = false;
    }

    return ss.str();
}

}

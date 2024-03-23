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

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "incbin/incbin.h"
#include "misc.h"
#include "nnue/evaluate_nnue.h"
#include "nnue/nnue_architecture.h"
#include "position.h"
#include "thread.h"
#include "types.h"
#include "uci.h"

// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.
#if !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUEBig, EvalFileDefaultNameBig);
INCBIN(EmbeddedNNUESmall, EvalFileDefaultNameSmall);
#else
const unsigned char        gEmbeddedNNUEBigData[1]   = {0x0};
const unsigned char* const gEmbeddedNNUEBigEnd       = &gEmbeddedNNUEBigData[1];
const unsigned int         gEmbeddedNNUEBigSize      = 1;
const unsigned char        gEmbeddedNNUESmallData[1] = {0x0};
const unsigned char* const gEmbeddedNNUESmallEnd     = &gEmbeddedNNUESmallData[1];
const unsigned int         gEmbeddedNNUESmallSize    = 1;
#endif


namespace Stockfish {

namespace Eval {

std::unordered_map<NNUE::NetSize, EvalFile> EvalFiles = {
  {NNUE::Big, {"EvalFile", EvalFileDefaultNameBig, "None"}},
  {NNUE::Small, {"EvalFileSmall", EvalFileDefaultNameSmall, "None"}}};


// Tries to load a NNUE network at startup time, or when the engine
// receives a UCI command "setoption name EvalFile value nn-[a-z0-9]{12}.nnue"
// The name of the NNUE network is always retrieved from the EvalFile option.
// We search the given network in three locations: internally (the default
// network may be embedded in the binary), in the active working directory and
// in the engine directory. Distro packagers may define the DEFAULT_NNUE_DIRECTORY
// variable to have the engine search in a special directory in their distro.
void NNUE::init() {

    for (auto& [netSize, evalFile] : EvalFiles)
    {
        // Replace with
        // Options[evalFile.option_name]
        // once fishtest supports the uci option EvalFileSmall
        std::string user_eval_file =
          netSize == Small ? evalFile.default_name : Options[evalFile.option_name];

        if (user_eval_file.empty())
            user_eval_file = evalFile.default_name;

#if defined(DEFAULT_NNUE_DIRECTORY)
        std::vector<std::string> dirs = {"<internal>", "", CommandLine::binaryDirectory,
                                         stringify(DEFAULT_NNUE_DIRECTORY)};
#else
        std::vector<std::string> dirs = {"<internal>", "", CommandLine::binaryDirectory};
#endif

        for (const std::string& directory : dirs)
        {
            if (evalFile.selected_name != user_eval_file)
            {
                if (directory != "<internal>")
                {
                    std::ifstream stream(directory + user_eval_file, std::ios::binary);
                    if (NNUE::load_eval(user_eval_file, stream, netSize))
                        evalFile.selected_name = user_eval_file;
                }

                if (directory == "<internal>" && user_eval_file == evalFile.default_name)
                {
                    // C++ way to prepare a buffer for a memory stream
                    class MemoryBuffer: public std::basic_streambuf<char> {
                       public:
                        MemoryBuffer(char* p, size_t n) {
                            setg(p, p, p + n);
                            setp(p, p + n);
                        }
                    };

                    MemoryBuffer buffer(
                      const_cast<char*>(reinterpret_cast<const char*>(
                        netSize == Small ? gEmbeddedNNUESmallData : gEmbeddedNNUEBigData)),
                      size_t(netSize == Small ? gEmbeddedNNUESmallSize : gEmbeddedNNUEBigSize));
                    (void) gEmbeddedNNUEBigEnd;  // Silence warning on unused variable
                    (void) gEmbeddedNNUESmallEnd;

                    std::istream stream(&buffer);
                    if (NNUE::load_eval(user_eval_file, stream, netSize))
                        evalFile.selected_name = user_eval_file;
                }
            }
        }
    }
}

// Verifies that the last net used was loaded successfully
void NNUE::verify() {

    for (const auto& [netSize, evalFile] : EvalFiles)
    {
        // Replace with
        // Options[evalFile.option_name]
        // once fishtest supports the uci option EvalFileSmall
        std::string user_eval_file =
          netSize == Small ? evalFile.default_name : Options[evalFile.option_name];
        if (user_eval_file.empty())
            user_eval_file = evalFile.default_name;

        if (evalFile.selected_name != user_eval_file)
        {
            std::string msg1 =
              "Network evaluation parameters compatible with the engine must be available.";
            std::string msg2 =
              "The network file " + user_eval_file + " was not loaded successfully.";
            std::string msg3 = "The UCI option EvalFile might need to specify the full path, "
                               "including the directory name, to the network file.";
            std::string msg4 = "The default net can be downloaded from: "
                               "https://tests.stockfishchess.org/api/nn/"
                             + evalFile.default_name;
            std::string msg5 = "The engine will be terminated now.";

            sync_cout << "info string ERROR: " << msg1 << sync_endl;
            sync_cout << "info string ERROR: " << msg2 << sync_endl;
            sync_cout << "info string ERROR: " << msg3 << sync_endl;
            sync_cout << "info string ERROR: " << msg4 << sync_endl;
            sync_cout << "info string ERROR: " << msg5 << sync_endl;

            exit(EXIT_FAILURE);
        }

        sync_cout << "info string NNUE evaluation using " << user_eval_file << sync_endl;
    }
}
}

// Returns a static, purely materialistic evaluation of the position from
// the point of view of the given color. It can be divided by PawnValue to get
// an approximation of the material advantage on the board in terms of pawns.
int Eval::simple_eval(const Position& pos, Color c) {
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c))
         + (pos.non_pawn_material(c) - pos.non_pawn_material(~c));
}


// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Position& pos) {

    assert(!pos.checkers());

    int simpleEval = simple_eval(pos, pos.side_to_move());
    bool smallNet = std::abs(simpleEval) > SmallNetThreshold;
    bool psqtOnly = std::abs(simpleEval) > PsqtOnlyThreshold;
    int nnueComplexity;

    Value nnue = smallNet ? NNUE::evaluate<NNUE::Small>(pos, true, &nnueComplexity, psqtOnly)
                          : NNUE::evaluate<NNUE::Big>(pos, true, &nnueComplexity, false);

    int optimism = pos.this_thread()->optimism[pos.side_to_move()];

    const auto adjustEval = [&](int optDiv, int nnueDiv, int pawnCountConstant, int pawnCountMul,
                                int npmConstant, int evalDiv, int shufflingConstant,
                                int shufflingDiv) -> int {
        int absDiff = std::abs(simpleEval - nnue);
        optimism += optimism * (nnueComplexity + absDiff) / optDiv;
        nnue -= nnue * (nnueComplexity + absDiff) / nnueDiv;

        int npm = pos.non_pawn_material() / 64;
        return (nnue * (npm + pawnCountConstant + pawnCountMul * pos.count<PAWN>()) +
                optimism * (npmConstant + npm))
               / evalDiv;
    };

    const int optDiv = smallNet ? 517 : 499;
    const int nnueDiv = smallNet ? 32857 : 32793;
    const int pawnCountConstant = smallNet ? 908 : 903;
    const int pawnCountMul = smallNet ? 7 : 9;
    const int npmConstant = smallNet ? 155 : 147;
    const int evalDiv = smallNet ? 1019 : 1067;
    const int shufflingConstant = smallNet ? 224 : 208;
    const int shufflingDiv = smallNet ? 238 : 211;

    int v = adjustEval(optDiv, nnueDiv, pawnCountConstant, pawnCountMul,
                       npmConstant, evalDiv, shufflingConstant, shufflingDiv);

    // Damp down the evaluation linearly when shuffling
    int shuffling = pos.rule50_count();
    v = v * (shufflingConstant - shuffling) / shufflingDiv;

    // Guarantee evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos) {

     // Check if the position is in check
     if (pos.checkers())
         return "Final evaluation: none (in check)";

     // Initialize a stringstream to build the trace string
     std::stringstream ss;
     ss << std::fixed << std::setprecision(2);

     // Add detailed trace of NNUE evaluation
     ss << '\n' << NNUE::trace(pos) << '\n';

     // Set the formatting for printing values
     ss << std::setw(15);

     // Calculate the NNUE rating from white's perspective
     Value nnueEval = NNUE::evaluate<NNUE::Big>(pos, false);
     nnueEval = pos.side_to_move() == WHITE ? nnueEval : -nnueEval;
     ss << "NNUE evaluation " << 0.01 * UCI::to_cp(nnueEval) << " (white side)\n";

     // Calculates the final rating from white's perspective and adds a note on the NNUE scale
     Value finalEval = evaluate(pos);
     finalEval = pos.side_to_move() == WHITE ? finalEval : -finalEval;
     ss << "Final evaluation " << 0.01 * UCI::to_cp(finalEval) << " (white side)";
     ss << " [with scaled NNUE, ...]";
     ss << "\n";

     // Returns the trace string
     return ss.str();
}

}  // namespace Stockfish

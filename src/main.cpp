#include "Simulation.hpp"
#include "FireGen.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <algorithm>

// Simple CLI:
//   --rounds N
//   --map PATH
//   --vision DIR
//   --paths DIR
//   --seed S
//   --emit-predicted   (optional flag; enables Predict::bumpDownwind calls)
//
// Example:
//   ./firedrones --rounds 20 --map map.txt --vision vision --paths paths --seed 42 --emit-predicted

int main(int argc, char** argv) {
    SimulationConfig sc;
    FireGenConfig    fc;

    // Defaults
    sc.numRounds  = 10;
    sc.mapPath    = "map.txt";
    sc.visionDir  = "vision";
    sc.pathsDir   = "paths";
    sc.emitPredicted = false;

    fc.numFires        = 5;
    fc.fireSize        = 8;
    fc.numCitizens     = 500;
    fc.numFirefighters = 25;
    fc.seed            = 1;

    // Parse very light CLI
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag)->bool { return a == flag && i+1 < argc; };

        if (need("--rounds"))           { sc.numRounds = std::max(1, std::atoi(argv[++i])); }
        else if (need("--map"))         { sc.mapPath = argv[++i]; }
        else if (need("--vision"))      { sc.visionDir = argv[++i]; }
        else if (need("--paths"))       { sc.pathsDir = argv[++i]; }
        else if (need("--seed"))        { fc.seed = static_cast<unsigned>(std::stoul(argv[++i])); }
        else if (a == "--emit-predicted") { sc.emitPredicted = true; }
        else if (a == "--help" || a == "-h") {
            std::cout <<
            "Usage: firedrones [options]\n"
            "  --rounds N          Number of rounds (default 10)\n"
            "  --map PATH          Path to map.txt (default map.txt)\n"
            "  --vision DIR        Directory for vision files (default vision)\n"
            "  --paths DIR         Directory for path files (default paths)\n"
            "  --seed S            RNG seed for generator (default 1)\n"
            "  --emit-predicted    Also run Predict::bumpDownwind each round\n";
            return 0;
        } else {
            std::cerr << "Unknown option: " << a << " (use --help)\n";
            return 1;
        }
    }

    try {
        Simulation sim(sc, fc);
        sim.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 2;
    } catch (...) {
        std::cerr << "Fatal: unknown error\n";
        return 2;
    }

    return 0;
}

#include "Simulation.hpp"
#include "Grid.hpp"
#include "FireGen.hpp"
#include "Drone.hpp"
#include "Rescue.hpp"
#include "Predict.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace fs = std::filesystem;

static std::string roundFileName(const std::string& dir, const std::string& stem, int round, const char* ext = ".txt") {
    std::ostringstream oss;
    oss << dir << "/" << stem << "_round" << std::setw(3) << std::setfill('0') << round << ext;
    return oss.str();
}

Simulation::Simulation(const SimulationConfig& sc, const FireGenConfig& fc)
: cfg_(sc), firegen_(fc) {
    firefighters_.clear();
}

void Simulation::setup() {
    fs::create_directories(fs::path(cfg_.visionDir));
    fs::create_directories(fs::path(cfg_.pathsDir));
    fs::create_directories(fs::path(cfg_.mapArchiveDir));

    // Initialize world
    firegen_.init(map_, firefighters_);

    // Place six refill stations (align with visualizer defaults)
    stations_ = {
        {"YAR", 0, 40}, {"LUN", 140, 25}, {"WIN", 250, 70},
        {"HFX", 265, 30}, {"NG", 400, 75}, {"SYD", 540, 50}
    };

    // One drone per station; patrol bounds cover whole map for now
    drones_.clear();
    for (size_t i = 0; i < stations_.size(); ++i) {
        const auto& st = stations_[i];
        int xMin = 0, yMin = 0, xMax = WIDTH - 1, yMax = HEIGHT - 1;
        drones_.emplace_back(int(i+1), st.x, st.y, xMin, yMin, xMax, yMax, st);
    }

    // Emit initial round 0 map for tools
    writeRoundFile(cfg_.mapPath, /*round*/ 0, map_);
    // Also archive round 0 full map
    writeRoundFile(roundFileName(cfg_.mapArchiveDir, "map", /*round*/ 0), 0, map_);
}

void Simulation::writeDroneVisionFile(int round) {
    fs::create_directories(fs::path(cfg_.visionDir));
    std::ostringstream name;
    name << cfg_.visionDir << "/vision_round" << std::setw(3) << std::setfill('0') << round << ".txt";
    std::ofstream out(name.str(), std::ios::trunc);

    // Gather and de-duplicate all FoV lines across drones this round
    std::vector<std::string> lines;
    lines.reserve(1024);
    for (const auto& d : drones_) d.logVision(map_, lines);

    std::unordered_set<std::string> seen;
    for (const auto& L : lines) {
        if (seen.insert(L).second) out << L << "\n";
    }
}

void Simulation::writeDronePathsFile(int round) {
    // Per-drone files (easy to consume in viz); you can also write a combined CSV if preferred.
    fs::create_directories(fs::path(cfg_.pathsDir));
    for (const auto& d : drones_) {
        d.endRoundFlushPath(cfg_.pathsDir, round);
    }
}

void Simulation::run() {
    setup();

    for (int round = 1; round <= cfg_.numRounds; ++round) {
        // Start-of-round bookkeeping
        for (auto& d : drones_) d.clearPath();

        // Determine if any drone's current FoV spots a severe (>=4) fire this round
        bool severeAlert = false;
        for (const auto& d : drones_) {
            auto [dx, dy] = d.position();
            for (int ix = dx - 1; ix <= dx + 1; ++ix) {
                for (int iy = dy - 1; iy <= dy + 1; ++iy) {
                    if (!map_.inBounds(ix, iy)) continue;
                    if (map_.cells[ix][iy].fire_sev >= 4) { severeAlert = true; break; }
                }
                if (severeAlert) break;
            }
            if (severeAlert) break;
        }

        // Per organizer rules, each drone takes one action per round
        for (auto& d : drones_) {
            d.actRound(map_, stations_, severeAlert);
        }

        // Environment evolves
    firegen_.spread(map_);
    firegen_.firefightersAct(map_, firefighters_);
    // After fire updates, change wind direction on every tile for the next round
    firegen_.randomizeWindDir(map_);

        // Write map.txt (truth) for this round
        writeRoundFile(cfg_.mapPath, round, map_);
    // Archive per-round full map
    writeRoundFile(roundFileName(cfg_.mapArchiveDir, "map", round), round, map_);

    // Emit your requested visualization artifacts
        writeDroneVisionFile(round);
        writeDronePathsFile(round);

        // (Optional) stdout progress
        std::cout << "Finished round " << round << "\n";
    }
}

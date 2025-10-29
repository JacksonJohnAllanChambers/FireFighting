// Simulation.hpp (sketch)
#pragma once
#include "Grid.hpp"
#include "FireGen.hpp"
#include "Drone.hpp"
#include <string>
struct SimulationConfig {
    int numRounds = 10;
    std::string mapPath     = "map.txt";
    std::string mapArchiveDir = "maps"; // per-round full maps
    std::string visionDir   = "vision";
    std::string pathsDir    = "paths";
    bool emitPredicted      = false;
};

class Simulation {
public:
    Simulation(const SimulationConfig&, const FireGenConfig&);
    void run();

private:
    SimulationConfig cfg_;
    Map map_;
    std::vector<std::pair<int,int>> firefighters_;
    std::vector<Drone> drones_;
    std::vector<RefillStation> stations_;
    FireGen firegen_;

    void setup();
    void writeDroneVisionFile(int round);
    void writeDronePathsFile  (int round);
};

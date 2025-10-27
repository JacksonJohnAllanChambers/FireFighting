// FireGen.hpp
#pragma once
#include "Grid.hpp"
#include <random>

struct FireGenConfig {
    int numFires       = 5;
    int fireSize       = 8;
    int numCitizens    = 500;
    int numFirefighters= 25;
    unsigned seed      = 1;
};

class FireGen {
public:
    explicit FireGen(FireGenConfig cfg);
    void init(Map& map, std::vector<std::pair<int,int>>& ffPositions);
    void spread(Map& map);
    void firefightersAct(Map& map, const std::vector<std::pair<int,int>>& ffPositions);

private:
    FireGenConfig cfg_;
    std::mt19937 rng_;
    int randint(int lo, int hi); // inclusive
};

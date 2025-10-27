// Grid.hpp
#pragma once
#include <array>
#include <string>
#include <vector>
#include <cstdint>

constexpr int WIDTH  = 550;
constexpr int HEIGHT = 100;

struct Tile {
    uint8_t fire_sev    = 0;  // C in comment: “fire” really means severity>0 ⇒ on fire
    uint8_t windspeed   = 0;  // D
    uint8_t winddir     = 0;  // E: 0=N,1=E,2=S,3=W
    uint8_t citizen     = 0;  // F
    uint8_t firefighter = 0;  // G

    // Extended, for your pipeline:
    uint8_t turns_since_seen = 0;  // n
    uint8_t trust            = 0;  // T
};

struct Map {
    std::array<std::array<Tile, HEIGHT>, WIDTH> cells{};

    bool inBounds(int x, int y) const {
        return x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT;
    }
};

// map.txt <> structs
std::string encodeLine(int x, int y, const Tile& t); // "AAABBCDEFG"
bool        decodeLine(const std::string& line, int& x, int& y, Tile& t);

// file I/O
void writeRoundFile(const std::string& path, int round, const Map& map);
bool readRoundFile (const std::string& path, int& outRound, Map& outMap);

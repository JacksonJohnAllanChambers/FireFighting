// Drone.hpp
#pragma once
#include "Grid.hpp"
#include <vector>
#include <utility>
#include <string>

struct RefillStation { std::string name; int x; int y; };

class Drone {
public:
    Drone(int id, int x0, int y0, int xMin, int yMin, int xMax, int yMax, RefillStation station);

    void logVision(const Map& map, std::vector<std::string>& outLines) const; // 3x3 FoV lines
    void clearPath();                  // start of round
    void endRoundFlushPath(const std::string& fpath, int round) const;
    // choose action per organizer rules; when alert=true, drones converge on severe fires
    void actRound(Map& map, const std::vector<RefillStation>& bases, bool alert);

    std::pair<int,int> position() const { return {x_, y_}; }

private:
    enum class State { SCOUT, TO_FIRE, EXTINGUISH, TO_BASE };
    int id_;
    int x_, y_;
    int xMin_, yMin_, xMax_, yMax_;
    int water_ = 10, capacity_ = 10;
    RefillStation station_;
    std::vector<std::pair<int,int>> path_; // append to each move
    int sweepDir_ = 1; // lawnmower direction: 1=right, -1=left (legacy)

    // Radiating scout state from base
    int dirX_ = 0; // -1,0,1
    int dirY_ = 0; // -1,0,1
    int homeX_ = 0;
    int homeY_ = 0;

    // Severe fire engagement state
    bool hasTarget_ = false;
    int targetX_ = -1;
    int targetY_ = -1;
    State state_ = State::SCOUT;

    void moveTo(int nx, int ny);
    bool needRefill() const { return water_ <= 2; }
    bool findBestTarget(const Map& map, int& outX, int& outY, int& outScore) const;
    void moveTowards(int tx, int ty, int steps);
    int  dumpWater(Map& map, int maxLiters);
    bool findNearestSevereFire(const Map& map, int& fx, int& fy) const;

    RefillStation nearestBase(const std::vector<RefillStation>& bases) const;
};

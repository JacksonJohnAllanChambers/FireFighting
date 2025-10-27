#include "Grid.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iomanip>

std::string encodeLine(int x, int y, const Tile& t) {
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << x
        << std::setw(2) << std::setfill('0') << y
        << int(t.fire_sev)
        << int(t.windspeed)
        << int(t.winddir)
        << int(t.citizen)
        << int(t.firefighter);
    return oss.str();
}

bool decodeLine(const std::string& line, int& x, int& y, Tile& t) {
    if (line.size() < 3+2+5) return false;
    x = std::stoi(line.substr(0,3));
    y = std::stoi(line.substr(3,2));
    t.fire_sev    = line[5]  - '0';
    t.windspeed   = line[6]  - '0';
    t.winddir     = line[7]  - '0';
    t.citizen     = line[8]  - '0';
    t.firefighter = line[9]  - '0';
    return true;
}

void writeRoundFile(const std::string& path, int round, const Map& map) {
    std::ofstream out(path, std::ios::trunc);
    out << round << "\n";
    for (int x = 0; x < WIDTH; ++x) {
        for (int y = 0; y < HEIGHT; ++y) {
            out << encodeLine(x, y, map.cells[x][y]) << "\n";
        }
    }
}

bool readRoundFile(const std::string& path, int& outRound, Map& outMap) {
    std::ifstream in(path);
    if (!in) return false;
    if (!(in >> outRound)) return false;
    std::string line;
    // consume endline of first line
    std::getline(in, line);
    int x, y; Tile t;
    for (int i = 0; i < WIDTH*HEIGHT; ++i) {
        if (!std::getline(in, line)) return false;
        if (!decodeLine(line, x, y, t)) return false;
        if (x >=0 && x < WIDTH && y >=0 && y < HEIGHT)
            outMap.cells[x][y] = t;
    }
    return true;
}

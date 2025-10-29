#include "Drone.hpp"
#include "Grid.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <climits>

static inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

Drone::Drone(int id, int x0, int y0, int xMin, int yMin, int xMax, int yMax, RefillStation station)
: id_(id), x_(x0), y_(y0),
  xMin_(xMin), yMin_(yMin), xMax_(xMax), yMax_(yMax),
  station_(std::move(station)) {
    path_.reserve(256);
    path_.push_back({x_, y_});
    // Initialize home and scouting direction radiating away from center
    homeX_ = station_.x; homeY_ = station_.y;
    int cx = WIDTH / 2, cy = HEIGHT / 2;
    dirX_ = (homeX_ < cx) ? 1 : (homeX_ > cx ? -1 : 0);
    dirY_ = (homeY_ < cy) ? 1 : (homeY_ > cy ? -1 : 0);
    if (dirX_ == 0 && dirY_ == 0) dirX_ = 1; // default to east if exactly at center
}

void Drone::clearPath() {
    path_.clear();
    path_.push_back({x_, y_});
}

void Drone::moveTo(int nx, int ny) {
    x_ = nx; y_ = ny;
    path_.push_back({x_, y_});
}

// removed legacy step/fight/refill helpers (superseded by actRound)

void Drone::logVision(const Map& map, std::vector<std::string>& outLines) const {
    for (int ix = x_ - 1; ix <= x_ + 1; ++ix) {
        for (int iy = y_ - 1; iy <= y_ + 1; ++iy) {
            if (!map.inBounds(ix, iy)) continue;
            const Tile& t = map.cells[ix][iy];
            outLines.push_back(encodeLine(ix, iy, t));
        }
    }
}

void Drone::endRoundFlushPath(const std::string& dir, int round) const {
    std::ostringstream name;
    name << dir << "/paths_round"
         << std::setw(3) << std::setfill('0') << round
         << "_drone" << id_ << ".csv";
    std::ofstream out(name.str(), std::ios::trunc);
    out << "drone_id,step,x,y\n";
    for (size_t i = 0; i < path_.size(); ++i) {
        out << id_ << "," << i << "," << path_[i].first << "," << path_[i].second << "\n";
    }
}

bool Drone::findBestTarget(const Map& map, int& outX, int& outY, int& outScore) const {
    int bestScore = -1, bx = x_, by = y_;
    // search radius 8 around current position
    for (int ix = x_ - 8; ix <= x_ + 8; ++ix) {
        for (int iy = y_ - 8; iy <= y_ + 8; ++iy) {
            if (!map.inBounds(ix, iy)) continue;
            if (ix < xMin_ || ix > xMax_ || iy < yMin_ || iy > yMax_) continue;
            const Tile& t = map.cells[ix][iy];
            if (t.fire_sev == 0) continue;
            int dist = std::abs(ix - x_) + std::abs(iy - y_);
            // Higher severity and windspeed preferred; penalize distance
            int score = int(t.fire_sev) * 20 + int(t.windspeed) * 2 - dist;
            if (score > bestScore) { bestScore = score; bx = ix; by = iy; }
        }
    }
    outX = bx; outY = by; outScore = bestScore;

    // If target is too far relative to remaining water, prefer refilling soon
    if (bestScore > 0) {
        int dist = std::abs(bx - x_) + std::abs(by - y_);
        if (water_ <= 2 || dist > water_) {
            // Encourage refilling by indicating no viable target
            outScore = -1;
        }
    }
    return outScore > 0;
}

void Drone::moveTowards(int tx, int ty, int steps) {
    while (steps-- > 0 && (x_ != tx || y_ != ty)) {
        int nx = x_, ny = y_;
        if (tx > x_) nx = x_ + 1; else if (tx < x_) nx = x_ - 1;
        if (ty > y_) ny = y_ + 1; else if (ty < y_) ny = y_ - 1;
        // clamp to patrol and global bounds
        nx = clampi(nx, xMin_, xMax_); ny = clampi(ny, yMin_, yMax_);
        nx = clampi(nx, 0, WIDTH - 1); ny = clampi(ny, 0, HEIGHT - 1);
        moveTo(nx, ny);
    }
}

int Drone::dumpWater(Map& map, int maxLiters) {
    if (!map.inBounds(x_, y_) || maxLiters <= 0 || water_ <= 0) return 0;
    Tile& t = map.cells[x_][y_];
    if (t.fire_sev == 0) return 0;
    int allow = std::min({maxLiters, water_, int(t.fire_sev)});
    t.fire_sev = static_cast<uint8_t>(int(t.fire_sev) - allow);
    water_ -= allow;
    return allow;
}

RefillStation Drone::nearestBase(const std::vector<RefillStation>& bases) const {
    if (bases.empty()) return station_;
    int best = INT_MAX; size_t bi = 0;
    for (size_t i = 0; i < bases.size(); ++i) {
        int d = std::abs(bases[i].x - x_) + std::abs(bases[i].y - y_);
        if (d < best) { best = d; bi = i; }
    }
    return bases[bi];
}

void Drone::actRound(Map& map, const std::vector<RefillStation>& bases, bool alert) {
    // Organizer action budgets
    struct Budget { int move; int dump; };
    static const Budget budgets[] = {
        {50, 0}, {0, 10}, {15, 7}, {25, 5}, {40, 2}
    };

    // Helper: refresh target if any severe fire exists (global truth)
    int fx = -1, fy = -1;
    if (!hasTarget_)
        hasTarget_ = findNearestSevereFire(map, fx, fy);
    if (hasTarget_) { targetX_ = fx < 0 ? targetX_ : fx; targetY_ = fy < 0 ? targetY_ : fy; }

    // State transitions based on water and location
    if (x_ == station_.x && y_ == station_.y && water_ < capacity_) {
        water_ = capacity_;
        // After refill: if we have a target, head to it; else resume scouting
        state_ = hasTarget_ ? State::TO_FIRE : State::SCOUT;
        return;
    }
    if (needRefill()) { state_ = State::TO_BASE; }

    // Act according to state
    switch (state_) {
        case State::TO_BASE: {
            if (alert) {
                auto nb = nearestBase(bases);
                moveTowards(nb.x, nb.y, budgets[0].move);
                if (x_ == nb.x && y_ == nb.y) {
                    water_ = capacity_;
                    state_ = hasTarget_ ? State::TO_FIRE : State::SCOUT;
                }
            } else {
                moveTowards(station_.x, station_.y, budgets[0].move);
                if (x_ == station_.x && y_ == station_.y) {
                    water_ = capacity_;
                    state_ = hasTarget_ ? State::TO_FIRE : State::SCOUT;
                }
            }
            return;
        }
        case State::TO_FIRE: {
            // If target lost or not severe anymore, drop it
            if (!(map.inBounds(targetX_, targetY_) && map.cells[targetX_][targetY_].fire_sev >= 4)) {
                hasTarget_ = findNearestSevereFire(map, targetX_, targetY_);
                if (!hasTarget_) { state_ = State::SCOUT; break; }
            }
            int dist = std::abs(targetX_ - x_) + std::abs(targetY_ - y_);
            if (dist <= budgets[2].move) {
                moveTowards(targetX_, targetY_, budgets[2].move);
                // If arrived, switch to extinguish
                if (x_ == targetX_ && y_ == targetY_) state_ = State::EXTINGUISH;
                return;
            } else if (dist <= budgets[4].move) {
                moveTowards(targetX_, targetY_, budgets[4].move);
                if (x_ == targetX_ && y_ == targetY_) state_ = State::EXTINGUISH;
                return;
            } else {
                // Far: use pure move to close distance
                moveTowards(targetX_, targetY_, budgets[0].move);
                return;
            }
        }
        case State::EXTINGUISH: {
            // Dump as much as allowed; if not on fire cell anymore or water low, decide next
            if (map.inBounds(x_, y_) && map.cells[x_][y_].fire_sev > 0) {
                dumpWater(map, budgets[1].dump);
            }
            if (water_ <= 2) { state_ = State::TO_BASE; return; }
            if (!(map.inBounds(x_, y_) && map.cells[x_][y_].fire_sev >= 4)) {
                // Target no longer severe; look for another nearby severe fire
                int nfx, nfy;
                if (findNearestSevereFire(map, nfx, nfy)) { targetX_ = nfx; targetY_ = nfy; state_ = State::TO_FIRE; }
                else { hasTarget_ = false; state_ = State::SCOUT; }
            }
            return;
        }
        case State::SCOUT: {
            // If alert is raised or a severe fire exists, converge
            if (alert) {
                int nfx, nfy;
                if (findNearestSevereFire(map, nfx, nfy)) {
                    hasTarget_ = true; targetX_ = nfx; targetY_ = nfy; state_ = State::TO_FIRE;
                    int dist = std::abs(targetX_ - x_) + std::abs(targetY_ - y_);
                    if (dist <= budgets[2].move) moveTowards(targetX_, targetY_, budgets[2].move);
                    else if (dist <= budgets[4].move) moveTowards(targetX_, targetY_, budgets[4].move);
                    else moveTowards(targetX_, targetY_, budgets[0].move);
                    return;
                }
            } else {
                // Legacy behavior: scout radiating outward
                int nfx, nfy;
                if (findNearestSevereFire(map, nfx, nfy)) {
                    hasTarget_ = true; targetX_ = nfx; targetY_ = nfy; state_ = State::TO_FIRE;
                    int dist = std::abs(targetX_ - x_) + std::abs(targetY_ - y_);
                    if (dist <= budgets[2].move) moveTowards(targetX_, targetY_, budgets[2].move);
                    else if (dist <= budgets[4].move) moveTowards(targetX_, targetY_, budgets[4].move);
                    else moveTowards(targetX_, targetY_, budgets[0].move);
                    return;
                }
                int steps = budgets[0].move;
                while (steps-- > 0) {
                    int nx = x_ + dirX_;
                    int ny = y_ + dirY_;
                    // bounce on boundaries
                    if (nx < 0 || nx >= WIDTH) { dirX_ = -dirX_; nx = x_ + dirX_; }
                    if (ny < 0 || ny >= HEIGHT) { dirY_ = -dirY_; ny = y_ + dirY_; }
                    moveTo(clampi(nx, 0, WIDTH-1), clampi(ny, 0, HEIGHT-1));
                    if (map.cells[x_][y_].fire_sev >= 4) { hasTarget_ = true; targetX_ = x_; targetY_ = y_; state_ = State::EXTINGUISH; break; }
                }
                return;
            }
        }
    }
}

bool Drone::findNearestSevereFire(const Map& map, int& fx, int& fy) const {
    int bestD = 1e9, bx = -1, by = -1;
    for (int ix = 0; ix < WIDTH; ++ix) {
        for (int iy = 0; iy < HEIGHT; ++iy) {
            const Tile& t = map.cells[ix][iy];
            if (t.fire_sev >= 4) {
                int d = std::abs(ix - x_) + std::abs(iy - y_);
                if (d < bestD) { bestD = d; bx = ix; by = iy; }
            }
        }
    }
    fx = bx; fy = by; return bx >= 0;
}

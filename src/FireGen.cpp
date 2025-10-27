#include "FireGen.hpp"
#include "Grid.hpp"
#include <algorithm>
#include <random>

static inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

FireGen::FireGen(FireGenConfig cfg)
: cfg_(cfg), rng_(cfg.seed ? cfg.seed : std::random_device{}()) {}

int FireGen::randint(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

void FireGen::init(Map& map, std::vector<std::pair<int,int>>& ffPositions) {
    // Clear
    for (int x = 0; x < WIDTH; ++x)
        for (int y = 0; y < HEIGHT; ++y)
            map.cells[x][y] = Tile{};

    // Random wind direction fields (coarse patches)
    for (int x = 0; x < WIDTH; ++x) {
        for (int y = 0; y < HEIGHT; ++y) {
            Tile& t = map.cells[x][y];
            t.windspeed = static_cast<uint8_t>(randint(0, 3));  // mild to start
            t.winddir   = static_cast<uint8_t>(randint(0, 3));  // 0..3
        }
    }

    // Seed clustered fires
    for (int i = 0; i < cfg_.numFires; ++i) {
        int cx = randint(20, WIDTH - 21);
        int cy = randint(10, HEIGHT - 11);
        for (int k = 0; k < cfg_.fireSize; ++k) {
            int x = clampi(cx + randint(-5, 5), 0, WIDTH - 1);
            int y = clampi(cy + randint(-5, 5), 0, HEIGHT - 1);
            Tile& t = map.cells[x][y];
            t.fire_sev = static_cast<uint8_t>(std::min(9, int(t.fire_sev) + randint(2, 4)));
        }
    }

    // Citizens
    for (int i = 0; i < cfg_.numCitizens; ++i) {
        int x = randint(0, WIDTH - 1);
        int y = randint(0, HEIGHT - 1);
        map.cells[x][y].citizen = 1;
    }

    // Firefighters (also mark on map)
    ffPositions.clear();
    ffPositions.reserve(cfg_.numFirefighters);
    for (int i = 0; i < cfg_.numFirefighters; ++i) {
        int x = randint(0, WIDTH - 1);
        int y = randint(0, HEIGHT - 1);
        map.cells[x][y].firefighter = 1;
        ffPositions.emplace_back(x, y);
    }
}

void FireGen::firefightersAct(Map& map, const std::vector<std::pair<int,int>>& ffPositions) {
    // Each firefighter reduces severity on its cell and 4-neighbors
    const int dx[5] = {0, 1, -1, 0, 0};
    const int dy[5] = {0, 0, 0, 1, -1};
    for (auto [x, y] : ffPositions) {
        for (int i = 0; i < 5; ++i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (!map.inBounds(nx, ny)) continue;
            Tile& t = map.cells[nx][ny];
            if (t.fire_sev > 0) {
                int red = 1;
                if (i == 0) red = 2; // center stronger
                t.fire_sev = static_cast<uint8_t>(std::max(0, int(t.fire_sev) - red));
            }
        }
    }
}

void FireGen::spread(Map& map) {
    // Organizer-specified spread rules:
    // - If severity 0-3: no spread
    // - If severity 4-9: spread one tile in wind direction
    //   * If target already has fire (>0), attempt +1 (up to 9). If that would exceed 9, do nothing.
    //   * If target has no fire, set target severity to floor(source/2).
    // Source tile severity does not decrease.

    Map next = map;

    for (int x = 0; x < WIDTH; ++x) {
        for (int y = 0; y < HEIGHT; ++y) {
            const Tile& src = map.cells[x][y];
            int s = int(src.fire_sev);
            if (s < 4) continue; // 0-3: no spread

            int dir = src.winddir % 4; // 0=N,1=E,2=S,3=W
            int dx = (dir == 1) - (dir == 3);
            int dy = (dir == 2) - (dir == 0);
            int nx = x + dx, ny = y + dy;
            if (!map.inBounds(nx, ny)) continue;

            const Tile& tgt0 = map.cells[nx][ny];
            Tile&       tgtN = next.cells[nx][ny];

            if (tgt0.fire_sev > 0) {
                // Increase by +1 unless that would exceed 9
                if (tgtN.fire_sev < 9) {
                    tgtN.fire_sev = static_cast<uint8_t>(std::min(9, int(tgtN.fire_sev) + 1));
                }
                // else do nothing
            } else {
                int half = s / 2; // floor
                if (half > int(tgtN.fire_sev)) {
                    tgtN.fire_sev = static_cast<uint8_t>(std::min(9, half));
                }
            }
        }
    }

    map = std::move(next);
}

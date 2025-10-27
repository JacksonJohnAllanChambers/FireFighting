#include "Predict.hpp"
#include "Grid.hpp"
#include <algorithm>

// A light, safe heuristic inspired by your Python FirePrediction:
// - If a tile has severity >= 4 AND windspeed >= 3, add +1 severity to the
//   immediate downwind neighbor (clamped), without exceeding 9.
// - This is an in-place "hint" overlay; feel free to point it at a separate Map
//   if you want a distinct Predicted_map.

namespace Predict {

static inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

void bumpDownwind(Map& map) {
    Map next = map; // work on a copy, then commit

    for (int x = 0; x < WIDTH; ++x) {
        for (int y = 0; y < HEIGHT; ++y) {
            const Tile& t = map.cells[x][y];
            if (t.fire_sev < 4 || t.windspeed < 3) continue;

            int dir = t.winddir % 4; // 0=N,1=E,2=S,3=W
            int dx = (dir == 1) - (dir == 3);
            int dy = (dir == 2) - (dir == 0);
            int nx = x + dx, ny = y + dy;
            if (!map.inBounds(nx, ny)) continue;

            Tile& nt = next.cells[nx][ny];
            int bumped = clampi(int(nt.fire_sev) + 1, 0, 9);
            nt.fire_sev = static_cast<uint8_t>(bumped);
        }
    }

    map = std::move(next);
}

} // namespace Predict

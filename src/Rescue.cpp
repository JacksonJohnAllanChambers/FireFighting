#include "Rescue.hpp"
#include "Grid.hpp"
#include <queue>
#include <algorithm>

// Simple rule set:
// - If a citizen is on a burning tile (fire_sev > 0), try to move them to the first
//   adjacent safe tile (8-neighborhood) that is in bounds and fire_sev == 0.
// - If no adjacent safe tile exists, keep them in place.
// - If a firefighter is on a burning tile, we leave them (they're working there).
//   You can extend this to auto-move them or auto-reduce fire if you want.

namespace Rescue {

static const int K8X[8] = { -1,  0, 1, -1, 1, -1, 0, 1 };
static const int K8Y[8] = { -1, -1,-1,  0, 0,  1, 1, 1 };

void evacuateCitizens(Map& map) {
    // Collect moves first to avoid double moves in one sweep
    struct Move { int sx, sy, dx, dy; };
    std::vector<Move> planned;

    for (int x = 0; x < WIDTH; ++x) {
        for (int y = 0; y < HEIGHT; ++y) {
            Tile& t = map.cells[x][y];
            if (t.citizen == 0) continue;
            if (t.fire_sev == 0) continue; // already safe

            // Find a safe neighbor (first-fit policy)
            for (int k = 0; k < 8; ++k) {
                int nx = x + K8X[k], ny = y + K8Y[k];
                if (!map.inBounds(nx, ny)) continue;
                const Tile& nt = map.cells[nx][ny];
                if (nt.fire_sev == 0 && nt.citizen == 0) {
                    planned.push_back({x, y, nx, ny});
                    break;
                }
            }
        }
    }

    // Apply planned moves
    for (const auto& mv : planned) {
        Tile& from = map.cells[mv.sx][mv.sy];
        Tile& to   = map.cells[mv.dx][mv.dy];
        if (from.citizen == 1 && to.citizen == 0 && to.fire_sev == 0) {
            from.citizen = 0;
            to.citizen   = 1;
        }
    }
}

} // namespace Rescue

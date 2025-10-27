#pragma once
#include "Grid.hpp"

namespace Rescue {

// Moves citizens from burning tiles to adjacent safe tiles.
void evacuateCitizens(Map& map);

} // namespace Rescue

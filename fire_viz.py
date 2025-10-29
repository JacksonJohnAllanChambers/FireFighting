##-----------------------------------------------------------------------------
# Fire Viz - Visualization of FirefighterDEC fire mapping and drone data (AI assisted) 
#-----------------------------------------------------------------------------
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib import colors as mcolors
from matplotlib.transforms import Affine2D
from typing import Optional, Dict, List, Tuple
import re
import csv
# -----------------------------------------------------------------------------
# Local helpers (previously imported) inlined to avoid external self-loading
# -----------------------------------------------------------------------------
WIDTH = 550
HEIGHT = 100
UNMAPPED = '\0'

DIR_TO_VEC = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0)
}

# Base stations (name, x, y)
BASE_STATIONS: List[Tuple[str, int, int]] = [
    ("YAR", 0, 40),
    ("LUN", 140, 25),
    ("WIN", 250, 70),
    ("HFX", 265, 30),
    ("NG", 400, 75),
    ("SYD", 540, 50),
]

class Drone:
    def __init__(self, x: int, y: int, name: str = "D"):
        self.x = x
        self.y = y
        self.name = name
    def location(self) -> Tuple[int, int]:
        return (self.x, self.y)

def parse_tile(s: str) -> Optional[Dict]:
    """
    Parse a tile string. Supports two formats:
    - Mapping.py convention (preferred):
        x[0:3], y[3:5], [5:reserved], fire_flag[6], severity[7], wind[8], dir[9],
        citizen[10], firefighter[11], turns_since_seen[12], trust[13]
      Minimum length: 14
    - Legacy fire_viz format:
        x[0:3], y[3:5], severity[5], wind[6], dir[7], citizen[8], firefighter[9],
        optional n/trust from position 10 onward
      Minimum length: 10
    Returns dict with unified keys: x, y, fire(severity), windspeed, winddir,
    citizen, firefighter, n(turns), trust, and fire_flag when available.
    """
    if not s or s == UNMAPPED:
        return None
    try:
        if len(s) >= 14:
            # Mapping.py convention
            x = int(s[0:3]); y = int(s[3:5])
            # s[5] reserved/unused
            fire_flag = int(s[6])
            severity = int(s[7])
            wind = int(s[8])
            wdir = int(s[9])
            citizen = int(s[10])
            firefighter = int(s[11])
            n = int(s[12])
            trust = int(s[13])
            return dict(x=x, y=y, fire=severity, fire_flag=fire_flag,
                        windspeed=wind, winddir=wdir,
                        citizen=citizen, firefighter=firefighter,
                        n=n, trust=trust)
        elif len(s) >= 10:
            # Legacy convention
            x = int(s[0:3]); y = int(s[3:5])
            severity = int(s[5])
            wind = int(s[6]); wdir = int(s[7])
            citizen = int(s[8]); firefighter = int(s[9])
            n = -1; trust = -1
            if len(s) > 10:
                tail = s[10:]
                if len(tail) == 2:
                    n, trust = int(tail[0]), int(tail[1])
                elif len(tail) == 3:
                    n, trust = int(tail[0]), int(tail[1:])
                elif len(tail) >= 4:
                    n, trust = int(tail[0:2]), int(tail[2:4])
            fire_flag = 1 if severity > 0 else 0
            return dict(x=x, y=y, fire=severity, fire_flag=fire_flag,
                        windspeed=wind, winddir=wdir,
                        citizen=citizen, firefighter=firefighter,
                        n=n, trust=trust)
        else:
            return None
    except Exception:
        return None

def encode_tile(x:int,y:int,fire:int,wind:int,wd:int,cit:int,ff:int,n:int=-1,trust:int=-1) -> str:
    """
    Encode a tile using Mapping.py convention.
    fire parameter is interpreted as severity (0-9). fire_flag is derived as 1 if fire>0 else 0.
    Positions: x[0:3], y[3:5], reserved[5]='0', fire_flag[6], severity[7], wind[8], dir[9],
               citizen[10], firefighter[11], turns[12], trust[13]
    When n/trust are negative, default them to 0 to keep fixed length.
    """
    severity = int(fire) % 10
    fire_flag = 1 if severity > 0 else 0
    wind = int(wind) % 10
    wd = int(wd) % 10  # allow up to 9; mapping.py mentions direction at index 9
    cit = int(cit) % 10
    ff = int(ff) % 10
    n = 0 if n < 0 else int(n) % 10
    trust = 0 if trust < 0 else int(trust) % 10
    return f"{x:03d}{y:02d}0{fire_flag}{severity}{wind}{wd}{cit}{ff}{n}{trust}"

def tiles_to_arrays(matrix: List[List[str]]):
    H = len(matrix); W = len(matrix[0])
    fire = np.full((H, W), np.nan)
    windspeed = np.full((H, W), np.nan)
    winddir = np.full((H, W), np.nan)
    citizen = np.zeros((H, W), dtype=bool)
    firefighter = np.zeros((H, W), dtype=bool)
    trust = np.full((H, W), np.nan)
    for y in range(H):
        for x in range(W):
            t = parse_tile(matrix[y][x])
            if t is None: continue
            fire[y, x] = t["fire"]
            windspeed[y, x] = t["windspeed"]
            winddir[y, x] = t["winddir"]
            citizen[y, x] = bool(t["citizen"])
            firefighter[y, x] = bool(t["firefighter"])
            trust[y, x] = t["trust"] if t["trust"] >= 0 else np.nan
    mapped_mask = ~np.isnan(fire)
    return {"fire": fire, "windspeed": windspeed, "winddir": winddir,
            "citizen": citizen, "firefighter": firefighter,
            "trust": trust, "mapped": mapped_mask}

# -----------------------------------------------------------------------------
# Colormap utilities: make severity 0 clear, unmapped transparent
# -----------------------------------------------------------------------------
def make_fire_cmap_norm():
        """
        Returns (cmap, norm) so that:
            - NaN (unmapped) is fully transparent
            - severity 0 is a light gray
            - severity 1..9 use a warm gradient (YlOrRd)
        """
        base = plt.get_cmap('YlOrRd')
        # Build colors: index 0 for sev 0 (light gray), 1..9 sampled from base
        grad = np.linspace(0.3, 1.0, 9)
        colors = ['#f0f0f0'] + [base(v) for v in grad]
        cmap = mcolors.ListedColormap(colors, name='fire10')
        cmap.set_bad(alpha=0.0)  # NaN -> transparent
        bounds = [-0.5] + [i + 0.5 for i in range(0, 10)]  # -0.5, 0.5, 1.5, ..., 9.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        return cmap, norm

# -----------------------------------------------------------------------------
# Adapters for C++ outputs (map.txt + vision/ + paths/)
# -----------------------------------------------------------------------------

def read_map_txt(path: str) -> List[List[str]]:
    """
    Read the C++ map.txt format:
    - Line 1: round number (int)
    - Then WIDTH*HEIGHT lines, each: AAABBCDEFG as produced by encodeLine()
    Returns a 2D list [HEIGHT][WIDTH] of tile strings.
    """
    grid = [[UNMAPPED for _ in range(WIDTH)] for _ in range(HEIGHT)]
    if not os.path.exists(path):
        return grid
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()  # round number (ignored here)
        for line in f:
            s = line.rstrip('\n')
            if len(s) < 10:
                continue
            t = parse_tile(s)
            if not t:
                continue
            x, y = t['x'], t['y']
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                grid[y][x] = s
    return grid

def read_vision_file(path: str) -> List[List[str]]:
    """
    Read a vision_roundNNN.txt file that contains many AAABBCDEFG tile lines.
    Returns a sparse 2D list [HEIGHT][WIDTH] with only seen tiles filled.
    """
    grid = [[UNMAPPED for _ in range(WIDTH)] for _ in range(HEIGHT)]
    if not os.path.exists(path):
        return grid
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.rstrip('\n')
            if len(s) < 10:
                continue
            t = parse_tile(s)
            if not t:
                continue
            x, y = t['x'], t['y']
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                grid[y][x] = s
    return grid

def read_map_round_file(maps_dir: str, round_num: int) -> Optional[List[List[str]]]:
    """
    Read a per-round archived full map from maps_dir/map_roundNNN.txt, if present.
    Returns a 2D list or None if missing.
    """
    path = os.path.join(maps_dir, f"map_round{round_num:03d}.txt")
    if not os.path.exists(path):
        return None
    return read_map_txt(path)

def read_paths_last_positions(round_num: int, paths_dir: str) -> List[Drone]:
    """
    For a given round number, read per-drone CSVs like paths_roundNNN_droneK.csv
    and return Drone objects positioned at the last step.
    """
    drones: List[Drone] = []
    pattern = os.path.join(paths_dir, f"paths_round{round_num:03d}_drone")
    for k in range(1, 100):  # reasonable upper bound
        candidate = f"{pattern}{k}.csv"
        if not os.path.exists(candidate):
            # stop only when we miss the first one; else continue to allow gaps
            if k == 1:
                break
            else:
                continue
        last_row = None
        try:
            with open(candidate, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    last_row = row
            if last_row and len(last_row) >= 4:
                # columns: drone_id,step,x,y
                x = int(last_row[2]); y = int(last_row[3])
                name = f"D{int(last_row[0])}"
                drones.append(Drone(x, y, name))
        except Exception:
            continue
    return drones

def build_rounds_from_outputs(vision_dir: str = 'vision',
                              paths_dir: str = 'paths',
                              map_path: Optional[str] = None,
                              maps_dir: Optional[str] = None) -> Tuple[List[Tuple[List[List[str]], List[List[str]], List[Drone]]], Dict[str, List[Tuple[int,int]]]]:
    """
    Build (hist, pred, drones) tuples for each round present using vision/ and paths/ outputs.
    - hist: from map_path if provided and per-round maps available; otherwise from per-round vision files (sparse).
    - pred: not available from C++ outputs; we mirror hist for overlay or leave as hist.
    - drones: last position per drone from paths CSVs.
    Tracks are reconstructed across rounds using last positions per round.
    """
    rounds: List[Tuple[List[List[str]], List[List[str]], List[Drone]]] = []
    tracks: Dict[str, List[Tuple[int,int]]] = {}

    # Discover available rounds from vision files
    if not os.path.isdir(vision_dir):
        return rounds, tracks
    files = sorted([fn for fn in os.listdir(vision_dir) if fn.startswith('vision_round') and fn.endswith('.txt')])
    round_nums: List[int] = []
    for fn in files:
        m = re.match(r"vision_round(\d{3})\.txt$", fn)
        if m:
            round_nums.append(int(m.group(1)))
    round_nums.sort()

    for r in round_nums:
        # Prefer archived full map if available; otherwise use sparse vision
        hist_map = None
        if maps_dir:
            hist_map = read_map_round_file(maps_dir, r)
        if hist_map is None:
            vision_path = os.path.join(vision_dir, f"vision_round{r:03d}.txt")
            hist_map = read_vision_file(vision_path)

        pred_map = [row[:] for row in hist_map]  # no predicted overlay from outputs
        drones = read_paths_last_positions(r, paths_dir)

        rounds.append((hist_map, pred_map, drones))
        # update tracks
        for d in drones:
            if d.name not in tracks:
                tracks[d.name] = []
            tracks[d.name].append((d.x, d.y))

    return rounds, tracks

def generate_dummy_maps(width=WIDTH, height=HEIGHT, seed: Optional[int]=42):
    rng = np.random.default_rng(seed)
    def empty_mat():
        return [['\\0' for _ in range(width)] for _ in range(height)]
    hist = empty_mat(); pred = empty_mat()
    clusters = [(100, 20, 6, 15), (300, 60, 7, 10), (480, 40, 5, 12)]
    for (cx, cy, base, radius) in clusters:
        for y in range(max(0, cy-radius), min(height, cy+radius+1)):
            for x in range(max(0, cx-radius), min(width, cx+radius+1)):
                dist = math.hypot(x-cx, y-cy)
                if dist <= radius:
                    sev = max(0, min(9, int(base - (dist / radius) * base + rng.integers(-1, 2))))
                    wind = int(rng.integers(2, 7)); wdir = int(rng.integers(0, 4))
                    cit = int(1 if rng.random() < 0.005 else 0); ff = int(1 if rng.random() < 0.003 else 0)
                    hist[y][x] = encode_tile(x, y, sev, wind, wdir, cit, ff, n=int(rng.integers(0, 20)), trust=int(rng.integers(0, 20)))
    for _ in range(2000):
        x = int(rng.integers(0, width)); y = int(rng.integers(0, height))
        if hist[y][x] == '\\0':
            wind = int(rng.integers(0, 4)); wdir = int(rng.integers(0, 4))
            cit = int(1 if rng.random() < 0.002 else 0); ff = int(1 if rng.random() < 0.002 else 0)
            hist[y][x] = encode_tile(x, y, 0, wind, wdir, cit, ff, n=int(rng.integers(0, 20)), trust=int(rng.integers(0, 20)))
    _DIR_TO_VEC = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}
    for y in range(height):
        for x in range(width):
            if hist[y][x] != '\\0':
                t = parse_tile(hist[y][x])
                sev = t["fire"]; wind = t["windspeed"]; wdir = t["winddir"]; cit = t["citizen"]; ff = t["firefighter"]
                px, py = x, y
                if sev >= 4:
                    dx, dy = _DIR_TO_VEC[wdir]
                    px = max(0, min(width-1, x + dx)); py = max(0, min(height-1, y + dy))
                    new_sev = max(0, min(9, sev // 2))
                    if pred[py][px] == '\\0':
                        pred[py][px] = encode_tile(px, py, new_sev, wind, wdir, cit, ff, n=0, trust=5)
                    else:
                        pt = parse_tile(pred[py][px])
                        merged_sev = max(pt["fire"], new_sev)
                        pred[py][px] = encode_tile(px, py, merged_sev, wind, wdir, max(cit, pt["citizen"]), max(ff, pt["firefighter"]), n=0, trust=5)
                if pred[y][x] == '\\0':
                    pred[y][x] = encode_tile(x, y, sev, wind, wdir, cit, ff, n=0, trust=8)
                else:
                    pt = parse_tile(pred[y][x])
                    merged_sev = max(pt["fire"], sev)
                    pred[y][x] = encode_tile(x, y, merged_sev, wind, wdir, max(cit, pt["citizen"]), max(ff, pt["firefighter"]), n=0, trust=8)
    return hist, pred

def generate_dummy_drones(num=4, seed: Optional[int]=7) -> List[Drone]:
    rng = np.random.default_rng(seed)
    drones = []
    for i in range(num):
        x = int(rng.integers(0, WIDTH)); y = int(rng.integers(0, HEIGHT))
        drones.append(Drone(x, y, name=chr(ord('A') + i)))
    return drones
# -----------------------------------------------------------------------------
# Simple per-round overlay plotter (static figure) for live GUI updating
# -----------------------------------------------------------------------------
def draw_scan_window(ax, x:int, y:int, color_alpha:float=0.2):
    rect = Rectangle((x-1, y-1), 3, 3, linewidth=1, edgecolor=None, facecolor='grey', alpha=color_alpha)
    ax.add_patch(rect)

def _ensure_parent_dir(path: Optional[str]):
    if not path:
        return
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def plot_fire_maps(hist_map: List[List[str]],
                   pred_map: List[List[str]],
                   drones: List[Drone],
                   show_scan_windows: bool = True,
                   pred_alpha: float = 0.35,
                   wind_stride: int = 5,
                   figsize: Tuple[int,int] = (18, 5),
                   save_path: Optional[str] = None,
                   title: str = "Firefighting Drone Swarm - Historical (base) + Predicted (overlay)"):
    arrays_hist = tiles_to_arrays(hist_map)
    arrays_pred = tiles_to_arrays(pred_map)
    cmap, norm = make_fire_cmap_norm()
    H, W = arrays_hist["fire"].shape

    fig, ax = plt.subplots(figsize=figsize)
    # Ensure no grid/ticks are shown
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Base = historical fire severity, overlay = predicted
    base_fire = np.ma.masked_invalid(arrays_hist["fire"])  # nan -> transparent
    im1 = ax.imshow(base_fire, origin='upper', interpolation='nearest', cmap=cmap, norm=norm)
    pred_fire = np.ma.masked_invalid(arrays_pred["fire"])  # nan -> transparent
    im2 = ax.imshow(pred_fire, origin='upper', interpolation='nearest', alpha=pred_alpha, cmap=cmap, norm=norm)

    # Wind arrows disabled by request

    # Citizens and firefighters
    citizen_mask = (arrays_hist["citizen"] | arrays_pred["citizen"])
    ff_mask = (arrays_hist["firefighter"] | arrays_pred["firefighter"])
    cy, cx = np.where(citizen_mask)
    fy, fx = np.where(ff_mask)
    ax.scatter(cx, cy, s=12, marker='^', label='Citizen', edgecolors='none', linewidths=0)
    ax.scatter(fx, fy, s=12, marker='s', label='Firefighter', edgecolors='none', linewidths=0)

    # Drones
    for d in drones:
        dx, dy = d.location()
        ax.scatter([dx], [dy], s=80, marker='X', label=f"Drone {d.name}")
        ax.text(dx+1, dy-1, d.name, fontsize=8, ha='left', va='center')
        if show_scan_windows:
            draw_scan_window(ax, dx, dy, color_alpha=0.15)

    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5); ax.set_aspect('equal')
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title(title)

    # Legend for points and tiles
    handles = [
        Line2D([0],[0], marker="^", linestyle="None", markersize=6, label="Citizen"),
        Line2D([0],[0], marker="s", linestyle="None", markersize=6, label="Firefighter"),
        Line2D([0],[0], marker="X", linestyle="None", markersize=8, label="Drone"),
        Patch(facecolor="#f0f0f0", edgecolor='none', label="Severity 0 (no fire)")
    ]
    ax.legend(handles=handles, loc="upper right")

    # Colorbar with discrete ticks 0..9 and note about unmapped
    cbar = fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.02, ticks=list(range(0,10)))
    cbar.set_label("Fire severity (0–9); unmapped tiles are transparent")
    if save_path:
        _ensure_parent_dir(save_path)
        plt.tight_layout(); plt.savefig(save_path, dpi=200)
    return fig, ax

# -----------------------------------------------------------------------------
# In-place updating visualizer for long-running loops
# -----------------------------------------------------------------------------
class FireViz:
    def __init__(self,
                 figsize: Tuple[int, int] = (18, 5),
                 pred_alpha: float = 0.35,
                 wind_stride: int = 5):
        self.pred_alpha = pred_alpha
        self.wind_stride = wind_stride

        # Base figure/axes
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect('equal')
        # Ensure no grid/ticks are shown
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Initialize empty layers with custom colormap/norm
        self.cmap, self.norm = make_fire_cmap_norm()
        base = np.full((HEIGHT, WIDTH), np.nan)
        self.im_hist = self.ax.imshow(np.ma.masked_invalid(base), origin='upper', interpolation='nearest', cmap=self.cmap, norm=self.norm)
        self.im_pred = self.ax.imshow(np.ma.masked_invalid(base), origin='upper', interpolation='nearest', alpha=self.pred_alpha, cmap=self.cmap, norm=self.norm)

        # Overlays
        self.cit_scatter = self.ax.scatter([], [], s=12, marker='^', label='Citizen', edgecolors='none', linewidths=0)
        self.ff_scatter = self.ax.scatter([], [], s=12, marker='s', label='Firefighter', edgecolors='none', linewidths=0)
        self.drone_scatter = self.ax.scatter([], [], s=70, marker='X', label='Drone', edgecolors='black', linewidths=0.5, alpha=0.9)
        self.drone_labels = {}
        self.scan_patches = []
        self.wind_artists = []

        self.ax.set_xlim(-0.5, WIDTH-0.5)
        self.ax.set_ylim(HEIGHT-0.5, -0.5)
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y")

        handles = [Line2D([0],[0], marker="^", linestyle="None", markersize=6, label="Citizen"),
                   Line2D([0],[0], marker="s", linestyle="None", markersize=6, label="Firefighter"),
                   Line2D([0],[0], marker="X", linestyle="None", markersize=8, label="Drone")]
        self.ax.legend(handles=handles, loc="upper right")
        self.cbar = self.fig.colorbar(self.im_hist, ax=self.ax, fraction=0.025, pad=0.02, ticks=list(range(0,10)))
        self.cbar.set_label("Fire severity (0–9); unmapped tiles are transparent")

    def _clear_artists(self):
        for p in self.scan_patches:
            try:
                p.remove()
            except Exception:
                pass
        self.scan_patches.clear()
        for a in self.wind_artists:
            try:
                a.remove()
            except Exception:
                pass
        self.wind_artists.clear()

    def draw(self,
             hist_map: List[List[str]],
             pred_map: List[List[str]],
             drones: List[Drone],
             title: Optional[str] = None,
             save_path: Optional[str] = None,
             show_scan_windows: bool = True):
        arrays_hist = tiles_to_arrays(hist_map)
        arrays_pred = tiles_to_arrays(pred_map)

        # Update raster layers
        self.im_hist.set_data(np.ma.masked_invalid(arrays_hist["fire"]))
        self.im_pred.set_data(np.ma.masked_invalid(arrays_pred["fire"]))

        # Update scatters
        cy, cx = np.where(arrays_hist["citizen"] | arrays_pred["citizen"])
        fy, fx = np.where(arrays_hist["firefighter"] | arrays_pred["firefighter"])
        self.cit_scatter.set_offsets(np.c_[cx, cy] if len(cx) else np.empty((0,2)))
        self.ff_scatter.set_offsets(np.c_[fx, fy] if len(fx) else np.empty((0,2)))

        # Drones and labels
        dxs, dys = [], []
        names_present = set()
        for d in drones:
            x, y = d.location()
            dxs.append(x); dys.append(y)
            names_present.add(d.name)
            if d.name not in self.drone_labels:
                self.drone_labels[d.name] = self.ax.text(x+1, y-1, d.name, fontsize=8, ha='left', va='center')
            else:
                self.drone_labels[d.name].set_position((x+1, y-1))
        # Remove labels for drones that disappeared
        for name in list(self.drone_labels.keys()):
            if name not in names_present:
                try:
                    self.drone_labels[name].remove()
                except Exception:
                    pass
                del self.drone_labels[name]
        self.drone_scatter.set_offsets(np.c_[dxs, dys] if len(dxs) else np.empty((0,2)))

        # Clear and redraw scan windows (wind arrows disabled)
        self._clear_artists()
        if show_scan_windows:
            for d in drones:
                x, y = d.location()
                rect = Rectangle((x-1, y-1), 3, 3, linewidth=1, edgecolor=None, facecolor='grey', alpha=0.15)
                self.ax.add_patch(rect)
                self.scan_patches.append(rect)
        # Wind arrows removed entirely

        if title:
            self.ax.set_title(title)
        if save_path:
            _ensure_parent_dir(save_path)
            self.fig.tight_layout()
            self.fig.savefig(save_path, dpi=200)
        # Nudge the GUI to render this update; caller can still call plt.pause()
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass
        return self.fig, self.ax

# --- helper to compute bounds that fully contain the rotated W×H rectangle ---
def rotated_bounds(w, h, deg):
    cx, cy = w / 2.0, h / 2.0
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    corners[:, 0] -= cx; corners[:, 1] -= cy
    R = np.array([[c, -s], [s, c]])
    rc = corners @ R.T
    rc[:, 0] += cx; rc[:, 1] += cy
    xmin, ymin = rc.min(axis=0)
    xmax, ymax = rc.max(axis=0)
    return xmin, xmax, ymin, ymax

# --- demo rounds generator (swap with your real per-round data) ---
def make_rounds(num_rounds=25, seed=135):
    rng = np.random.default_rng(seed)
    rounds = []
    hist, pred = generate_dummy_maps(seed=seed)
    drones = [Drone(x, y, name) for (name, x, y) in BASE_STATIONS]
    tracks = {d.name: [(d.x, d.y)] for d in drones}

    for r in range(num_rounds):
        new_pred = [row[:] for row in pred]

        # simple fire propagation
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if hist[y][x] != "\0":
                    t = parse_tile(hist[y][x])
                    if not t:
                        continue
                    sev, wdir, wind = t["fire"], t["winddir"], t["windspeed"]
                    if sev >= 4:
                        dx, dy = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}[wdir]
                        nx = max(0, min(WIDTH-1, x+dx))
                        ny = max(0, min(HEIGHT-1, y+dy))
                        t2 = parse_tile(new_pred[ny][nx]) if new_pred[ny][nx] != "\0" else None
                        sev2 = max(0, min(9, sev//2))
                        if t2 is None or sev2 > t2["fire"]:
                            new_pred[ny][nx] = encode_tile(nx, ny, sev2, wind, wdir,
                                                            t["citizen"], t["firefighter"], n=r, trust=5)

        # move drones (demo)
        for d in drones:
            dx, dy = int(np.random.randint(-3, 4)), int(np.random.randint(-2, 3))
            d.x = int(np.clip(d.x + dx, 0, WIDTH-1))
            d.y = int(np.clip(d.y + dy, 0, HEIGHT-1))
            tracks[d.name].append((d.x, d.y))

        pred = new_pred
        rounds.append((hist, pred, [Drone(d.x, d.y, d.name) for d in drones]))

        # reveal newly scanned tiles (3×3)
        for d in drones:
            for yy in range(max(0, d.y-1), min(HEIGHT, d.y+2)):
                for xx in range(max(0, d.x-1), min(WIDTH, d.x+2)):
                    if hist[yy][xx] == "\0":
                        hist[yy][xx] = encode_tile(xx, yy, 0,
                                                    int(np.random.randint(0, 4)),
                                                    int(np.random.randint(0, 4)),
                                                    0, 0, n=r, trust=2)
    return rounds, tracks

# --- main animation builder ---
def build_anim_data_only(
    rounds,
    tracks,
    rotate_deg=360.0,                  # data rotated CCW; 300° = 15° clockwise from 315°
    pred_alpha=0.35,
    wind_stride=8,
    out_path="fire_viz_anim_rot135_basemap.gif",
    bg_image_path: Optional[str] = None,
    bg_alpha: float = 0.55,            # basemap opacity
    bg_extent: Optional[tuple] = None, # (xmin, xmax, ymin, ymax) in data tile coordinates
):
    H, W = HEIGHT, WIDTH
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_axis_off()
    ax.set_facecolor('none')  # transparent so background basemap shows through

    # Fit the rotated data fully in view
    xmin, xmax, ymin, ymax = rotated_bounds(W, H, rotate_deg)
    pad = 5
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymax + pad, ymin - pad)  # y inverted for image-style view
    ax.set_aspect('equal')               # preserve aspect -> no squish/stretch

    # Rotation transform for data layers ONLY
    base_trans = Affine2D().rotate_deg_around(W/2.0, H/2.0, rotate_deg) + ax.transData

    # --- Basemap as full-figure background (NO rotation, not constrained to grid) ---
    if bg_image_path and os.path.exists(bg_image_path):
        img = plt.imread(bg_image_path)
        ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-100)
        ax_bg.imshow(img, origin="upper", alpha=bg_alpha)
        ax_bg.set_axis_off()
        # Note: This background fills the figure; it preserves image aspect and is not tied to grid extents.

    # First-frame arrays
    first_hist, first_pred, _ = rounds[0]
    arrays_hist = tiles_to_arrays(first_hist)
    arrays_pred = tiles_to_arrays(first_pred)

    # Fire layers (rotated) with custom colormap/norm
    cmap, norm = make_fire_cmap_norm()
    hist_img = ax.imshow(np.ma.masked_invalid(arrays_hist["fire"]),
                         origin='upper', interpolation='nearest',
                         zorder=-5, clip_on=False, cmap=cmap, norm=norm)
    pred_img = ax.imshow(np.ma.masked_invalid(arrays_pred["fire"]),
                         origin='upper', interpolation='nearest',
                         alpha=pred_alpha, zorder=-4, clip_on=False, cmap=cmap, norm=norm)
    hist_img.set_transform(base_trans)
    pred_img.set_transform(base_trans)

    # Overlays (rotated)
    wind_artists = []
    cit_scatter = ax.scatter([], [], s=12, marker='^', zorder=5, transform=base_trans, clip_on=False,
                             edgecolors='none', linewidths=0)
    ff_scatter  = ax.scatter([], [], s=12, marker='s', zorder=5, transform=base_trans, clip_on=False,
                             edgecolors='none', linewidths=0)
    drone_scatter = ax.scatter([], [], s=70, marker='X', zorder=6, transform=base_trans, clip_on=False,
                               edgecolors='black', linewidths=0.5, alpha=0.9)

    # Static base stations (blue squares)
    bs_x = [x for (name, x, y) in BASE_STATIONS]
    bs_y = [y for (name, x, y) in BASE_STATIONS]
    base_scatter = ax.scatter(bs_x, bs_y, s=60, marker='s', color='blue', zorder=6, transform=base_trans, clip_on=False)

    drone_labels: Dict[str, any] = {}
    path_lines: Dict[str, Line2D] = {}
    for name in tracks.keys():
        line, = ax.plot([], [], linewidth=0.7, alpha=0.4, zorder=4,
                         solid_capstyle='round', transform=base_trans, clip_on=False)
        path_lines[name] = line

    # HUD (screen-aligned; comment out if you want pure data)
    hud = ax.text(0.01, 0.99, "Round 0", transform=fig.transFigure,
                  ha='left', va='top', fontsize=12,
                  bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    def update(i):
        for a in wind_artists: a.remove()
        wind_artists.clear()

        hist, pred, drones = rounds[i]
        arrays_hist = tiles_to_arrays(hist)
        arrays_pred = tiles_to_arrays(pred)

        hist_img.set_data(np.ma.masked_invalid(arrays_hist["fire"]))
        pred_img.set_data(np.ma.masked_invalid(arrays_pred["fire"]))

        cy, cx = np.where(arrays_hist["citizen"] | arrays_pred["citizen"])
        fy, fx = np.where(arrays_hist["firefighter"] | arrays_pred["firefighter"])
        cit_scatter.set_offsets(np.c_[cx, cy])
        ff_scatter.set_offsets(np.c_[fx, fy])

        dxs, dys = [], []
        for d in drones:
            x, y = d.location()
            dxs.append(x); dys.append(y)
            if d.name not in drone_labels:
                drone_labels[d.name] = ax.text(x+1, y-1, d.name, fontsize=8,
                                               ha='left', va='center',
                                               zorder=7, transform=base_trans, clip_on=False)
            else:
                drone_labels[d.name].set_position((x+1, y-1))
        drone_scatter.set_offsets(np.c_[dxs, dys])

        # Drone paths
        for name, line in path_lines.items():
            pts = tracks[name][:i+1]
            if pts:
                xs, ys = zip(*pts)
                line.set_data(xs, ys)

        # Wind arrows disabled by request

        hud.set_text(f"Round {i}")


    total_frames = len(rounds)
    anim = FuncAnimation(fig, update, frames=total_frames, interval=180, blit=False)
    writer = PillowWriter(fps=6)
    _ensure_parent_dir(out_path)

    # Progress indicator: prints percentage as frames are encoded
    last_pct = {"p": -1}
    def _progress_cb(i, n):
        try:
            pct = int(((i + 1) * 100) / max(1, n))
        except Exception:
            pct = 0
        if pct != last_pct["p"]:
            print(f"\rSaving GIF: {pct}% ({i+1}/{n})", end="", flush=True)
            last_pct["p"] = pct

    try:
        anim.save(out_path, writer=writer, dpi=120, progress_callback=_progress_cb)
        print()  # newline after progress
    except TypeError:
        # Older Matplotlib without progress_callback support
        anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    return out_path

# --- run demo (provide your basemap here) ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fire Viz Animation")
    parser.add_argument("--rotate", type=float, default=-30, help="Data rotation in degrees CCW (e.g., 300 is 15° clockwise from 315)")
    parser.add_argument("--rounds", type=int, default=25, help="Number of rounds/frames (only for demo data mode)")
    parser.add_argument("--outfile", type=str, default="GIFs/fire_viz_anim.gif", help="Output GIF path (default: GIFs/fire_viz_anim.gif)")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of frames rendered from outputs (0 = all)")
    parser.add_argument("--bg", type=str, default="", help="Background image path (optional)")
    parser.add_argument("--from-outputs", action='store_true', help="Use C++ simulator outputs (vision/, paths/, maps/) instead of demo data")
    parser.add_argument("--vision-dir", type=str, default="vision", help="Directory containing vision_roundNNN.txt files")
    parser.add_argument("--paths-dir", type=str, default="paths", help="Directory containing paths_roundNNN_droneK.csv files")
    parser.add_argument("--map", type=str, default="", help="Optional map.txt path (uses latest map only)")
    parser.add_argument("--maps-dir", type=str, default="maps", help="Directory containing per-round full maps (map_roundNNN.txt)")
    parser.add_argument("--vision-only", action='store_true', help="Ignore archived full maps and visualize only drone vision (FoV)")
    args = parser.parse_args()

    if args.from_outputs:
        # If vision-only requested, ignore maps inputs explicitly
        maps_dir = None if args.vision_only else (args.maps_dir if args.maps_dir else None)
        map_path = None if args.vision_only else (args.map if args.map else None)
        rounds, tracks = build_rounds_from_outputs(
            vision_dir=args.vision_dir,
            paths_dir=args.paths_dir,
            map_path=map_path,
            maps_dir=maps_dir
        )
        # Optionally limit frames from outputs
        if args.max_frames and args.max_frames > 0 and len(rounds) > args.max_frames:
            limit = max(1, args.max_frames)
            rounds = rounds[:limit]
            for name in list(tracks.keys()):
                tracks[name] = tracks[name][:limit]
        if not rounds:
            print("No rounds discovered under vision/ and paths/.")
            raise SystemExit(1)
    else:
        rounds, tracks = make_rounds(num_rounds=args.rounds, seed=135)

    out = build_anim_data_only(
        rounds, tracks,
        rotate_deg=(args.rotate if args.rotate is not None else -30),
        pred_alpha=0.35,
        wind_stride=8,
        out_path=args.outfile,
        bg_image_path=(args.bg if args.bg else None),
        bg_alpha=0.55,
        bg_extent=(0, WIDTH, 0, HEIGHT)
    )
    print(f"Saved to: {out}")

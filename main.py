"""
Subak Irrigation System Simulation
===================================
Modern reimplementation of Lansing & Kremer (1993)
"Emergent Properties of Balinese Water Temple Networks"

Simulates the tradeoff between pest synchronization and water competition
in Balinese rice terrace networks. Demonstrates emergent self-organization
from simple local decision rules.
"""

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


# --- Model Parameters ---

GRID_SIZE = 12          # 12x12 grid of subaks
N_PHASES = 6            # Planting phase options (bi-monthly within a year)
N_STEPS = 50            # Simulation years
Y_MAX = 8.6             # Maximum yield (tons/ha) under ideal conditions
ALPHA = 0.6             # Pest damage sensitivity
BETA = 0.4              # Water stress sensitivity
PEST_SYNC_MIN = 0.02    # Minimum pest loss at full synchrony
PEST_SYNC_MAX = 0.50    # Maximum pest loss at zero synchrony
WATER_BASE_FLOW = 1.0   # Normalized water flow at top of watershed
WATER_NEED = 0.15       # Water needed per subak per phase
MUTATION_RATE = 0.05    # Probability of random schedule exploration


@dataclass
class Subak:
    """A single irrigation community (agent)."""
    row: int
    col: int
    phase: int = 0           # Current planting phase (0 to N_PHASES-1)
    yield_val: float = 0.0   # Last harvest yield
    pest_loss: float = 0.0
    water_loss: float = 0.0

    def __post_init__(self):
        self.phase = random.randint(0, N_PHASES - 1)


def get_neighbors(row, col, grid_size):
    """Return 4-connected neighbor coordinates."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            neighbors.append((nr, nc))
    return neighbors


def compute_pest_loss(subak, grid, grid_size):
    """
    Pest damage depends on synchronization with neighbors.
    More neighbors sharing the same planting phase = lower pest pressure
    (synchronized fallow periods starve pests).
    """
    neighbors = get_neighbors(subak.row, subak.col, grid_size)
    if not neighbors:
        return PEST_SYNC_MAX

    sync_count = 0
    for nr, nc in neighbors:
        phase_diff = abs(grid[nr][nc].phase - subak.phase)
        phase_diff = min(phase_diff, N_PHASES - phase_diff)
        if phase_diff <= 1:
            sync_count += 1

    sync_ratio = sync_count / len(neighbors)
    pest_loss = PEST_SYNC_MAX - (PEST_SYNC_MAX - PEST_SYNC_MIN) * sync_ratio
    return pest_loss


def compute_water_loss(subak, grid, grid_size):
    """
    Water stress depends on upstream demand.
    Water flows from row 0 (upstream) to row N (downstream).
    """
    row = subak.row
    phase = subak.phase

    upstream_demand = 0
    upstream_count = 0
    for r in range(row):
        for c in range(grid_size):
            upstream_count += 1
            phase_diff = abs(grid[r][c].phase - phase)
            phase_diff = min(phase_diff, N_PHASES - phase_diff)
            if phase_diff == 0:
                upstream_demand += 1

    if upstream_count == 0:
        return 0.0

    demand_ratio = upstream_demand / upstream_count
    position_factor = row / (grid_size - 1)
    water_loss = demand_ratio * position_factor * 0.8

    return min(water_loss, 0.9)


def compute_yield(subak, grid, grid_size):
    """Compute harvest yield given pest and water losses."""
    pest_loss = compute_pest_loss(subak, grid, grid_size)
    water_loss = compute_water_loss(subak, grid, grid_size)

    subak.pest_loss = pest_loss
    subak.water_loss = water_loss
    subak.yield_val = Y_MAX * (1 - ALPHA * pest_loss) * (1 - BETA * water_loss)
    return subak.yield_val


def update_decisions(grid, grid_size):
    """
    Each subak observes its neighbors' yields and copies the strategy
    of the most successful neighbor (best-neighbor imitation rule).
    """
    new_phases = [[0] * grid_size for _ in range(grid_size)]

    for r in range(grid_size):
        for c in range(grid_size):
            if random.random() < MUTATION_RATE:
                new_phases[r][c] = random.randint(0, N_PHASES - 1)
                continue

            best_yield = grid[r][c].yield_val
            best_phase = grid[r][c].phase

            for nr, nc in get_neighbors(r, c, grid_size):
                if grid[nr][nc].yield_val > best_yield:
                    best_yield = grid[nr][nc].yield_val
                    best_phase = grid[nr][nc].phase

            new_phases[r][c] = best_phase

    for r in range(grid_size):
        for c in range(grid_size):
            grid[r][c].phase = new_phases[r][c]


def run_simulation():
    """Run the full simulation and collect data for visualization."""
    grid = [[Subak(row=r, col=c) for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]

    history = []

    for step in range(N_STEPS):
        yields = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                y = compute_yield(grid[r][c], grid, GRID_SIZE)
                yields.append(y)

        # Compute phase distribution for ritual calendar visualization
        phase_counts = [0] * N_PHASES
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                phase_counts[grid[r][c].phase] += 1

        # Compute synchrony index (entropy-based)
        total = GRID_SIZE * GRID_SIZE
        entropy = 0
        for count in phase_counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        max_entropy = math.log2(N_PHASES)
        synchrony = 1 - (entropy / max_entropy) if max_entropy > 0 else 1

        snapshot = {
            "step": step,
            "avg_yield": sum(yields) / len(yields),
            "min_yield": min(yields),
            "max_yield": max(yields),
            "avg_pest_loss": sum(grid[r][c].pest_loss for r in range(GRID_SIZE) for c in range(GRID_SIZE)) / total,
            "avg_water_loss": sum(grid[r][c].water_loss for r in range(GRID_SIZE) for c in range(GRID_SIZE)) / total,
            "phases": [[grid[r][c].phase for c in range(GRID_SIZE)] for r in range(GRID_SIZE)],
            "yields": [[round(grid[r][c].yield_val, 2) for c in range(GRID_SIZE)] for r in range(GRID_SIZE)],
            "phase_counts": phase_counts,
            "synchrony": round(synchrony, 3),
        }
        history.append(snapshot)

        update_decisions(grid, GRID_SIZE)

    return history


def generate_html(history):
    """Generate an interactive HTML visualization with Miratuku CI."""
    html_path = Path(__file__).parent / 'template.html'
    return html_path.read_text(encoding='utf-8')


def main():
    print("Running Subak simulation...")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE} = {GRID_SIZE**2} subaks")
    print(f"  Phases: {N_PHASES}, Steps: {N_STEPS}")
    print()

    history = run_simulation()

    initial = history[0]
    final = history[-1]
    print(f"  Year  0: avg yield = {initial['avg_yield']:.2f} t/ha "
          f"(pest: {initial['avg_pest_loss']:.1%}, water: {initial['avg_water_loss']:.1%}, sync: {initial['synchrony']:.3f})")
    print(f"  Year {N_STEPS-1}: avg yield = {final['avg_yield']:.2f} t/ha "
          f"(pest: {final['avg_pest_loss']:.1%}, water: {final['avg_water_loss']:.1%}, sync: {final['synchrony']:.3f})")
    print(f"  Improvement: {((final['avg_yield'] / initial['avg_yield']) - 1) * 100:+.1f}%")
    print()

    html = generate_html(history)
    html = html.replace('__SIMULATION_DATA__', json.dumps(history))
    html = html.replace('__GRID_SIZE__', str(GRID_SIZE))
    html = html.replace('__N_PHASES__', str(N_PHASES))
    html = html.replace('__Y_MAX__', str(Y_MAX))
    html = html.replace('__MAX_STEP__', str(N_STEPS - 1))

    out_path = Path(__file__).parent / 'subak_simulation.html'
    out_path.write_text(html, encoding='utf-8')
    print(f"  Output: {out_path}")
    print("  Open in browser to explore the simulation.")


if __name__ == '__main__':
    main()

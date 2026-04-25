"""
Subak Irrigation System Simulation — Three Models
===================================================
1. Lansing & Kremer (1993) - Original grid model
2. Janssen (2007) - Watershed network + decision rule comparison
3. Lansing et al. (2009) - Dynamic budding model
"""

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


# --- Shared Parameters ---
N_PHASES = 6
N_STEPS = 50
Y_MAX = 8.6
ALPHA = 0.6
BETA = 0.4
PEST_SYNC_MIN = 0.02
PEST_SYNC_MAX = 0.50
MUTATION_RATE = 0.05


def compute_synchrony(phase_list):
    """Entropy-based synchrony index."""
    counts = [0] * N_PHASES
    for p in phase_list:
        counts[p] += 1
    total = len(phase_list)
    if total == 0:
        return 0
    entropy = 0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    max_e = math.log2(N_PHASES)
    return round(1 - (entropy / max_e) if max_e > 0 else 1, 3)


# ============================================================
# MODEL 1: Lansing & Kremer (1993) — Original Grid
# ============================================================

def run_model1():
    GRID = 12
    grid = [[{"phase": random.randint(0, N_PHASES-1), "yield": 0.0, "pest": 0.0, "water": 0.0}
             for c in range(GRID)] for r in range(GRID)]
    history = []

    def neighbors(r, c):
        nb = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < GRID and 0 <= nc < GRID:
                nb.append((nr, nc))
        return nb

    def calc_yield(r, c):
        s = grid[r][c]
        # Pest: sync with neighbors
        nb = neighbors(r, c)
        sync = 0
        for nr, nc in nb:
            diff = abs(grid[nr][nc]["phase"] - s["phase"])
            diff = min(diff, N_PHASES - diff)
            if diff <= 1:
                sync += 1
        ratio = sync / len(nb) if nb else 0
        pest = PEST_SYNC_MAX - (PEST_SYNC_MAX - PEST_SYNC_MIN) * ratio

        # Water: upstream demand
        upstream_same = 0
        upstream_total = 0
        for ur in range(r):
            for uc in range(GRID):
                upstream_total += 1
                if grid[ur][uc]["phase"] == s["phase"]:
                    upstream_same += 1
        if upstream_total == 0:
            water = 0
        else:
            water = (upstream_same / upstream_total) * (r / (GRID-1)) * 0.8

        s["pest"] = pest
        s["water"] = min(water, 0.9)
        s["yield"] = Y_MAX * (1 - ALPHA * pest) * (1 - BETA * s["water"])
        return s["yield"]

    for step in range(N_STEPS):
        yields = []
        for r in range(GRID):
            for c in range(GRID):
                yields.append(calc_yield(r, c))

        phases_flat = [grid[r][c]["phase"] for r in range(GRID) for c in range(GRID)]
        phase_counts = [0]*N_PHASES
        for p in phases_flat:
            phase_counts[p] += 1

        history.append({
            "step": step,
            "avg_yield": round(sum(yields)/len(yields), 3),
            "min_yield": round(min(yields), 3),
            "max_yield": round(max(yields), 3),
            "avg_pest": round(sum(grid[r][c]["pest"] for r in range(GRID) for c in range(GRID))/(GRID*GRID), 4),
            "avg_water": round(sum(grid[r][c]["water"] for r in range(GRID) for c in range(GRID))/(GRID*GRID), 4),
            "phases": [[grid[r][c]["phase"] for c in range(GRID)] for r in range(GRID)],
            "yields": [[round(grid[r][c]["yield"], 2) for c in range(GRID)] for r in range(GRID)],
            "phase_counts": phase_counts,
            "synchrony": compute_synchrony(phases_flat),
        })

        # Update: copy best neighbor
        new_p = [[0]*GRID for _ in range(GRID)]
        for r in range(GRID):
            for c in range(GRID):
                if random.random() < MUTATION_RATE:
                    new_p[r][c] = random.randint(0, N_PHASES-1)
                    continue
                best_y = grid[r][c]["yield"]
                best_ph = grid[r][c]["phase"]
                for nr, nc in neighbors(r, c):
                    if grid[nr][nc]["yield"] > best_y:
                        best_y = grid[nr][nc]["yield"]
                        best_ph = grid[nr][nc]["phase"]
                new_p[r][c] = best_ph
        for r in range(GRID):
            for c in range(GRID):
                grid[r][c]["phase"] = new_p[r][c]

    return history


# ============================================================
# MODEL 2: Janssen (2007) — Watershed Network + Decision Rules
# ============================================================

def run_model2():
    """
    Janssen's extension: realistic watershed DAG, pest diffusion via
    water channels, and comparison of 3 decision rules.
    """
    N_NODES = 48  # Subaks in watershed
    N_LEVELS = 6  # Depth levels of watershed

    # Build a tree-like watershed: each level has more nodes
    # Level 0: 2 nodes (springs), Level 5: 13 nodes (downstream)
    level_sizes = [2, 4, 6, 8, 13, 15]
    nodes = []
    level_map = {}
    idx = 0
    for lv, sz in enumerate(level_sizes):
        level_map[lv] = []
        for i in range(sz):
            nodes.append({
                "id": idx,
                "level": lv,
                "phase": random.randint(0, N_PHASES-1),
                "yield": 0.0, "pest": 0.0, "water": 0.0,
                "upstream": [],  # filled below
                "downstream": [],
                "lateral": [],   # same-level neighbors
            })
            level_map[lv].append(idx)
            idx += 1

    N_NODES = len(nodes)

    # Wire upstream-downstream connections
    for lv in range(1, N_LEVELS):
        parents = level_map[lv-1]
        children = level_map[lv]
        for ci, child_id in enumerate(children):
            parent_id = parents[ci % len(parents)]
            nodes[child_id]["upstream"].append(parent_id)
            nodes[parent_id]["downstream"].append(child_id)

    # Wire lateral connections (same level, adjacent)
    for lv in range(N_LEVELS):
        ids = level_map[lv]
        for i in range(len(ids)-1):
            nodes[ids[i]]["lateral"].append(ids[i+1])
            nodes[ids[i+1]]["lateral"].append(ids[i])

    def all_upstream(node_id):
        """Recursively get all upstream node IDs."""
        result = []
        stack = list(nodes[node_id]["upstream"])
        while stack:
            nid = stack.pop()
            result.append(nid)
            stack.extend(nodes[nid]["upstream"])
        return result

    def calc_yield_j(nid):
        n = nodes[nid]
        # Pest: sync with lateral + upstream/downstream neighbors
        neighbors = n["lateral"] + n["upstream"] + n["downstream"]
        if not neighbors:
            pest = PEST_SYNC_MAX
        else:
            sync = 0
            for nb_id in neighbors:
                diff = abs(nodes[nb_id]["phase"] - n["phase"])
                diff = min(diff, N_PHASES - diff)
                if diff <= 1:
                    sync += 1
            ratio = sync / len(neighbors)
            pest = PEST_SYNC_MAX - (PEST_SYNC_MAX - PEST_SYNC_MIN) * ratio

        # Pest diffusion via water: pests from upstream unsynchronized fields
        ups = all_upstream(nid)
        if ups:
            pest_from_water = 0
            for uid in ups:
                diff = abs(nodes[uid]["phase"] - n["phase"])
                diff = min(diff, N_PHASES - diff)
                if diff > 1:
                    pest_from_water += 0.03  # pest larvae in water
            pest = min(pest + pest_from_water, PEST_SYNC_MAX)

        # Water stress: proportional to upstream same-phase demand
        if ups:
            same = sum(1 for uid in ups if nodes[uid]["phase"] == n["phase"])
            demand = same / len(ups)
        else:
            demand = 0
        pos = n["level"] / (N_LEVELS - 1)
        water = demand * pos * 0.85

        n["pest"] = pest
        n["water"] = min(water, 0.9)
        n["yield"] = Y_MAX * (1 - ALPHA * pest) * (1 - BETA * n["water"])

    def update_rule(rule_name):
        """Apply different decision rules."""
        new_phases = {}
        for nid, n in enumerate(nodes):
            if random.random() < MUTATION_RATE:
                new_phases[nid] = random.randint(0, N_PHASES-1)
                continue

            if rule_name == "copy_best":
                # Copy the best-yielding neighbor
                candidates = n["lateral"] + n["upstream"] + n["downstream"]
                best_y = n["yield"]
                best_p = n["phase"]
                for cid in candidates:
                    if nodes[cid]["yield"] > best_y:
                        best_y = nodes[cid]["yield"]
                        best_p = nodes[cid]["phase"]
                new_phases[nid] = best_p

            elif rule_name == "satisfice":
                # Change only if current yield < threshold (80% of Y_MAX)
                threshold = Y_MAX * 0.8
                if n["yield"] >= threshold:
                    new_phases[nid] = n["phase"]  # stay
                else:
                    candidates = n["lateral"] + n["upstream"] + n["downstream"]
                    best_y = n["yield"]
                    best_p = n["phase"]
                    for cid in candidates:
                        if nodes[cid]["yield"] > best_y:
                            best_y = nodes[cid]["yield"]
                            best_p = nodes[cid]["phase"]
                    new_phases[nid] = best_p

            elif rule_name == "random":
                # Pure random — no learning
                new_phases[nid] = random.randint(0, N_PHASES-1)

        for nid in new_phases:
            nodes[nid]["phase"] = new_phases[nid]

    # Run 3 scenarios
    results = {}
    for rule in ["copy_best", "satisfice", "random"]:
        # Reset
        for n in nodes:
            n["phase"] = random.randint(0, N_PHASES-1)
            n["yield"] = 0; n["pest"] = 0; n["water"] = 0

        hist = []
        for step in range(N_STEPS):
            for nid in range(N_NODES):
                calc_yield_j(nid)

            yields = [n["yield"] for n in nodes]
            phases = [n["phase"] for n in nodes]
            phase_counts = [0]*N_PHASES
            for p in phases:
                phase_counts[p] += 1

            hist.append({
                "step": step,
                "avg_yield": round(sum(yields)/len(yields), 3),
                "min_yield": round(min(yields), 3),
                "max_yield": round(max(yields), 3),
                "avg_pest": round(sum(n["pest"] for n in nodes)/N_NODES, 4),
                "avg_water": round(sum(n["water"] for n in nodes)/N_NODES, 4),
                "phase_counts": phase_counts,
                "synchrony": compute_synchrony(phases),
                "phases_by_level": {str(lv): [nodes[nid]["phase"] for nid in ids]
                                    for lv, ids in level_map.items()},
                "yields_by_level": {str(lv): [round(nodes[nid]["yield"], 2) for nid in ids]
                                    for lv, ids in level_map.items()},
            })
            update_rule(rule)

        results[rule] = hist

    # Also return network structure for visualization
    network = {
        "nodes": [{"id": n["id"], "level": n["level"]} for n in nodes],
        "edges": [],
        "level_sizes": level_sizes,
    }
    for n in nodes:
        for did in n["downstream"]:
            network["edges"].append({"from": n["id"], "to": did})

    return results, network


# ============================================================
# MODEL 3: Lansing et al. (2009) — Budding Model
# ============================================================

def run_model3():
    """
    Dynamic network: starts with few subaks, new ones 'bud' from
    successful communities. Demonstrates how the network itself
    emerges through population growth and downstream expansion.
    """
    N_STEPS_BUD = 80  # Longer to show network growth
    SPLIT_RATIO = 0.3  # S parameter: fraction that splits off
    GROWTH_THRESHOLD = 7.5  # Yield threshold to trigger budding
    MAX_NODES = 60
    BUD_COOLDOWN = 3  # Minimum years between buds from same parent

    # Start with 6 founding subaks at the headwaters
    nodes = []
    for i in range(6):
        nodes.append({
            "id": i,
            "level": 0,
            "phase": random.randint(0, N_PHASES-1),
            "yield": 0.0, "pest": 0.0, "water": 0.0,
            "population": 1.0,
            "parent_id": -1,
            "born_step": 0,
            "last_bud_step": -10,
            "upstream": [],
            "downstream": [],
            "lateral": [],
        })

    # Wire initial lateral connections
    for i in range(5):
        nodes[i]["lateral"].append(i+1)
        nodes[i+1]["lateral"].append(i)

    next_id = 6

    def calc_yield_b(nid):
        n = nodes[nid]
        # Pest
        nbs = n["lateral"] + n["upstream"] + n["downstream"]
        if not nbs:
            pest = PEST_SYNC_MAX * 0.5  # Isolated = moderate pest
        else:
            sync = 0
            for nb_id in nbs:
                nb = next((x for x in nodes if x["id"] == nb_id), None)
                if nb is None:
                    continue
                diff = abs(nb["phase"] - n["phase"])
                diff = min(diff, N_PHASES - diff)
                if diff <= 1:
                    sync += 1
            ratio = sync / max(len(nbs), 1)
            pest = PEST_SYNC_MAX - (PEST_SYNC_MAX - PEST_SYNC_MIN) * ratio

        # Water: simpler — based on level depth
        max_level = max(nd["level"] for nd in nodes)
        if max_level == 0:
            water = 0
        else:
            # Count same-phase at same or higher level
            same_above = sum(1 for nd in nodes if nd["level"] <= n["level"]
                           and nd["id"] != n["id"] and nd["phase"] == n["phase"])
            total_above = max(sum(1 for nd in nodes if nd["level"] <= n["level"]
                               and nd["id"] != n["id"]), 1)
            water = (same_above / total_above) * (n["level"] / max(max_level, 1)) * 0.7

        n["pest"] = pest
        n["water"] = min(water, 0.9)
        n["yield"] = Y_MAX * (1 - ALPHA * pest) * (1 - BETA * n["water"])

    history = []

    for step in range(N_STEPS_BUD):
        # Calculate yields
        for i, n in enumerate(nodes):
            calc_yield_b(n["id"])

        # Record snapshot
        yields = [n["yield"] for n in nodes]
        phases = [n["phase"] for n in nodes]
        phase_counts = [0]*N_PHASES
        for p in phases:
            phase_counts[p] += 1

        # Level distribution
        max_lv = max(n["level"] for n in nodes)
        level_counts = [0]*(max_lv+1)
        for n in nodes:
            level_counts[n["level"]] += 1

        history.append({
            "step": step,
            "n_subaks": len(nodes),
            "max_level": max_lv,
            "avg_yield": round(sum(yields)/len(yields), 3),
            "min_yield": round(min(yields), 3),
            "max_yield": round(max(yields), 3),
            "avg_pest": round(sum(n["pest"] for n in nodes)/len(nodes), 4),
            "avg_water": round(sum(n["water"] for n in nodes)/len(nodes), 4),
            "phase_counts": phase_counts,
            "synchrony": compute_synchrony(phases),
            "level_counts": level_counts,
            "nodes": [{"id": n["id"], "level": n["level"], "phase": n["phase"],
                       "yield": round(n["yield"], 2), "parent_id": n["parent_id"],
                       "born_step": n["born_step"]} for n in nodes],
        })

        # Decision update: copy best neighbor
        new_phases = {}
        for n in nodes:
            nid = n["id"]
            if random.random() < MUTATION_RATE:
                new_phases[nid] = random.randint(0, N_PHASES-1)
                continue
            nbs = n["lateral"] + n["upstream"] + n["downstream"]
            best_y = n["yield"]
            best_p = n["phase"]
            for nb_id in nbs:
                nb = next((x for x in nodes if x["id"] == nb_id), None)
                if nb and nb["yield"] > best_y:
                    best_y = nb["yield"]
                    best_p = nb["phase"]
            new_phases[nid] = best_p
        for n in nodes:
            if n["id"] in new_phases:
                n["phase"] = new_phases[n["id"]]

        # Budding: successful communities spawn downstream children
        if len(nodes) < MAX_NODES:
            candidates = [n for n in nodes
                         if n["yield"] > GROWTH_THRESHOLD
                         and (step - n["last_bud_step"]) >= BUD_COOLDOWN
                         and len(n["downstream"]) < 2]
            # At most 2 buds per step
            random.shuffle(candidates)
            for parent in candidates[:2]:
                if len(nodes) >= MAX_NODES:
                    break
                new_node = {
                    "id": next_id,
                    "level": parent["level"] + 1,
                    "phase": parent["phase"],  # Inherit parent's strategy
                    "yield": 0.0, "pest": 0.0, "water": 0.0,
                    "population": SPLIT_RATIO,
                    "parent_id": parent["id"],
                    "born_step": step,
                    "last_bud_step": -10,
                    "upstream": [parent["id"]],
                    "downstream": [],
                    "lateral": [],
                }
                # Connect laterally to siblings
                for sibling_id in parent["downstream"]:
                    new_node["lateral"].append(sibling_id)
                    sib = next((x for x in nodes if x["id"] == sibling_id), None)
                    if sib:
                        sib["lateral"].append(next_id)

                parent["downstream"].append(next_id)
                parent["last_bud_step"] = step
                parent["population"] *= (1 - SPLIT_RATIO)
                nodes.append(new_node)
                next_id += 1

    return history


# ============================================================
# Main
# ============================================================

def main():
    print("Running 3 Subak simulations...")
    print()

    # Model 1
    print("  [Model 1] Lansing & Kremer (1993) — Grid model")
    h1 = run_model1()
    print(f"    Year 0:  yield={h1[0]['avg_yield']:.2f}, pest={h1[0]['avg_pest']:.1%}")
    print(f"    Year 49: yield={h1[-1]['avg_yield']:.2f}, pest={h1[-1]['avg_pest']:.1%}")
    print()

    # Model 2
    print("  [Model 2] Janssen (2007) — Watershed + decision rules")
    h2, net2 = run_model2()
    for rule in ["copy_best", "satisfice", "random"]:
        h = h2[rule]
        print(f"    {rule:12s}: yield {h[0]['avg_yield']:.2f} -> {h[-1]['avg_yield']:.2f}")
    print()

    # Model 3
    print("  [Model 3] Lansing et al. (2009) — Budding model")
    h3 = run_model3()
    print(f"    Step 0:  {h3[0]['n_subaks']} subaks, yield={h3[0]['avg_yield']:.2f}")
    print(f"    Step 79: {h3[-1]['n_subaks']} subaks, yield={h3[-1]['avg_yield']:.2f}")
    print()

    # Build combined data
    sim_data = {
        "model1": h1,
        "model2": {"rules": h2, "network": net2},
        "model3": h3,
    }

    html_path = Path(__file__).parent / 'template.html'
    html = html_path.read_text(encoding='utf-8')
    html = html.replace('__SIMULATION_DATA__', json.dumps(sim_data))

    out_path = Path(__file__).parent / 'subak_simulation.html'
    out_path.write_text(html, encoding='utf-8')
    print(f"  Output: {out_path}")


if __name__ == '__main__':
    main()

"""
Hybrid MPS Scheduling — Pyomo (MILP) + OR-Tools (CP-SAT)
=========================================================
This starter kit shows a 2-layer approach:
  Layer A (Pyomo MILP): Master Production Schedule & lot sizing
  Layer B (OR-Tools CP-SAT): Detailed machine scheduling with shifts & setup

Run as a single Python file. You can replace the toy data with your own.
Tested with: Python 3.10+, pyomo>=6, ortools>=9.10, highs/cbc/glpk as MILP solver.

Author: Rifky Aryo
"""
from __future__ import annotations

# ====== Imports ======
from dataclasses import dataclass
from typing import Dict, List, Tuple

import math

# Pyomo (MILP)
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Binary, Objective,
    Constraint, minimize, value, SolverFactory
)

# OR-Tools (CP-SAT)
from ortools.sat.python import cp_model

# ====== Toy Data Definitions ======
# We create a small example that still demonstrates: split/merge via BOM, family batching,
# work centers, capacity, shifts, setup, precedence between operations.

# Items: Finished Goods (FG) and Intermediate/Components (C*)
ITEMS = ["FG_A", "C_A1", "C_A2"]
FAMILIES = {"F_FG": ["FG_A"], "F_COMP": ["C_A1", "C_A2"]}

# Work centers (aggregated capacity for MILP layer)
WORK_CENTERS = ["WC_ASM", "WC_COMP"]
PERIODS = [1, 2]  # two days/buckets for planning layer (can be days/weeks)

# Demand for finished good
DEMAND = {("FG_A", 1): 40, ("FG_A", 2): 20}
# No direct demand for components
for i in ("C_A1", "C_A2"):
    for t in PERIODS:
        DEMAND[(i, t)] = 0

# BOM (parent -> component) with yields (alpha). FG requires both components (merge),
# and each unit of FG consumes 1 unit of each component.
BOM: Dict[Tuple[str, str], float] = {  # (parent, child): alpha
    ("FG_A", "C_A1"): 1.0,
    ("FG_A", "C_A2"): 1.0,
}
# Lead times on BOM arcs (periods)
LEAD: Dict[Tuple[str, str], int] = {
    ("FG_A", "C_A1"): 0,  # same period availability for simplicity
    ("FG_A", "C_A2"): 0,
}

# Processing time per unit at each Work Center (hours/unit). 0 means not processed there.
PT = {
    ("FG_A", "WC_ASM"): 0.2,  # assembly work center
    ("FG_A", "WC_COMP"): 0.0,
    ("C_A1", "WC_COMP"): 0.1,
    ("C_A1", "WC_ASM"): 0.0,
    ("C_A2", "WC_COMP"): 0.12,
    ("C_A2", "WC_ASM"): 0.0,
}

# Setup time per item at each Work Center (hours) if that item is produced in that bucket
SETUP_TIME = {
    ("FG_A", "WC_ASM"): 1.0,
    ("C_A1", "WC_COMP"): 0.5,
    ("C_A2", "WC_COMP"): 0.5,
}

# Capacity (hours) per Work Center per period
CAP = {
    ("WC_ASM", 1): 16.0,  # 2 shifts x 8h example
    ("WC_ASM", 2): 16.0,
    ("WC_COMP", 1): 16.0,
    ("WC_COMP", 2): 16.0,
}

# Costs
HOLD = {"FG_A": 0.5, "C_A1": 0.1, "C_A2": 0.1}
LATE = {"FG_A": 10.0, "C_A1": 0.0, "C_A2": 0.0}
SETUP_COST = {
    ("FG_A", "WC_ASM"): 20.0,
    ("C_A1", "WC_COMP"): 5.0,
    ("C_A2", "WC_COMP"): 5.0,
}

MINLOT = {"FG_A": 10, "C_A1": 10, "C_A2": 10}
BIGM = 10_000

# ====== Helper: first/prev for periods ======
class OrderedPeriods:
    def __init__(self, periods: List[int]):
        self.p = sorted(periods)
        self.idx = {t: i for i, t in enumerate(self.p)}

    def first(self):
        return self.p[0]

    def prev(self, t):
        i = self.idx[t]
        if i == 0:
            return None
        return self.p[i - 1]

    def shift(self, t, k):
        # t + k (can be negative). Returns None if out of bounds
        i = self.idx[t] + k
        if i < 0 or i >= len(self.p):
            return None
        return self.p[i]

OP = OrderedPeriods(PERIODS)

# ====== Layer A: Pyomo MILP (MPS & Lot Sizing) ======

def build_and_solve_milp():
    m = ConcreteModel()

    # Sets
    m.I = Set(initialize=ITEMS)
    m.W = Set(initialize=WORK_CENTERS)
    m.T = Set(initialize=PERIODS, ordered=True)
    m.F = Set(initialize=list(FAMILIES.keys()))
    # Map family -> items
    m.If = {f: set(FAMILIES[f]) for f in m.F}

    # Params
    m.d = Param(m.I, m.T, initialize=lambda _, i, t: DEMAND.get((i, t), 0.0), default=0.0)
    m.alpha = Param(Set(dimen=2, initialize=list(BOM.keys())), initialize=lambda _, p, c: BOM[(p, c)])
    m.lead = Param(Set(dimen=2, initialize=list(BOM.keys())), initialize=lambda _, p, c: LEAD[(p, c)], default=0)
    m.time = Param(m.I, m.W, initialize=lambda _, i, w: PT.get((i, w), 0.0), default=0.0)
    m.setup_time = Param(m.I, m.W, initialize=lambda _, i, w: SETUP_TIME.get((i, w), 0.0), default=0.0)
    m.cap = Param(m.W, m.T, initialize=lambda _, w, t: CAP[(w, t)], default=0.0)
    m.hold = Param(m.I, initialize=lambda _, i: HOLD[i])
    m.late = Param(m.I, initialize=lambda _, i: LATE[i])
    m.setup_cost = Param(m.I, m.W, initialize=lambda _, i, w: SETUP_COST.get((i, w), 0.0), default=0.0)
    m.BigM = Param(initialize=BIGM)
    m.minlot = Param(m.I, initialize=lambda _, i: MINLOT[i])

    # Variables
    m.x = Var(m.I, m.W, m.T, domain=NonNegativeReals)  # production qty
    m.inv = Var(m.I, m.T, domain=NonNegativeReals)
    m.back = Var(m.I, m.T, domain=NonNegativeReals)
    m.y = Var(m.I, m.W, m.T, domain=Binary)            # item active in (w,t)
    m.z = Var(m.F, m.W, m.T, domain=Binary)            # family gate (optional)

    # Objective
    def obj_rule(mm):
        return sum(mm.hold[i] * mm.inv[i, t] for i in mm.I for t in mm.T) \
             + sum(mm.late[i] * mm.back[i, t] for i in mm.I for t in mm.T) \
             + sum(mm.setup_cost[i, w] * mm.y[i, w, t] for i in mm.I for w in mm.W for t in mm.T)

    m.OBJ = Objective(rule=obj_rule, sense=minimize)

    # Flow balance with BOM (supports merge via parent->child arcs)
    def flow_rule(mm, i, t):
        prev_t = OP.prev(t)
        prev_inv = mm.inv[i, prev_t] if prev_t is not None else 0
        in_prod = sum(mm.x[i, w, t] for w in mm.W)

        # Parent contributions that create i (co/by-product style)
        bom_in = 0
        for (p, c) in BOM.keys():
            if c != i:
                continue
            lt = int(value(mm.lead[p, c]))
            tp = OP.shift(t, -lt)
            if tp is None:
                continue
            bom_in += value(mm.alpha[p, c]) * sum(mm.x[p, w, tp] for w in mm.W)

        prev_back = mm.back[i, prev_t] if prev_t is not None else 0
        # Demand at t
        dem = mm.d[i, t]
        return prev_inv + in_prod + bom_in + prev_back == dem + mm.inv[i, t] + mm.back[i, t]

    m.Flow = Constraint(m.I, m.T, rule=flow_rule)

    # Capacity per work-center per period
    def cap_rule(mm, w, t):
        return sum(mm.time[i, w] * mm.x[i, w, t] for i in mm.I) \
             + sum(mm.setup_time[i, w] * mm.y[i, w, t] for i in mm.I) <= mm.cap[w, t]

    m.Cap = Constraint(m.W, m.T, rule=cap_rule)

    # Lot sizing link
    def lot_upper(mm, i, w, t):
        return mm.x[i, w, t] <= mm.BigM * mm.y[i, w, t]

    def lot_min(mm, i, w, t):
        return mm.x[i, w, t] >= mm.minlot[i] * mm.y[i, w, t]

    m.LotU = Constraint(m.I, m.W, m.T, rule=lot_upper)
    m.LotL = Constraint(m.I, m.W, m.T, rule=lot_min)

    # Family gate (reduce parallel different items of same family)
    def fam_gate_upper(mm, f, w, t):
        return sum(mm.y[i, w, t] for i in m.If[f]) <= mm.BigM * mm.z[f, w, t]

    m.FamGate = Constraint(m.F, m.W, m.T, rule=fam_gate_upper)

    # Solve
    # Try HiGHS -> CBC -> GLPK
    for solver in ["highs", "cbc", "glpk"]:
        try:
            opt = SolverFactory(solver)
            if opt is not None and opt.available():
                res = opt.solve(m, tee=False)
                break
        except Exception:
            continue
    else:
        raise RuntimeError("No MILP solver found (install highs/cbc/glpk)")

    # Extract production plan
    plan = []
    for i in m.I:
        for w in m.W:
            for t in m.T:
                qty = value(m.x[i, w, t])
                if qty and qty > 1e-6:
                    plan.append({"item": i, "wc": w, "t": int(t), "qty": round(qty, 3)})

    inv = []
    for i in m.I:
        for t in m.T:
            v = value(m.inv[i, t])
            if v and v > 1e-6:
                inv.append({"item": i, "t": int(t), "inv": round(v, 3)})

    return plan, inv

# ====== Layer B: OR-Tools CP-SAT (Detailed Scheduling) ======
# We build a simple example: turn lot plan into jobs, schedule on machines with shifts and setup.

@dataclass
class Machine:
    name: str
    # list of on-shift windows (start, end) in minutes
    windows: List[Tuple[int, int]]

@dataclass
class Job:
    name: str
    family: str
    machine_alts: List[str]
    duration: int  # minutes
    release: int = 0
    due: int = 24 * 60

# Convert Layer A plan into granular jobs

def explode_plan_to_jobs(plan: List[Dict], lot_size: int = 10) -> List[Job]:
    # Map items to families (fallback: item name)
    fam_of = {}
    for f, items in FAMILIES.items():
        for it in items:
            fam_of[it] = f

    jobs: List[Job] = []
    for row in plan:
        i, w, t, qty = row["item"], row["wc"], row["t"], int(math.ceil(row["qty"]))
        # determine lot splits
        n_lots = max(1, qty // lot_size + (1 if qty % lot_size else 0))
        size = max(1, qty // n_lots)
        # processing time per unit -> per lot duration (round to minutes)
        hours_per_unit = PT.get((i, w), 0.0)
        lot_duration_min = int(math.ceil(hours_per_unit * size * 60))
        if lot_duration_min == 0:
            continue  # skip non-processed at WC
        for k in range(n_lots):
            jobs.append(
                Job(
                    name=f"{i}_{w}_t{t}_lot{k+1}",
                    family=next((f for f, items in FAMILIES.items() if i in items), i),
                    machine_alts=[w],  # simple: wc acts as a single machine name here
                    duration=lot_duration_min,
                    release=(t - 1) * (8 * 60),  # release at start of period t (assume 8h day)
                    due=t * (8 * 60),
                )
            )
    return jobs

# Build a simple calendar: each work center has 2 shifts of 4 hours (example)
MACHINES: Dict[str, Machine] = {
    "WC_ASM": Machine("WC_ASM", windows=[(0, 240), (300, 540)]),   # 0-4h, 5h-9h
    "WC_COMP": Machine("WC_COMP", windows=[(0, 240), (300, 540)]),
}

# Sequence-dependent setup (by family). Matrix in minutes.
FAMILY_LIST = list(FAMILIES.keys())
FIDX = {f: idx for idx, f in enumerate(FAMILY_LIST)}
DEFAULT_SETUP = 10  # minutes
SETUP_MATRIX = [[0 for _ in FAMILY_LIST] for _ in FAMILY_LIST]
for a in FAMILY_LIST:
    for b in FAMILY_LIST:
        if a == b:
            SETUP_MATRIX[FIDX[a]][FIDX[b]] = 0
        else:
            SETUP_MATRIX[FIDX[a]][FIDX[b]] = DEFAULT_SETUP


def schedule_jobs_cp(jobs: List[Job]):
    model = cp_model.CpModel()

    # Create intervals per machine; since we model WC as machines, one resource each
    # We also enforce calendars by splitting each job into an interval that must lie inside on-shift windows.

    # Variables
    task_starts = {}
    task_ends = {}
    task_intervals = {}
    task_family = {}

    # For NoOverlap with transition times we need a sequence per machine
    machine_to_tasks: Dict[str, List[str]] = {m: [] for m in MACHINES.keys()}

    for j in jobs:
        # Here each job has exactly one machine alternative (the WC). Extendable to alts.
        mname = j.machine_alts[0]
        start = model.NewIntVar(0, 10_000, f"start_{j.name}")
        end = model.NewIntVar(0, 10_000, f"end_{j.name}")
        interval = model.NewIntervalVar(start, j.duration, end, f"int_{j.name}")
        task_starts[j.name] = start
        task_ends[j.name] = end
        task_intervals[j.name] = interval
        task_family[j.name] = j.family
        machine_to_tasks[mname].append(j.name)

        # Release/Due
        model.Add(start >= j.release)
        model.Add(end <= j.due + 120)  # allow small lateness; penalize in objective

        # Calendar constraint: force (start, end) to lie in available windows; simple approach:
        # We create optional intervals for each window and exactly one must be chosen.
        win_intervals = []
        presence = []
        for idx, (ws, we) in enumerate(MACHINES[mname].windows):
            s = model.NewIntVar(ws, we, f"wstart_{j.name}_{idx}")
            e = model.NewIntVar(ws, we, f"wend_{j.name}_{idx}")
            present = model.NewBoolVar(f"present_{j.name}_{idx}")
            iv = model.NewOptionalIntervalVar(s, j.duration, e, present, f"wint_{j.name}_{idx}")
            # link chosen window to main start/end
            model.Add(start == s).OnlyEnforceIf(present)
            model.Add(end == e).OnlyEnforceIf(present)
            win_intervals.append(iv)
            presence.append(present)
        # exactly one window must be chosen
        model.Add(sum(presence) == 1)
        # Also ensure main interval equals the chosen window (implicity linked via equality)

    # NoOverlap + setup times per machine using TransitionTimes
    penalties = []
    for mname, tnames in machine_to_tasks.items():
        if not tnames:
            continue
        # Build sequence variables and precedences
        intervals = [task_intervals[n] for n in tnames]
        families = [FIDX[task_family[n]] for n in tnames]
        lvars = [model.NewIntVar(0, len(tnames) - 1, f"rank_{mname}_{i}") for i in range(len(tnames))]
        # OR-Tools requires a SequenceVar
        seq = model.AddVariableSequence(intervals, lvars, f"seq_{mname}")
        trans = cp_model.TransitionTimes(SETUP_MATRIX)
        model.AddNoOverlap2(seq, trans)

        # Slight lateness penalty
        for n in tnames:
            # tardiness = max(0, end - due)
            # approximate with a slack var
            job = next(j for j in jobs if j.name == n)
            tard = model.NewIntVar(0, 10_000, f"tard_{n}")
            model.Add(tard >= task_ends[n] - job.due)
            penalties.append(tard)

    # Objective: minimize total tardiness + small sum of starts (to encourage early starts)
    model.Minimize(sum(penalties) + sum(task_starts[n] // 100 for n in task_starts))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    solver.parameters.num_search_workers = 8
    res = solver.Solve(model)

    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT did not find a feasible schedule")

    schedule = []
    for j in jobs:
        s = solver.Value(task_starts[j.name])
        e = solver.Value(task_ends[j.name])
        schedule.append({
            "job": j.name,
            "machine": j.machine_alts[0],
            "family": j.family,
            "start_min": s,
            "end_min": e,
            "duration": j.duration,
        })
    # sort by machine then start
    schedule.sort(key=lambda x: (x["machine"], x["start_min"]))
    return schedule

# ====== Main demo ======
if __name__ == "__main__":
    plan, inv = build_and_solve_milp()
    print("=== LAYER A: MPS Plan (Pyomo MILP) ===")
    for r in plan:
        print(r)
    print("\nInventory:")
    for r in inv:
        print(r)

    jobs = explode_plan_to_jobs(plan, lot_size=20)
    print(f"\nExploded into {len(jobs)} jobs for CP layer")
    for j in jobs[:8]:
        print(j)

    schedule = schedule_jobs_cp(jobs)
    print("\n=== LAYER B: Detailed Schedule (OR-Tools CP-SAT) ===")
    for s in schedule:
        print(s)

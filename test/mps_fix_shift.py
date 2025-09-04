#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPS Job-Shop (Bucketed) with Locks — Fixed Shift 08:00–18:00
------------------------------------------------------------
Perubahan penting:
- Mengabaikan payload "capacities". Kapasitas harian tiap mesin = 600 menit (08:00–18:00).
- Mesin diambil dari daftar ops pada "jobs".
- locks.capacity tetap mengurangi kapasitias final.
- pins tetap memaksa operasi start pada bucket hari tertentu.
"""

import sys
import json
import math
from ortools.sat.python import cp_model

def load_input():
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            return json.load(f)
    return json.load(sys.stdin)

# ---------- Util ----------

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _collect_machine_ids_from_jobs(jobs):
    mset = set()
    for job in (jobs or []):
        for op in (job.get("ops") or []):
            mset.add(_safe_int(op.get("machine")))
    return sorted(mset)

def _build_fixed_cap_map(days, machine_ids, lock_caps, shift_minutes=600):
    """
    Bangun kapasitas harian tetap: tiap mesin, tiap hari = shift_minutes.
    Lalu kurangi dengan locks.capacity bila ada.
    """
    day_set = set(days)
    cap_map = {}
    for m in machine_ids:
        for d in days:
            cap_map[(m, d)] = shift_minutes

    # reduce by locks
    for lc in (lock_caps or []):
        d = lc.get("date")
        if d not in day_set:
            continue
        m = _safe_int(lc.get("machine_id"))
        minutes = max(0, _safe_int(lc.get("minutes")))
        if (m, d) in cap_map:
            cap_map[(m, d)] = max(0, cap_map[(m, d)] - minutes)
        # jika mesin tidak ada di jobs, lock diabaikan (tidak relevan)

    return cap_map, set(machine_ids)

# ---------- Solver ----------

def solve():
    data = load_input()

    days = data.get("horizon", [])
    if not days:
        print(json.dumps({"status": "MODEL_INVALID", "error": "Empty horizon."}))
        return 0

    day_idx = {d: i for i, d in enumerate(days)}
    params = data.get("params", {}) or {}
    time_limit = float(params.get("time_limit_sec", 15))
    late_pen = _safe_int(params.get("late_penalty", 10))
    setup_pen = _safe_int(params.get("setup_penalty", 1))
    split_chunk = max(1, _safe_int(params.get("split_chunk_max", 240)))

    # (opsional) izinkan override menit shift via params
    shift_minutes = _safe_int(params.get("shift_minutes", 600))  # default 600 = 10 jam

    locks = data.get("locks", {}) or {}
    cap_locks = locks.get("capacity", []) or []
    pin_locks = locks.get("pins", []) or []

    jobs = data.get("jobs", []) or []

    # Kapasitas akhir: fixed 600 menit per hari per mesin (mesin diambil dari jobs), minus locks.capacity
    machine_ids = _collect_machine_ids_from_jobs(jobs)
    cap_map, machines = _build_fixed_cap_map(days, machine_ids, cap_locks, shift_minutes=shift_minutes)

    # Siapkan model
    model = cp_model.CpModel()

    # Simpan indikator b per (machine, day) untuk setiap chunk
    chunk_registry = []

    # Untuk precedence: "first day" per operasi
    op_first_day = {}         # (j_idx, o_idx) -> IntVar
    job_due_idx = {}          # j_idx -> due day index (clamped)
    job_cp_id = {}            # j_idx -> co_product_id
    job_ops_count = {}        # j_idx -> len(ops)
    job_info = {}             # j_idx -> job dict (untuk output)

    # Map bantu untuk pins
    job_by_cp = {}
    op_index_in_job = {}

    # Bangun variabel chunk & precedence
    for j_idx, job in enumerate(jobs):
        job_info[j_idx] = job
        cp_id = _safe_int(job.get("co_product_id"))
        job_by_cp[cp_id] = j_idx
        ops = job.get("ops", []) or []
        job_ops_count[j_idx] = len(ops)

        # due index
        due_str = job.get("due_date")
        due_idx = day_idx.get(due_str, len(days) - 1)
        job_due_idx[j_idx] = due_idx
        job_cp_id[j_idx] = cp_id

        # Build op -> chunk
        for o_idx, op in enumerate(ops):
            op_id = _safe_int(op.get("operation_id"))
            op_index_in_job[(j_idx, op_id)] = o_idx

            machine = _safe_int(op.get("machine"))
            total_minutes = max(1, _safe_int(op.get("minutes"), 1))

            # Hari valid: hari dengan kapasitas > 0 untuk mesin ini
            allowed_day_indices = [i for i, d in enumerate(days) if cap_map.get((machine, d), 0) > 0]
            if not allowed_day_indices:
                allowed_day_indices = []  # akan memicu infeasible

            # split operasi panjang
            n_chunks = max(1, math.ceil(total_minutes / split_chunk))
            sizes = [split_chunk] * n_chunks
            sizes[-1] = total_minutes - split_chunk * (n_chunks - 1)

            # first day var
            fd = model.NewIntVar(0, len(days) - 1, f"fd_j{j_idx}_o{o_idx}")
            op_first_day[(j_idx, o_idx)] = fd

            chosen_days_this_op = []

            for c_idx, mins in enumerate(sizes):
                if allowed_day_indices:
                    b_vars = [model.NewBoolVar(f"b_j{j_idx}_o{o_idx}_c{c_idx}_d{d}")
                              for d in allowed_day_indices]
                    model.Add(sum(b_vars) == 1)
                    cd = model.NewIntVar(0, len(days) - 1, f"cd_j{j_idx}_o{o_idx}_c{c_idx}")
                    model.Add(cd == sum(b_vars[k] * allowed_day_indices[k] for k in range(len(allowed_day_indices))))
                else:
                    b_vars = [model.NewBoolVar(f"b_zero_j{j_idx}_o{o_idx}_c{c_idx}")]
                    model.Add(b_vars[0] == 1)
                    cd = model.NewIntVar(0, 0, f"cd_zero_j{j_idx}_o{o_idx}_c{c_idx}")
                    model.Add(cd == 0)

                chosen_days_this_op.append(cd)

                chunk_registry.append({
                    "j_idx": j_idx,
                    "job": job,
                    "o_idx": o_idx,
                    "op": op,
                    "machine": machine,
                    "minutes": mins,
                    "allowed_days": allowed_day_indices,
                    "b_per_day": b_vars,
                    "chosen_day": cd
                })

            # fd <= setiap chosen_day
            for cd in chosen_days_this_op:
                model.Add(fd <= cd)
            # ada minimal satu chunk yang sama dengan fd
            bmin = [model.NewBoolVar(f"fd_eq_j{j_idx}_o{o_idx}_c{c}") for c in range(len(chosen_days_this_op))]
            for c, cd in enumerate(chosen_days_this_op):
                model.Add(fd == cd).OnlyEnforceIf(bmin[c])
                model.Add(fd != cd).OnlyEnforceIf(bmin[c].Not())
            model.Add(sum(bmin) >= 1)

        # precedence antar operasi
        for o_idx in range(len(ops) - 1):
            model.Add(op_first_day[(j_idx, o_idx)] <= op_first_day[(j_idx, o_idx + 1)])

    # Terapkan pins (LOCK)
    for pin in pin_locks:
        cp_id = _safe_int(pin.get("co_product_id"))
        op_id = _safe_int(pin.get("operation_id"))
        day_str = pin.get("date")
        if day_str not in day_idx:
            continue
        j_idx = job_by_cp.get(cp_id)
        if j_idx is None:
            continue
        o_idx = op_index_in_job.get((j_idx, op_id))
        if o_idx is None:
            continue
        target = day_idx[day_str]
        model.Add(op_first_day[(j_idx, o_idx)] == target)

    # Kapasitas per mesin-hari
    for (m, d), cap in cap_map.items():
        if cap <= 0:
            continue
        d_id = day_idx[d]
        terms = []
        for reg in chunk_registry:
            if reg["machine"] != m:
                continue
            if d_id in reg["allowed_days"]:
                idx = reg["allowed_days"].index(d_id)
                b = reg["b_per_day"][idx]
                terms.append((reg["minutes"], b))
        if terms:
            model.Add(sum(w * b for (w, b) in terms) <= cap)

    # Objective: minimize total lateness + kecilkan penalti setup
    obj_terms = []
    for j_idx in range(len(jobs)):
        if job_ops_count[j_idx] == 0:
            continue
        last_fd = op_first_day[(j_idx, job_ops_count[j_idx] - 1)]
        due = job_due_idx[j_idx]
        L = model.NewIntVar(0, len(days), f"late_job{j_idx}")
        model.Add(last_fd - due <= L)
        obj_terms.append(L * late_pen)

    # penalti setup (sekali per operasi bertanda)
    seen_setup = set()
    for reg in chunk_registry:
        if _safe_int(reg["op"].get("is_setup", 0)) == 1:
            key = (reg["j_idx"], reg["o_idx"])
            if key not in seen_setup:
                seen_setup.add(key)
                b = model.NewBoolVar(f"setup_pen_j{reg['j_idx']}_o{reg['o_idx']}")
                model.Add(b == 1)
                obj_terms.append(b * setup_pen)

    model.Minimize(sum(obj_terms))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }

    # Build output
    result = {
        "status": status_map.get(status, "UNKNOWN"),
        "assignments": [],
        "machine_load": [],
        "job_kpi": []
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Machine load
        for (m, d), cap in cap_map.items():
            if cap <= 0:
                continue
            d_id = day_idx[d]
            used = 0
            for reg in chunk_registry:
                if reg["machine"] != m:
                    continue
                if d_id in reg["allowed_days"]:
                    idx = reg["allowed_days"].index(d_id)
                    if solver.Value(reg["b_per_day"][idx]) == 1:
                        used += reg["minutes"]
            result["machine_load"].append({
                "date": d,
                "machine_id": m,
                "used": used,
                "capacity": cap
            })

        # Assignments
        for reg in chunk_registry:
            job = reg["job"]
            op = reg["op"]
            d_id = solver.Value(reg["chosen_day"])
            if d_id < 0 or d_id >= len(days):
                continue
            d = days[d_id]
            if cap_map.get((reg["machine"], d), 0) <= 0:
                continue
            result["assignments"].append({
                "date": d,
                "machine_id": reg["machine"],
                "co_product_id": _safe_int(job.get("co_product_id")),
                "product_id": _safe_int(job.get("product_id")),
                "product_code": job.get("code"),
                "operation_id": _safe_int(op.get("operation_id")),
                "op_index": reg["o_idx"],
                "pp_id": _safe_int(op.get("pp_id")),
                "minutes": _safe_int(reg["minutes"])
            })

        # KPI per job
        for j_idx in range(len(jobs)):
            if job_ops_count[j_idx] == 0:
                result["job_kpi"].append({
                    "co_product_id": job_cp_id.get(j_idx),
                    "due_date": jobs[j_idx].get("due_date"),
                    "first_day": None,
                    "lateness_days": 0
                })
                continue
            last_fd = solver.Value(op_first_day[(j_idx, job_ops_count[j_idx] - 1)])
            due = job_due_idx[j_idx]
            late = max(0, last_fd - due)
            result["job_kpi"].append({
                "co_product_id": job_cp_id.get(j_idx),
                "due_date": jobs[j_idx].get("due_date"),
                "first_day": last_fd,
                "lateness_days": late
            })

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(solve())

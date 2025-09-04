#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPS Job-Shop (Bucketed) with Locks
----------------------------------
Input: JSON via stdin dengan skema:

{
  "horizon": ["2025-09-10","2025-09-11", ...],         // daftar tanggal (harian)
  "capacities": [                                       // kapasitas menit per mesin-hari (SEBELUM dikurangi lock)
    {"machine_id": 1, "date": "2025-09-10", "minutes": 420},
    ...
  ],
  "jobs": [                                             // unit penjadwalan = co_product
    {
      "co_product_id": 77,
      "product_id": 10,
      "code": "PRD-10",
      "due_date": "2025-09-12",
      "ops": [
        {
          "pp_id": 123,                                 // urutan dasar (jika tak ada kolom sequence)
          "operation_id": 55,
          "machine": 1,
          "minutes": 180,                               // durasi operasi (menit)
          "is_setup": 0
        },
        ...
      ]
    },
    ...
  ],
  "locks": {
    "capacity": [                                       // kapasitas yang SUDAH terpakai (jadwal real + simulasi yang dikunci)
      {"machine_id": 1, "date": "2025-09-10", "minutes": 120},
      ...
    ],
    "pins": [                                           // operasi yang DIPAKSA start di tanggal tertentu (bucket)
      {"co_product_id": 77, "operation_id": 55, "machine_id": 1, "date": "2025-09-11", "minutes": 180}
    ]
  },
  "params": {
    "time_limit_sec": 15,
    "late_penalty": 10,                                 // bobot keterlambatan (per hari)
    "setup_penalty": 1,                                 // bobot setup (opsional, kecil)
    "split_chunk_max": 240                              // menit maksimum per chunk saat operasi dipecah
  }
}

Output: JSON ke stdout:
{
  "status": "OPTIMAL|FEASIBLE|INFEASIBLE|UNKNOWN|MODEL_INVALID",
  "assignments": [
    {"date":"2025-09-10","machine_id":1,"co_product_id":77,"product_id":10,
     "product_code":"PRD-10","operation_id":55,"op_index":0,"pp_id":123,"minutes":180}
    ...
  ],
  "machine_load": [
    {"date":"2025-09-10","machine_id":1,"used":300,"capacity":420},
    ...
  ],
  "job_kpi": [
    {"co_product_id":77,"due_date":"2025-09-12","first_day":2,"lateness_days":0}
  ]
}

Catatan:
- Kapasitas final = capacities - locks.capacity. (locks.pins TIDAK mengurangi kapasitas, diasumsikan sudah tercakup di locks.capacity.)
- Precedence di-level hari: operasi berikutnya tidak boleh mulai SEBELUM operasi sebelumnya (berbasis "first day" operasi).
- Operasi panjang otomatis dipecah menjadi beberapa chunk (<= split_chunk_max menit).
- Jika suatu operasi tidak punya satupun hari yang memiliki kapasitas > 0 untuk mesinnya, model bisa menjadi INFEASIBLE.
"""

import sys
import json
import math
from ortools.sat.python import cp_model
from datetime import datetime, timedelta

def load_input():
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            return json.load(f)
    return json.load(sys.stdin)


def _parse_dt(date_str, time_str):
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

def _intersect_windows(day_windows, op_windows):
    if not op_windows:
        return day_windows
    out = []
    for dw in day_windows:
        for ow in op_windows:
            s = max(dw["start"], ow["start"])
            e = min(dw["end"],   ow["end"])
            if s < e:
                out.append({"start": s, "end": e, "shift_id": dw["shift_id"], "is_overtime": dw["is_overtime"]})
    return out

def pack_assignments_to_segments(assignments, windows_payload, op_time_windows_payload):
    # index windows
    win_map = {}  # (m,date) -> [ {start,end,shift_id,is_overtime} ... ]
    for row in (windows_payload or []):
        m = int(row["machine_id"]); d = row["date"]
        arr = []
        for itv in row["intervals"]:
            arr.append({
                "start": _parse_dt(d, itv["start"]),
                "end":   _parse_dt(d, itv["end"]),
                "shift_id": itv.get("shift_id"),
                "is_overtime": int(itv.get("is_overtime", 0))
            })
        arr.sort(key=lambda x:x["start"])
        win_map[(m,d)] = arr

    # index op-time windows (optional)
    opw_map = {}  # (co_product_id, operation_id, date) -> [ {start,end} ... ]
    for ow in (op_time_windows_payload or []):
        key = (int(ow["co_product_id"]), int(ow["operation_id"]), ow["date"])
        opw_map.setdefault(key, []).append({
            "start": _parse_dt(ow["date"], ow["start"]),
            "end":   _parse_dt(ow["date"], ow["end"]),
        })

    segments = []
    leftovers = []  # jika menit tumpah (kurang slot)
    # urutkan biar stabil: per machine, date, lalu op_index
    assignments_sorted = sorted(assignments, key=lambda a: (a["machine_id"], a["date"], a["op_index"]))
    for a in assignments_sorted:
        m = int(a["machine_id"]); d = a["date"]; need = int(a["minutes"])
        day_wins = win_map.get((m,d), [])
        if not day_wins:
            leftovers.append({**a, "reason": "no_windows"})
            continue

        # terapkan op time windows (jika ada) -> intersect
        owins = opw_map.get((int(a["co_product_id"]), int(a["operation_id"]), d), [])
        if owins:
            # ubah ke format sama (punya shift_id/is_overtime)
            ow_full = []
            for ow in owins:
                ow_full.append({"start": ow["start"], "end": ow["end"], "shift_id": None, "is_overtime": 0})
            work_wins = _intersect_windows(day_wins, ow_full)
        else:
            work_wins = day_wins

        # pack
        i = 0
        while need > 0 and i < len(work_wins):
            w = work_wins[i]
            cap_min = int((w["end"] - w["start"]).total_seconds() // 60)
            if cap_min <= 0:
                i += 1
                continue
            use = min(need, cap_min)
            seg_start = w["start"]
            seg_end   = seg_start + timedelta(minutes=use)
            segments.append({
                "date": d,
                "machine_id": m,
                "co_product_id": a["co_product_id"],
                "product_id": a["product_id"],
                "product_code": a["product_code"],
                "operation_id": a["operation_id"],
                "op_index": a["op_index"],
                "pp_id": a["pp_id"],
                "minutes": use,
                "start_time": seg_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": seg_end.strftime("%Y-%m-%d %H:%M:%S"),
                "shift_id": w["shift_id"],
                "is_overtime": w["is_overtime"]
            })
            # geser window start
            w["start"] = seg_end
            need -= use
            if use == cap_min:
                i += 1

        if need > 0:
            leftovers.append({**a, "remaining_minutes": need, "reason": "not_enough_window"})

    return segments, leftovers

# ---------- Util ----------

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _build_cap_map(days, capacities, lock_caps):
    """Bangun map kapasitas akhir: (machine_id, date) -> minutes (>=0)."""
    day_set = set(days)
    cap_map = {}
    machines = set()
    # base
    for c in capacities:
        d = c.get("date")
        if d not in day_set:
            continue
        m = _safe_int(c.get("machine_id"))
        minutes = max(0, _safe_int(c.get("minutes")))
        cap_map[(m, d)] = cap_map.get((m, d), 0) + minutes
        machines.add(m)
    # reduce by locks
    for lc in (lock_caps or []):
        d = lc.get("date")
        if d not in day_set:
            continue
        m = _safe_int(lc.get("machine_id"))
        minutes = max(0, _safe_int(lc.get("minutes")))
        if (m, d) in cap_map:
            cap_map[(m, d)] = max(0, cap_map[(m, d)] - minutes)
        else:
            # jika tidak ada kapasitas base, tetap 0 (jangan negatif)
            cap_map[(m, d)] = 0
        machines.add(m)
    return cap_map, machines

# ---------- Solver ----------

def solve():
    # data = json.load(sys.stdin)
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

    locks = data.get("locks", {}) or {}
    cap_locks = locks.get("capacity", []) or []
    pin_locks = locks.get("pins", []) or []

    # Kapasitas akhir
    cap_map, machines = _build_cap_map(days, data.get("capacities", []) or [], cap_locks)

    # Siapkan model
    model = cp_model.CpModel()

    # Untuk tiap mesin-hari, kita akan bangun batas kapasitas sebagai:
    # sum(minutes * assignment_bool) <= capacity
    # Kita tidak perlu variabel agregat y[(m,d)] (opsional).
    # Namun untuk reporting, kita akan hitung "used" dari solusi.

    # Simpan indikator b per (machine, day) untuk setiap chunk
    # Struktur:
    # chunk_registry = [
    #   {
    #     "j_idx": int, "job": job_json,
    #     "o_idx": int, "op": op_json,
    #     "machine": int,
    #     "minutes": int,
    #     "allowed_days": [day_indices ...],
    #     "b_per_day": [BoolVar ...],     # sebaran per hari
    #     "chosen_day": IntVar            # day index
    #   },
    #   ...
    # ]
    chunk_registry = []

    jobs = data.get("jobs", []) or []

    # Untuk precedence, kita butuh "first day" per operasi (min chosen_day dari chunk-chunknya)
    op_first_day = {}         # (j_idx, o_idx) -> IntVar
    job_due_idx = {}          # j_idx -> due day index (clamped)
    job_cp_id = {}            # j_idx -> co_product_id
    job_ops_count = {}        # j_idx -> len(ops)
    job_info = {}             # j_idx -> job dict (untuk output)

    # Map bantu untuk pins (co_product_id -> j_idx, dan (j_idx, operation_id) -> o_idx)
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

            # hari yang boleh dipakai untuk mesin ini = hari-hari yang kapasitasnya > 0
            allowed_day_indices = [i for i, d in enumerate(days) if cap_map.get((machine, d), 0) > 0]

            # kalau bener-bener tidak ada hari dengan kapasitas > 0 untuk mesin ini,
            # model akan infeasible (memang benar — tidak ada slot)
            if not allowed_day_indices:
                # tetap buat var dan nanti constraints akan memaksa infeasible
                allowed_day_indices = []

            # split jadi beberapa chunk
            n_chunks = max(1, math.ceil(total_minutes / split_chunk))
            sizes = [split_chunk] * n_chunks
            sizes[-1] = total_minutes - split_chunk * (n_chunks - 1)

            # chosen day untuk "first day" operasi
            # first day = min(chosen_day semua chunk). Kita bangun dengan trick reified.
            fd = model.NewIntVar(0, len(days) - 1, f"fd_j{j_idx}_o{o_idx}")
            op_first_day[(j_idx, o_idx)] = fd

            # kumpul chosen day tiap chunk, lalu link "fd <= each chosen_day"
            chosen_days_this_op = []

            for c_idx, mins in enumerate(sizes):
                # jika tidak ada allowed day sama sekali, kita tetap definisikan satu dummy pilihan,
                # tapi ini akan memicu infeasible karena tidak ada kapasitas harian untuk mesin tsb.
                if allowed_day_indices:
                    b_vars = [model.NewBoolVar(f"b_j{j_idx}_o{o_idx}_c{c_idx}_d{d}")
                              for d in allowed_day_indices]
                    # exactly one
                    model.Add(sum(b_vars) == 1)
                    # chosen_day int
                    cd = model.NewIntVar(0, len(days) - 1, f"cd_j{j_idx}_o{o_idx}_c{c_idx}")
                    # tautkan cd == allowed_day_indices[k] saat b_vars[k]==1
                    # cd == sum(b_k * allowed_day_index_k)
                    # Realisasikan via equality linear:
                    model.Add(
                        cd == sum(b_vars[k] * allowed_day_indices[k] for k in range(len(allowed_day_indices)))
                    )
                else:
                    # Tidak ada hari diperbolehkan (akan infeasible), buat placeholder:
                    b_vars = [model.NewBoolVar(f"b_zero_j{j_idx}_o{o_idx}_c{c_idx}")]
                    model.Add(b_vars[0] == 1)  # paksa 1 (tak ada hari real)
                    cd = model.NewIntVar(0, 0, f"cd_zero_j{j_idx}_o{o_idx}_c{c_idx}")
                    model.Add(cd == 0)

                chosen_days_this_op.append(cd)

                # Simpan registry
                chunk_registry.append({
                    "j_idx": j_idx,
                    "job": job,
                    "o_idx": o_idx,
                    "op": op,
                    "machine": machine,
                    "minutes": mins,
                    "allowed_days": allowed_day_indices,   # mungkin kosong
                    "b_per_day": b_vars,
                    "chosen_day": cd
                })

            # fd <= setiap chosen_day
            for cd in chosen_days_this_op:
                model.Add(fd <= cd)

            # pastikan ada minimal satu chunk yang punya chosen_day == fd
            bmin = [model.NewBoolVar(f"fd_eq_j{j_idx}_o{o_idx}_c{c}") for c in range(len(chosen_days_this_op))]
            for c, cd in enumerate(chosen_days_this_op):
                model.Add(fd == cd).OnlyEnforceIf(bmin[c])
                model.Add(fd != cd).OnlyEnforceIf(bmin[c].Not())
            model.Add(sum(bmin) >= 1)

        # precedence antar operasi (berbasis first-day)
        for o_idx in range(len(ops) - 1):
            model.Add(op_first_day[(j_idx, o_idx)] <= op_first_day[(j_idx, o_idx + 1)])

    # Terapkan pins (LOCK posisi operasi pada hari tertentu)
    # Catatan: pins TIDAK otomatis mengurangi kapasitas. Pastikan Laravel memasukkan porsi menitnya ke locks.capacity jika memang terpakai.
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
        # paksa first day == target
        model.Add(op_first_day[(j_idx, o_idx)] == target)

    # Kapasitas: untuk setiap (mesin, hari), sum semua chunk yg dipilih pada hari tsb <= capacity
    for (m, d) in cap_map.keys():
        cap = cap_map[(m, d)]
        if cap <= 0:
            continue
        d_id = day_idx[d]
        # kumpulkan semua b_var untuk chunk di mesin m & hari d_id
        terms = []
        for reg in chunk_registry:
            if reg["machine"] != m:
                continue
            # cari indeks hari d_id dalam allowed_days
            if d_id in reg["allowed_days"]:
                idx = reg["allowed_days"].index(d_id)
                b = reg["b_per_day"][idx]
                terms.append((reg["minutes"], b))
        if terms:
            model.Add(sum(w * b for (w, b) in terms) <= cap)
        # kalau tidak ada terms, tidak perlu constraint (otomatis 0 <= cap)

    # Objective:
    # 1) Minimize total lateness (hari) per job (co_product)
    # 2) Penalize setup ops sedikit (opsional)
    obj_terms = []

    for j_idx in range(len(jobs)):
        if job_ops_count[j_idx] == 0:
            # job tanpa operasi: tidak memberi beban, lateness 0
            continue
        last_fd = op_first_day[(j_idx, job_ops_count[j_idx] - 1)]
        due = job_due_idx[j_idx]
        # Lateness var (>= 0), approx: L >= last_fd - due, L >= 0
        L = model.NewIntVar(0, len(days), f"late_job{j_idx}")
        # last_fd - due <= L  => last_fd - L <= due
        # Implementasi lebih langsung:
        model.Add(last_fd - due <= L)
        obj_terms.append(L * late_pen)

    # setup penalty (ringan)
    for reg in chunk_registry:
        # Hanya kasih penalti 1x per operasi (bukan per chunk)
        # Jadi cek reg['op']['is_setup'] dan beri penalti pada fd op tsb (pakai bool dummy = 1)
        if _safe_int(reg["op"].get("is_setup", 0)) == 1 and reg["o_idx"] == 0:
            # cukup satu kali per operasi pertama (atau bisa pakai set op seen)
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
        use_map = {}  # (m,d) -> used minutes
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
                    b = reg["b_per_day"][idx]
                    if solver.Value(b) == 1:
                        used += reg["minutes"]
            use_map[(m, d)] = used
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
            j_idx = reg["j_idx"]
            d_id = solver.Value(reg["chosen_day"])
            if d_id < 0 or d_id >= len(days):
                continue
            d = days[d_id]
            # hanya laporkan jika hari tsb memang punya kapasitas (defensif)
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
            # setelah result["assignments"] terisi:
            segments, leftovers = pack_assignments_to_segments(
                result["assignments"],
                data.get("windows"),
                data.get("op_time_windows")
            )
            result["segments"] = segments  # sudah ada start_time & end_time fix
            if leftovers:
                result["leftovers"] = leftovers  # warning kalau ada yang tak kebagian slot


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

from flask import Flask, request, jsonify
from ortools.sat.python import cp_model
from datetime import datetime, timedelta

app = Flask(__name__)


def _to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "ya", "yes", "y")
    return False


def _parse_abs(dt_str):
    # "%Y-%m-%d %H:%M" -> datetime
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M")


def _normalize_jobs(jobs_data):
    """
    jobs_data: list of jobs
      job: list of tasks
        task: [operation_id, duration, (optional) is_locked]
              or {"operation_id":.., "duration":.., "is_locked":..}
    -> returns list of jobs with tuples (op:int, dur:int, locked:bool)
    """
    norm = []
    for job in jobs_data:
        nj = []
        for t in job:
            if isinstance(t, dict):
                op = int(t.get("operation_id", 0))
                dur = int(t.get("duration", 0))
                locked = _to_bool(t.get("is_locked", False))
            else:
                # list/tuple
                op = int(t[0])
                dur = int(t[1])
                locked = _to_bool(t[2]) if len(t) > 2 else False
            nj.append((op, dur, locked))
        norm.append(nj)
    return norm


def schedule(now, jobs_data, shipment_deadlines, products):
    # Normalisasi agar tiap task = (operation, duration, locked)
    jobs_data = _normalize_jobs(jobs_data)

    # Deadline relatif (menit)
    shipment_deadlines_minutes = [
        int((deadline - now).total_seconds() // 60) for deadline in shipment_deadlines
    ]

    # Kumpulkan semua operation (mesin)
    all_operations = set()
    for job in jobs_data:
        for op, _, _ in job:
            all_operations.add(op)
    operations_count = (max(all_operations) + 1) if all_operations else 0

    # Estimasi horizon dasar
    estimated_total_duration = sum(sum(dur for _, dur, _ in job) for job in jobs_data)
    horizon = int(max(1, estimated_total_duration) * 1.5)

    # --- Lock absolut (opsional) bila ada waktu fixed di payload product ---
    fixed_starts = {}  # (job_id, task_id) -> start_minute
    fixed_ends = {}    # (job_id, task_id) -> end_minute
    for idx, p in enumerate(products):
        locked_tasks = p.get("locked_tasks") or []
        if locked_tasks:
            for t_idx, item in enumerate(locked_tasks):
                if t_idx >= len(jobs_data[idx]):
                    break
                if item.get("start_time"):
                    s_abs = _parse_abs(item["start_time"])
                    s_min = int((s_abs - now).total_seconds() // 60)
                    fixed_starts[(idx, t_idx)] = s_min
                    # index 1 = duration pada jobs_data ternormalisasi
                    horizon = max(horizon, s_min + jobs_data[idx][t_idx][1] + 1)
                if item.get("end_time"):
                    e_abs = _parse_abs(item["end_time"])
                    e_min = int((e_abs - now).total_seconds() // 60)
                    fixed_ends[(idx, t_idx)] = e_min
                    horizon = max(horizon, e_min + 1)
        elif p.get("locked_start_time"):
            # Job-level lock: pakukan start task pertama
            s_abs = _parse_abs(p["locked_start_time"])
            s_min = int((s_abs - now).total_seconds() // 60)
            fixed_starts[(idx, 0)] = s_min
            total_job = sum(d for _, d, _ in jobs_data[idx]) if idx < len(jobs_data) else 0
            horizon = max(horizon, s_min + total_job + 1)

    # Pastikan horizon juga menutup deadline
    if shipment_deadlines_minutes:
        horizon = max(horizon, max(shipment_deadlines_minutes) + 1)

    # --- Build model ---
    model = cp_model.CpModel()
    all_tasks = {}
    operation_to_intervals = [[] for _ in range(operations_count)]
    job_ends = []

    # Variabel interval per task
    for job_id, job in enumerate(jobs_data):
        for task_id, (operation, duration, _) in enumerate(job):
            suffix = f"_{job_id}_{task_id}"
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            interval = model.NewIntervalVar(start_var, duration, end_var, "interval" + suffix)
            all_tasks[(job_id, task_id)] = (start_var, end_var, interval)
            operation_to_intervals[operation].append(interval)

    # NoOverlap per operation (mesin)
    for operation in range(operations_count):
        model.AddNoOverlap(operation_to_intervals[operation])

    # Urutan task dalam satu job
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            _, end_var, _ = all_tasks[(job_id, task_id)]
            next_start_var, _, _ = all_tasks[(job_id, task_id + 1)]
            model.Add(next_start_var >= end_var)

    # Deadline per job
    for job_id, job in enumerate(jobs_data):
        if not job:
            continue
        last_task_id = len(job) - 1
        _, end_var, _ = all_tasks[(job_id, last_task_id)]
        model.Add(end_var <= shipment_deadlines_minutes[job_id])
        job_ends.append(end_var)

    # Terapkan lock absolut (jika ada waktu fixed)
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job)):
            start_var, end_var, _ = all_tasks[(job_id, task_id)]
            if (job_id, task_id) in fixed_starts:
                model.Add(start_var == fixed_starts[(job_id, task_id)])
            if (job_id, task_id) in fixed_ends:
                model.Add(end_var == fixed_ends[(job_id, task_id)])

    # Objective: minimize makespan
    makespan = model.NewIntVar(0, horizon, "makespan")
    if job_ends:
        model.AddMaxEquality(makespan, job_ends)
    else:
        model.Add(makespan == 0)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)

    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    result = {
        "status": "ok" if feasible else "infeasible",
        "makespan_minutes": int(solver.ObjectiveValue()) if feasible else None,
        "start_time": now.strftime("%Y-%m-%d %H:%M"),
        "tasks": [],
    }

    if feasible:
        all_tasks_list = []
        global_task_id = 1
        for job_id, job in enumerate(jobs_data):
            product = products[job_id]
            product_id = product.get("id")
            product_name = product.get("name")

            for seq_id, (operation, duration, locked_flag) in enumerate(job, start=1):
                start = solver.Value(all_tasks[(job_id, seq_id - 1)][0])
                end = solver.Value(all_tasks[(job_id, seq_id - 1)][1])
                all_tasks_list.append({
                    "id": global_task_id,
                    "task_id": f"{product_id}.{seq_id}",
                    "product_id": product_id,
                    "product_name": product_name,
                    "operation_id": operation,
                    "duration": duration,
                    "start_minute": start,
                    "end_minute": end,
                    "start_time": (now + timedelta(minutes=start)).strftime("%Y-%m-%d %H:%M"),
                    "end_time": (now + timedelta(minutes=end)).strftime("%Y-%m-%d %H:%M"),
                    "previous_task": f"{product_id}.{seq_id - 1}" if seq_id > 1 else None,
                    "process_dependency": f"{product_id}.{seq_id}",
                    "is_locked": _to_bool(locked_flag),
                })
                global_task_id += 1

        all_tasks_list.sort(key=lambda x: x["start_minute"])
        result["tasks"] = all_tasks_list

    return result


@app.route("/schedule", methods=["POST"])
def schedule_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or empty JSON"}), 400

        now = datetime.strptime(data["start_time"], "%Y-%m-%d %H:%M")
        products = data["products"]

        # Bentuk jobs_data & deadlines; tasks bisa 2 atau 3 kolom
        jobs_data = []
        shipment_deadlines = []
        for product in products:
            jobs_data.append(product["tasks"])  # dinormalisasi di schedule()
            shipment_deadlines.append(
                datetime.strptime(product["shipment_deadline"], "%Y-%m-%d %H:%M")
            )

        result = schedule(now, jobs_data, shipment_deadlines, products)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # application = app  # untuk WSGI/cPanel, uncomment baris ini
    app.run(debug=True, host="0.0.0.0", port=5000)

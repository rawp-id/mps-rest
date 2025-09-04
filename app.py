from flask import Flask, request, jsonify
from ortools.sat.python import cp_model
from datetime import datetime, timedelta, time

app = Flask(__name__)

WORK_START = time(8, 0)
WORK_END   = time(17, 0)
WORK_MINUTES_PER_DAY = (WORK_END.hour*60 + WORK_END.minute) - (WORK_START.hour*60 + WORK_START.minute)


def _to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "ya", "yes", "y")
    return False


def _parse_abs(dt_str):
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M")


def _normalize_jobs(jobs_data):
    """
    tasks: [operation_id, duration] atau {"operation_id":..,"duration":..}
    (is_locked per-task diabaikan; lock ada di level product)
    -> [(op:int, dur:int)]
    """
    norm = []
    for job in jobs_data:
        nj = []
        for t in job:
            if isinstance(t, dict):
                op = int(t.get("operation_id", 0))
                dur = int(t.get("duration", 0))
            else:
                op = int(t[0]); dur = int(t[1])
            nj.append((op, dur))
        norm.append(nj)
    return norm


# =========================
# Kalender kerja (08:00–17:00)
# =========================
def _align_to_next_work_minute(dt: datetime) -> datetime:
    """Jika di luar jam kerja, geser ke slot kerja terdekat berikutnya."""
    if dt.time() < WORK_START:
        return dt.replace(hour=WORK_START.hour, minute=WORK_START.minute, second=0, microsecond=0)
    if dt.time() >= WORK_END:
        next_day = (dt + timedelta(days=1)).replace(hour=WORK_START.hour, minute=WORK_START.minute, second=0, microsecond=0)
        return next_day
    return dt.replace(second=0, microsecond=0)


def _business_minutes_between(start_dt: datetime, end_dt: datetime) -> int:
    """Hitung menit kerja antara start_dt dan end_dt (non-negatif)."""
    if end_dt <= start_dt:
        return 0
    cur = _align_to_next_work_minute(start_dt)
    end_dt = end_dt.replace(second=0, microsecond=0)
    total = 0
    while cur < end_dt:
        day_work_start = cur.replace(hour=WORK_START.hour, minute=WORK_START.minute)
        day_work_end   = cur.replace(hour=WORK_END.hour, minute=WORK_END.minute)
        # jika end_dt sebelum mulai kerja hari ini, lompat ke hari berikutnya
        if end_dt <= day_work_start:
            break
        # segmen yang dihitung pada hari ini
        seg_start = max(cur, day_work_start)
        seg_end   = min(end_dt, day_work_end)
        if seg_end > seg_start:
            total += int((seg_end - seg_start).total_seconds() // 60)
        # lompat ke hari kerja berikutnya 08:00
        cur = day_work_end if end_dt > day_work_end else end_dt
        if cur >= day_work_end:
            cur = _align_to_next_work_minute(cur + timedelta(minutes=1))
    return total


def _add_business_minutes(start_dt: datetime, minutes: int) -> datetime:
    """Tambahkan menit kerja ke start_dt, lompat malam di luar jam kerja."""
    cur = _align_to_next_work_minute(start_dt)
    remaining = int(minutes)
    while remaining > 0:
        day_work_end = cur.replace(hour=WORK_END.hour, minute=WORK_END.minute)
        usable = int((day_work_end - cur).total_seconds() // 60)
        if usable <= 0:
            cur = _align_to_next_work_minute(cur + timedelta(minutes=1))
            continue
        take = min(usable, remaining)
        cur += timedelta(minutes=take)
        remaining -= take
        if cur >= day_work_end and remaining > 0:
            cur = _align_to_next_work_minute(cur + timedelta(minutes=1))
    return cur


def schedule(now, jobs_data, shipment_deadlines, products):
    # Normalisasi task → menit proses (durasi tetap menit kalender; kita anggap durasi = menit kerja)
    jobs_data = _normalize_jobs(jobs_data)

    # Pastikan start berada di jam kerja
    now = _align_to_next_work_minute(now)

    # Status lock per product
    product_locked = [_to_bool(p.get("locked", False)) for p in products]

    # Deadline dalam skala "menit kerja" dari now
    shipment_deadlines_minutes = [
        _business_minutes_between(now, _align_to_next_work_minute(dl)) for dl in shipment_deadlines
    ]

    # Operasi (mesin)
    all_operations = set()
    for job in jobs_data:
        for op, _ in job:
            all_operations.add(op)
    operations_count = (max(all_operations) + 1) if all_operations else 0

    # Estimasi horizon dalam menit kerja
    total_duration_work_min = sum(sum(d for _, d in job) for job in jobs_data)
    horizon = max(1, int(total_duration_work_min * 2))
    if shipment_deadlines_minutes:
        horizon = max(horizon, max(shipment_deadlines_minutes) + 1)

    # Build model di domain "menit kerja"
    model = cp_model.CpModel()
    all_tasks = {}
    operation_to_intervals = [[] for _ in range(operations_count)]
    job_ends = []

    for job_id, job in enumerate(jobs_data):
        for task_id, (operation, duration) in enumerate(job):
            # duration sudah dianggap menit kerja
            suffix = f"_{job_id}_{task_id}"
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var   = model.NewIntVar(0, horizon, "end" + suffix)
            interval  = model.NewIntervalVar(start_var, duration, end_var, "interval" + suffix)
            all_tasks[(job_id, task_id)] = (start_var, end_var, interval)
            operation_to_intervals[operation].append(interval)

    # NoOverlap per operation
    for operation in range(operations_count):
        model.AddNoOverlap(operation_to_intervals[operation])

    # Urutan dalam job
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            _, end_var, _ = all_tasks[(job_id, task_id)]
            next_start, _, _ = all_tasks[(job_id, task_id + 1)]
            model.Add(next_start >= end_var)

    # Constraint deadline (di menit kerja)
    for job_id, job in enumerate(jobs_data):
        if not job:
            continue
        last_id = len(job) - 1
        _, end_var, _ = all_tasks[(job_id, last_id)]
        dl = shipment_deadlines_minutes[job_id]
        if product_locked[job_id]:
            model.Add(end_var == dl)   # HARUS tepat di deadline kerja
        else:
            model.Add(end_var <= dl)   # ≤ deadline kerja
        job_ends.append(end_var)

    # Objective
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
        "makespan_minutes": int(solver.ObjectiveValue()) if feasible else None,  # menit kerja
        "start_time": now.strftime("%Y-%m-%d %H:%M"),
        "tasks": [],
    }

    if feasible:
        all_tasks_list = []
        global_task_id = 1
        for job_id, job in enumerate(jobs_data):
            product = products[job_id]
            pid = product.get("id")
            pname = product.get("name")
            co_product_id = product.get("co_product_id")
            is_locked_job = product_locked[job_id]

            for seq_id, (operation, duration) in enumerate(job, start=1):
                start_w = solver.Value(all_tasks[(job_id, seq_id - 1)][0])  # menit kerja
                end_w   = solver.Value(all_tasks[(job_id, seq_id - 1)][1])  # menit kerja
                start_dt = _add_business_minutes(now, start_w)
                end_dt   = _add_business_minutes(now, end_w)

                all_tasks_list.append({
                    "id": global_task_id,
                    "task_id": f"{pid}.{seq_id}",
                    "product_id": pid,
                    "co_product_id": co_product_id,
                    "product_name": pname,
                    "operation_id": operation,
                    "duration": duration,  # menit kerja
                    "start_minute": start_w,   # menit kerja sejak start
                    "end_minute": end_w,       # menit kerja sejak start
                    "start_time": start_dt.strftime("%Y-%m-%d %H:%M"),
                    "end_time": end_dt.strftime("%Y-%m-%d %H:%M"),
                    "previous_task": f"{pid}.{seq_id - 1}" if seq_id > 1 else None,
                    "process_dependency": f"{pid}.{seq_id}",
                    "is_locked": is_locked_job,
                })
                global_task_id += 1

        all_tasks_list.sort(key=lambda x: (x["start_time"], x["operation_id"]))
        result["tasks"] = all_tasks_list

    return result


@app.route("/schedule", methods=["POST"])
def schedule_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or empty JSON"}), 400

        now = _parse_abs(data["start_time"])
        products = data["products"]

        jobs_data = []
        shipment_deadlines = []
        for p in products:
            jobs_data.append(p["tasks"])
            shipment_deadlines.append(_parse_abs(p["shipment_deadline"]))

        result = schedule(now, jobs_data, shipment_deadlines, products)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

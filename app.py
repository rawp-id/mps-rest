from flask import Flask, request, jsonify
from ortools.sat.python import cp_model
from datetime import datetime, timedelta

app = Flask(__name__)


def schedule(now, jobs_data, shipment_deadlines, products):
    shipment_deadlines_minutes = [
        int((deadline - now).total_seconds() // 60)
        for deadline in shipment_deadlines
    ]

    all_operations = set()
    for job in jobs_data:
        for operations_id, _ in job:
            all_operations.add(operations_id)
    operations_count = max(all_operations) + 1

    estimated_total_duration = sum(
        sum(task[1] for task in job) for job in jobs_data
    )
    horizon = int(estimated_total_duration * 1.5)

    model = cp_model.CpModel()
    all_tasks = {}
    operation_to_intervals = [[] for _ in range(operations_count)]
    job_ends = []

    for job_id, job in enumerate(jobs_data):
        for task_id, (operation, duration) in enumerate(job):
            suffix = f"_{job_id}_{task_id}"
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            interval = model.NewIntervalVar(
                start_var, duration, end_var, "interval" + suffix
            )
            all_tasks[(job_id, task_id)] = (start_var, end_var, interval)
            operation_to_intervals[operation].append(interval)

    for operation in range(operations_count):
        model.AddNoOverlap(operation_to_intervals[operation])

    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            _, end_var, _ = all_tasks[(job_id, task_id)]
            next_start_var, _, _ = all_tasks[(job_id, task_id + 1)]
            model.Add(next_start_var >= end_var)

    for job_id, job in enumerate(jobs_data):
        last_task_id = len(job) - 1
        _, end_var, _ = all_tasks[(job_id, last_task_id)]
        deadline = shipment_deadlines_minutes[job_id]
        model.Add(end_var <= deadline)
        job_ends.append(end_var)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)

    result = {
        "status": "ok" if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else "infeasible",
        "makespan_minutes": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "start_time": now.strftime("%Y-%m-%d %H:%M"),
        "tasks": [],
        # "products": [],
    }

    all_tasks_list = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        products_list = []
        global_task_id = 1
        for job_id, job in enumerate(jobs_data):
            product = products[job_id]
            product_id = product["id"]
            product_name = product["name"]
            product_tasks = []

            for task_id, (operation, duration) in enumerate(job):
                start = solver.Value(all_tasks[(job_id, task_id)][0])
                end = solver.Value(all_tasks[(job_id, task_id)][1])
                task_info = {
                    # "id": global_task_id,
                    "task_id": f"{product_id}.{task_id + 1}",
                    "product_id": product_id,
                    "product_name": product_name,
                    "operation_id": operation,
                    "duration": duration,
                    "start_minute": start,
                    "end_minute": end,
                    "start_time": (now + timedelta(minutes=start)).strftime("%Y-%m-%d %H:%M"),
                    "end_time": (now + timedelta(minutes=end)).strftime("%Y-%m-%d %H:%M"),
                    "previous_task": f"{product_id}.{operation - 1}" if operation > 1 else None,
                    "process_dependency": f"{product_id}.{operation}" if operation > 0 else None
                }
                global_task_id += 1
                all_tasks_list.append(task_info)
                product_tasks.append(task_info)

            products_list.append({
                "product_id": product_id,
                "name": product_name,
                "tasks": product_tasks
            })

        tasks = sorted(all_tasks_list, key=lambda x: x["start_minute"])
        for task in tasks:
            if "id" not in task:
                global_task_id = 1
                while any(t.get("id") == global_task_id for t in all_tasks_list):
                    global_task_id += 1
        task["id"] = global_task_id

        result["tasks"] = tasks
        # result["products"] = products_list

    return result


@app.route("/schedule", methods=["POST"])
def schedule_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or empty JSON"}), 400

        now = datetime.strptime(data["start_time"], "%Y-%m-%d %H:%M")
        products = data["products"]

        jobs_data = []
        shipment_deadlines = []
        for product in products:
            jobs_data.append(product["tasks"])
            shipment_deadlines.append(datetime.strptime(
                product["shipment_deadline"], "%Y-%m-%d %H:%M"))

        result = schedule(now, jobs_data, shipment_deadlines, products)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

# application = app  # <- tambahkan baris ini untuk cPanel WSGI


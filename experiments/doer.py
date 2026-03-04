#!/usr/bin/env python3
"""The Doer — always-on experiment daemon that fills idle compute.

You (the human + Claude) are the thinker. This is the doer.
When CPU is free, it runs the next CPU job. When GPU is free, it runs
the next GPU job. Both can run simultaneously.

Usage:
    # Start the doer (runs forever, polls every 60s)
    nohup python experiments/doer.py > /tmp/doer.log 2>&1 &

    # Start with custom poll interval
    python experiments/doer.py --poll 30

    # Dry run (show what would launch, don't actually start anything)
    python experiments/doer.py --dry-run

    # Show current status
    python experiments/doer.py --status

Edit experiments/job_queue.json while running to add/reorder/remove jobs.
The doer re-reads the file every poll cycle.

Job queue format (experiments/job_queue.json):
[
    {
        "id": "34m_train",
        "name": "Train 34M scale ladder",
        "resource": "cpu",
        "command": "python experiments/scale_ladder/train_model.py --size 34M --seed 42 --device cpu",
        "cwd": "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity",
        "priority": 1,
        "status": "pending",
        "depends_on": null,
        "notify": true
    }
]

Fields:
    id:          Unique identifier (used for depends_on references)
    name:        Human-readable description
    resource:    "cpu" or "gpu" — which slot this job occupies
    command:     Shell command to run
    cwd:         Working directory (default: project root)
    priority:    Lower number = higher priority (1 = first)
    status:      "pending" | "running" | "done" | "failed" | "skipped"
    depends_on:  null, or an id string — won't start until that job is "done"
    notify:      If true, writes to /tmp/doer_notify.txt when job finishes
    pid:         (set by doer) PID of running process
    started_at:  (set by doer) ISO timestamp
    finished_at: (set by doer) ISO timestamp
    exit_code:   (set by doer) process exit code
    log_file:    (set by doer) path to stdout/stderr log
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_FILE = PROJECT_ROOT / "experiments" / "job_queue.json"
LOG_DIR = Path("/tmp/doer_logs")
NOTIFY_FILE = Path("/tmp/doer_notify.txt")
STATUS_FILE = Path("/tmp/doer_status.json")


def load_queue() -> list[dict]:
    """Load job queue from JSON. Returns empty list if missing."""
    if not QUEUE_FILE.exists():
        return []
    try:
        return json.loads(QUEUE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [doer] WARNING: Failed to read queue: {e}")
        return []


def save_queue(jobs: list[dict]) -> None:
    """Atomically write job queue back to JSON."""
    tmp = QUEUE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2) + "\n")
    tmp.rename(QUEUE_FILE)


def is_pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_next_job(jobs: list[dict], resource: str) -> dict | None:
    """Find the highest-priority pending job for a resource slot."""
    candidates = []
    for job in jobs:
        if job.get("status") != "pending":
            continue
        if job.get("resource") != resource:
            continue
        # Check dependency
        dep = job.get("depends_on")
        if dep:
            dep_job = next((j for j in jobs if j["id"] == dep), None)
            if dep_job is None or dep_job.get("status") != "done":
                continue
        candidates.append(job)

    if not candidates:
        return None
    # Sort by priority (lower = higher priority)
    candidates.sort(key=lambda j: j.get("priority", 999))
    return candidates[0]


def launch_job(job: dict) -> subprocess.Popen:
    """Launch a job as a subprocess."""
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{job['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    cwd = job.get("cwd", str(PROJECT_ROOT))

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        job["command"],
        shell=True,
        cwd=cwd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Survives if doer is killed
    )

    job["status"] = "running"
    job["pid"] = proc.pid
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    job["log_file"] = str(log_path)

    return proc


def check_running_job(job: dict, procs: dict) -> None:
    """Check if a running job has finished."""
    pid = job.get("pid")
    if pid is None:
        return

    proc = procs.get(job["id"])
    if proc is not None:
        # We have the Popen object
        retcode = proc.poll()
        if retcode is not None:
            job["status"] = "done" if retcode == 0 else "failed"
            job["exit_code"] = retcode
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            del procs[job["id"]]

            if job.get("notify", False):
                notify(job)
    else:
        # Process was started by a previous doer instance
        if not is_pid_alive(pid):
            # Process ended but we don't know the exit code
            job["status"] = "done"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            job["exit_code"] = "unknown (adopted)"

            if job.get("notify", False):
                notify(job)


def notify(job: dict) -> None:
    """Write a notification when a job finishes."""
    status = job["status"].upper()
    msg = f"[{datetime.now().strftime('%H:%M')}] {status}: {job['name']}"
    if job.get("exit_code") not in (0, None, "unknown (adopted)"):
        msg += f" (exit code {job['exit_code']})"
    if job.get("log_file"):
        msg += f"\n  Log: {job['log_file']}"
    msg += "\n"

    with open(NOTIFY_FILE, "a") as f:
        f.write(msg)

    print(f"  [doer] {msg.strip()}")
    # Terminal bell
    sys.stdout.write("\a")
    sys.stdout.flush()


def write_status(jobs: list[dict], cpu_job: str | None, gpu_job: str | None) -> None:
    """Write current status to a file for quick checking."""
    status = {
        "updated": datetime.now().isoformat(),
        "cpu_slot": cpu_job or "idle",
        "gpu_slot": gpu_job or "idle",
        "pending": sum(1 for j in jobs if j["status"] == "pending"),
        "running": sum(1 for j in jobs if j["status"] == "running"),
        "done": sum(1 for j in jobs if j["status"] == "done"),
        "failed": sum(1 for j in jobs if j["status"] == "failed"),
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2) + "\n")


def show_status() -> None:
    """Print current doer status."""
    if STATUS_FILE.exists():
        status = json.loads(STATUS_FILE.read_text())
        print(f"  Doer Status (as of {status['updated']})")
        print(f"    CPU: {status['cpu_slot']}")
        print(f"    GPU: {status['gpu_slot']}")
        print(f"    Pending: {status['pending']}  Running: {status['running']}  "
              f"Done: {status['done']}  Failed: {status['failed']}")
    else:
        print("  Doer not running (no status file)")

    print()
    jobs = load_queue()
    if not jobs:
        print("  No jobs in queue. Edit experiments/job_queue.json to add some.")
        return

    for j in jobs:
        marker = {"pending": " ", "running": ">", "done": "+", "failed": "!", "skipped": "-"}
        m = marker.get(j.get("status", "?"), "?")
        dep = f" (after {j['depends_on']})" if j.get("depends_on") else ""
        pid = f" [PID {j['pid']}]" if j.get("pid") and j["status"] == "running" else ""
        print(f"  [{m}] {j.get('priority', '?'):>2d}. {j['name']} ({j['resource']}){dep}{pid}")


def run_loop(poll_interval: int = 60, dry_run: bool = False) -> None:
    """Main event loop."""
    print(f"{'='*60}")
    print(f"  THE DOER — Experiment Daemon")
    print(f"  Queue: {QUEUE_FILE}")
    print(f"  Logs:  {LOG_DIR}/")
    print(f"  Poll:  every {poll_interval}s")
    print(f"  Mode:  {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}")
    print()

    # Track Popen objects for jobs we launched
    procs: dict[str, subprocess.Popen] = {}

    while True:
        jobs = load_queue()
        if not jobs:
            time.sleep(poll_interval)
            continue

        # Check status of running jobs
        for job in jobs:
            if job.get("status") == "running":
                check_running_job(job, procs)

        # Find what's currently running per resource
        # Count ALL running jobs per resource (not just first) to prevent OOM
        cpu_running_jobs = [j["id"] for j in jobs if j["status"] == "running" and j["resource"] == "cpu"]
        gpu_running_jobs = [j["id"] for j in jobs if j["status"] == "running" and j["resource"] == "gpu"]
        cpu_running = cpu_running_jobs[0] if cpu_running_jobs else None
        gpu_running = gpu_running_jobs[0] if gpu_running_jobs else None

        # Safety: never launch if ANY job is already running on that resource
        # GPU jobs use 40+ GB each — two simultaneous = OOM
        cpu_busy = len(cpu_running_jobs) > 0
        gpu_busy = len(gpu_running_jobs) > 0

        # Try to fill empty slots
        launched = False
        for resource, busy in [("cpu", cpu_busy), ("gpu", gpu_busy)]:
            if busy:
                continue
            next_job = get_next_job(jobs, resource)
            if next_job is None:
                continue

            if dry_run:
                print(f"  [dry-run] Would launch on {resource}: {next_job['name']}")
                print(f"            Command: {next_job['command']}")
            else:
                print(f"  [{resource.upper()}] Launching: {next_job['name']}", flush=True)
                proc = launch_job(next_job)
                procs[next_job["id"]] = proc
                print(f"    PID {proc.pid} → {next_job.get('log_file')}", flush=True)
                launched = True

                if resource == "cpu":
                    cpu_running = next_job["id"]
                else:
                    gpu_running = next_job["id"]

        if launched or any(j["status"] != j.get("_prev_status") for j in jobs):
            save_queue(jobs)

        # Track status changes for next iteration
        for j in jobs:
            j["_prev_status"] = j["status"]

        write_status(jobs, cpu_running, gpu_running)

        # Check if all done
        all_terminal = all(j.get("status") in ("done", "failed", "skipped") for j in jobs)
        if all_terminal and jobs:
            print(f"\n  [doer] All jobs complete! Exiting.")
            save_queue(jobs)
            break

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(
        prog="doer",
        description="The Doer — always-on experiment daemon",
    )
    parser.add_argument("--poll", type=int, default=60,
                        help="Poll interval in seconds (default: 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would launch without actually starting anything")
    parser.add_argument("--status", action="store_true",
                        help="Show current status and exit")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    # Handle graceful shutdown
    def handle_signal(sig, frame):
        print(f"\n  [doer] Caught signal {sig}. Saving queue and exiting.")
        print(f"  [doer] Running jobs will continue (detached).")
        jobs = load_queue()
        save_queue(jobs)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    run_loop(poll_interval=args.poll, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

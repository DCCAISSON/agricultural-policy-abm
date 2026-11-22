"""Run the remaining seed-17 paths in at most two independent processes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def run(manifest, release, first, output):
    jobs = json.loads(manifest.read_text(encoding="utf-8"))["jobs"]
    completed = json.loads((first / "COMPLETED.json").read_text(encoding="utf-8"))
    if completed["years"] != 30 or completed["seed"] != 17:
        raise ValueError("first full path has not passed")
    first_indices = [i for i, job in enumerate(jobs) if job["original_job"] == completed["original_job"]
                     and job["mode"] == completed["mode"]]
    if len(jobs) != 48 or len(first_indices) != 1:
        raise ValueError("expected 48 paths and one already completed first path")
    output.mkdir(parents=True, exist_ok=False)
    pending = sorted((i for i in range(len(jobs)) if i not in first_indices),
                     key=lambda i: (jobs[i]["matrix"] != "p3", i))
    running, results = {}, []
    failed = False
    started = time.monotonic()
    metadata = {"manifest": str(manifest), "release": str(release), "first_path": str(first),
        "workers_max": 2, "seed": 17, "planned_paths_including_first": 48,
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "batch_input.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    while running or (pending and not failed):
        while pending and not failed and len(running) < 2:
            available = psutil.virtual_memory().available / 1024**3
            if available < 4:
                if not running:
                    failed = True
                    results.append({"status": "BLOCKED_MEMORY", "available_gb": available})
                break
            index = pending.pop(0)
            target = output / f"job_{index:02d}"
            log = (output / f"job_{index:02d}.log").open("x", encoding="utf-8")
            command = [sys.executable, "-u", "-B", "-X", "utf8", "-m",
                "scripts.run_national_formal_revision_20261118", str(manifest), str(index), str(target),
                "--release", str(release)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            running[index] = (process, log, target, time.monotonic())
            print(json.dumps({"started": index, "pid": process.pid, "job": jobs[index]["original_job"],
                              "mode": jobs[index]["mode"], "available_gb": available}), flush=True)
        for index, (process, log, target, launch) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            success = code == 0 and (target / "COMPLETED.json").exists()
            record = {"index": index, "exit_code": code, "status": "COMPLETED" if success else "FAILED",
                      "output": str(target), "elapsed_seconds": time.monotonic() - launch}
            results.append(record)
            print(json.dumps(record), flush=True)
            failed |= not success
            del running[index]
        if running or (pending and not failed):
            time.sleep(2)
    record = {**metadata, "results": results, "not_started_indices": pending,
              "elapsed_seconds": time.monotonic() - started,
              "completed_paths_including_first": 1 + sum(r["status"] == "COMPLETED" for r in results)}
    (output / ("FAILED.json" if failed else "COMPLETED.json")).write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "release", "first", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    sys.exit(run(*(getattr(args, k).resolve() for k in ("manifest", "release", "first", "output"))))

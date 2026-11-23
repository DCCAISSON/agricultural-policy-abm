"""From the authorized development root: python -B recheck_p0.py NEW_OUTPUT."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root))
here = Path(__file__).resolve().parent
d = root / "results/diagnostics/national_v1_20261122"
dest = Path(sys.argv[1]).resolve()
lock = json.loads((here / "source_before.json").read_bytes())
manifest = d / "worktree/results/diagnostics/national_v1_20261122/source_fingerprints.csv"
expected = {root / r["file"]: r["parent_sha256"] for r in
            csv.DictReader(manifest.read_text(encoding="utf-8-sig").splitlines())}
expected[root / lock["file"]] = lock["sha256"]
expected[root / "scripts/audit_2012_grain_area_gap.py"] = lock["area_gap_script_sha256"]

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check():
    changed = [p.name for p, v in expected.items() if digest(p) != v]
    if changed:
        raise RuntimeError(f"Source or original artifact changed: {changed}")

if dest == d or dest.is_relative_to(d) or d.is_relative_to(dest):
    raise ValueError("Refusing an output overlapping the original V1")
private = dest / "PRIVATE_DO_NOT_SHARE"
behavior = dest / "behavior"
if any(p.exists() for p in (private, behavior, dest / "national_v1_area_gap_2012")):
    raise FileExistsError("Use a fresh output directory")
originals = {p: digest(p) for p in d.iterdir() if p.is_file()}
originals.update({p: digest(p) for p in (d.parent/"national_v1_area_gap_2012").glob("*.csv")})
expected.update(originals)
check()
from scripts import run_national_household_portfolio as run
from scripts import audit_household_portfolio_behavior as audit
from scripts import audit_2012_grain_area_gap as gap

original = run.ROOT
gate = json.loads((d / "behavior_gate.json").read_bytes())
old = {r["seed"]: Path(r["episode"]).resolve().parent for r in gate["episodes"] if r["rule"] == "old"}
before = {r["seed"]: Path(r["episode"]).resolve().parent for r in gate["episodes"] if r["rule"] == "v1"}
for n in ("PROTOCOL.md", "reference/reference_households.csv"):
    expected[original / n] = gate["hashes"][n]
shares = Path("reference/initial_grain_shares.json")
expected[original / shares] = json.loads((before[103]/"design.json").read_bytes())["share_sha256"]
check()
run.ROOT = private / "national_household_portfolio"
(run.ROOT / shares).parent.mkdir(parents=True)
shutil.copyfile(original / shares, run.ROOT / shares)
assert digest(run.ROOT / shares) == expected[original / shares]
audit.V1_OUT = gap.V1_OUT = behavior
new = []
for seed in lock["seeds"]:
    check()
    run.run(seed, "v1", policy="base", paired_old=old[seed], organizations=60)
    check()
    job, = (private / "national_v1_20261122/runs").glob(f"s{seed}_v1_base_n60_*")
    design, base = (json.loads((p/"design.json").read_bytes()) for p in (job, before[seed]))
    assert all(design[k] == base[k] for k in ("runner_sha256","inputs","share_sha256","initial_parcels_sha256"))
    src = dict(base["source_hashes"])
    src[str(root/lock["file"])] = lock["sha256"]
    assert design["source_hashes"] == src and json.loads((job/"audit.json").read_bytes())["ok"]
    assert json.loads((job/"episode/effective_config.json").read_bytes()) == json.loads((before[seed]/"episode/effective_config.json").read_bytes())
    new.append(job/"episode")
for year in (2011, 2012):
    check()
    sys.argv = ["behavior", str(year), "--v1", "--old", *[str(old[s]/"episode") for s in lock["seeds"]],
                "--new", *map(str, new)]
    audit.main()
check()
gap.main()
check()
print("Recheck complete; source fingerprints and original artifacts unchanged.", flush=True)

"""Legacy baseline rerun, or read-only V1 comparison: python -m scripts.verify_first_paper_baseline --compare-v1."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

def main():
    from scripts import run_national_household_portfolio as experiment
    root = experiment.ROOT
    with tempfile.TemporaryDirectory(prefix="portfolio_baseline_") as tmp:
        out = Path(tmp)
        (out/"reference").mkdir()
        shutil.copy2(root/"reference/initial_grain_shares.json",out/"reference")
        paired = root/"runs/s103_old_base_20261121_215842_328673"
        archived = root/"runs/s103_new_base_20261121_220058_199520/episode"
        with patch.object(experiment,"ROOT",out):
            experiment.run(103,"new",paired_old=paired)
        episode, = (out/"runs").glob("*/episode")
        paths = [Path(f) for f in ("initial_operators.json","initial_households.json","effective_config.json")]
        paths += [Path(f"year_{year:02}")/f for year in (1,2,3) for f in
                  (*experiment.FILES,"market_outcomes.json","exogenous_shocks.json","actor_trace.jsonl")]
        hashes = {p.as_posix():experiment.old.digest(episode/p) for p in paths}
        assert all(v==experiment.old.digest(archived/p) for p,v in hashes.items())
        print(json.dumps(dict(seed=103,years=3,matched_files=len(paths),sha256=hashes),sort_keys=True))


def compare_v1():
    root = Path(__file__).resolve().parents[1] / 'results/diagnostics/national_v1_20261122'
    runs = sorted((root/'runs').glob('s103_v1_base_n60_*'))[:2]
    if len(runs) != 2 or not all((p/'audit.json').is_file() for p in runs):
        raise SystemExit('Both independent 103/base/n60 audit.json files must exist; no model was started.')
    for p in runs:
        audit = json.loads((p/'audit.json').read_bytes())
        config = json.loads((p/'episode/effective_config.json').read_bytes())
        if not (audit['ok'] is True and audit['seed'] == config['experiment']['seed'] == 103
                and audit['rule'] == 'v1' and config['runtime_fixed_rule']['national_v1'] is True
                and config['scale']['n_enterprises'] == 60):
            raise SystemExit(f'Not an audited V1 baseline: {p.name}')

    annual = ('actor_trace.jsonl actors.jsonl actual_harvests actual_plot_use annual_checks cash_flow '
              'decision_trace exogenous_shocks fiscal_account government_sar initial_rent_settlements land '
              'lease_transactions market_outcomes operator_snapshot plot_production_status preproduction_rent_ledger '
              'price_inputs production project_payments service_providers service_tasks').split()
    required = {'audit.json', 'design.json', 'reference_counts.json'}
    required.update(f'episode/{s}.json' for s in ('COMPLETED effective_config initial_exogenous initial_households '
                    'initial_operators learning_parameters q_after q_before').split())
    required.update(f'episode/year_{y:02}/{s}' + ('' if '.' in s else '.json')
                    for y in (1, 2, 3) for s in annual)
    paths = required.copy()
    for p in runs:
        paths.update(f.relative_to(p).as_posix() for f in p.rglob('*') if f.is_file())

    def local_key(s, run):
        # Strip only identifiable local roots, never a file name or content hash.
        if not isinstance(s, str) or not Path(s).is_absolute():
            return s
        path = Path(s)
        for base, prefix in ((run, '@run'), (root.parents[2], '@project')):
            if path.is_relative_to(base):
                return prefix + '/' + path.relative_to(base).as_posix()
        return s

    rows = []
    for rel in sorted(paths):
        a, b = (p/rel for p in runs)
        raw = [p.read_bytes() if p.is_file() else None for p in (a, b)]
        digests = [hashlib.sha256(data).hexdigest() if data is not None else '' for data in raw]
        differences = path_count = 0
        max_error = 0
        elapsed = None

        def compare(x, y):
            nonlocal differences, path_count, max_error
            if type(x) is type(y) and x == y:
                return
            if isinstance(x, dict) and isinstance(y, dict):
                left, right = ({local_key(k, p): k for k in d} for d, p in zip((x, y), runs))
                if len(left) != len(x) or len(right) != len(y):
                    raise ValueError(f'Ambiguous local paths in {rel}')
                differences += len(left.keys() ^ right.keys())
                for k in left.keys() & right.keys():
                    path_count += left[k] != right[k]
                    compare(x[left[k]], y[right[k]])
            elif isinstance(x, list) and isinstance(y, list):
                differences += abs(len(x) - len(y))
                for u, v in zip(x, y):
                    compare(u, v)
            elif isinstance(x, str) and isinstance(y, str) and local_key(x, runs[0]) == local_key(y, runs[1]):
                path_count += 1
            else:
                differences += 1
                if type(x) in (int, float) and type(y) in (int, float):
                    max_error = max(max_error, abs(x-y))

        if None in raw:
            status, differences = 'missing_file', 1
        elif digests[0] == digests[1]:
            status = 'byte_identical'
        elif a.suffix in ('.json', '.jsonl'):
            left, right = (json.loads(v) if a.suffix == '.json' else [json.loads(line) for line in v.splitlines()]
                           for v in raw)
            if rel == 'episode/COMPLETED.json' and 'elapsed_seconds' in left and 'elapsed_seconds' in right:
                elapsed = round(right.pop('elapsed_seconds') - left.pop('elapsed_seconds'), 3)
            if rel == 'audit.json' and left.get('baseline') != right.get('baseline'):
                refs = [root.parents[2] / rec['baseline'] for rec in (left, right)]
                if all(p.is_dir() for p in refs) and refs[0].samefile(refs[1]):
                    path_count += 1
                    del left['baseline'], right['baseline']
            compare(left, right)
            status = ('different' if differences else 'metadata_only' if elapsed or path_count else 'encoding_only')
        else:
            status, differences = 'different', 1
        rows.append((a.relative_to(root).as_posix(), b.relative_to(root).as_posix(), *digests,
                     status, differences, max_error, path_count, elapsed))
    with (root/'v1_replay.csv').open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(('first_file', 'replay_file', 'first_sha256', 'replay_sha256', 'comparison',
                         'substantive_differences', 'max_abs_numeric_error', 'local_path_differences',
                         'elapsed_seconds_replay_minus_first'))
        writer.writerows(rows)
    print(f'files={len(rows)} byte_identical={sum(r[4]=="byte_identical" for r in rows)} '
          f'different_files={sum(bool(r[5]) for r in rows)} max_abs_error={max(r[6] for r in rows)}')
    if any(r[5] for r in rows):
        raise SystemExit(1)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-v1', action='store_true', help='compare completed runs; never start a model')
    if parser.parse_args().compare_v1:
        compare_v1()
    else:
        main()

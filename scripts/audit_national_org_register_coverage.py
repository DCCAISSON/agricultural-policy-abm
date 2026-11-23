"""Recount the two R2 entity tables; inspect the saved workbook-header inventory."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/empirical/three_paper_execution_20260908/business_registry_full_parse"
OLD = BASE / "organization_runtime_input_20260908_r2/organization_runtime_input_verification.json"
HEADERS = ROOT / "project_management/evidence/next_execution_batch_20260907/business_file_schema_inventory.csv"
OUT = ROOT / "results/diagnostics/national_org_register_coverage_20261121"
REGIONS = {
    "东部": "北京 天津 河北 上海 江苏 浙江 福建 山东 广东 海南",
    "中部": "山西 安徽 江西 河南 湖北 湖南",
    "西部": "内蒙古 广西 重庆 四川 贵州 云南 西藏 陕西 甘肃 青海 宁夏 新疆",
    "东北": "辽宁 吉林 黑龙江",
}
REGION = {p: r for r, provinces in REGIONS.items() for p in provinces.split()}
REQUIRED = set("所属省份 成立日期 企业类型 经营状态 经营范围 所属行业 统一社会信用代码".split())


def save(name, header, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header.split())
        writer.writerows(rows)


def main():
    old = json.loads(OLD.read_text(encoding="utf-8"))
    hashes = {Path(k.replace("\\", "/")).name: v for k, v in old["source_hashes_before"].items()}
    totals, candidates, types = Counter(), Counter(), Counter()
    partitions, inputs = [], []
    for part, file_id in [("p041_final", "f0042"), ("p178_final_v3", "f0179")]:
        suffix = f"{part}_{file_id}_D1F_READ_20260908_v3"
        source = BASE / f"business_entities_{suffix}.csv"
        manifest = BASE / f"business_file_manifest_{suffix}.json"
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if hashlib.sha256(source.read_bytes()).hexdigest() != hashes[source.name]:
            raise ValueError(f"Entity table differs from R2: {source}")
        years, included = Counter(), Counter()
        with source.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                province, year, kind = (r[k] for k in ("provinces", "registered_years", "legal_forms"))
                key = (part, province, year, kind)
                years[year] += 1
                totals[key] += 1
                if kind in ("company", "cooperative") and "crop_production" in {
                    s.strip() for s in r["activity_labels_candidate"].split(";")
                }:
                    candidates[key] += 1
                    included[kind] += 1
        if sum(years.values()) != int(meta["entities"]):
            raise ValueError(f"Entity count differs from partition manifest: {part}")
        types.update(included)
        partitions.append((part, meta["relative_path"], meta["parsed_data_rows"], meta["nonempty_rows"],
                           sum(years.values()), json.dumps(dict(sorted(years.items()))),
                           included["company"], included["cooperative"], sum(included.values())))
        inputs.extend([source, manifest])

    actual = sum(totals.values()), sum(candidates.values()), types["company"], types["cooperative"]
    original = (old["total_entities_scanned"], old["runtime_candidate_entities"],
                old["runtime_type_counts"]["AGRI_CORPORATION"], old["runtime_type_counts"]["COOPERATIVE"])
    if not actual == original == (162119, 4154, 4085, 69):
        raise ValueError(f"Reconciliation failed: recount={actual}, R2={original}")

    # File coverage comes from workbook headers, not from entity counts or worksheet dimensions.
    files = []
    with HEADERS.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            region = REGION[r["province_prefix"]]
            missing = REQUIRED.difference(json.loads(r["fields_json"]))
            files.append((region, r["province_prefix"], r["path"], r["sheet"], r["header_row"],
                          r["read_status"], ";".join(sorted(missing)), int(Path(r["path"]).is_file())))
    if len({row[2] for row in files}) != len(files):
        raise ValueError("Header inventory has repeated workbook paths; count workbooks separately.")

    OUT.mkdir(exist_ok=True, parents=True)
    save("partition_totals.csv", "partition raw_workbook raw_rows nonempty_raw_rows entity_rows registered_year_counts company_candidates cooperative_candidates candidates", partitions)
    save("coverage.csv", "partition region province_raw registered_year_raw legal_form entity_rows candidates",
         ((*k[:1], REGION[k[1].removesuffix("省").removesuffix("自治区")], *k[1:], n, candidates[k])
          for k, n in sorted(totals.items())))
    save("file_inventory.csv", "region province_prefix original_path sheet header_row archived_read_status missing_required_fields original_path_exists", files)
    strata = []
    for region in REGIONS:
        rows = [r for r in files if r[0] == region]
        strata.append((region, len({r[1] for r in rows}), len(rows),
                       sum(r[5] == "ok" and not r[6] for r in rows), sum(r[7] for r in rows),
                       sum(n for (_, p, y, _), n in candidates.items()
                           if y == "2010" and REGION[p.removesuffix("省").removesuffix("自治区")] == region)))
    save("four_regions.csv", "region provinces inventoried_files files_with_required_headers original_paths_present old_candidates_registered_2010", strata)

    paths = "\n".join(f"- `{p.relative_to(ROOT).as_posix()}`" for p in [*inputs, OLD, HEADERS])
    report = f"""# 工商登记初始化试验：两分区核验

在项目根目录运行：`python -B -X utf8 scripts/audit_national_org_register_coverage.py`。脚本只读下列输入，覆盖本诊断目录中的同名产物：

{paths}

**计数口径。** 逐条读取两张分区实体表，不跨文件去重，不按经营状态筛选。以原始 provinces、registered_years、legal_forms 分组；仅当 legal_forms 等于 company 或 cooperative，且 activity_labels_candidate 按分号拆分、去除首尾空格后含完整标签 crop_production 时计为候选。登记文本标签不是实际种植记录。实体表的 SHA-256 已与旧 R2 记录核对，分区实体行数已与各自文件清单核对。

**结果。** partition_totals.csv 区分工作表数据行、非空原始行和分区实体行。后者合计 {actual[0]:,}；候选 {actual[1]:,}，公司 {actual[2]:,}、合作社 {actual[3]:,}，四项与 R2 一致。coverage.csv 保留所有实际出现的省名×登记年份×法律形式组合，未入选的组记 0，不把未观察的组补成零。内蒙古自治区 2023 年 12 条、候选 0；安徽省 2023 年 162,105 条，含全部候选；另有“安徽”2 条，年份为空、类型 unresolved，均非候选。未合并省名或填补年份。该比例仅描述两分区内的候选构成；旧试验把它用作固定 60 家组织的类型抽取概率，而非用登记数确定组织总量。

**2010 年四区能否抽样。** file_inventory.csv 逐文件复核历史表头清单中的省份前缀、工作表、表头行和字段。所查字段为{'、'.join(sorted(REQUIRED))}；four_regions.csv 按东、中、西、东北列出文件数和通过字段检查的文件数。原清单中的 {len(files)} 个路径本次可访问 {sum(r[7] for r in files)} 个；字段证据来自已存清单，不是本次重新打开工作簿所得。文件名可用于按地区选取待核分区，表头只能证明有相应列，不说明各列完整或文件中有 2010 年候选。

两张实体表在四区均未观察到 2010 年登记候选，不表示四区当年没有此类单位。若抽取当年新登记单位，须在各区文件内核出 2010 年登记、法律形式和活动均符合的实体，并核对跨文件重复；若抽取 2010 年在营单位，还须纳入此前成立且当年仍在营的实体，核实当年经营状态、退出时点和种植活动。当前登记状态不能反推当年状态。现有清单不足以计算四区抽样权重或替换模型参数。
"""
    (OUT / "核验说明.md").write_text(report, encoding="utf-8")
    print(f"PASS entities={actual[0]} candidates={actual[1]} company={actual[2]} cooperative={actual[3]} -> {OUT}")


if __name__ == "__main__":
    main()

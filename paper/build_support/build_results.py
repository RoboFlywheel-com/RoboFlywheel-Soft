# SPDX-License-Identifier: Apache-2.0
"""Regenerate manuscript tables from the shared, hashed experimental CSVs."""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import json
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
SHARED_DATA = PAPER.parent / "paper_submission_assets/data"
DATA = PAPER / "analysis_data" if (PAPER / "analysis_data").is_dir() else SHARED_DATA
OUT = PAPER / "results"
METHODS = {
    "fixed_profile": "Fixed profile",
    "one_shot_prior": "One-shot prior",
    "ruletune": "Rule-based update",
    "seeded_sobol": "Sobol search",
    "vid2sid_style_reimplementation": "Vid2Sid-style (text)",
    "swim2real_style_reimplementation": "Swim2Real-style (text)",
    "full_numeric_agent": "Full numerical agent",
}
FAMILY = {"solid": "Soft solids", "cloth_sheet": "Cloth sheets", "shell": "Shells", "garment": "Garments", "bag": "Bags"}
INPUTS = {}


def read(name):
    p = DATA / name
    INPUTS[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return list(csv.DictReader(p.open()))


def table(name, spec, head, rows):
    s = "\\begin{tabular}{" + spec + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n"
    s += "\n".join(("\\midrule\n" if row[0] == 'Privileged reference' else '') + " & ".join(map(str, row)) + r" \\" for row in rows)
    s += "\n\\bottomrule\n\\end{tabular}\n"
    (OUT / name).write_text(s)


def main():
    OUT.mkdir(exist_ok=True)
    macros = {}
    census = read("verification_by_family.csv")
    by_family = {r["family"]: r for r in census}
    fields = ["category_n", "pool_n", "physics_pass_n", "runtime_pass_n", "intersection_n"]
    lines = [[FAMILY[f]] + [f"{int(by_family[f][k]):,}" for k in fields] for f in FAMILY]
    totals = [sum(int(r[k]) for r in census) for k in fields]
    lines.append([r"\textbf{Total}"] + [r"\textbf{" + f"{n:,}" + "}" for n in totals])
    table("T02_dataset.tex", "lrrrrr", "Family & Types & Pool & Physical & Runtime & Both", lines)
    for key, value in zip(["types", "pool", "physical", "runtime", "both"], totals):
        macros[key] = f"{value:,}"
    states = read("agent_state_counts.csv")
    state_names = {"accepted_first_pass": "Accepted at first candidate", "corrected_then_accepted": "Accepted after correction", "rejected_model_limit": "Model-limit rejection", "rejected_numerical_failure": "Numerical rejection", "uncertain_human_escalation": "Unresolved judgment"}
    table("A01_states.tex", "lr", "Construction outcome & Assets", [[state_names[r["agent_status"]],f"{int(r['n']):,}"] for r in states])
    family_states=read('production_states_by_family.csv')
    fs={(r['family'],r['agent_status']):int(r['n']) for r in family_states}
    table('A09_family_states.tex','lrrrrr','Family & First pass & Corrected & Model limit & Numerical & Unresolved',[[FAMILY[f]]+[f"{fs.get((f,state),0):,}" for state in state_names] for f in FAMILY])
    summary, paired, transitions, transitions_by_method = [], [], [], []
    for experiment in ["recovery", "filter"]:
        rows = read(experiment + "_long.csv")
        ids = sorted({r['sample_id'] for r in rows})
        rng = np.random.default_rng(20260926)
        draws = rng.integers(0,len(ids),(5000,len(ids)))
        groups = defaultdict(list)
        for r in rows:
            groups[r["method"]].append(r)
        def cross(rs):
            counts = Counter()
            for r in rs:
                if not r['calibration_error'] or not r['heldout_error']:
                    counts['missing'] += 1
                else:
                    counts[(float(r['calibration_error']) <= 0.05, r['status_success'] == 'True')] += 1
            return [counts[True,True], counts[True,False], counts[False,True], counts[False,False], counts['missing']]
        counts = cross(rows)
        assert sum(counts) == len(rows)
        assert counts == ([682,49,3,83,23] if experiment == 'recovery' else [8075,325,0,0,0])
        transitions.append(["30 targets" if experiment == 'recovery' else "300 bags"] + [f'{n:,}' for n in counts] + [f'{len(rows):,}'])
        for method in METHODS:
            transitions_by_method.append([experiment.capitalize(), METHODS[method]] + cross(groups[method]))
        for method in METHODS:
            rs = groups[method]
            valid = [float(r["heldout_error"]) for r in rs if r["heldout_error"]]
            summary.append(dict(experiment=experiment, method=method, n=len(rs), success=sum(r["status_success"] == "True" for r in rs), error_n=len(valid), mean_error=float(np.mean(valid))))
        budget_rows = read(experiment + "_by_method_budget.csv")
        lookup = {(r["method"], int(r["budget_allowed"])): r for r in budget_rows}
        table("A_" + experiment + "_budgets.tex", "lrrrr", "Method & Budget 1 & Budget 2 & Budget 4 & Budget 8", [[METHODS[m]] + [f"{float(lookup[m,b]['heldout_error_mean']):.4f} ({lookup[m,b]['heldout_error_n']})" for b in [1,2,4,8]] for m in METHODS])
        for budget in [1,2,4,8]:
            pair = defaultdict(dict)
            for r in rows:
                if int(r["budget_allowed"]) == budget and r["method"] in ("fixed_profile", "full_numeric_agent") and r["heldout_error"]:
                    key = r["sample_id"]
                    assert r["method"] not in pair[key]
                    pair[key][r["method"]] = float(r["heldout_error"])
            delta = np.array([pair[sid]['full_numeric_agent']-pair[sid]['fixed_profile'] if len(pair[sid])==2 else np.nan for sid in ids])
            boot = np.nanmean(delta[draws],axis=1)
            lo,hi = np.nanquantile(boot,[.025,.975])
            item = dict(experiment=experiment,budget=budget,n_pairs=int(np.isfinite(delta).sum()),mean_difference=float(np.nanmean(delta)),ci_low=float(lo),ci_high=float(hi))
            paired.append(item)
            if budget == 8:
                macros[experiment + "-delta"] = f"{np.nanmean(delta):.4f}"
                macros[experiment + "-ci"] = f"[{lo:.4f}, {hi:.4f}]"
                macros[experiment + "-pairs"] = str(int(np.isfinite(delta).sum()))
    lookup = {(r["experiment"],r["method"]):r for r in summary}
    main_rows = []
    for m,label in METHODS.items():
        n,f = lookup["recovery",m],lookup["filter",m]
        main_rows.append([label, f"{n['success']}/{n['n']}", f"{n['mean_error']:.4f}", n['error_n'],f"{f['success']}/{f['n']}",f"{f['mean_error']:.4f}"])
    table("T04_parameters.tex", "lrrrrr", "Method & Recovery pass & Error & Valid n & Expanded pass & Error", main_rows)
    table("T04_verification.tex", "lrrrrrr", r"Population & Both pass & Cal. only & Indep. only & Neither & Missing & Total", transitions)
    table("A07_verification_methods.tex", "llrrrrr", r"Experiment & Method & Both & Cal. only & Indep. only & Neither & Missing", transitions_by_method)
    structure = [r for r in read("structure_cases.csv") if r["status"] == "simulated_for_review"]
    variants = {"shape_preserving": "Shape-preserving", "softer_material": "Softer material", "flexible_cloth": "Flexible cloth"}
    struct_rows = []
    for probe in ["drop","compression"]:
        for v,label in variants.items():
            rs = [r for r in structure if r["probe"]==probe and r["variant"]==v]
            struct_rows.append([probe.capitalize(),label,f"{np.mean([float(r['height_ratio']) for r in rs]):.3f}",f"{np.mean([float(r['final_max_speed']) for r in rs]):.3f}"])
    table("A02_structure.tex","llrr","Probe & Configuration & Height ratio & Max. speed (m/s)",struct_rows)
    compact_structure=[]
    for v,label in variants.items():
        line=[label]
        for probe in ['drop','compression']:
            rs=[r for r in structure if r['probe']==probe and r['variant']==v]
            line.extend([f"{np.mean([float(r['height_ratio']) for r in rs]):.3f}",f"{np.mean([float(r['final_max_speed']) for r in rs]):.5f}"])
        compact_structure.append(line)
    table('T03_structure.tex','lrrrr',r'& \multicolumn{2}{c}{Drop} & \multicolumn{2}{c}{Compression} \\ Configuration & Height ratio & Speed (m/s) & Height ratio & Speed (m/s)',compact_structure)
    detail_rows=[]
    for sid in sorted({r['sample_id'] for r in structure}):
        for probe in ['drop','compression']:
            rs={r['variant']:r for r in structure if r['sample_id']==sid and r['probe']==probe}
            detail_rows.append([sid.rsplit('_',1)[-1],probe.capitalize()]+[f"{float(rs[v]['height_ratio']):.3f} / {float(rs[v]['final_max_speed']):.3f}" for v in variants])
    table('A03_structure_individual.tex','llrrr','Garment & Probe & Shape-preserving & Softer material & Flexible cloth',detail_rows)
    human=read('human_rating_counts.csv')
    dims={'q1_settle':'Settling','q2_deformation':'Deformation','q3_recovery':'Recovery','q4_contact':'Contact','q5_overall':'Overall'}
    human_rows=[]
    for dim,name in dims.items():
        rs=[r for r in human if r['dimension']==dim]
        n=int(rs[0]['n']); accept=sum(int(r['count']) for r in rs if int(r['score'])>=4)
        human_rows.append([name,n,accept,f'{100*accept/n:.1f}'+r'\%'])
    table('A04_experts.tex','lrrr','Dimension & Rated clips & Score 4 or 5 & Proportion',human_rows)
    routing=read('routing_execution.csv')
    route_names={'pbd_cloth':'PBD cloth','mpm_elastic':'Elastic MPM','mpm_elastoplastic':'Elastoplastic MPM'}
    table('A05_routing.tex','lrr','Executed solver & Assets & Physically verified',[[name,sum(r['executed_solver']==mode for r in routing),sum(r['executed_solver']==mode and r['physical_status']=='verified' for r in routing)] for mode,name in route_names.items()])
    apps=read('application_long.csv')
    app_names={'fixed':'Fixed command','reactive_slip':'Slip feedback','deligrasp_prior':'Force prior (local)','privileged_oracle':'Privileged reference'}
    assert len(apps)==4000 and sum(r['task_success']=='True' for r in apps)==3079
    assert sum(r['intact_success']=='True' for r in apps)==0 and all(r['damage_failure']=='True' for r in apps)
    app_main=[]; app_detail=[]
    for method,label in app_names.items():
        pooled=[r for r in apps if r['method']==method]
        hidden=[r for r in pooled if r['source_split']=='hidden_test']
        cal=[r for r in pooled if r['source_split']=='calibration']
        assert len(hidden)==340 and len(cal)==660
        def task_rate(rs):
            n=sum(r['task_success']=='True' for r in rs)
            return f'{n}/{len(rs)} ({100*n/len(rs):.1f}'+r'\%)'
        app_main.append([label,task_rate(cal),task_rate(hidden),'0/340',f"{np.median([float(r['damage_proxy']) for r in hidden]):.3f}"])
        for split,rs in [('Calibration',cal),('Held-out',hidden),('All',pooled)]:
            app_detail.append([split,label,len(rs),sum(r['task_success']=='True' for r in rs),0,len(rs),f"{np.median([float(r['damage_proxy']) for r in rs]):.3f}",f"{np.median([float(r['integrity_proxy']) for r in rs]):.3f}"])
    table('T06_application.tex','lrrrr','Controller & Cal. task & Held-out task & Intact & Damage',app_main)
    table('A08_application.tex','llrrrrrr','Split & Controller & N & Task & Intact & Flag & Damage & Integrity',app_detail)
    table('A06_paired.tex','lrrrr',r'Experiment & Budget & Pairs & Difference & 95\% interval',[[r['experiment'].capitalize(),r['budget'],r['n_pairs'],f"{r['mean_difference']:.4f}",f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"] for r in paired])
    (OUT/'results.tex').write_text('% Generated by build_support/build_results.py; source hashes in analysis_manifest.json.\n'+''.join(r'\setresult{'+k+'}{'+v+'}\n' for k,v in macros.items()))
    for name,rows in [('method_summary.csv',summary),('paired_differences.csv',paired)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'analysis_manifest.json').write_text(json.dumps(dict(inputs=INPUTS,bootstrap_seed=20260926,bootstrap_replicates=5000,paired_unit='asset/target within each budget',difference='full_numeric_agent minus fixed_profile; shared valid pairs only',calibration_tolerance=0.05,independent_tolerance=0.08,application_status='completed_deformation_proxy_evaluation',application_canonical_sha256='8ad6158f90e5b1c750d8ab630a2add5f570a5a08680c88bc01c179cc3f65336a',source_registry='../analysis_data/source_registry.json'),indent=2)+'\n')
    print(json.dumps(paired,indent=2))


if __name__ == '__main__':
    main()

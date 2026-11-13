#!/usr/bin/env python3
"""Reproduce the exploratory analysis of compiled microplastic decision records.

Read-only input: the archived XLSX is never written. Calculations use literal
eligibility and gate cells; formula caches are audit comparators, not inputs.
All supplementary diagnostics are post hoc and descriptive. They do not assess
decision accuracy or authenticate reviewer identities or primary sources.

Usage:
  python analysis.py --input Four_Reviewer_Analyst_Reference_Aligned.xlsx --output results
Dependencies: Python 3.10+, numpy, openpyxl.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np
import openpyxl
from openpyxl.utils import get_column_letter

RATERS = list("ABCD")
GATES = [f"G{i:02d}" for i in range(1, 7)]
STATES = ["Pass", "Not required", "Fail", "Unresolved"]
OUTCOMES = ["Supported", "Not supported", "Unresolved"]
PAIRS = list(combinations(range(4), 2))
EXPECTED_SHA256 = "ce03124d396f53a6caa82c4e5f5b03d593ad2878c4de06ede9db63809f6b0640"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows([{k: json.dumps(clean(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple))
                          else clean(v) for k, v in row.items()} for row in records])


def normalize(value):
    if value is None:
        return None
    value = value.strip() if isinstance(value, str) else value
    return "Fail" if value == "Fl" else value


def outcome(states, eligibility="Eligible"):
    if eligibility == "Ineligible":
        return "Ineligible"
    if eligibility != "Eligible":
        return "Pending eligibility"
    if any(x is None or x == "" for x in states):
        return "Incomplete"
    if "Fail" in states:
        return "Not supported"
    if "Unresolved" in states:
        return "Unresolved"
    return "Supported"


def agreement(values, categories):
    a = np.asarray(values, dtype=object)
    n, nr = a.shape
    pairs = list(combinations(range(nr), 2))
    matches = sum(int(np.sum(a[:, r] == a[:, s])) for r, s in pairs)
    counts = np.asarray([[np.sum(a[:, r] == c) for c in categories] for r in range(nr)], dtype=float)
    p = counts / n
    po = matches / (len(pairs) * n)
    pe_conger = float(np.mean([np.dot(p[r], p[s]) for r, s in pairs]))
    pe_fleiss = float(np.sum(np.mean(p, axis=0) ** 2))
    def kappa(pe):
        return None if np.isclose(pe, 1, rtol=0, atol=1e-14) else (po - pe) / (1 - pe)
    unanimity = int(np.sum(np.all(a == a[:, :1], axis=1)))
    at_least_three = int(sum(max(Counter(row).values()) >= 3 for row in a))
    return dict(n=n, raters=nr, unanimous=unanimity, at_least_three=at_least_three,
                matching_pairs=matches, possible_pairs=len(pairs) * n, observed=po,
                expected_conger=pe_conger, conger=kappa(pe_conger),
                expected_fleiss=pe_fleiss, fleiss=kappa(pe_fleiss),
                categories=categories, rater_counts={RATERS[r]: dict(zip(categories, counts[r].astype(int)))
                                                     for r in range(nr)})


def bootstrap(values, types, dois, cluster_order, nrep, seed):
    """Same cluster draws for Overall and Limited, fixed raters, PCG64 RNG."""
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.choice(len(cluster_order), size=(nrep, len(cluster_order)), replace=True)
    weights = np.zeros((nrep, len(cluster_order)), dtype=np.int64)
    for b in range(nrep):
        weights[b] = np.bincount(draws[b], minlength=len(cluster_order))
    records, results = [], {}
    for stratum in ("Overall", "Limited"):
        all_mask = np.ones(len(values), dtype=bool) if stratum == "Overall" else types == stratum
        sizes, match, margins = [], [], []
        for doi in cluster_order:
            x = values[(dois == doi) & all_mask]
            sizes.append(len(x))
            match.append(sum(int(np.sum(x[:, r] == x[:, s])) for r, s in PAIRS))
            margins.append([[int(np.sum(x[:, r] == c)) for c in OUTCOMES] for r in range(4)])
        sample_n = weights @ np.asarray(sizes)
        po = (weights @ np.asarray(match)) / (6 * sample_n)
        p = np.einsum("bc,crk->brk", weights, np.asarray(margins)) / sample_n[:, None, None]
        pe_c = np.mean(np.stack([np.sum(p[:, r] * p[:, s], axis=1) for r, s in PAIRS]), axis=0)
        pe_f = np.sum(np.mean(p, axis=1) ** 2, axis=1)
        summary = {"n_min": int(sample_n.min()), "n_max": int(sample_n.max())}
        kappas = {}
        for name, pe in (("conger", pe_c), ("fleiss", pe_f)):
            finite = ~np.isclose(pe, 1, rtol=0, atol=1e-14)
            k = np.full(nrep, np.nan)
            k[finite] = (po[finite] - pe[finite]) / (1 - pe[finite])
            kappas[name] = k
            summary[name] = dict(finite=int(finite.sum()), undefined=int((~finite).sum()),
                undefined_draws_one_based=(np.where(~finite)[0] + 1).tolist(),
                interval=np.quantile(k[finite], [0.025, 0.975], method="linear").tolist())
        results[stratum] = summary
        for b in range(nrep):
            records.append(dict(draw=b + 1, stratum=stratum, sample_n=int(sample_n[b]),
                                conger=kappas["conger"][b], fleiss=kappas["fleiss"][b]))
    return results, records


def expected_formula(case_row, gate_row, r):
    e = f"{get_column_letter(8 + r)}{case_row}"
    first, last = get_column_letter(4 + 6 * r), get_column_letter(9 + 6 * r)
    s = f"'03_GATE_MATRIX'!{first}{gate_row}:{last}{gate_row}"
    return (f'=IF({e}="Ineligible","Ineligible",IF({e}<>"Eligible","Pending eligibility",'
            f'IF(COUNTBLANK({s})>0,"Incomplete",IF(COUNTIF({s},"Fail")>0,"Not supported",'
            f'IF(COUNTIF({s},"Unresolved")>0,"Unresolved","Supported")))))')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    wf = openpyxl.load_workbook(args.input, data_only=False)
    wc = openpyxl.load_workbook(args.input, data_only=True)
    ws, gm = wf["02_CASES"], wf["03_GATE_MATRIX"]
    gate_rows = {gm.cell(row, 1).value: row for row in range(2, gm.max_row + 1)}
    if len(gate_rows) != gm.max_row - 1:
        raise ValueError("Duplicate Question_ID in gate matrix")
    evidence_rows = {(wf["04_GATE_EVIDENCE"].cell(row, 1).value,
                      wf["04_GATE_EVIDENCE"].cell(row, 3).value): row
                     for row in range(2, wf["04_GATE_EVIDENCE"].max_row + 1)}
    questions, rating_rows, observations, formula_audit = [], [], [], []
    normalizations, redundant_mismatches, prefix_mismatches = [], [], []
    all_gates, all_elig, all_outcomes = [], [], []
    for row in range(2, ws.max_row + 1):
        qid, paper, doi, typ, reuse, matrix = [ws.cell(row, col).value for col in range(1, 7)]
        gr = gate_rows[qid]
        assert [gm.cell(gr, col).value for col in range(1, 4)] == [qid, paper, doi]
        eligibility = [normalize(ws.cell(row, 8 + r).value) for r in range(4)]
        assert set(eligibility) <= {"Eligible", "Ineligible"}
        common = all(x == "Eligible" for x in eligibility)
        x = np.empty((6, 4), dtype=object)
        for g in range(6):
            er = evidence_rows[(qid, GATES[g])]
            for r in range(4):
                cell = gm.cell(gr, 4 + 6 * r + g)
                raw = cell.value
                x[g, r] = normalize(raw)
                if raw != x[g, r]:
                    normalizations.append(dict(cell=f"03_GATE_MATRIX!{cell.coordinate}", old=raw, new=x[g, r]))
                if x[g, r] is not None and x[g, r] not in STATES:
                    raise ValueError(f"Unknown state in {cell.coordinate}: {raw}")
                ev = wf["04_GATE_EVIDENCE"].cell(er, 8 + r)
                if normalize(ev.value) != x[g, r]:
                    redundant_mismatches.append(dict(question_id=qid, gate=GATES[g], rater=RATERS[r],
                                                     matrix_state=x[g, r], evidence_state=ev.value))
                rationale = wf["04_GATE_EVIDENCE"].cell(er, 12 + r).value
                m = re.match(r"^(Pass|Fail|Unresolved|Not required)\b", rationale or "", re.IGNORECASE)
                if m and m.group(1).lower() != str(x[g, r]).lower():
                    prefix_mismatches.append(dict(question_id=qid, gate=GATES[g], rater=RATERS[r],
                        state=x[g, r], rationale_prefix=m.group(1), cell=f"04_GATE_EVIDENCE!{get_column_letter(12+r)}{er}", rationale=rationale))
                rating_rows.append(dict(question_id=qid, paper_id=paper, doi=doi, question_type=typ,
                    matrix=matrix, rater=RATERS[r], gate=GATES[g], eligibility=eligibility[r],
                    common_eligible=common, raw_state=raw, state=x[g, r],
                    source_cell=f"03_GATE_MATRIX!{cell.coordinate}", rationale=rationale,
                    rationale_cell=f"04_GATE_EVIDENCE!{get_column_letter(12+r)}{er}"))
        outcomes = [outcome(x[:, r].tolist(), eligibility[r]) for r in range(4)]
        for r in range(4):
            actual_formula = ws.cell(row, 12 + r).value
            cache = wc["02_CASES"].cell(row, 12 + r).value
            expected = expected_formula(row, gr, r)
            formula_audit.append(dict(question_id=qid, rater=RATERS[r],
                formula_cell=f"02_CASES!{get_column_letter(12+r)}{row}",
                is_formula=isinstance(actual_formula, str) and actual_formula.startswith("="),
                formula_matches_rule=actual_formula == expected, cache=cache,
                recalculated=outcomes[r], cache_matches=cache == outcomes[r], common_eligible=common))
            observations.append(dict(question_id=qid, paper_id=paper, doi=doi, question_type=typ,
                matrix=matrix, specified_reuse=reuse, rater=RATERS[r], eligibility=eligibility[r],
                common_eligible=common, **dict(zip(GATES, x[:, r].tolist())), outcome=outcomes[r]))
        questions.append(dict(question_id=qid, paper_id=paper, doi=doi, question_type=typ,
            matrix=matrix, specified_reuse=reuse, source_row=row, common_eligible=common,
            **{f"eligibility_{r}": eligibility[i] for i, r in enumerate(RATERS)},
            **{f"outcome_{r}": outcomes[i] for i, r in enumerate(RATERS)}))
        all_gates.append(x)
        all_elig.append(eligibility)
        all_outcomes.append(outcomes)
    assert len({q["question_id"] for q in questions}) == len(questions)
    gates_all = np.asarray(all_gates, dtype=object)
    elig_all = np.asarray(all_elig, dtype=object)
    outcomes_all = np.asarray(all_outcomes, dtype=object)
    common_mask = np.all(elig_all == "Eligible", axis=1)
    q = [row for row in questions if row["common_eligible"]]
    x = gates_all[common_mask]
    y = outcomes_all[common_mask]
    types = np.asarray([row["question_type"] for row in q])
    dois = np.asarray([row["doi"] for row in q])
    assert not any(v is None for v in x.ravel())
    for i in range(len(questions)):
        for r in range(4):
            if elig_all[i, r] == "Ineligible":
                assert all(v is None for v in gates_all[i, :, r]), "Ineligible gates unexpectedly populated"
            else:
                assert all(v in STATES for v in gates_all[i, :, r]), "Eligible record has missing or invalid gates"
    order_sorted = sorted(set(dois.tolist()))

    agreement_results = {"Eligibility": agreement(elig_all, ["Eligible", "Ineligible"])}
    gate_agreement, disagreement, qgate_records, q_diagnostics, pair_vectors = [], {}, [], [], []
    for stratum in ["Overall", "Limited", "Stronger"]:
        mask = np.ones(len(q), dtype=bool) if stratum == "Overall" else types == stratum
        agreement_results[stratum] = agreement(y[mask], OUTCOMES)
        for g, gate in enumerate(GATES):
            gate_agreement.append(dict(stratum=stratum, gate=gate, **agreement(x[mask, g, :], STATES)))
        unit_count, pair_count = Counter(), Counter()
        for local in np.where(mask)[0]:
            for g in range(6):
                states = set(x[local, g, :])
                category = "Unanimous" if len(states) == 1 else ("Pass/Not required only" if states <= {"Pass", "Not required"} else "Other disagreement")
                unit_count[category] += 1
                for r, s in PAIRS:
                    if x[local, g, r] == x[local, g, s]:
                        pair_count["Matching"] += 1
                    elif {x[local, g, r], x[local, g, s]} == {"Pass", "Not required"}:
                        pair_count["Pass/Not required only"] += 1
                    else:
                        pair_count["Other disagreement"] += 1
        disagreement[stratum] = dict(question_gate_units=dict(unit_count),
            pair_comparisons=dict(pair_count),
            full_vector_unanimous=int(np.sum(np.all(x[mask] == x[mask, :, :1], axis=(1, 2)))),
            not_required_by_rater={RATERS[r]: int(np.sum(x[mask, :, r] == "Not required")) for r in range(4)})
    for i, row in enumerate(q):
        unanimous_gates, common_fails = [], []
        for g, gate in enumerate(GATES):
            states = set(x[i, g])
            category = "Unanimous" if len(states) == 1 else ("Pass/Not required only" if states <= {"Pass", "Not required"} else "Other disagreement")
            if len(states) == 1:
                unanimous_gates.append(gate)
            if states == {"Fail"}:
                common_fails.append(gate)
            qgate_records.append(dict(question_id=row["question_id"], doi=row["doi"],
                question_type=row["question_type"], gate=gate, disagreement_type=category,
                **{RATERS[r]: x[i, g, r] for r in range(4)}))
        q_diagnostics.append(dict(question_id=row["question_id"], doi=row["doi"],
            question_type=row["question_type"], unanimous_gates=unanimous_gates,
            n_unanimous_gates=len(unanimous_gates), common_fail_gates=common_fails,
            outcome_unanimous=len(set(y[i])) == 1,
            unanimous_outcome=y[i, 0] if len(set(y[i])) == 1 else "",
            full_vector_unanimous=len(unanimous_gates) == 6))
        for r, s in PAIRS:
            pair_vectors.append(dict(question_id=row["question_id"], question_type=row["question_type"],
                pair=f"{RATERS[r]}-{RATERS[s]}", hamming=int(np.sum(x[i, :, r] != x[i, :, s])),
                outcome_match=y[i, r] == y[i, s]))

    boot_sorted, boot_records_sorted = bootstrap(y, types, dois, order_sorted, args.bootstrap_replicates, args.seed)
    # Specify deterministic ordering because a seeded bootstrap alone does not
    # fix the mapping from pseudorandom indices to source clusters.
    bootstrap_result = dict(replicates=args.bootstrap_replicates, seed=args.seed, generator="NumPy PCG64",
        raters="Fixed A-D", cluster_definition="Exact DOI; independence of underlying datasets not established",
        order="Lexicographic exact DOI", cluster_order=order_sorted,
        quantile_method="numpy.quantile(method='linear') on finite estimates",
        results=boot_sorted)

    ablation, ablation_changes = [], []
    for g, gate in enumerate(GATES):
        z = np.asarray([[outcome(np.delete(x[i, :, r], g).tolist()) for r in range(4)] for i in range(len(q))], dtype=object)
        for stratum in ["Overall", "Limited", "Stronger"]:
            mask = np.ones(len(q), dtype=bool) if stratum == "Overall" else types == stratum
            changes = z[mask] != y[mask]
            transitions = Counter(f"{a} -> {b}" for a, b in zip(y[mask].ravel(), z[mask].ravel()) if a != b)
            ablation.append(dict(removed_gate=gate, stratum=stratum,
                n_rater_question_records=int(mask.sum() * 4), changed_records=int(changes.sum()),
                changed_questions=int(np.sum(np.any(changes, axis=1))), transitions=dict(transitions),
                **agreement(z[mask], OUTCOMES)))
        for i, r in zip(*np.where(z != y)):
            ablation_changes.append(dict(removed_gate=gate, question_id=q[i]["question_id"],
                question_type=types[i], rater=RATERS[r], original_outcome=y[i, r], omitted_gate_outcome=z[i, r]))

    lodo = []
    for doi in order_sorted:
        for stratum in ["Overall", "Limited"]:
            mask = (dois != doi) & (np.ones(len(q), dtype=bool) if stratum == "Overall" else types == stratum)
            lodo.append(dict(omitted_doi=doi, omitted_paper_id=next(row["paper_id"] for row in q if row["doi"] == doi),
                             stratum=stratum, **agreement(y[mask], OUTCOMES)))
    lodo_summary = {}
    for stratum in ["Overall", "Limited"]:
        rs = [row for row in lodo if row["stratum"] == stratum]
        lodo_summary[stratum] = {stat: dict(min=min(row[stat] for row in rs), max=max(row[stat] for row in rs))
                                 for stat in ["n", "conger", "fleiss", "observed"]}

    reference_rows, reference_rule_counts, reference_mismatches, ref_outcomes = [], Counter(), [], []
    reference_map = {wf["07_PANEL_DECISIONS"].cell(row, 1).value: row
                     for row in range(2, wf["07_PANEL_DECISIONS"].max_row + 1)}
    for i, row in enumerate(questions):
        rr = reference_map[row["question_id"]]
        if not row["common_eligible"]:
            assert wf["07_PANEL_DECISIONS"].cell(rr, 5).value == "Disputed"
            assert all(wf["07_PANEL_DECISIONS"].cell(rr, 6 + g).value is None for g in range(6))
            continue
        predicted = []
        for g, gate in enumerate(GATES):
            votes = gates_all[i, g].tolist()
            counts = Counter(votes)
            state, votes_n = counts.most_common(1)[0]
            if row["question_id"] == "T03-L" and gate == "G01":
                state, rule = "Unresolved", "Explicit T03-L G01 override"
            elif votes_n == 4:
                rule = "Unanimous retention"
            elif votes_n >= 3:
                rule = "Three-vote mode"
            elif row["question_type"] == "Limited" and set(votes) <= {"Pass", "Not required"}:
                state, rule = "Not required", "Limited Pass/Not required tie"
            else:
                state, rule = "Unresolved", "Other mixed state without three votes"
            reference_rule_counts[rule] += 1
            predicted.append(state)
            recorded = wf["07_PANEL_DECISIONS"].cell(rr, 6 + g).value
            refrec = dict(question_id=row["question_id"], gate=gate, rule=rule,
                          recalculated=state, recorded=recorded, matches=state == recorded)
            reference_rows.append(refrec)
            if state != recorded:
                reference_mismatches.append(refrec)
        derived = outcome(predicted)
        recorded_outcome = wc["07_PANEL_DECISIONS"].cell(rr, 12).value
        assert derived == recorded_outcome
        ref_outcomes.append(derived)

    ai_labels = []
    for sheet in wf:
        for row in sheet:
            for cell in row:
                if isinstance(cell.value, str) and "original AI-B record" in cell.value:
                    ai_labels.append(dict(cell=f"{sheet.title}!{cell.coordinate}", text=cell.value))
    originals = ["Reviewer_A.xlsx", "Reviewer_B.xlsx", "Reviewer_C(1).xlsx", "Reviewer_D.xlsx"]
    paper_doi = {}
    for row in questions:
        paper_doi.setdefault(row["paper_id"], set()).add(row["doi"])
    assert all(len(v) == 1 for v in paper_doi.values())
    assert len({row["doi"] for row in questions}) == len(paper_doi)
    baseline = np.repeat(np.where(types == "Stronger", "Not supported", "Supported")[:, None], 4, axis=1)
    result = dict(
        input=dict(filename=args.input.name, sha256=input_sha, expected_hash_matches=input_sha == EXPECTED_SHA256,
            original_not_modified=True, sheets=[dict(name=s.title, rows=s.max_row, columns=s.max_column) for s in wf]),
        environment=dict(python=platform.python_version(), numpy=np.__version__, openpyxl=openpyxl.__version__),
        counts=dict(questions=len(questions), doi_publications=len(paper_doi), raters=4,
            eligibility_cells=elig_all.size, eligibility_counts=dict(Counter(elig_all.ravel())),
            common_eligible_questions=len(q), common_eligible_dois=len(order_sorted),
            common_question_types=dict(Counter(types)), gate_cells_total=gates_all.size,
            gate_states_present=sum(v is not None for v in gates_all.ravel()),
            gate_states_blank=sum(v is None for v in gates_all.ravel()), common_gate_states=x.size,
            paired_clusters=sum(np.sum(dois == d) == 2 for d in order_sorted),
            single_question_clusters=[dict(doi=d, question_ids=[row["question_id"] for row in q if row["doi"] == d])
                                      for d in order_sorted if np.sum(dois == d) == 1],
            mixed_eligibility_questions=[row["question_id"] for i, row in enumerate(questions) if len(set(elig_all[i])) > 1]),
        agreement=agreement_results, gate_agreement=gate_agreement, disagreement=disagreement,
        bootstrap=bootstrap_result,
        type_only_baseline=dict(matches=int(np.sum(baseline == y)), total=y.size,
            proportion=float(np.mean(baseline == y)), rule="Stronger -> Not supported; Limited -> Supported",
            interpretation="Post hoc descriptive match to rule-derived records; not decision accuracy or independent validation"),
        full_vector_hamming=dict(Counter(str(row["hamming"]) for row in pair_vectors)),
        pair_vector_diagnostics={stratum: dict(
            pairs=len(rs), identical_vectors=sum(row["hamming"] == 0 for row in rs),
            matching_outcomes=sum(row["outcome_match"] for row in rs),
            matching_outcomes_with_different_vectors=sum(row["outcome_match"] and row["hamming"] > 0 for row in rs),
            different_outcomes=sum(not row["outcome_match"] for row in rs))
            for stratum in ["Overall", "Limited", "Stronger"]
            for rs in [[row for row in pair_vectors if stratum == "Overall" or row["question_type"] == stratum]]},
        question_diagnostics=q_diagnostics, ablation=ablation,
        leave_one_doi_out_summary=lodo_summary,
        formula_audit=dict(total=len(formula_audit), formulas=sum(row["is_formula"] for row in formula_audit),
            formula_rule_matches=sum(row["formula_matches_rule"] for row in formula_audit),
            cache_matches=sum(row["cache_matches"] for row in formula_audit),
            common_eligible_cache_matches=sum(row["cache_matches"] and row["common_eligible"] for row in formula_audit),
            mismatches=[row for row in formula_audit if not row["formula_matches_rule"] or not row["cache_matches"]],
            interpretation="Python independently reconstructs outcomes; no Excel engine recalculation was performed"),
        normalization=dict(changed_cells=normalizations,
            interpretation="Compiled labels already normalized; this does not verify upstream changes to unavailable original forms"),
        redundant_gate_mismatches=redundant_mismatches,
        rationale_prefix_mismatches=prefix_mismatches,
        reference_audit=dict(checked_gates=len(reference_rows), rule_counts=dict(reference_rule_counts),
            mismatches=reference_mismatches, outcomes=dict(Counter(ref_outcomes)),
            excluded_mixed_eligibility=sum(not row["common_eligible"] for row in questions),
            all_source_checked_values=dict(Counter(wf["07_PANEL_DECISIONS"].cell(r, 13).value for r in range(2, len(questions) + 2))),
            all_panel_1_values=dict(Counter(wf["07_PANEL_DECISIONS"].cell(r, 14).value for r in range(2, len(questions) + 2))),
            all_panel_2_values=dict(Counter(wf["07_PANEL_DECISIONS"].cell(r, 15).value for r in range(2, len(questions) + 2))),
            interpretation="Derivative analyst comparison, not independent ground truth or a fifth rater"),
        provenance=dict(original_input_files=[dict(name=name, present_next_to_supplied_workbook=(args.input.parent/name).is_file()) for name in originals],
            residual_ai_labels=ai_labels,
            interpretation="Workbook attributes A-D to experts but cannot independently authenticate identities, rating process, source access, or record conversion"),
        source_identity=dict(one_to_one_paper_id_doi=True,
            independence="Distinct DOIs are publication identifiers, not verified independent dataset or sampling-campaign families",
            external_register_overlap="Not assessed; underlying development and international registers not supplied"),
        analytical_scope="Exploratory descriptive secondary analysis of archived compiled records; ratings and questions retained unchanged; post hoc diagnostics do not validate decisions"
    )
    assert input_sha == hashlib.sha256(args.input.read_bytes()).hexdigest(), "Input changed during analysis"
    (args.output / "results.json").write_text(json.dumps(clean(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tables = {"questions": questions, "ratings": rating_rows, "rater_question_outcomes": observations,
        "agreement": [dict(endpoint=k, **v) for k, v in agreement_results.items()],
        "gate_agreement": gate_agreement, "question_gate_disagreement": qgate_records,
        "question_diagnostics": q_diagnostics, "rater_pair_vectors": pair_vectors,
        "bootstrap_draws": boot_records_sorted, "leave_one_gate_out": ablation,
        "leave_one_gate_out_changes": ablation_changes, "leave_one_doi_out": lodo,
        "formula_audit": formula_audit, "reference_audit": reference_rows,
        "rationale_prefix_mismatches": prefix_mismatches}
    for name, records in tables.items():
        write_csv(args.output / f"{name}.csv", records)
    archived_codebook = [dict(zip([c.value for c in wf["12_CODEBOOK"][1]], row))
                         for row in list(wf["12_CODEBOOK"].values)[1:]]
    (args.output / "archived_codebook.json").write_text(json.dumps(archived_codebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "input_sha256": input_sha, "counts": result["counts"],
        "agreement": {k: {f: v[f] for f in ["n", "unanimous", "matching_pairs", "possible_pairs", "conger", "fleiss"]} for k, v in agreement_results.items()},
        "bootstrap": bootstrap_result["results"],
        "disagreement": disagreement, "type_only_baseline": result["type_only_baseline"],
        "ablation": [{k: v[k] for k in ["removed_gate", "stratum", "changed_records", "changed_questions", "transitions", "conger", "unanimous"]} for v in ablation],
        "leave_one_doi_out": lodo_summary,
        "formula_audit": result["formula_audit"], "reference_audit": result["reference_audit"],
        "rationale_prefix_mismatches": prefix_mismatches}
    (args.output / "audit_summary.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

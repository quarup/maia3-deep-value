"""Summarise evals/results/truncation.jsonl: for each way of choosing moves and each threshold, the mean gap
to the reference tree (points of value) and the model evaluations used.

    python evals/truncation_report.py
"""
import json
import statistics as st

rows = [json.loads(l) for l in open("evals/results/truncation.jsonl")]
print(f"{len(rows)} board states; reference: every legal move, eps 0.002, {st.mean(r['ref_nodes'] for r in rows):.0f} "
      f"evaluations on average; zero-hop is {st.mean(abs(r['zero_hop'] - r['ref']) * 100 for r in rows):.2f} points from it\n")
print("| Moves followed | eps | Evaluations | Mean gap to reference (points) | Mean signed gap | Worst gap |")
print("| --- | --- | --- | --- | --- | --- |")
names = list(dict.fromkeys(k.split("|")[0] for k in rows[0]["variants"]))
for n in names:
    for k in [k for k in rows[0]["variants"] if k.split("|")[0] == n]:
        d = [(r["variants"][k]["value"] - r["ref"]) * 100 for r in rows]
        nodes = st.mean(r["variants"][k]["nodes"] for r in rows)
        print(f"| {n} | {k.split('|')[1]} | {nodes:.0f} | {st.mean(map(abs, d)):.2f} | {st.mean(d):+.2f} | {max(map(abs, d)):.1f} |")

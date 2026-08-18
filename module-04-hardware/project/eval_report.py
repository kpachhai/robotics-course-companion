"""Turn a trial log into the numbers and the markdown you will publish.

Reads the JSONL written by run_eval.py and prints:

  - the headline rate with its 95% interval, which is the honest headline
  - how many trials were voided, because a hidden void is a hidden result
  - the failure histogram, which is the part that tells you what to fix
  - a per-condition breakdown, which is where an even-looking rate falls apart
  - a markdown block you can paste straight into the writeup

Standard library only. Run it as often as you like; it never mutates the log.

Run:
    python eval_report.py results.jsonl
    python eval_report.py results.jsonl --group cell
"""

import argparse
import json
import math
from collections import Counter, defaultdict

Z95 = 1.959963985


def wilson(successes, trials, z=Z95):
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return max(0.0, centre - half), min(1.0, centre + half)


def load(path):
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log")
    parser.add_argument("--group", help="key inside 'condition' to break down by")
    args = parser.parse_args()

    records = load(args.log)
    scored = [r for r in records if r["outcome"] in ("success", "failure")]
    voids = [r for r in records if r["outcome"] == "void"]
    successes = sum(1 for r in scored if r["outcome"] == "success")
    trials = len(scored)

    if trials == 0:
        raise SystemExit("no scored trials in that log")

    low, high = wilson(successes, trials)
    rate = successes / trials

    print(f"scored trials   {trials}")
    print(f"successes       {successes}")
    print(f"rate            {rate:.1%}   95% CI [{low:.1%}, {high:.1%}]")
    if voids:
        print(f"voided          {len(voids)}  <-- these must appear in the writeup")
        for record in voids:
            print(f"                trial {record['trial']}: "
                  f"{record.get('void_reason', 'no reason given')}")

    durations = [r["seconds"] for r in scored if "seconds" in r]
    if durations:
        print(f"median trial    {sorted(durations)[len(durations) // 2]:.0f} s")

    failures = Counter(r.get("failure_mode", "untagged")
                       for r in scored if r["outcome"] == "failure")
    if failures:
        print("\nfailure modes")
        for mode, count in failures.most_common():
            share = count / max(len(scored) - successes, 1)
            print(f"  {count:3d}  {share:5.0%} of failures  {mode}")
        top_mode, top_count = failures.most_common(1)[0]
        ceiling = (successes + top_count) / trials
        print(f"\nFixing '{top_mode}' alone would move the rate to at most "
              f"{ceiling:.0%}. That is where the next episodes go.")

    if args.group:
        print(f"\nby {args.group}")
        buckets = defaultdict(lambda: [0, 0])
        for record in scored:
            key = record.get("condition", {}).get(args.group, "?")
            buckets[str(key)][1] += 1
            if record["outcome"] == "success":
                buckets[str(key)][0] += 1
        for key in sorted(buckets):
            got, total = buckets[key]
            print(f"  {key:>12}  {got}/{total}  {got / total:5.0%}")
        print("  A rate that is even here is a policy. A rate that is 90% in "
              "one bucket and 20% in another is two policies wearing a coat.")

    print("\n--- paste into the writeup -------------------------------------")
    print(f"**{rate:.0%} success ({successes}/{trials} trials, "
          f"95% CI {low:.0%}-{high:.0%})**")
    if voids:
        print(f"\n{len(voids)} trial(s) voided and excluded; reasons listed below.")
    if failures:
        print("\n| failure mode | count |")
        print("|---|---|")
        for mode, count in failures.most_common():
            print(f"| {mode} | {count} |")


if __name__ == "__main__":
    main()

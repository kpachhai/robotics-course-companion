"""What a success rate over N trials actually entitles you to say (Lesson 4.17).

Twenty trials is the smallest evaluation anyone will take seriously, and it is
still a very small number. This prints the interval around your headline figure
so that the number you publish carries its own error bars.

Standard library only, no hardware, no dataset. It is arithmetic about counting.

Run:
    python eval_stats.py rate 14 20
    python eval_stats.py plan --rate 0.7 --width 0.20
    python eval_stats.py compare 14 20 17 20

Interval method is the Wilson score interval, which is the right default for
small n and near-certain rates. The textbook 'p +/- 1.96 * sqrt(p(1-p)/n)'
interval is wrong here in a way that matters: at 20 out of 20 it reports a
width of zero, telling you that you have proven perfection with twenty tries.
"""

import argparse
import math

Z95 = 1.959963985


def wilson(successes, trials, z=Z95):
    """Wilson score interval for a binomial rate. Returns (low, high)."""
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return max(0.0, centre - half), min(1.0, centre + half)


def trials_for_width(rate, target_width, z=Z95, cap=5000):
    """Smallest n whose interval is no wider than target_width at this rate."""
    for trials in range(5, cap + 1):
        low, high = wilson(round(rate * trials), trials, z)
        if high - low <= target_width:
            return trials
    return None


def trials_to_clear(rate, floor, z=Z95, cap=5000):
    """Smallest n whose interval lower bound clears `floor` at this rate."""
    for trials in range(5, cap + 1):
        low, _ = wilson(round(rate * trials), trials, z)
        if low > floor:
            return trials
    return None


def bar(low, high, width=44):
    """A one-line 0-100% ruler with the interval drawn on it."""
    cells = ["."] * width
    start, end = int(low * width), min(int(high * width), width - 1)
    for i in range(start, end + 1):
        cells[i] = "="
    return "0% |" + "".join(cells) + "| 100%"


def cmd_rate(args):
    low, high = wilson(args.successes, args.trials)
    point = args.successes / args.trials
    print(f"{args.successes}/{args.trials} = {point:.1%}")
    print(f"95% interval  [{low:.1%}, {high:.1%}]   width {100 * (high - low):.0f} points")
    print(bar(low, high))
    print()
    print(f"Publish it as: \"{point:.0%} ({args.successes}/{args.trials}, "
          f"95% CI {low:.0%}-{high:.0%})\".")
    if low < 0.5 <= point:
        print("Note: the lower bound is below 50%, so this run does not "
              "establish that the policy works more often than not.")


def cmd_plan(args):
    for_width = trials_for_width(args.rate, args.width)
    to_clear = trials_to_clear(args.rate, args.floor)
    print(f"assuming the true rate is about {args.rate:.0%}")
    print(f"  trials for an interval {args.width:.0%} wide : {for_width}")
    print(f"  trials for the lower bound to clear {args.floor:.0%} : {to_clear}")
    print()
    print("Both numbers are why a 20-trial evaluation is a claim about the "
          "order of magnitude of your success rate, not about its value.")


def cmd_compare(args):
    a_low, a_high = wilson(args.a_successes, args.a_trials)
    b_low, b_high = wilson(args.b_successes, args.b_trials)
    print(f"A  {args.a_successes}/{args.a_trials} = "
          f"{args.a_successes / args.a_trials:.1%}  [{a_low:.1%}, {a_high:.1%}]")
    print(bar(a_low, a_high))
    print(f"B  {args.b_successes}/{args.b_trials} = "
          f"{args.b_successes / args.b_trials:.1%}  [{b_low:.1%}, {b_high:.1%}]")
    print(bar(b_low, b_high))
    print()
    if a_high < b_low or b_high < a_low:
        print("The intervals do not overlap: this is a real difference.")
    else:
        print("The intervals overlap. You have not shown that these two "
              "policies differ. Do not write 'B is better' in the post.")
    print("Overlap is a conservative test. Non-overlap proves a difference; "
          "overlap does not prove sameness.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    rate = sub.add_parser("rate", help="interval around one measured rate")
    rate.add_argument("successes", type=int)
    rate.add_argument("trials", type=int)
    rate.set_defaults(func=cmd_rate)

    plan = sub.add_parser("plan", help="how many trials would you need")
    plan.add_argument("--rate", type=float, default=0.7)
    plan.add_argument("--width", type=float, default=0.20)
    plan.add_argument("--floor", type=float, default=0.5)
    plan.set_defaults(func=cmd_plan)

    compare = sub.add_parser("compare", help="two policies, side by side")
    compare.add_argument("a_successes", type=int)
    compare.add_argument("a_trials", type=int)
    compare.add_argument("b_successes", type=int)
    compare.add_argument("b_trials", type=int)
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

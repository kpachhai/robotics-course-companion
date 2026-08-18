"""Lesson 4.1 - price an SO-101 build from the published bill of materials.

Every unit price below is transcribed from the SO-ARM100 repository's own BOM
table, read on 2026-08-09. The script's job is to let you re-derive the totals
rather than trust them: it asserts that its own arithmetic reproduces the two
published figures ($229.88 for the pair, $121.94 for a follower on its own).

If an assertion fires, the prices in this file have drifted from the source and
need re-checking against the repository. That is the intended failure mode.

Runs anywhere. No hardware, no dependencies beyond the standard library.

Run:  python bom_cost.py
      python bom_cost.py --follower-only
      python bom_cost.py --with-extras
      python bom_cost.py --with-extras --print-service
"""
import argparse
from dataclasses import dataclass

SOURCE = "SO-ARM100 repository BOM, read 2026-08-09"

# Published totals, used as self-checks rather than as output.
PUBLISHED_BOTH_ARMS = 229.88
PUBLISHED_FOLLOWER = 121.94


@dataclass(frozen=True)
class Line:
    part: str
    qty: int
    unit: float

    @property
    def total(self) -> float:
        return self.qty * self.unit


# The two arms, exactly as published. Twelve servos in three gear ratios:
# the follower takes six C001, the leader takes 1x C001 + 2x C044 + 3x C046.
BOTH_ARMS = [
    Line("STS3215 servo 7.4V, 345:1 (C001)", 7, 13.89),
    Line("STS3215 servo 7.4V, 191:1 (C044)", 2, 13.89),
    Line("STS3215 servo 7.4V, 147:1 (C046)", 3, 13.89),
    Line("Motor control board (Waveshare)", 2, 10.60),
    Line("USB-C cable, 2 pack", 1, 7.00),
    Line("Power supply", 2, 10.00),
    Line("Table clamps, 4 pack", 1, 9.00),
    Line("Screwdriver set, Phillips #0 and #1", 1, 6.00),
]

# One follower on its own. The clamp line is two clamps rather than four; that
# is the only way the published $121.94 reconciles, and it is an inference from
# the total rather than a separately published price.
FOLLOWER_ONLY = [
    Line("STS3215 servo 7.4V, 345:1 (C001)", 6, 13.89),
    Line("Motor control board (Waveshare)", 1, 10.60),
    Line("USB-C cable, 2 pack", 1, 7.00),
    Line("Power supply", 1, 10.00),
    Line("Table clamps, 2", 1, 5.00),
    Line("Screwdriver set, Phillips #0 and #1", 1, 6.00),
]

# NOT published anywhere. These are placeholders you should overwrite with real
# quotes from your own suppliers before believing the number they produce.
EXTRAS = [
    Line("PLA+ filament, 1 kg (estimate)", 1, 25.00),
    Line("USB webcam (estimate)", 2, 30.00),
    Line("Spare STS3215, 345:1 (estimate)", 1, 13.89),
]
PRINT_SERVICE = Line("Print service for both arms (wide estimate)", 1, 120.00)


def render(title: str, lines: list[Line]) -> float:
    width = max(len(line.part) for line in lines)
    print(f"\n{title}")
    print("-" * (width + 22))
    for line in lines:
        print(f"{line.part:<{width}}  {line.qty:>3} x {line.unit:>6.2f} = {line.total:>7.2f}")
    total = sum(line.total for line in lines)
    print("-" * (width + 22))
    print(f"{'total':<{width}}  {'':>3}   {'':>6}   {total:>7.2f}")
    return total


def check_published() -> None:
    """Fail loudly if the transcribed prices stop reproducing the published totals."""
    both = round(sum(line.total for line in BOTH_ARMS), 2)
    follower = round(sum(line.total for line in FOLLOWER_ONLY), 2)
    assert both == PUBLISHED_BOTH_ARMS, (
        f"two-arm total is {both}, published figure is {PUBLISHED_BOTH_ARMS}. "
        "Re-check the unit prices against the repository BOM."
    )
    assert follower == PUBLISHED_FOLLOWER, (
        f"follower-only total is {follower}, published figure is {PUBLISHED_FOLLOWER}. "
        "Re-check the unit prices against the repository BOM."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--follower-only", action="store_true",
                        help="price one follower arm instead of the leader/follower pair")
    parser.add_argument("--with-extras", action="store_true",
                        help="add filament, two webcams and a spare motor (all estimates)")
    parser.add_argument("--print-service", action="store_true",
                        help="add a print service instead of assuming you own a printer")
    args = parser.parse_args()

    check_published()

    if args.follower_only:
        parts_total = render("Follower arm only, self-sourced", FOLLOWER_ONLY)
        published = PUBLISHED_FOLLOWER
    else:
        parts_total = render("Follower + leader, self-sourced", BOTH_ARMS)
        published = PUBLISHED_BOTH_ARMS

    print(f"\nmatches the published total of {published:.2f}   ({SOURCE})")

    running = parts_total
    if args.with_extras or args.print_service:
        extras = list(EXTRAS) if args.with_extras else []
        if args.print_service:
            extras.append(PRINT_SERVICE)
        running = parts_total + render("Everything the BOM leaves out (estimates)", extras)
        print("\nThe lines above are estimates, not published prices. Replace them with")
        print("real quotes before treating the number as a budget.")

    print(f"\nall in: {running:.2f}")
    print("Kit and marketplace prices move constantly. Check the vendor list in the")
    print("SO-ARM100 repository for your region before committing to a route.")


if __name__ == "__main__":
    main()

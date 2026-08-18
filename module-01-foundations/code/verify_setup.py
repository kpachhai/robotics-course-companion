"""Verify the course environment. Run:  python verify_setup.py"""
import sys
from pathlib import Path

def check(name, fn):
    try:
        v = fn()
        print(f"  ✅ {name}{f' {v}' if v else ''}")
        return True
    except Exception as e:  # noqa: BLE001 - report anything
        print(f"  ❌ {name}: {e}")
        return False

def main():
    print(f"Python {sys.version.split()[0]}")
    ok = True
    ok &= check("numpy", lambda: __import__("numpy").__version__)
    ok &= check("scipy", lambda: __import__("scipy").__version__)
    ok &= check("matplotlib", lambda: __import__("matplotlib").__version__)
    if not ok:
        print("\nFix the ❌ lines (pip install -r requirements.txt) and re-run.")
        sys.exit(1)

    # Tiny end-to-end test: math + a saved plot.
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R

    q = R.from_euler("z", 90, degrees=True)
    assert np.allclose(q.apply([1, 0, 0]), [0, 1, 0], atol=1e-12)

    t = np.linspace(0, 2 * np.pi, 200)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(np.cos(t), np.sin(t), lw=2)
    ax.set_aspect("equal"); ax.set_title("setup check: a circle")
    # Beside this script, not in the current directory: the README tells you to
    # run it from the repo root, and a bare filename would drop an untracked
    # PNG there. `module-*/code/*.png` is already gitignored.
    out = Path(__file__).with_name("setup_check.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  ✅ end-to-end (rotation math + plot) - wrote {out.name}")
    print("\nAll good. Start with module-00-orientation/01-what-is-a-robot.md")

if __name__ == "__main__":
    main()

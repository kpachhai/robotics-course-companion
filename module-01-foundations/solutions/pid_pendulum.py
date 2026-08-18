"""Solution - Lesson 1.17: the PID.update implementation (rest identical to starter)."""


class PID:
    def __init__(self, kp, ki, kd, dt=0.001, i_clamp=2.0):
        self.kp, self.ki, self.kd, self.dt, self.i_clamp = kp, ki, kd, dt, i_clamp
        self.integ, self.prev_err = 0.0, None

    def reset(self):
        self.integ, self.prev_err = 0.0, None

    def update(self, err):
        # P
        p = self.kp * err
        # I with anti-windup clamp
        self.integ += err * self.dt
        self.integ = max(-self.i_clamp, min(self.i_clamp, self.integ))
        i = self.ki * self.integ
        # D on error (setpoint is constant here; on real hardware, filter this
        # and/or differentiate the measurement instead; lesson, Check q2)
        d = 0.0 if self.prev_err is None else self.kd * (err - self.prev_err) / self.dt
        self.prev_err = err
        return p + i + d


if __name__ == "__main__":
    # Quick behavioral check: P-only droops below target; PID converges to it.
    import importlib.util, pathlib, sys
    spec = importlib.util.spec_from_file_location(
        "starter", pathlib.Path(__file__).resolve().parents[0].parent / "code" / "pid_pendulum.py")
    starter = importlib.util.module_from_spec(spec); sys.modules["starter"] = starter
    spec.loader.exec_module(starter)
    starter.PID = PID
    import numpy as np
    _, th_p = starter.simulate(PID(8.0, 0.0, 0.0))
    _, th_pid = starter.simulate(PID(8.0, 6.0, 1.2))
    droop = np.degrees(starter.TARGET - np.mean(th_p[-500:]))
    resid = abs(np.degrees(starter.TARGET - np.mean(th_pid[-500:])))
    assert droop > 3.0, "P-only should droop visibly under gravity"
    assert resid < 0.5, "PID should erase the droop"
    print(f"✅ solution PID: P-only droop {droop:.1f}°, PID residual {resid:.2f}°.")

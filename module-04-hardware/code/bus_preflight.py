"""Lesson 4.3 - look at the bus before you change anything on it.

What this does:
  * lists the serial ports the machine can currently see, so you can tell a
    board that enumerated from one that did not
  * prints the SO-101 ID plan and the order the setup tool walks it
  * prints the exact lerobot-setup-motors commands for your ports, so you paste
    rather than type
  * prints the LED triage table, because most 'timeout' errors are physical

What this deliberately does NOT do: talk to the motors. Nothing here opens a
port, pings a device or writes a register. Discovery and ID assignment belong to
LeRobot's own tools, which are the tested path.

HONESTY NOTE: the author has no SO-101 on the machine where this was written,
so this script has never been run against a physical arm. The command strings
are transcribed from the LeRobot v0.6.1 documentation; the port listing uses
pyserial, which is a standard dependency of the Feetech stack. Read it before
you trust it, and treat its output as a prompt rather than an authority.

Run:  python bus_preflight.py
      python bus_preflight.py --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1
"""
import argparse
import platform

# LeRobot's hardcoded SO-101 mapping, read from the library source 2026-08-09.
JOINTS = [
    ("shoulder_pan", 1),
    ("shoulder_lift", 2),
    ("elbow_flex", 3),
    ("wrist_flex", 4),
    ("wrist_roll", 5),
    ("gripper", 6),
]

# Bus defaults, read from the Feetech driver source 2026-08-09.
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_PROTOCOL = 0
DEFAULT_TIMEOUT_MS = 1000
ENCODER_COUNTS_PER_TURN = 4096

LED_TRIAGE = [
    ("all steady red, gripper through to base",
     "wiring is fine; the problem is in software or in the port you chose"),
    ("one or more dark, or the chain stops part way",
     "wiring: reseat the 3-pin cables, check the board's power lead, click each connector home"),
    ("blinking",
     "error state: usually overload, or the wrong supply voltage"),
]


def list_ports() -> list[tuple[str, str]]:
    """Serial ports the OS can see right now, or an empty list if pyserial is absent."""
    try:
        from serial.tools import list_ports as pyserial_ports
    except ImportError:
        return []
    return [(port.device, port.description or "no description")
            for port in pyserial_ports.comports()]


def print_ports() -> None:
    print("Serial ports visible now")
    print("-" * 64)
    ports = list_ports()
    if not ports:
        print("  pyserial is not installed here, or no ports are present.")
        print("  Use LeRobot's own tool instead, which is the documented path:")
        print("      lerobot-find-port")
        print("  It asks you to unplug a device and reports which port disappeared.")
    else:
        for device, description in ports:
            print(f"  {device:<28} {description}")
        print()
        print("  A board that is powered but missing from this list did not enumerate.")
        print("  Check the USB cable and, on a Waveshare board, that both jumpers are")
        print("  on the B channel.")
    if platform.system() == "Linux":
        print()
        print("  On Linux you may need permission before anything can open the port:")
        print("      sudo chmod 666 /dev/ttyACM0")


def print_id_plan() -> None:
    print("\nID plan (LeRobot hardcodes this; it is not yours to choose)")
    print("-" * 64)
    for name, motor_id in JOINTS:
        units = "0 to 100" if name == "gripper" else "degrees"
        print(f"  ID {motor_id}   {name:<16} reported as {units}")
    order = " -> ".join(f"{name} ({motor_id})" for name, motor_id in reversed(JOINTS))
    print(f"\n  setup order: {order}")
    print("  One motor connected at a time. Every motor ships as ID 1, so two on the")
    print("  bus at once is an address collision with no error message.")

    print(f"\n  bus defaults: {DEFAULT_BAUDRATE:,} baud, protocol {DEFAULT_PROTOCOL}, "
          f"{DEFAULT_TIMEOUT_MS} ms timeout")
    print(f"  encoder: {ENCODER_COUNTS_PER_TURN} counts per turn, absolute")


def print_commands(follower_port: str | None, leader_port: str | None) -> None:
    follower = follower_port or "<FOLLOWER_PORT>"
    leader = leader_port or "<LEADER_PORT>"
    print("\nCommands to run (LeRobot v0.6.1)")
    print("-" * 64)
    print(f"  lerobot-setup-motors --robot.type=so101_follower --robot.port={follower}")
    print(f"  lerobot-setup-motors --teleop.type=so101_leader  --teleop.port={leader}")
    if not (follower_port and leader_port):
        print("\n  Pass --follower-port and --leader-port to have these filled in.")
        print("  Find the ports with: lerobot-find-port")


def print_triage() -> None:
    print("\nBefore you read a log, read the LEDs")
    print("-" * 64)
    for symptom, meaning in LED_TRIAGE:
        print(f"  {symptom}")
        print(f"      -> {meaning}")
    print("\n  Most 'timeout' errors are physical, not code.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--follower-port", help="serial port of the follower's control board")
    parser.add_argument("--leader-port", help="serial port of the leader's control board")
    args = parser.parse_args()

    print_ports()
    print_id_plan()
    print_commands(args.follower_port, args.leader_port)
    print_triage()
    print("\nNothing above touched the motors. Run the commands yourself when the")
    print("LEDs look right and one motor is connected on its own.")


if __name__ == "__main__":
    main()


    
  
"""SCARA neural-network sphere tracking demo for Webots.

Place shared_nn_large.pkl beside this controller script.
Load pickle files only from your own trusted training run.
"""

import math
import pickle
from pathlib import Path

import numpy as np
from controller import Supervisor

ARM_DEF = "ARM"
TARGET_DEF = "TARGET"
TCP_DEF = "TCP"
#MODEL_FILE = Path(__file__).resolve().parent / "shared_nn_medium.pkl"
MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "shared_nn_medium.pkl"

# Match the joint ranges used for training the shared network.
# q1, q2 in radians; q4 in metres.
SCARA_LIMITS = ((-0.73, 0.73), (0.0, 0.83), (-0.17, 0.0))
COMMAND_PERIOD = 0.05
PRINT_PERIOD = 0.5


def relative_xyz(node, reference):
    pose = node.getPose(reference)
    return np.array((pose[3], pose[7], pose[11]), dtype=float)


def predict_joints(model, target_xyz):
    x, y, z = target_xyz
    radius = math.hypot(x, y)
    features = np.array([[0.0, 1.0, x, y, z, radius]], dtype=float)

    raw = np.asarray(model.predict(features)[0], dtype=float)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"Invalid neural network prediction: {raw}")

    clipped = np.clip(raw, -1.0, 1.0)
    lower = np.array([pair[0] for pair in SCARA_LIMITS])
    upper = np.array([pair[1] for pair in SCARA_LIMITS])
    joints = lower + ((clipped + 1.0) / 2.0) * (upper - lower)
    return joints, raw, bool(np.any(raw != clipped))


def main():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            "Copy shared_nn_large.pkl into this controller folder."
        )

    with MODEL_FILE.open("rb") as handle:
        model = pickle.load(handle)

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    arm = robot.getFromDef(ARM_DEF)
    target = robot.getFromDef(TARGET_DEF)
    tcp = robot.getFromDef(TCP_DEF)

    if arm is None or target is None:
        raise RuntimeError("Expected DEF ARM and DEF TARGET in the Webots world.")
    if tcp is None:
        print("WARNING: DEF TCP not found. Tracking error will not be measured.")

    motor_names = ("base_arm_motor", "arm_motor", "shaft_linear_motor")
    motors = [robot.getDevice(name) for name in motor_names]
    for name, motor in zip(motor_names, motors):
        if motor is None:
            raise RuntimeError(f"Missing SCARA motor: {name}")

    # These are conservative movement speeds; q4 uses m/s.
    motors[0].setVelocity(0.7)
    motors[1].setVelocity(0.7)
    motors[2].setVelocity(0.05)

    shaft_rotation = robot.getDevice("shaft_rotation_motor")
    if shaft_rotation is not None:
        shaft_rotation.setPosition(0.0)

    last_command = -float("inf")
    last_print = -float("inf")
    latest_joints = None
    latest_raw = None
    latest_clipped = False

    print("SCARA NN sphere tracking started (shared large model).")
    while robot.step(timestep) != -1:
        now = robot.getTime()
        target_xyz = relative_xyz(target, arm)

        if now - last_command >= COMMAND_PERIOD:
            latest_joints, latest_raw, latest_clipped = predict_joints(model, target_xyz)
            for motor, joint in zip(motors, latest_joints):
                motor.setPosition(float(joint))
            last_command = now

        if now - last_print >= PRINT_PERIOD and latest_joints is not None:
            print("\n--- SCARA NN sphere tracking ---")
            print(f"Target ARM xyz (m): {np.round(target_xyz, 4)}")
            print(f"Raw normalized prediction: {np.round(latest_raw, 4)}")
            print(f"Commanded q1 (rad): {latest_joints[0]:.4f}")
            print(f"Commanded q2 (rad): {latest_joints[1]:.4f}")
            print(f"Commanded q4 (m):   {latest_joints[2]:.4f}")
            if latest_clipped:
                print("WARNING: NN prediction exceeded [-1,1]; clipped to joint limits.")
            if tcp is not None:
                actual_xyz = relative_xyz(tcp, arm)
                error_mm = float(np.linalg.norm(actual_xyz - target_xyz) * 1000.0)
                print(f"Actual TCP ARM xyz (m): {np.round(actual_xyz, 4)}")
                print(f"Live tracking error: {error_mm:.2f} mm")
            print("Note: NN clipping alone does not determine target reachability.")
            last_print = now


if __name__ == "__main__":
    main()


"""Re-run historical IK tuning experiments on the original train/validation CSVs.

Place this script alongside the 12 CSV files (ned/scara train/val small/medium/large),
or edit DATA_DIR. Requires numpy and scikit-learn.

Writes tuning_results.csv; blank values in failed runs, with errors in status.
All TCP errors are validation errors (not independent-test Task 4.A errors).
"""
from pathlib import Path
import csv
import math
import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor

DATA_DIR = Path(__file__).resolve().parent[1] / "datasets" / "nedAndSCARA"  # Change to your datasets folder if needed
OUTPUT_FILE = Path(__file__).resolve().parent[1] / "datasets"/ "tuning_results.csv"
SIZES = ("small", "medium", "large")
# Set to ("large",) to rerun only the originally documented large-dataset trials.

# (experiment, architecture, activation, learning rate, normalize XYZ,
#  add radius, clip predictions, original stopping settings)
CONFIGS = [
    (1, (64, 64), "relu", .001, False, False, False, True),
    (2, (64, 64), "relu", .001, False, False, False, False),
    (3, (64, 64), "relu", .001, True,  False, False, False),
    (4, (64, 64), "relu", .001, True,  False, True,  False),
    (5, (128, 128), "relu", .001, True, False, True, False),
    (6, (128, 128), "relu", .001, True, True, True, False),
    (7, (128, 128), "relu", .001, False, True, True, False),
    (8, (128, 128), "relu", .001, False, False, True, False),
    (9, (128, 128), "relu", .001, False, True, True, False),
    (10, (256, 256), "relu", .001, False, True, True, False),
    (11, (256, 256), "relu", .0005, False, True, True, False),
    (12, (256, 256), "tanh", .001, False, True, True, False),
    (14, (128, 128, 128), "relu", .001, False, True, True, False),
    (15, (256, 256, 256), "relu", .001, False, True, True, False),
    (16, (384, 384), "relu", .001, False, True, True, False),
    ("FINAL", (256, 256), "relu", .001, False, True, True, False),
]

# Physical joint limits: first two revolute, third Ned revolute / SCARA prismatic.
LIMITS = {
    "ned": ((-2.8, 2.8), (-0.8, 0.8), (0.0, 1.3)),
    "scara": ((-0.73, 0.73), (0.0, 0.83), (-0.17, 0.0)),
}


def load_csv(robot, split, size):
    """Read generated CSVs with x,y,z,q1,q2,q3/q4,q1_norm,q2_norm,q3/q4_norm."""
    filename = DATA_DIR / f"{robot}_{split}_{size}.csv"
    if not filename.exists():
        filename = DATA_DIR / robot / f"{robot}_{split}_{size}.csv"
    if not filename.exists():
        raise FileNotFoundError(f"Cannot find {robot}_{split}_{size}.csv under {DATA_DIR}")
    with filename.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    third = "q3_norm" if robot == "ned" else "q4_norm"
    xyz = np.array([[float(row[k]) for k in ("x", "y", "z")] for row in rows])
    joints = np.array([[float(row[k]) for k in ("q1_norm", "q2_norm", third)] for row in rows])
    flags = np.tile([1., 0.] if robot == "ned" else [0., 1.], (len(rows), 1))
    return flags, xyz, joints


def prepare(flags, xyz, add_radius, bounds=None):
    xyz_input = xyz.copy()
    if bounds is not None:
        low, high = bounds
        span = np.where(high > low, high - low, 1.0)
        xyz_input = 2 * (xyz_input - low) / span - 1
    features = np.column_stack((flags, xyz_input))
    if add_radius:
        # Radius is calculated from ORIGINAL XYZ even if XYZ inputs are normalized.
        radius = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        features = np.column_stack((features, radius))
    return features


def denormalize(predicted, robot):
    limits = np.asarray(LIMITS[robot], dtype=float)
    return limits[:, 0] + (predicted + 1) / 2 * (limits[:, 1] - limits[:, 0])


def fk_ned(q):
    q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]
    H, A, B, U, V = .1065 + .0650, .221, .012, .0318503164, .22999994
    wx = A + U * np.cos(q3) - V * np.sin(q3)
    wy = B + U * np.sin(q3) + V * np.cos(q3)
    r = wy * np.cos(q2) + wx * np.sin(q2)
    z = H - wy * np.sin(q2) + wx * np.cos(q2)
    return np.column_stack((-r * np.sin(q1), r * np.cos(q1), z))


def fk_scara(q):
    q1, q2, q4 = q[:, 0], q[:, 1], q[:, 2]
    x1, y1, z1 = .310, .005, -.004
    x2, y2 = .2545, -.005
    x2r = x2 * np.cos(q2) - y2 * np.sin(q2)
    y2r = x2 * np.sin(q2) + y2 * np.cos(q2)
    xa, ya = x1 + x2r, y1 + y2r
    x = .060 + xa * np.cos(q1) - ya * np.sin(q1)
    y = ya * np.cos(q1) + xa * np.sin(q1)
    z = .220 + z1 + q4 - .034
    return np.column_stack((x, y, z))


def run_experiment(config, size, cached_data):
    number, layers, activation, lr, normalize, radius, clip, old_stopping = config
    data = cached_data[size]
    train = data["train"]
    val = data["val"]
    train_xyz = np.vstack([train[robot][1] for robot in ("ned", "scara")])
    bounds = (train_xyz.min(axis=0), train_xyz.max(axis=0)) if normalize else None

    X_train = np.vstack([prepare(*train[robot][:2], radius, bounds) for robot in ("ned", "scara")])
    Y_train = np.vstack([train[robot][2] for robot in ("ned", "scara")])

    # Deterministic combined shuffle, matching the intended historical training flow.
    order = np.random.RandomState(42).permutation(len(X_train))
    X_train, Y_train = X_train[order], Y_train[order]

    kwargs = dict(hidden_layer_sizes=layers, activation=activation,
                  solver="adam", learning_rate_init=lr,
                  max_iter=1000 if old_stopping else 2000,
                  early_stopping=False, random_state=42)
    if not old_stopping:
        kwargs.update(tol=1e-7, n_iter_no_change=30)
    model = MLPRegressor(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(X_train, Y_train)

    result = dict(experiment=number, size=size, architecture=str(layers),
                  activation=activation, learning_rate=lr, normalize_xyz=normalize,
                  radius=radius, clipping=clip, iterations=model.n_iter_,
                  training_loss=model.loss_, status="OK")

    for robot in ("ned", "scara"):
        flags, xyz, targets = val[robot]
        X_val = prepare(flags, xyz, radius, bounds)
        predicted = model.predict(X_val)
        result[f"{robot}_joint_mae_normalized"] = float(np.mean(np.abs(predicted - targets)))
        result[f"{robot}_invalid_predictions"] = int(np.sum(np.any((predicted < -1) | (predicted > 1), axis=1)))
        if clip:
            predicted = np.clip(predicted, -1, 1)
        joints = denormalize(predicted, robot)
        positions = fk_ned(joints) if robot == "ned" else fk_scara(joints)
        errors_mm = np.linalg.norm(positions - xyz, axis=1) * 1000
        result[f"{robot}_mean_mm"] = float(np.mean(errors_mm))
        result[f"{robot}_std_mm"] = float(np.std(errors_mm))
        result[f"{robot}_max_mm"] = float(np.max(errors_mm))
    return result


def main():
    cached = {}
    for size in SIZES:
        cached[size] = {
            split: {robot: load_csv(robot, split, size) for robot in ("ned", "scara")}
            for split in ("train", "val")
        }

    columns = ["experiment", "size", "architecture", "activation", "learning_rate",
               "normalize_xyz", "radius", "clipping", "iterations", "training_loss",
               "ned_mean_mm", "ned_std_mm", "ned_max_mm", "scara_mean_mm",
               "scara_std_mm", "scara_max_mm", "ned_joint_mae_normalized",
               "scara_joint_mae_normalized", "ned_invalid_predictions",
               "scara_invalid_predictions", "status"]
    # One model fit for each unique config + size. Experiments 7 and 9 are
    # identical, as are 10 and FINAL. Experiments 3/4 differ only at evaluation.
    cache = {}
    with OUTPUT_FILE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for config in CONFIGS:
            for size in SIZES:
                number, layers, act, lr, normalize, radius, clip, old = config
                # Reuse identical full experiments; clipping variants still retrain.
                key = (layers, act, lr, normalize, radius, clip, old, size)
                try:
                    if key not in cache:
                        cache[key] = run_experiment(config, size, cached)
                    result = dict(cache[key])
                    result["experiment"] = number
                    print(f"{str(number):>5} {size:>6}: Ned {result['ned_mean_mm']:8.3f} mm | "
                          f"SCARA {result['scara_mean_mm']:8.3f} mm", flush=True)
                except Exception as exc:
                    result = dict(experiment=number, size=size, status=f"ERROR: {exc}")
                    print(f"{number} {size}: {result['status']}", flush=True)
                writer.writerow(result)
                handle.flush()
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

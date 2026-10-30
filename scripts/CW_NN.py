from pathlib import Path
import numpy as np
import sys, os

from CW_fk_ned import fk_ned
from CW_fk_scara import fk_scara

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


#Helper Denormalization Functions
def denormalize(value, minimum, maximum):
    return minimum + ((value + 1) / 2) * (maximum - minimum)


#DATASET PREPROCESSING
#Load in ned and scara datasets
#Modify them to include flags: is_scara, is_ned
#OPTIONAL: Normalize the TCP x,y,z values; do this after training on unnormalized TCP
#values to see if the improve the results or not
#Combine the two to form one dataset:
# (is_ned, is_scara, x, y, z ; joint1_norm, joint2_norm, joint3_norm)
#Shuffle the dataset to randomize the order

DATASET_DIR = "../datasets"

#Load the dataset and shuffle the samples
def load_dataset(split, size=None, shuffle=True):

    if split not in ("train", "val", "test"):
        raise ValueError("Split must be 'train', 'val', or 'test'.")

    if split != "test" and size not in ("small", "medium", "large"):
        raise ValueError("Size must be 'small', 'medium', or 'large'.")

    filename = (
        "{robot}_test.csv"
        if split == "test"
        else "{robot}_" + split + "_" + size + ".csv"
    )

    def load_robot(robot, is_ned, is_scara):
        path = f"{DATASET_DIR}/{robot}/{filename.format(robot=robot)}"

        data = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            ndmin=2
        )

        if data.shape[1] != 9:
            raise ValueError(
                f"Expected 9 CSV columns in {path}, got {data.shape[1]}"
            )

        flags = np.tile([is_ned, is_scara], (len(data), 1))

        return np.column_stack((
            flags,
            data[:, 0:3],
            data[:, 6:9]
        ))

    ned = load_robot("ned", 1, 0)
    scara = load_robot("scara", 0, 1)

    combined = np.vstack((ned, scara))

    if shuffle:
        rng = np.random.RandomState(42)
        return combined[rng.permutation(len(combined))]

    return combined

#Separate samples into Input(X) and Output(Y) sets
#Also add derived horizontal XY radius to input set
def prepare_data(dataset, normalize_tcp=False, tcp_bounds=None):
    X = dataset[:, :5].copy()
    Y = dataset[:, 5:8].copy()

    if normalize_tcp:
        tcp = X[:, 2:5]
        if tcp_bounds is None:
            tcp_bounds = (tcp.min(axis=0), tcp.max(axis=0))
        tcp_min, tcp_max = tcp_bounds
        X[:, 2:5] = 2 * (tcp - tcp_min) / (tcp_max - tcp_min) - 1

    #Calculate radius as additional feature using raw TCP coordinates.
    x = dataset[:, 2]
    y = dataset[:, 3]

    radius = np.sqrt(x**2 + y**2)
    X = np.column_stack((X, radius))
    return X, Y, tcp_bounds



#Training the model with current hyperparameters
def train_network(X_train, Y_train):
    model = MLPRegressor(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        #alpha=0.001,
        max_iter=2000,
        tol=1e-7,
        n_iter_no_change=30,
        early_stopping=False,
        random_state=42
    )

    model.fit(X_train, Y_train)

    return model

#Evaluation
def evaluate_network(model, X_val, Y_val):
    """
    - MAE (Mean Absolute Error): The average absolute difference between predicted and expected normalized joint values.
    - RMSE (Root Mean Squared Error): Like MAE but penalizes larger errors more heavily.
    - Per-joint MAE: Helps identify which of the three outputs the network struggles with most.
    """
    predictions = model.predict(X_val)

    mae = mean_absolute_error(Y_val, predictions)
    rmse = np.sqrt(mean_squared_error(Y_val, predictions))

    print("\nValidation Results:")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    print("\nPer-joint MAE:")
    joint_mae = np.mean(np.abs(Y_val - predictions), axis=0)

    for i, error in enumerate(joint_mae, start=1):
        print(f"Joint {i}: {error:.6f}")

    return predictions

#Like evaluate_network but it evaluates per robot
#This is just to know which one needs to be tended to
def evaluate_per_robot(model, X_val, Y_val):
    predictions = model.predict(X_val)

    robots = {
        "Ned": X_val[:, 0] == 1,
        "SCARA": X_val[:, 1] == 1
    }

    for name, mask in robots.items():
        actual = Y_val[mask]
        predicted = predictions[mask]

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        joint_mae = np.mean(
            np.abs(actual - predicted),
            axis=0
        )

        print(f"\n{name} Validation Results:")
        print(f"Samples: {len(actual)}")
        print(f"MAE:  {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")

        print("Per-joint MAE:")
        print(f"Joint 1: {joint_mae[0]:.6f}")
        print(f"Joint 2: {joint_mae[1]:.6f}")
        print(f"Joint 3: {joint_mae[2]:.6f}")

#Evaluate the difference in TCP position between actual and predicted
def evaluate_tcp_error(model, X_val, raw_tcp, clip_predictions=False):
    """
    1. Denormalize the predicted joint values
    2. Apply FK to the predicted joint values using CW_fk_*robotname*.py
    3. Subtract to get error
    """

    predictions = model.predict(X_val)

    if clip_predictions:
        """
        Clipping ensures the predicted joints stay within their permitted ranges but it can make
        a configuration physically valid without making it accurate.
        It could improve some TCP predictions and worsen others.
        However in our findings, the difference is negligible.
        """
        predictions = np.clip(predictions, -1.0, 1.0)

    errors = {
        "Ned": [],
        "SCARA": []
    }

    #for inputs, predicted_joints in zip(X_val, predictions):
        #is_ned, is_scara, x, y, z = inputs

    for inputs, predicted_joints, target_tcp in zip(X_val, predictions, raw_tcp):
        is_ned, is_scara = inputs[:2]

        if is_ned == 1:
            q1 = denormalize(predicted_joints[0], -2.8, 2.8)
            q2 = denormalize(predicted_joints[1], -0.8, 0.8)
            q3 = denormalize(predicted_joints[2], 0.0, 1.3)

            predicted_tcp = fk_ned(q1, q2, q3)
            robot = "Ned"

        else:
            q1 = denormalize(predicted_joints[0], -0.73, 0.73)
            q2 = denormalize(predicted_joints[1], 0.0, 0.83)
            q4 = denormalize(predicted_joints[2], -0.17, 0.0)

            predicted_tcp = fk_scara(q1, q2, q4)
            robot = "SCARA"

        #target_tcp = np.array([x, y, z])
        predicted_tcp = np.array(predicted_tcp)

        error = np.linalg.norm(predicted_tcp - target_tcp)

        errors[robot].append(error * 1000)

    for robot, robot_errors in errors.items():
        robot_errors = np.array(robot_errors)

        print(f"\n{robot} TCP Positioning Error:")
        print(f"Mean:   {np.mean(robot_errors):.3f} mm")
        print(f"Median: {np.median(robot_errors):.3f} mm")
        print(f"Max:    {np.max(robot_errors):.3f} mm")

#Finds the top 5 worst samples based on TCP error by sorting
def evaluate_worst_samples(model, X_val, Y_val, raw_tcp, top_n=5):
    predictions = model.predict(X_val)

    results = {
        "Ned": [],
        "SCARA": []
    }

    for i, (inputs, predicted, expected, target_tcp) in enumerate(
        zip(X_val, predictions, Y_val, raw_tcp)
    ):
        is_ned = inputs[0] == 1

        if is_ned:
            robot = "Ned"

            joints = [
                denormalize(predicted[0], -2.8, 2.8),
                denormalize(predicted[1], -0.8, 0.8),
                denormalize(predicted[2], 0.0, 1.3)
            ]

            predicted_tcp = fk_ned(*joints)

        else:
            robot = "SCARA"

            joints = [
                denormalize(predicted[0], -0.73, 0.73),
                denormalize(predicted[1], 0.0, 0.83),
                denormalize(predicted[2], -0.17, 0.0)
            ]

            predicted_tcp = fk_scara(*joints)

        error_mm = np.linalg.norm(
            np.array(predicted_tcp) - target_tcp
        ) * 1000

        results[robot].append({
            "index": i,
            "error_mm": error_mm,
            "target_tcp": target_tcp,
            "predicted_joints": predicted,
            "expected_joints": expected,
            "invalid": np.any(
                (predicted < -1) | (predicted > 1)
            )
        })

    for robot, samples in results.items():
        worst = sorted(
            samples,
            key=lambda sample: sample["error_mm"],
            reverse=True
        )[:top_n]

        print(f"\n{robot} Worst {top_n} TCP Predictions:")

        for sample in worst:
            print(f"\nSample index: {sample['index']}")
            print(f"TCP error: {sample['error_mm']:.3f} mm")
            print(f"Target TCP: {sample['target_tcp']}")
            print(f"Expected joints: {sample['expected_joints']}")
            print(f"Predicted joints: {sample['predicted_joints']}")
            print(f"Joint limit violation: {sample['invalid']}")


#Joint-limit validation function
def evaluate_joint_limits(model, X_val):
    """
    What does this check?
    Since our training targets are normalized to [-1, 1], any predicted value outside that range
    corresponds to a joint value outside its permitted limits.
    For example:
    Normalized prediction	Meaning
    0.50	                Valid
    -0.85	                Valid
    1.08	                Above maximum joint limit
    -1.12	                Below minimum joint limit
    The function reports how many validation samples contain at least one invalid joint,
    which joints are responsible, and the minimum and maximum predicted values.
    """
    predictions = model.predict(X_val)

    robots = {
        "Ned": X_val[:, 0] == 1,
        "SCARA": X_val[:, 1] == 1
    }

    for name, mask in robots.items():
        robot_predictions = predictions[mask]

        #Normalized joint limits are [-1, 1]

        violations = (
            (robot_predictions < -1) |
            (robot_predictions > 1)
        )

        #Count samples where at least one joint is invalid
        invalid_samples = np.any(violations, axis=1)

        total = len(robot_predictions)
        invalid_count = np.sum(invalid_samples)

        print(f"\n{name} Joint Limit Check:")
        print(f"Total samples: {total}")
        print(f"Invalid samples: {invalid_count}")
        print(f"Invalid percentage: {invalid_count / total * 100:.2f}%")

        print("Violations per joint:")
        for i in range(3):
            print(f"Joint {i + 1}: {np.sum(violations[:, i])}")

        print("Prediction ranges:")
        for i in range(3):
            minimum = np.min(robot_predictions[:, i])
            maximum = np.max(robot_predictions[:, i])

            print(
                f"Joint {i + 1}: "
                f"Min = {minimum:.4f}, "
                f"Max = {maximum:.4f}"
            )


#Region analysis for ned by genAI
#This is what told us that the upper region struggles
#and that adding radius as 6th input helped reduce that
def evaluate_ned_joint1_regions(model, X_val, Y_val, raw_tcp):
    predictions = model.predict(X_val)

    # Select Ned samples only.
    ned_mask = X_val[:, 0] == 1

    ned_expected = Y_val[ned_mask]
    ned_predicted = predictions[ned_mask]
    ned_tcp = raw_tcp[ned_mask]

    # Five regions across the normalized Joint 1 range.
    regions = [
        (-1.0, -0.6),
        (-0.6, -0.2),
        (-0.2, 0.2),
        (0.2, 0.6),
        (0.6, 1.0)
    ]

    print("\nNed Joint 1 Region Analysis:")

    for lower, upper in regions:
        mask = (
            (ned_expected[:, 0] >= lower) &
            (ned_expected[:, 0] <= upper)
        )

        expected = ned_expected[mask]
        predicted = ned_predicted[mask]
        targets = ned_tcp[mask]

        if len(expected) == 0:
            continue

        joint1_mae = np.mean(
            np.abs(expected[:, 0] - predicted[:, 0])
        )

        tcp_errors = []

        for joints, target in zip(predicted, targets):
            q1 = denormalize(joints[0], -2.8, 2.8)
            q2 = denormalize(joints[1], -0.8, 0.8)
            q3 = denormalize(joints[2], 0.0, 1.3)

            predicted_tcp = np.array(fk_ned(q1, q2, q3))

            error = np.linalg.norm(
                predicted_tcp - target
            ) * 1000

            tcp_errors.append(error)

        print(f"\nJoint 1 range: [{lower}, {upper}]")
        print(f"Samples: {len(expected)}")
        print(f"Joint 1 MAE: {joint1_mae:.6f}")
        print(f"Mean TCP error: {np.mean(tcp_errors):.3f} mm")
        print(f"Max TCP error: {np.max(tcp_errors):.3f} mm")





if __name__ == "__main__":

    for size in ("small", "medium", "large"):
    #sys.exit(0)
    #for size in ("large",):
        print(f"\n{'=' * 20} {size.upper()} DATASET {'=' * 20}")

        #Load the corresponding training and validation datasets
        train = load_dataset("train", size)
        val = load_dataset("val", size)

        #Prepare inputs and normalized joint targets
        normalize_tcp = False  #False = raw TCP, True = normalized TCP

        #Prepare training data and calculate TCP bounds
        X_train, Y_train, tcp_bounds = prepare_data(
            train,
            normalize_tcp=normalize_tcp
        )

        #Prepare validation data using training bounds
        X_val, Y_val, _ = prepare_data(
            val,
            normalize_tcp=normalize_tcp,
            tcp_bounds=tcp_bounds
        )

        #Train a new shared network for this dataset size
        model = train_network(X_train, Y_train)

        print(f"\nTraining complete!")
        print(f"Iterations: {model.n_iter_}")
        print(f"Final training loss: {model.loss_:.6f}")

        #Evaluate on the corresponding validation dataset
        evaluate_network(model, X_val, Y_val)
        evaluate_per_robot(model, X_val, Y_val)

        raw_tcp = val[:, 2:5].copy()
        print("\n--- WITHOUT CLIPPING ---")
        evaluate_tcp_error(
            model, X_val, raw_tcp,
            clip_predictions=False
        )
        
        print("\n--- WITH CLIPPING ---")
        evaluate_tcp_error(
            model, X_val, raw_tcp,
            clip_predictions=True
        )
        
        #See if the predicted joint values go outside the their permitted ranges and count how many
        evaluate_joint_limits(model, X_val)

        #Extra for figuring out which robot is struggling and which joints
        evaluate_worst_samples(model, X_val, Y_val, raw_tcp)
        evaluate_ned_joint1_regions(model, X_val, Y_val, raw_tcp)

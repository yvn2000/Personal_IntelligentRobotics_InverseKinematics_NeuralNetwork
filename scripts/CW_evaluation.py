from pathlib import Path
import numpy as np
import pickle

from CW_fk_ned import fk_ned
from CW_fk_scara import fk_scara


from CW_NN import load_dataset
from CW_NN import prepare_data
from CW_NN import train_network


#Helper Function
def denormalize(value, minimum, maximum):
    return minimum + ((value + 1) / 2) * (maximum - minimum)



def evaluate_task4a(model, test, clip_predictions=True):
    """
    Task 4.A: Evaluate a trained model on both robots' test sets.

    For each robot, calculate:
    - TCP position error: Euclidean distance (metres)
    - Revolute joint error: signed wrapped angular difference (radians)
    - Prismatic joint error: squared displacement difference (m^2)

    Report the mean and standard deviation of every error type.
    """

    X_test, Y_test, _ = prepare_data(test, normalize_tcp=False)

    predictions = model.predict(X_test)

    #Count invalid predictions before clipping
    ned_invalid = 0
    scara_invalid = 0

    ned_total = 0
    scara_total = 0

    for i in range(len(predictions)):

        #Identify which robot this prediction belongs to
        if X_test[i][0] == 1:
            robot = "Ned"
            ned_total += 1
        else:
            robot = "SCARA"
            scara_total += 1

        #Check whether any of the three joints is outside [-1, 1]
        invalid = False
        for joint in predictions[i]:
            if joint < -1.0 or joint > 1.0:
                invalid = True
                break

        #Count the invalid prediction for the correct robot
        if invalid:
            if robot == "Ned":
                ned_invalid += 1
            else:
                scara_invalid += 1

    #Calculate the percentages
    ned_percentage = (ned_invalid / ned_total) * 100
    scara_percentage = (scara_invalid / scara_total) * 100
    print(f"Ned invalid predictions: {ned_invalid}/{ned_total} ({ned_percentage:.2f}%)")
    print(f"SCARA invalid predictions: {scara_invalid}/{scara_total} ({scara_percentage:.2f}%)")


    #Clipping safety measure: restrict predictions to [-1, 1]
    if clip_predictions:
        predictions = np.clip(predictions, -1.0, 1.0)

    #Robot-specific joint limits, types, and FK functions
    robots = {
        "Ned": {
            "mask": X_test[:, 0] == 1,
            "limits": [
                (-2.8, 2.8),
                (-0.8, 0.8),
                (0.0, 1.3)
            ],
            "types": ["revolute", "revolute", "revolute"],
            "names": ["q1", "q2", "q3"],
            "fk": fk_ned
        },
        "SCARA": {
            "mask": X_test[:, 1] == 1,
            "limits": [
                (-0.73, 0.73),
                (0.0, 0.83),
                (-0.17, 0.0)
            ],
            "types": ["revolute", "revolute", "prismatic"],
            "names": ["q1", "q2", "q4"],
            "fk": fk_scara
        }
    }

    results = {}

    
    #Evaluate Ned and SCARA separately
    for robot_name in robots:

        config = robots[robot_name]
        mask = config["mask"]

        #Select the test samples belonging to this robot
        actual_norm = Y_test[mask]
        predicted_norm = predictions[mask]

        target_tcp = test[mask, 2:5]

        #Denormalize actual and predicted joint values
        actual_joints = []
        predicted_joints = []
        for sample in range(len(actual_norm)):

            actual_sample = []
            predicted_sample = []

            for joint in range(3):

                minimum = config["limits"][joint][0]
                maximum = config["limits"][joint][1]

                actual_value = denormalize(
                    actual_norm[sample][joint],
                    minimum,
                    maximum
                )

                predicted_value = denormalize(
                    predicted_norm[sample][joint],
                    minimum,
                    maximum
                )

                actual_sample.append(actual_value)
                predicted_sample.append(predicted_value)

            actual_joints.append(actual_sample)
            predicted_joints.append(predicted_sample)

        #Apply FK to each
        predicted_tcp = []
        for joints in predicted_joints:
            q1 = joints[0]
            q2 = joints[1]
            q3 = joints[2]

            x, y, z = config["fk"](q1, q2, q3)
            predicted_tcp.append([x, y, z])

        #Calculate Euclidean TCP position errors
        position_errors = []
        for sample in range(len(predicted_tcp)):

            dx = predicted_tcp[sample][0] - target_tcp[sample][0]
            dy = predicted_tcp[sample][1] - target_tcp[sample][1]
            dz = predicted_tcp[sample][2] - target_tcp[sample][2]

            error = np.sqrt(dx**2 + dy**2 + dz**2)
            position_errors.append(error)

        #Calculate the joint errors
        joint_errors = [[], [], []]
        for sample in range(len(actual_joints)):

            for joint in range(3):

                actual = actual_joints[sample][joint]
                predicted = predicted_joints[sample][joint]

                difference = actual - predicted
                if config["types"][joint] == "revolute":

                    #Signed wrapped angular error, acc to CW specs
                    error = np.arctan2(
                        np.sin(difference),
                        np.cos(difference)
                    )

                else:
                    #Squared displacement error, acc to CW specs
                    error = difference**2

                joint_errors[joint].append(error)

        #Calculate mean and standard deviation
        position_mean = np.mean(position_errors)
        position_std = np.std(position_errors)

        joint_means = []
        joint_stds = []

        for joint in range(3):
            mean = np.mean(joint_errors[joint])
            std = np.std(joint_errors[joint])
            joint_means.append(mean)
            joint_stds.append(std)

        results[robot_name] = {
            "position_mean": position_mean,
            "position_std": position_std,
            "joint_means": np.array(joint_means),
            "joint_stds": np.array(joint_stds)
        }

        print(f"\n--- {robot_name} ({len(target_tcp)} test samples) ---")
        print("\nTCP Position Error:")
        print(f"Mean: {position_mean * 1000:.3f} mm")
        print(f"Std:  {position_std * 1000:.3f} mm")

        print("\nJoint Errors:")
        for joint in range(3):
            name = config["names"][joint]

            if config["types"][joint] == "revolute":
                unit = "rad"
            else:
                unit = "m²"

            print(
                f"{name}: "
                f"Mean = {joint_means[joint]:.8f} {unit}, "
                f"Std = {joint_stds[joint]:.8f} {unit}"
            )

    return results



def export_task4b(model, size, test):

    #Robot-specific joint limits
    robots = {
        "Ned": {
            "flag": 0,
            "limits": [(-2.8, 2.8), (-0.8, 0.8), (0.0, 1.3)]
        },
        "SCARA": {
            "flag": 1,
            "limits": [(-0.73, 0.73), (0.0, 0.83), (-0.17, 0.0)]
        }
    }

    for robot_name in robots:

        config = robots[robot_name]

        #Select samples belonging to this robot
        robot_test = test[test[:, config["flag"]] == 1]

        #Use 100 samples per robot.
        robot_test = robot_test[:100]

        X_test, Y_test, _ = prepare_data(
            robot_test,
            normalize_tcp=False
        )

        predictions = model.predict(X_test)

        #Keep predictions within the valid joint limits
        predictions = np.clip(predictions, -1.0, 1.0)

        #Convert normalized predictions to physical joint values
        predicted_joints = []

        for sample in predictions:

            joints = []

            for joint in range(3):

                minimum = config["limits"][joint][0]
                maximum = config["limits"][joint][1]

                value = denormalize(
                    sample[joint],
                    minimum,
                    maximum
                )

                joints.append(value)

            predicted_joints.append(joints)

        #Combine target XYZ and predicted joints
        target_tcp = robot_test[:, 2:5]

        output = np.column_stack((
            target_tcp,
            predicted_joints
        ))

        filename = f"{robot_name.lower()}_task4b_{size}.csv"

        if robot_name == "Ned":
            header = "x_true,y_true,z_true,q1_pred,q2_pred,q3_pred"
        else:
            header = "x_true,y_true,z_true,q1_pred,q2_pred,q4_pred"

        np.savetxt(
            filename,
            output,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.10f"
        )

        print(f"Saved {filename}")



if __name__ == "__main__":

    #Load the independent test sets, these are the same test samples for all three models
    test = load_dataset("test")
    test_4b = load_dataset("test", shuffle=False)

    all_results = {}

    #Train and evaluate all three models
    for size in ("small", "medium", "large"):

        print(f"\n{'=' * 20} {size.upper()} MODEL {'=' * 20}")

        #Load the training dataset for this model size
        train = load_dataset("train", size)

        #Prepare the network inputs and expected outputs
        X_train, Y_train, _ = prepare_data(
            train,
            normalize_tcp=False
        )

        #Re-train the shared Ned + SCARA neural network
        model = train_network(X_train, Y_train)

        """
        #IDK IF WE CAN USE THIS OR NOT, pickle NOT SAID IN PERMITTED LIBRARIES
        model_path = f"../models/shared_nn_{size}.pkl"
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        print(f"Saved model to {model_path}")
        """

        print("\nTraining complete!")
        print(f"Iterations: {model.n_iter_}")
        print(f"Final training loss: {model.loss_:.6f}")

        #Evaluate the trained model using the independent test sets
        all_results[size] = evaluate_task4a(
            model,
            test,
            clip_predictions=True
        )

        export_task4b(model, size, test_4b)



"""
FINAL RESULTS:


==================== SMALL MODEL ====================

Training complete!
Iterations: 496
Final training loss: 0.001243
Ned invalid predictions: 13/500 (2.60%)
SCARA invalid predictions: 23/500 (4.60%)

--- Ned (500 test samples) ---
TCP Position Error:
Mean: 7.846 mm
Std:  7.897 mm
Joint Errors:
q1: Mean = -0.02817573 rad, Std = 0.13889719 rad
q2: Mean = 0.01301809 rad, Std = 0.01254360 rad
q3: Mean = 0.00535628 rad, Std = 0.00758941 rad

--- SCARA (500 test samples) ---
TCP Position Error:
Mean: 4.551 mm
Std:  2.793 mm
Joint Errors:
q1: Mean = -0.01090443 rad, Std = 0.01864747 rad
q2: Mean = 0.02645669 rad, Std = 0.04343613 rad
q4: Mean = 0.00000068 m², Std = 0.00000550 m²

==================== MEDIUM MODEL ====================

Training complete!
Iterations: 248
Final training loss: 0.000583
Ned invalid predictions: 10/500 (2.00%)
SCARA invalid predictions: 3/500 (0.60%)

--- Ned (500 test samples) ---
TCP Position Error:
Mean: 7.576 mm
Std:  5.666 mm
Joint Errors:
q1: Mean = -0.01918530 rad, Std = 0.03844344 rad
q2: Mean = 0.01488653 rad, Std = 0.01004488 rad
q3: Mean = 0.00360348 rad, Std = 0.00484201 rad

--- SCARA (500 test samples) ---
TCP Position Error:
Mean: 4.750 mm
Std:  2.314 mm
Joint Errors:
q1: Mean = -0.01803674 rad, Std = 0.01365893 rad
q2: Mean = 0.02196036 rad, Std = 0.02737905 rad
q4: Mean = 0.00000016 m², Std = 0.00000043 m²

==================== LARGE MODEL ====================

Training complete!
Iterations: 186
Final training loss: 0.000628
Ned invalid predictions: 2/500 (0.40%)
SCARA invalid predictions: 5/500 (1.00%)

--- Ned (500 test samples) ---
TCP Position Error:
Mean: 8.380 mm
Std:  7.194 mm
Joint Errors:
q1: Mean = 0.02623717 rad, Std = 0.04913066 rad
q2: Mean = 0.01074399 rad, Std = 0.00803839 rad
q3: Mean = 0.00057779 rad, Std = 0.00508965 rad

--- SCARA (500 test samples) ---
TCP Position Error:
Mean: 2.582 mm
Std:  1.595 mm
Joint Errors:
q1: Mean = -0.00825505 rad, Std = 0.01046150 rad
q2: Mean = 0.00967037 rad, Std = 0.02232347 rad
q4: Mean = 0.00000016 m², Std = 0.00000125 m²

"""

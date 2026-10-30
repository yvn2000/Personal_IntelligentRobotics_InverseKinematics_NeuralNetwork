
from controller import Supervisor
import csv
import math
from pathlib import Path


#Create the Webots supervisor
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())


#Get Ned's motors and position sensors
motors = [
    robot.getDevice("joint_1"),
    robot.getDevice("joint_2"),
    robot.getDevice("joint_3")
]

#Enable the position sensors
sensors = []
for motor in motors:
    sensor = motor.getPositionSensor()
    sensor.enable(timestep)
    sensors.append(sensor)


#Get the ARM and TCP nodes
arm = robot.getFromDef("ARM")
tcp = robot.getFromDef("TCP")
if arm is None or tcp is None:
    raise RuntimeError("Could not find ARM or TCP in the Webots world.")


#Folder containing the prediction CSV files
#folder = Path(__file__).resolve().parent
folder = Path(__file__).resolve().parents[2] / "datasets" / "task4"

#Moves the robot acc to the joints, code from genAI
def move_robot(joints):

    for i in range(3):
        motors[i].setPosition(joints[i])

    #Wait until the motors have reached their targets
    tolerance = 0.001
    max_steps = 300
    stable_steps = 0

    for step in range(max_steps):

        if robot.step(timestep) == -1:
            raise RuntimeError("Webots simulation stopped.")

        reached = True

        for i in range(3):
            actual = sensors[i].getValue()
            target = joints[i]

            if abs(actual - target) > tolerance:
                reached = False

        if reached:
            stable_steps += 1
        else:
            stable_steps = 0

        #Require several consecutive successful checks
        if stable_steps >= 5:
            return True

    return False


def evaluate_model(size):

    filename = folder / f"ned_task4b_{size}.csv"

    print(f"\n{'=' * 20} {size.upper()} MODEL {'=' * 20}")

    #Read the 100 prediction samples
    with open(filename, newline="") as file:
        reader = csv.DictReader(file)
        samples = list(reader)

    if len(samples) != 100:
        raise ValueError(
            f"Expected 100 samples in {filename}, got {len(samples)}"
        )

    errors = []
    results = []

    for index, sample in enumerate(samples):

        target = [
            float(sample["x_true"]),
            float(sample["y_true"]),
            float(sample["z_true"])
        ]

        #Predicted joint values
        joints = [
            float(sample["q1_pred"]),
            float(sample["q2_pred"]),
            float(sample["q3_pred"])
        ]

        #Move the robot to the predicted configuration
        reached = move_robot(joints)

        #Measure TCP in ARM coordinate frame
        pose = tcp.getPose(arm)

        measured = [
            pose[3],
            pose[7],
            pose[11]
        ]

        #TCP position error.
        dx = measured[0] - target[0]
        dy = measured[1] - target[1]
        dz = measured[2] - target[2]
        error = math.sqrt(dx**2 + dy**2 + dz**2)
        errors.append(error)

        results.append([
            index + 1,
            target[0], target[1], target[2],
            measured[0], measured[1], measured[2],
            error * 1000,
            reached
        ])

        status = "OK" if reached else "TIMEOUT"

        print(
            f"{size.upper()} sample {index + 1}/100: "
            f"{error * 1000:.3f} mm [{status}]"
        )

    #Calculate mean and standard deviation
    mean = sum(errors) / len(errors)

    variance = sum(
        (error - mean)**2 for error in errors
    ) / len(errors)

    std = math.sqrt(variance)

    print(f"\n--- Ned {size.upper()} Results ---")
    print(f"Samples: {len(errors)}")
    print(f"Mean TCP error: {mean * 1000:.3f} mm")
    print(f"Std TCP error:  {std * 1000:.3f} mm")

    #Save
    output_file = folder / f"ned_task4b_results_{size}.csv"
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "sample",
            "target_x", "target_y", "target_z",
            "measured_x", "measured_y", "measured_z",
            "error_mm",
            "reached"
        ])

        writer.writerows(results)

    print(f"Saved {output_file.name}")


for size in ("small", "medium", "large"):
    evaluate_model(size)

print("\nNed Task 4.B evaluation complete.")


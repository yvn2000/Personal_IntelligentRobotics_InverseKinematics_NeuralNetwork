import random
import csv
import os

from CW_fk_scara import fk_scara



#SCARA joint limits
q1_min = -0.73
q1_max = 0.73

#q2 must be strictly positive for elbow-up
q2_min = 0.0
q2_max = 0.83

#Specification gives q4 range [-0.20, 0.0] however
#webots validation showed ground collision near the lower limit
#Need to restrict dataset to collision-free configurations
q4_min = -0.17
q4_max = 0.0



#Dataset sizes
small_train_size = 800
small_val_size = 200

medium_train_size = 4000
medium_val_size = 1000

large_train_size = 8000
large_val_size = 2000

test_size = 500

#Normalize a joint value to the range [-1, 1]
def normalize(q, q_min, q_max):
    return 2 * ((q - q_min) / (q_max - q_min)) - 1

def denormalize(q_norm, q_min, q_max):
    return ((q_norm + 1) / 2) * (q_max - q_min) + q_min

#Generate one valid elbow-up SCARA configuration
def generate_scara_sample():

    #Generat random joint config
    q1 = random.uniform(q1_min, q1_max)

    #Strictly positive for elbow-up
    q2 = random.uniform(q2_min, q2_max)

    q4 = random.uniform(q4_min, q4_max)

    #Calculate TCP position using FK
    x, y, z = fk_scara(q1, q2, q4)

    #Normalize joint values
    q1_norm = normalize(q1, q1_min, q1_max)
    q2_norm = normalize(q2, q2_min, q2_max)
    q4_norm = normalize(q4, q4_min, q4_max)

    return {
        "x": x,
        "y": y,
        "z": z,

        "q1": q1,
        "q2": q2,
        "q4": q4,

        "q1_norm": q1_norm,
        "q2_norm": q2_norm,
        "q4_norm": q4_norm
    }

#Generate the requested number of SCARA samples
def generate_samples(amount):
    return [generate_scara_sample() for _ in range(amount)]

#Save samples to a CSV file
def save_csv(filename, samples):

    fieldnames = [
        "x", "y", "z",
        "q1", "q2", "q4",
        "q1_norm", "q2_norm", "q4_norm"
    ]

    with open(filename, "w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(samples)



#makes generation reproducible
random.seed(42)

os.makedirs("../datasets/scara", exist_ok=True)

#TRAINING DATA

#Generate the largest training dataset once
large_train = generate_samples(large_train_size)

#Smaller datasets are subsets of it
medium_train = large_train[:medium_train_size]
small_train = large_train[:small_train_size]


#VALIDATION DATA

#Generate validation data separately from training data
large_val = generate_samples(large_val_size)

#Smaller validation sets are subsets of it
medium_val = large_val[:medium_val_size]
small_val = large_val[:small_val_size]



#TEST DATA

#Generated only after train/validation data so these samples are independent of both
test = generate_samples(test_size)


#Saving
save_csv("../datasets/scara/scara_train_small.csv", small_train)
save_csv("../datasets/scara/scara_val_small.csv", small_val)

save_csv("../datasets/scara/scara_train_medium.csv", medium_train)
save_csv("../datasets/scara/scara_val_medium.csv", medium_val)

save_csv("../datasets/scara/scara_train_large.csv", large_train)
save_csv("../datasets/scara/scara_val_large.csv", large_val)

save_csv("../datasets/scara/scara_test.csv", test)


print("SCARA datasets generated successfully.")

#Get counts
print()
print("Small:")
print(f"  Training:   {len(small_train)}")
print(f"  Validation: {len(small_val)}")

print()
print("Medium:")
print(f"  Training:   {len(medium_train)}")
print(f"  Validation: {len(medium_val)}")

print()
print("Large:")
print(f"  Training:   {len(large_train)}")
print(f"  Validation: {len(large_val)}")

print()
print("Independent test:")
print(f"  Test:       {len(test)}")
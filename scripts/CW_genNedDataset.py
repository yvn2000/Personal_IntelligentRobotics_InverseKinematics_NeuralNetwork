import random
import csv
import os

from CW_fk_ned import fk_ned


#joint limits
q1_min = -2.8
q1_max = 2.8

q2_min = -0.8
q2_max = 0.8

#joint3 > 0 (elbow-up configuration)
#0 is accepted to include max stretched configs
q3_min = 0.0
q3_max = 1.3


#dataset sizes
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

#Generate one random valid Ned configuration and calculate its TCP position using forward kinematics
def generate_ned_sample():

    #Generate random joint configuration
    q1 = random.uniform(q1_min, q1_max)
    q2 = random.uniform(q2_min, q2_max)
    q3 = random.uniform(q3_min, q3_max)

    #Calculate TCP position using FK
    x, y, z = fk_ned(q1, q2, q3)

    #Normalize joint values
    q1_norm = normalize(q1, q1_min, q1_max)
    q2_norm = normalize(q2, q2_min, q2_max)
    q3_norm = normalize(q3, q3_min, q3_max)

    return {
        "x": x,
        "y": y,
        "z": z,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q1_norm": q1_norm,
        "q2_norm": q2_norm,
        "q3_norm": q3_norm
    }

#Generate a list containing the requested number of samples
def generate_samples(amount):
    return [generate_ned_sample() for _ in range(amount)]

#Save samples to a CSV file
def save_csv(filename, samples):
    fieldnames = [
        "x", "y", "z",
        "q1", "q2", "q3",
        "q1_norm", "q2_norm", "q3_norm"
    ]

    with open(filename, "w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(samples)


#makes generation reproducible
random.seed(42)
os.makedirs("../datasets/ned", exist_ok=True)


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
save_csv("../datasets/ned/ned_train_small.csv", small_train)
save_csv("../datasets/ned/ned_val_small.csv", small_val)

save_csv("../datasets/ned/ned_train_medium.csv", medium_train)
save_csv("../datasets/ned/ned_val_medium.csv", medium_val)

save_csv("../datasets/ned/ned_train_large.csv", large_train)
save_csv("../datasets/ned/ned_val_large.csv", large_val)

save_csv("../datasets/ned/ned_test.csv", test)


print("Ned datasets generated successfully.")

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
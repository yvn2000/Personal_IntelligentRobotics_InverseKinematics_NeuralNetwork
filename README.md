
# Code Details & Instructions

The following descriptions will indicate the file structure of the project code along with instructions on how to execute the right files for every task.

This document is divided into 2 primary sections:
- Python Code Scripts
- Webots Robot Scripts

The order in which the files should be executed is represented by the order of the subheading in this document.

### Dependency

A library called pickle was used to save the trained models as a separate file. If this library is not installed on the machine, it must be by running "pip install pickle" or "pip install pickle5" or whichever installation process works for you.

### Preliminary Step
Note: All datasets generated are stored in the datasets folder. The generated models for each dataset size are stored in the models folder.

The preliminary code executed are the validation controllers in Webots. Their purpose is to take a set of joint values and using the TCP supervisor, prints the TCP coordinates for those joints. These were used for validation to compare the FK values for the next step.

To run this, first open the Webots world for either robot arm. Open the arm descriptor on the left and ensure supervisor is set to True. Then click on the controller part and select either the "ned_validation" and "scara_validation", depending on the robot arm, and then click play to see the results be printed.







## Python Code


This concerns the files that are not executed in Webots. They are all ran from scripts directory so make sure that that is the current working directory for the console. This includes:
- Forward Kinematics
- Dataset Generation
- Neural Network Training
- Evaluating TCP and Joint Errors



### Forward Kinematics

Within the scripts directory, there are two files named:
- CW_fk_ned.py
- CW_fk_scara.py

For their respective robot arm, they are responsible for calculating the Forward Kinematics for various different joint configurations. These FK TCP positions are then compared against the TCP positions reported by the TCP supervisor in Webots, from which the errors are obtained. The results are saved as comments within the same file.

To execute them simply type "python CW_fk_ned.py" or "python CW_fk_scara.py" in the console and the results will be printed. If "python" does not work, try "python3".


### Dataset Generation

Also in the scripts directory, there are two files named:
- CW_genNedDataset.py
- CW_genScaraDataset.py

Each file creates the small, medium, large and test datasets for each arm by calling the FK function from before on joint values generated randomly within their joint limits. The datasets generated are stored in the datasets directory found in the root directory.

Run them similarly to the fk files by running "python" or "python3" followed by the file names for each robot arm.


### Neural Network Training

The file responsible for training the Neural Network is CW_NN.py

Execute it by running it in the console just like before. The file contains the preprocessing and error reporting for all 3 datasets. Several evaluation functions were written for hyperparameter tuning on the validation sets.

Upon execution, 3 different models will begin to train on the training set and also be evaluated on the validation sets, with the results being printed.

### Evaluating TCP and Joint Errors

Within the scripts directory, there's a file called CW_evaluation.py which is responsible for evaluating the models on the test dataset. It is executed similarly as before by running it in the console. 

The NN is re-trained on the training dataset and the models are saved using pickle. After this, the models are evaluated on the test sets using the evaluation metrics for TCP Position Error Mean & Standard Deviation and Joint Errors, specified by the coursework.

The models are saved in the models directory which can be found in the root directory.

Additionally, the 100 sample datasets are also generated which will later be used in Webots for evaluating TCP Position Errors via the simulation TCP coordinates for task 4.B. These sets can be found in datasets/task4/


### Optional

There's a file in the scripts directory called "rerun_ik_tuning.py" which was used to re-run all the 15+ different hyper parameter configurations used for experimenting for all 3 datasets. The results are saved in the datasets directory.

## Webots Robots

This concerns running the files needed for simulating the Ned & SCARA robot arms in Webots. Their purpose includes:
- TCP Position Validation
- Controllers for evaluating TCP Position Error for Task 4.B
- Controllers for demonstrating the Neural Network's efficiency in live Webots environment using thr red sphere.



### TCP Position Validation

This is the same as the preliminary step, executing the validation controllers for desired joint values using the TCP supervisor. See that section above for more detail.


### Controllers for evaluating TCP Position Error for Task 4.B

Simply switch the controller to the one titled "ned_evaluation" or "scara_evaluation", based on the robot. Then hit play and you should see the robot trying out various joint poses from the 100 samples set and reporting the errors for them.

### Controllers for demonstrating the Neural Network's efficiency in live Webots environment using thr red sphere.

Simply switch the controller to the one titled "ned_nn_sphere" or "scara_nn_sphere", based on the robot. Then hit play. Once it begins to run, move the sphere anywhere you wish and the robot arm's TCP should follow it, if its in the permitted range. If its not in the permitted range, a message will be displayed in the console indicating so. The console also displays all other information.
"""Install the external Python dependencies used by the robot NN controllers."""

import subprocess
import sys


def main():
    packages = ["numpy", "scikit-learn"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

    import numpy
    import sklearn
    import pickle  # noqa: F401 - part of Python's standard library

    print(f"NumPy: {numpy.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print("pickle: included with Python")
    print("All dependencies are available.")


if __name__ == "__main__":
    main()

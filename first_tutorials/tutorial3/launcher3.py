import subprocess

# Path of the file to execute
file = "./tutorial3.py"

# Execute the file
for i in range(50):
    subprocess.run(["python3", file])

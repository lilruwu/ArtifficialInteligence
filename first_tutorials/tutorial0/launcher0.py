import subprocess

# Path of the file to execute
file = "./tutorial0.py"

# Execute the file
for i in range(1000000):
    subprocess.run(["python", file])

import subprocess

# Path of the file to execute
file = "/home/ruwu/pyTorch/ArtifficialInteligence/tutorial0.py"

# Execute the file
for i in range(1000000):
    subprocess.run(["python", file])

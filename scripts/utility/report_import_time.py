import subprocess

if __name__ == '__main__':
    with open("temp_import.prof", "w") as f:
        subprocess.run([
            "python", "-X", "importtime", "-c", "from rfstudio import visualization",
        ], check=False, stderr=f)
    subprocess.run([
        "tuna", "temp_import.prof",
    ], check=False)

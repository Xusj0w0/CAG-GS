import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)

    args = parser.parse_args()

    path = Path(args.dir)
    if path.exists():
        for py_file in path.rglob("*.py"):
            print("Formatting {}".format(str(py_file.relative_to(path))))
            subprocess.run(["python", "-m", "black", str(py_file), "--line-length", "120"])
            subprocess.run(["python", "-m", "isort", str(py_file)])
        print("Finished.")

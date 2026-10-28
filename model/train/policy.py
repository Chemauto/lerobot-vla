"""Model-specific aliases for the shared train/evaluate pipeline."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(default_model='act'):
    command = [sys.executable, str(ROOT/'model/run_experiment.py'),
               '--model', default_model, *sys.argv[1:]]
    os.chdir(ROOT)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()

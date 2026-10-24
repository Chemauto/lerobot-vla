"""Launch the official LeRobot trainer with a local config; no custom training loop."""

import argparse
from datetime import datetime
import os
import json
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def main(default_model="act"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"model/configs/{default_model}.json")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without training")
    args, overrides = parser.parse_known_args()
    if any(arg.startswith("--config_path") for arg in overrides):
        parser.error("Use --config PATH to select the official training configuration")
    config = args.config.resolve()
    if not config.is_file():
        parser.error(f"Configuration does not exist: {config}")
    model_type = json.loads(config.read_text())["policy"]["type"]
    # New preset runs get a separate folder; resume and custom configs keep their paths.
    has_output = any(arg == "--output_dir" or arg.startswith("--output_dir=") for arg in overrides)
    resuming = any(arg in ("--resume", "--resume=true", "--resume=True") for arg in overrides)
    if config == ROOT / f"model/configs/{model_type}.json" and not has_output and not resuming:
        day = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
        batch = ROOT / "outputs" / f"{model_type}{day}"
        output = batch / "train"
        number = 2
        while output.exists():
            output = batch / f"train_{number:02d}"
            number += 1
        overrides.append(f"--output_dir={output.relative_to(ROOT)}")
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train", f"--config_path={config}", *overrides]
    print(f"Project: {ROOT}\n{shlex.join(command)}", flush=True)
    if not args.dry_run:
        os.chdir(ROOT)
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()

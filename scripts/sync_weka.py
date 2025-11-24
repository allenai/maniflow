import getpass
import os

# Usage
# python scripts/sync_weka.py


def main() -> None:
    dest_adr = f"weka://prior-default/{getpass.getuser()}"

    command = f"rclone  sync --progress --copy-links\
         --exclude .idea \
         --exclude __pycache__/ \
         --exclude .DS_Store \
         --exclude .envrc \
         --exclude .git/ \
         --exclude output/ \
         --exclude */static/ \
         --exclude third_party/ \
         --exclude src/ \
         --exclude debug/ \
         --exclude debug_output/ \
         --exclude experiment_output/\
         --exclude data/ \
         --exclude *.ipynb \
         --exclude *.pth\
         --exclude collected_trajectories/ \
         --exclude experiment_output/ \
         --exclude test_debug_images/ \
         --exclude .venv/ \
         --exclude assets/ \
         ../maniflow {dest_adr}/maniflow"
    print(command)
    os.system(command)


if __name__ == "__main__":
    main()

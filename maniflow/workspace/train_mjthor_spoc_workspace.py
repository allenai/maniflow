"""
Workspace for training MjThorToSpoc dataset.
This is a thin wrapper around the RoboTwin workspace that handles the different config structure.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hydra
import pathlib
from maniflow.workspace.train_maniflow_robotwin_workspace import TrainManiFlowRoboTwinWorkspace

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config'))
)
def main(cfg):
    workspace = TrainManiFlowRoboTwinWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()


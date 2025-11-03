"""
Module: rlbench_demos_gen.py
Description:
    Utility script to generate and persist RLBench task demonstrations as HDF5 files,
    plus optionally export a sample episode to an MP4 video for quick visual inspection.

    The script:
      1) Launches an RLBench Environment with the requested action mode.
      2) Samples 'num_episode' demonstrations for a given task.
      3) Saves raw RLBench demos (pickle) for reproducibility.
      4) Converts each episode to a compact training dictionary and writes HDF5 files.
      5) Optionally renders a specific episode to a video for quick review.

    Notes:
      - The task list and observation / demo configs are read from `config_const`.
      - The helper utilities for HDF5 IO live in `demo_utils.py`.
"""

# ----------------------------
# Standard / third-party imports
# ----------------------------
import os
import pickle
from tqdm import tqdm  # (Imported but not used in this file; retained if used by your utils)

# RLBench imports
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, CloseDoor, HangFrameOnHanger, StackCups  # noqa: F401 (examples/import side effects)

# Local utilities
from utils.rlbench_demos_utils import (
    episode2dict,
    dump_episode2hdf5,
    load_all_hdf52episode,
    create_folder_if_not_exists,
)

# Media utils
import numpy as np  # noqa: F401 (kept in case downstream uses it)
import mediapy as media

# Configs & tasks (wildcard import is kept intentionally for eval(task_name) below)
from config_const import Obs_config, Demo_conifg
from rlbench.tasks import *  # noqa: F403
from typing import Union

def sampleNsave_demos(
    demo_task=ReachTarget,
    num_episode: int = 2,
    obs_cfg: Union[ObservationConfig, None] = None,
    show_demo: bool = False,
    dataset_dir: Union[str, None] = None,
    act_type: str = "joint",
) -> None:
    """
    Generate RLBench demonstrations for a single task and persist them.

    Args:
        demo_task: RLBench task class (e.g., ReachTarget). Not an instance.
        num_episode: Number of episodes/demonstrations to sample.
        obs_cfg: RLBench ObservationConfig. If None, a default config is used.
        show_demo: If True, launches a visible simulator window; otherwise headless.
        dataset_dir: Directory where per-episode HDF5 files and demos.pkl are saved.
        act_type: Action type for the arm; one of {"joint", "eepose"}.

    Raises:
        ValueError: If an unsupported `act_type` is provided.

    Side effects:
        - Launches and shuts down the RLBench Environment.
        - Writes a 'demos.pkl' containing the raw RLBench Observation lists.
        - Writes per-episode HDF5 files named 'episode_XXX.hdf5' to `dataset_dir`.
    """
    # Default observation configuration if none provided
    if obs_cfg is None:
        obs_cfg = ObservationConfig()

    print("Constructing the environment ...")

    # Select arm action mode based on requested action type
    if act_type == "joint":
        arm_action_mode = JointPosition()
    elif act_type == "eepose":
        arm_action_mode = EndEffectorPoseViaIK()
    else:
        print(f"No action type: {act_type}.")
        raise ValueError

    # Build and launch environment (Discrete gripper: open=1, close=0)
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=arm_action_mode, gripper_action_mode=Discrete()
        ),
        obs_config=obs_cfg,
        headless=not show_demo,
    )
    env.launch()

    print("Generating demos ...")
    task = env.get_task(demo_task)
    # `live_demos=True` ensures demonstrations are generated in the simulator now
    demos = task.get_demos(int(num_episode), live_demos=True)  # List[List[Observation]]

    # Save raw demos for exact reproducibility and potential environment resets
    if dataset_dir is not None:
        with open(os.path.join(dataset_dir, "demos.pkl"), "wb") as f:
            pickle.dump(demos, f)

    print("Parsing the recorded demos ...")
    for idx, episode in enumerate(demos):
        # Convert RLBench Observations to a compact training dictionary
        episode_dict = episode2dict(episode)

        # Persist the data to HDF5 (one file per episode)
        if dataset_dir is not None:
            dump_episode2hdf5(
                episode_dict=episode_dict,
                save_path=os.path.join(dataset_dir, "episode_" + str(idx).zfill(3)),
                sim=True,
            )

    print("Done!")
    env.shutdown()


def save_task_video(demo_filepath: str, savepath: str, fps: int = 30) -> None:
    """
    Render and save a recorded demo episode to a video (e.g., MP4).

    Args:
        demo_filepath: Path to a single episode HDF5 file (episode_XXX.hdf5).
        savepath: Output path for the rendered video file.
        fps: Frames per second for the output video.

    Notes:
        - This uses the 'front_rgb' stream stored in the episode HDF5.
        - The helper `load_all_hdf52episode` reconstructs the episode dict.
    """
    # Load episode dictionary containing RGB frames and state signals
    episode_dict = load_all_hdf52episode(filepath=demo_filepath)

    # Write the front camera frames to a video
    media.write_video(savepath, episode_dict.front_rgb, fps=fps)
    print(f"Saved demo {demo_filepath} to {savepath} ...")


# ----------------------------
# Script entry point
# ----------------------------
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task demos generation")
    print("Start generating multi-task demos ...")
    
    parser.add_argument("--exp-one-demo", action="store_true", help="export a demo example")
    parser.add_argument("--show-demo", action="store_true", help="display the recorded demo")
    parser.add_argument("--check-shape", action="store_true", help="display the recorded demo")
    args = parser.parse_args()

    # Iterate over tasks defined in your config (string names of RLBench task classes)
    for task_name in tqdm(Demo_conifg.task_list, desc="Generating demos", ncols=100): # Demo_conifg.task_list:
        
        print("\ntask: ", task_name)

        # Choose action type and derive dataset subfolder name
        act_type = Demo_conifg.action_type
        task_suffix = "_joint" if act_type == "joint" else "_eepose"

        # Resolve directories relative to this script location
        cur_filedir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(cur_filedir, "dataset", task_name + task_suffix)
        create_folder_if_not_exists(dataset_dir)

        # ---- Generate demonstrations and write HDF5 files ----
        sampleNsave_demos(
            demo_task=eval(task_name),  # e.g., ReachTarget, HangFrameOnHanger
            num_episode=Demo_conifg.num_episode,
            obs_cfg=Obs_config,
            show_demo=args.show_demo,
            dataset_dir=dataset_dir,
            act_type=act_type,
        )

        # ---- Optionally export a sample episode as a video ----
        if args.exp_one_demo:
            demo_dir = os.path.join(cur_filedir, "dataset", "demo_videos")
            create_folder_if_not_exists(demo_dir)
            savepath = os.path.join(demo_dir, f"{task_name}_demo.mp4")

            # this tries to render episode index 3; adjust as needed
            filepath = os.path.join(dataset_dir, "episode_" + str(3).zfill(3) + ".hdf5")
            save_task_video(filepath, savepath)

        if args.check_shape:
            # Debug helpers 
            filepath = os.path.join(dataset_dir, "episode_" + str(3).zfill(3) + ".hdf5")
            episode_dict = load_all_hdf52episode(filepath=filepath)
            print('front rgb shape: ', episode_dict.front_rgb.shape)
            print('wrist rgb shape: ', episode_dict.wrist_rgb.shape)
            print('joint pos shape: ', episode_dict.joint_pos.shape)
            print('gripper pose shape: ', episode_dict.ee_pose.shape)
            print('gripper open shape: ', episode_dict.ee_open.shape)
            print('data keys:', episode_dict.keys())
        

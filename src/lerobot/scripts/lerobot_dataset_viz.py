#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episodes 0
```

- Visualize multiple episodes:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episodes 0-15,17,19-20
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episodes 0-15,17,19-20 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episodes_0_to_15_17_19_20.rrd .
local$ rerun lerobot_pusht_episodes_0_to_15_17_19_20.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episodes 0-15,17,19-20 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

"""

import argparse
import gc
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD


def parse_episode_ranges(episodes_str: str | None) -> list[int] | None:
    """Parse episode string like '0-15,17,19-20' into a list of episode indices."""
    if episodes_str is None:
        return None
    episode_indices = []
    parts = episodes_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            episode_indices.extend(range(int(start), int(end) + 1))
        else:
            episode_indices.append(int(part))
    return sorted(set(episode_indices))


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_indices: list[int]):
        self.frame_ids = []
        # Iterate through all episodes in the filtered dataset
        for ep_idx in range(len(dataset.meta.episodes)):
            from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
            to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
            self.frame_ids.extend(range(from_idx, to_idx))

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    # Add logging to verify the dataset contains the requested episodes
    print(f"Requested episodes: {episode_indices}")
    print(f"Dataset contains {len(dataset.meta.episodes)} episodes")
    
    # Verify that all episodes are included
    if len(dataset.meta.episodes) != len(episode_indices):
        print(f"Dataset contains {len(dataset.meta.episodes)} episodes, but {len(episode_indices)} were requested")
        print("Some episodes might not be available in the dataset")

    # Display information about loaded episodes
    episode_info = []
    for i in range(len(dataset.meta.episodes)):
        from_idx = dataset.meta.episodes["dataset_from_index"][i]
        to_idx = dataset.meta.episodes["dataset_to_index"][i]
        frames = to_idx - from_idx
        episode_info.append(f"Episode {i}: {frames} frames")
    print("Loaded episodes info:")
    print("\n".join(episode_info))

    print("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_indices)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    print("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    episodes_str = "_".join(map(str, episode_indices)) if len(episode_indices) <= 5 else f"{episode_indices[0]}_to_{episode_indices[-1]}"
    rr.init(f"{repo_id}/episodes_{episodes_str}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)  # type: ignore[attr-defined]

    print("Logging to Rerun")

    # Create a mapping from dataset indices to original episode indices for proper naming
    dataset_to_original_ep = {}
    for i, ep_idx in enumerate(episode_indices):
        if i < len(dataset.meta.episodes):
            dataset_to_original_ep[i] = ep_idx
    
    # Organize frames by episode
    frames_by_episode = {}
    for ep_idx in range(len(dataset.meta.episodes)):
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
        original_ep_idx = dataset_to_original_ep.get(ep_idx, ep_idx)
        frames_by_episode[ep_idx] = {
            "original_idx": original_ep_idx,
            "frames": list(range(from_idx, to_idx)),
            "data": []
        }

    # Build a blueprint where each episode has its own container with the associated camera views.
    preferred_camera_order = ["image", "wrist_image"]
    # Preserve dataset ordering but make sure the preferred keys appear first when available.
    ordered_camera_keys = [key for key in preferred_camera_order if key in dataset.meta.camera_keys]
    ordered_camera_keys.extend(key for key in dataset.meta.camera_keys if key not in ordered_camera_keys)

    camera_view_map: dict[str, list[rr.blueprint.View]] = {key: [] for key in ordered_camera_keys}
    for episode_info in frames_by_episode.values():
        episode_root = f"episode_{episode_info['original_idx']}"
        for camera_key in ordered_camera_keys:
            camera_view_map[camera_key].append(
                rr.blueprint.views.Spatial2DView(
                    origin=f"cameras/{camera_key}/{episode_root}",
                    name=episode_root,
                )
            )

    camera_containers: list[rr.blueprint.Container] = []
    for camera_key in ordered_camera_keys:
        views = camera_view_map.get(camera_key, [])
        if not views:
            continue
        camera_containers.append(
            rr.blueprint.Vertical(
                name=camera_key,
                contents=views,
            )
        )

    if camera_containers:
        rr.send_blueprint(
            rr.blueprint.Blueprint(
                rr.blueprint.Grid(
                    name="Cameras",
                    contents=camera_containers,
                    grid_columns=len(camera_containers) if len(camera_containers) > 1 else None,
                ),
                auto_views=False,
                auto_layout=False,
            )
        )

    # Collect all frame data first, grouped by episode
    print("Collecting frame data...")
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            dataset_index = batch["index"][i].item()
            
            # Determine which episode this frame belongs to
            current_episode = None
            for ep_idx in range(len(dataset.meta.episodes)):
                from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
                to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
                if from_idx <= dataset_index < to_idx:
                    current_episode = ep_idx
                    break
            
            if current_episode is not None:
                # Save this frame's data to the corresponding episode
                frame_data = {
                    "frame_index": batch["frame_index"][i].item(),
                    "timestamp": batch["timestamp"][i].item(),
                    "dataset_index": dataset_index,
                }
                
                # Extract camera images
                frame_data["images"] = {}
                for key in dataset.meta.camera_keys:
                    frame_data["images"][key] = to_hwc_uint8_numpy(batch[key][i])
                
                # Extract other data
                frame_data["actions"] = {}
                if ACTION in batch:
                    for dim_idx, val in enumerate(batch[ACTION][i]):
                        frame_data["actions"][dim_idx] = val.item()
                
                frame_data["state"] = {}
                if OBS_STATE in batch:
                    for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                        frame_data["state"][dim_idx] = val.item()
                
                frame_data["metrics"] = {}
                if DONE in batch:
                    frame_data["metrics"]["done"] = batch[DONE][i].item()
                if REWARD in batch:
                    frame_data["metrics"]["reward"] = batch[REWARD][i].item()
                if "next.success" in batch:
                    frame_data["metrics"]["next_success"] = batch["next.success"][i].item()
                
                frames_by_episode[current_episode]["data"].append(frame_data)
    
    # Now log each episode's data separately with clean entity paths
    print("Logging episodes to Rerun...")
    for ep_idx, ep_data in frames_by_episode.items():
        original_ep_idx = ep_data["original_idx"]
        print(f"Logging episode {original_ep_idx} with {len(ep_data['data'])} frames")
        
        # Process each frame in the episode
        for frame_data in tqdm.tqdm(ep_data["data"]):
            episode_root = f"episode_{original_ep_idx}"
            
            # Set the time for this frame
            rr.set_time_sequence("frame", frame_data["frame_index"])
            
            # Log all camera images directly under the episode entity
            for key, img_data in frame_data["images"].items():
                rr.log(f"cameras/{key}/{episode_root}", rr.Image(img_data))
            
            # Log actions
            for dim_idx, val in frame_data["actions"].items():
                rr.log(f"{episode_root}/actions/{dim_idx}", rr.Scalars([float(val)]))
            
            # Log state
            for dim_idx, val in frame_data["state"].items():
                rr.log(f"{episode_root}/state/{dim_idx}", rr.Scalars([float(val)]))
            
            # Log metrics
            for metric_name, val in frame_data["metrics"].items():
                rr.log(f"{episode_root}/metrics/{metric_name}", rr.Scalars([float(val)]))

    if mode == "local" and save:
        # save .rrd locally
        assert output_dir is not None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        episodes_str = "_".join(map(str, episode_indices)) if len(episode_indices) <= 5 else f"{episode_indices[0]}_to_{episode_indices[-1]}"
        rrd_path = output_dir / f"{repo_id_str}_episodes_{episodes_str}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help=(
            "Episodes to visualize. Can be a single episode (e.g. '0'), a range (e.g. '0-15'), or a combination (e.g. '0-15,17,19-20'). "
            "If omitted, all episodes are shown."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")
    episodes_str = kwargs.pop("episodes")

    episode_indices = parse_episode_ranges(episodes_str)

    print("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=episode_indices, root=root, tolerance_s=tolerance_s)

    resolved_episode_indices = (
        episode_indices
        if episode_indices is not None
        else list(range(len(dataset.meta.episodes)))
    )

    visualize_dataset(dataset, resolved_episode_indices, **kwargs)


if __name__ == "__main__":
    main()

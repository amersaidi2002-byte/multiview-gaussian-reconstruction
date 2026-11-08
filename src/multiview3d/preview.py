"""Render a saved Gaussian scene at camera poses from a JSON manifest."""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import torch

from .renderer import Camera, quaternion_rotation, render

if TYPE_CHECKING:
    from .reconstruction import View


def save_cameras(output: Path, views: list["View"]) -> None:
    manifest = [
        {
            "name": view.name,
            "height": view.image.shape[0],
            "width": view.image.shape[1],
            "intrinsic": view.intrinsic.tolist(),
            "rotation": view.rotation.tolist(),
            "translation": view.translation.tolist(),
        }
        for view in views
    ]
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def preview(scene_path: Path, camera_path: Path, output: Path) -> None:
    with np.load(scene_path, allow_pickle=False) as arrays:
        scene = {key: torch.tensor(arrays[key], dtype=torch.float32) for key in arrays.files}
    factors = quaternion_rotation(scene["quaternion_wxyz"]) * scene["scales"][:, None, :]
    output.mkdir(parents=True, exist_ok=True)
    for record in json.loads(camera_path.read_text(encoding="utf-8")):
        camera = Camera(
            torch.tensor(record["intrinsic"], dtype=torch.float32),
            torch.tensor(record["rotation"], dtype=torch.float32),
            torch.tensor(record["translation"], dtype=torch.float32),
            record["height"], record["width"],
        )
        rgb = render(camera, scene["xyz"], factors, scene["rgb"], scene["opacity"], scene["background"])
        image = (rgb.clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy()
        Image.fromarray(image).save(output / f"{Path(record['name']).stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    preview(args.scene, args.cameras, args.output)


if __name__ == "__main__":
    main()

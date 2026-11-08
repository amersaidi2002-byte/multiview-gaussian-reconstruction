"""Fit a scene from training images and score renders at held-out camera poses."""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pycolmap
import torch

from .preview import save_cameras
from .reconstruction import View, initialize_cloud, load_views, split_indices
from .renderer import Camera, GaussianScene


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)


def _camera(view: View, origin: np.ndarray, scale: float) -> Camera:
    height, width = view.image.shape[:2]
    return Camera(
        _tensor(view.intrinsic), _tensor(view.rotation),
        _tensor((view.rotation @ origin + view.translation) / scale),
        height, width,
    )


def _quality(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mse = np.mean((prediction - target) ** 2)
    psnr = -10 * np.log10(max(float(mse), 1e-10))
    mu_pred = cv2.GaussianBlur(prediction, (11, 11), 1.5)
    mu_target = cv2.GaussianBlur(target, (11, 11), 1.5)
    var_pred = cv2.GaussianBlur(prediction**2, (11, 11), 1.5) - mu_pred**2
    var_target = cv2.GaussianBlur(target**2, (11, 11), 1.5) - mu_target**2
    covariance = cv2.GaussianBlur(prediction * target, (11, 11), 1.5) - mu_pred * mu_target
    ssim = (
        (2 * mu_pred * mu_target + 0.01**2) * (2 * covariance + 0.03**2)
        / ((mu_pred**2 + mu_target**2 + 0.01**2) * (var_pred + var_target + 0.03**2))
    )
    return {"psnr_db": float(psnr), "ssim": float(ssim[5:-5, 5:-5].mean())}


def _score(images: list[np.ndarray], targets: list[np.ndarray], names: list[str]) -> dict:
    rows = [
        {"image": name, **_quality(image, target)}
        for image, target, name in zip(images, targets, names, strict=True)
    ]
    return {
        "views": rows,
        **{key: float(np.mean([row[key] for row in rows])) for key in ("psnr_db", "ssim")},
    }


@torch.no_grad()
def _predict(scene: GaussianScene, cameras: list[Camera]) -> list[np.ndarray]:
    return [scene(camera).clamp(0, 1).numpy() for camera in cameras]


def _train(
    scene: GaussianScene, cameras: list[Camera], images: list[torch.Tensor],
    steps: int, seed: int, label: str,
) -> dict:
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam([
        {"params": [scene.centers], "lr": 0.0008},
        {"params": [scene.log_scales, scene.rotation], "lr": 0.009},
        {"params": [scene.color_logits, scene.opacity_logits, scene.background_logits], "lr": 0.025},
    ])
    trace = []
    losses = []
    start = time.perf_counter()
    for step, index in enumerate(rng.integers(len(cameras), size=steps), start=1):
        optimizer.zero_grad(set_to_none=True)
        prediction = scene(cameras[index])
        residual = prediction - images[index]
        loss = residual.abs().mean() + 0.5 * residual.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == steps:
            average = float(np.mean(losses))
            trace.append({"step": step, "mean_training_loss": average})
            print(f"{label} seed={seed} step={step}/{steps} loss={average:.4f}", flush=True)
            losses.clear()
    return {"optimization_seconds": time.perf_counter() - start, "training_trace": trace}


def _save_gallery(output: Path, columns: dict[str, list[np.ndarray]], names: list[str]) -> None:
    height, width = next(iter(columns.values()))[0].shape[:2]
    cell_width, cell_height = width * 2, height * 2
    gutter, label_height = 12, 23
    canvas = Image.new(
        "RGB",
        (gutter + len(columns) * (cell_width + gutter), gutter + len(names) * (cell_height + label_height + gutter)),
        "#101920",
    )
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(names):
        top = gutter + row * (cell_height + label_height + gutter)
        for column, (label, images) in enumerate(columns.items()):
            left = gutter + column * (cell_width + gutter)
            draw.text((left, top), f"{name}  /  {label}", fill="#f4f6f7")
            pixels = (np.clip(images[row], 0, 1) * 255 + 0.5).astype(np.uint8)
            panel = Image.fromarray(pixels).resize((cell_width, cell_height), Image.Resampling.NEAREST)
            canvas.paste(panel, (left, top + label_height))
    canvas.save(output, optimize=True)


def _export_scene(output: Path, scene: GaussianScene, origin: np.ndarray, scale: float) -> None:
    with torch.no_grad():
        np.savez_compressed(
            output,
            xyz=scene.centers.numpy() * scale + origin,
            rgb=scene.color_logits.sigmoid().numpy(),
            opacity=scene.opacity_logits.sigmoid().numpy(),
            scales=scene.log_scales.clamp(-6, -1).exp().numpy() * scale,
            quaternion_wxyz=torch.nn.functional.normalize(scene.rotation, dim=-1).numpy(),
            background=scene.background_logits.sigmoid().numpy(),
        )


def benchmark(args: argparse.Namespace) -> dict:
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    views = load_views(args.data, args.width)
    if len({view.image.shape for view in views}) != 1:
        raise ValueError("The benchmark requires a common image size across views.")
    train_ids, test_ids = split_indices(len(views))
    train = [views[index] for index in train_ids]
    test = [views[index] for index in test_ids]
    train_names = [view.name for view in train]
    test_names = [view.name for view in test]
    features = load_views(args.data, args.feature_width, include=set(train_names))
    assert [view.name for view in features] == train_names
    xyz, colors = initialize_cloud(features, args.max_points)
    origin = np.median(xyz, axis=0)
    scale = float(np.quantile(np.linalg.norm(xyz - origin, axis=1), 0.8))
    if scale <= 1e-8:
        raise ValueError("Degenerate initialization cloud.")
    centers = _tensor((xyz - origin) / scale)
    colors = _tensor(colors)
    train_cameras = [_camera(view, origin, scale) for view in train]
    test_cameras = [_camera(view, origin, scale) for view in test]
    train_rgb = [view.image.astype(np.float32) / 255 for view in train]
    test_rgb = [view.image.astype(np.float32) / 255 for view in test]
    backgrounds = [
        np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]]) for image in train_rgb
    ]
    background = _tensor(np.median(np.concatenate(backgrounds), axis=0))
    train_tensors = [_tensor(image) for image in train_rgb]
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Triangulated {len(xyz)} points using {len(train)} train views; test={test_names}", flush=True)

    nearest = [min(range(len(train)), key=lambda i: np.linalg.norm(train[i].center - view.center)) for view in test]
    nearest_images = [train_rgb[index] for index in nearest]
    initial = GaussianScene(centers, colors, background, anisotropic=False)
    initial_images = _predict(initial, test_cameras)
    result = {
        "scene": args.data.name,
        "image_size": list(test[0].image.shape[:2]),
        "feature_width": args.feature_width,
        "initial_points": len(xyz),
        "max_points": args.max_points,
        "steps_per_fit": args.steps,
        "seeds": args.seeds,
        "threads": args.threads,
        "train_images": train_names,
        "test_images": test_names,
        "environment": {
            "python": platform.python_version(), "pytorch": torch.__version__,
            "opencv": cv2.__version__, "pycolmap": pycolmap.__version__, "numpy": np.__version__,
        },
        "nearest_training_view": {
            **_score(nearest_images, test_rgb, test_names),
            "source_images": [train_names[index] for index in nearest],
        },
        "initial_splats": _score(initial_images, test_rgb, test_names),
        "trained": {},
    }
    examples = {}
    for label, anisotropic in (("isotropic", False), ("anisotropic", True)):
        runs = []
        for seed in args.seeds:
            scene = GaussianScene(centers, colors, background, anisotropic=anisotropic)
            log = _train(scene, train_cameras, train_tensors, args.steps, seed, label)
            renders = _predict(scene, test_cameras)
            runs.append({"seed": seed, **log, **_score(renders, test_rgb, test_names)})
            if seed == args.seeds[0]:
                examples[label] = renders
                _export_scene(args.output / f"{label}_scene.npz", scene, origin, scale)
        result["trained"][label] = {
            "runs": runs,
            **{
                key: {"mean": float(np.mean([run[key] for run in runs])),
                      "std": float(np.std([run[key] for run in runs]))}
                for key in ("psnr_db", "ssim", "optimization_seconds")
            },
        }
    _save_gallery(
        args.output / "held_out_views.png",
        {
            "Photo": test_rgb, "Nearest input": nearest_images, "Initial splats": initial_images,
            "Learned isotropic": examples["isotropic"], "Learned anisotropic": examples["anisotropic"],
        },
        test_names,
    )
    save_cameras(args.output / "test_cameras.json", test)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing images/ and sparse/0/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--feature-width", type=int, default=760)
    parser.add_argument("--max-points", type=int, default=900)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        args.width < 32 or args.feature_width < args.width or args.max_points < 16
        or args.steps < 1 or args.threads < 1 or min(args.seeds) < 0
        or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("Use width >= 32, feature-width >= width, max-points >= 16, positive steps/threads and distinct nonnegative seeds.")
    report = benchmark(args)
    for label in ("nearest_training_view", "initial_splats"):
        print(f"{label}: {report[label]['psnr_db']:.2f} dB")
    for label, summary in report["trained"].items():
        print(f"{label}: {summary['psnr_db']['mean']:.2f} dB across {len(args.seeds)} seeds")


if __name__ == "__main__":
    main()

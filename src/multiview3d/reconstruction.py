"""Load calibrated cameras and triangulate image features from training views."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap


@dataclass(frozen=True)
class View:
    name: str
    image: np.ndarray
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def projection(self) -> np.ndarray:
        return self.intrinsic @ np.column_stack((self.rotation, self.translation))

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.translation


def load_views(root: Path, width: int, include: set[str] | None = None) -> list[View]:
    calibration = pycolmap.Reconstruction(str(root / "sparse" / "0"))
    views = []
    maps = {}
    for image in sorted(calibration.images.values(), key=lambda item: item.name):
        if include is not None and image.name not in include:
            continue
        camera = calibration.cameras[image.camera_id]
        if camera.camera_id not in maps:
            height = round(width * camera.height / camera.width)
            intrinsic = camera.calibration_matrix().copy()
            intrinsic[0] *= width / camera.width
            intrinsic[1] *= height / camera.height
            intrinsic[:2, 2] -= 0.5  # COLMAP pixel centers start at (0.5, 0.5).
            if camera.model.name == "SIMPLE_RADIAL":
                distortion = np.array([camera.params[3], 0, 0, 0])
            elif camera.model.name in {"SIMPLE_PINHOLE", "PINHOLE"}:
                distortion = np.zeros(4)
            else:
                raise ValueError(f"Unsupported camera model {camera.model.name}")
            xy = cv2.initUndistortRectifyMap(
                intrinsic, distortion, None, intrinsic, (width, height), cv2.CV_32FC1
            )
            maps[camera.camera_id] = (intrinsic, height, xy)
        intrinsic, height, xy = maps[camera.camera_id]
        pixels = cv2.imread(str(root / "images" / image.name), cv2.IMREAD_COLOR)
        if pixels is None:
            raise FileNotFoundError(root / "images" / image.name)
        pixels = cv2.resize(pixels, (width, height), interpolation=cv2.INTER_AREA)
        pixels = cv2.remap(pixels, *xy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        pose = image.cam_from_world().matrix()
        views.append(View(image.name, rgb, intrinsic, pose[:, :3], pose[:, 3]))
    return views


def split_indices(count: int) -> tuple[list[int], list[int]]:
    test = list(range(2, count, 5))
    if not test or count < 6:
        raise ValueError("At least six registered images are required.")
    train = [index for index in range(count) if index not in test]
    return train, test


def _project(view: View, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_xyz = points @ view.rotation.T + view.translation
    homogeneous = camera_xyz @ view.intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, camera_xyz[:, 2]


def triangulate_pair(
    left: View, right: View, left_xy: np.ndarray, right_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = cv2.triangulatePoints(
        left.projection, right.projection, left_xy.T, right_xy.T
    ).T
    with np.errstate(divide="ignore", invalid="ignore"):
        xyz = homogeneous[:, :3] / homogeneous[:, 3:4]
    first, first_depth = _project(left, xyz)
    second, second_depth = _project(right, xyz)
    error = np.maximum(
        np.linalg.norm(first - left_xy, axis=1),
        np.linalg.norm(second - right_xy, axis=1),
    )
    ray_a = xyz - left.center
    ray_b = xyz - right.center
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_angle = (ray_a * ray_b).sum(1) / (
            np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
        )
    valid = (
        np.isfinite(xyz).all(1)
        & (first_depth > 0)
        & (second_depth > 0)
        & (error < 2)
        & (cos_angle < np.cos(np.deg2rad(1.5)))
    )
    return xyz, valid


def _ratio_matches(matcher, first, second) -> dict[int, int]:
    matches = {}
    if first is not None and second is not None and len(second) >= 2:
        for pair in matcher.knnMatch(first, second, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                matches[pair[0].queryIdx] = pair[0].trainIdx
    return matches


def initialize_cloud(views: list[View], max_points: int) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=2500)
    features = [
        sift.detectAndCompute(cv2.cvtColor(view.image, cv2.COLOR_RGB2GRAY), None)
        for view in views
    ]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    points, colors = [], []
    for i, (keypoints, descriptors) in enumerate(features):
        for j in range(i + 1, min(i + 4, len(views))):
            other_keypoints, other_descriptors = features[j]
            forward = _ratio_matches(matcher, descriptors, other_descriptors)
            backward = _ratio_matches(matcher, other_descriptors, descriptors)
            pairs = [(a, b) for a, b in forward.items() if backward.get(b) == a]
            if not pairs:
                continue
            xy_a = np.array([keypoints[a].pt for a, _ in pairs])
            xy_b = np.array([other_keypoints[b].pt for _, b in pairs])
            xyz, valid = triangulate_pair(views[i], views[j], xy_a, xy_b)
            pixels_a = np.rint(xy_a[valid]).astype(int)
            pixels_b = np.rint(xy_b[valid]).astype(int)
            height, width = views[i].image.shape[:2]
            pixels_a = np.clip(pixels_a, [0, 0], [width - 1, height - 1])
            height, width = views[j].image.shape[:2]
            pixels_b = np.clip(pixels_b, [0, 0], [width - 1, height - 1])
            rgb_a = views[i].image[pixels_a[:, 1], pixels_a[:, 0]].astype(float)
            rgb_b = views[j].image[pixels_b[:, 1], pixels_b[:, 0]].astype(float)
            points.append(xyz[valid])
            colors.append((rgb_a + rgb_b) / 510)
    if not points:
        raise ValueError("No matching image features could be triangulated.")
    xyz, rgb = np.concatenate(points), np.concatenate(colors)
    if len(xyz) < 16:
        raise ValueError("Too few valid triangulated points.")
    radius = np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1)
    inliers = radius <= np.quantile(radius, 0.95)
    xyz, rgb = xyz[inliers], rgb[inliers]
    cell = max(np.quantile(radius, 0.75) * 0.012, 1e-6)
    _, unique = np.unique(np.floor(xyz / cell), axis=0, return_index=True)
    if len(unique) < 16:
        raise ValueError("Too few distinct points after voxel deduplication.")
    rng = np.random.default_rng(0)
    selected = rng.choice(unique, min(max_points, len(unique)), replace=False)
    return xyz[selected], rgb[selected]

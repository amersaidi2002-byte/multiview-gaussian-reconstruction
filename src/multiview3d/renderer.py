"""Differentiable projection and visibility-aware compositing in PyTorch."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Camera:
    intrinsic: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
    height: int
    width: int


def quaternion_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = F.normalize(quaternion, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ), dim=-1
    ).reshape(-1, 3, 3)


def project(
    camera: Camera, centers: torch.Tensor, factors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz = centers @ camera.rotation.T + camera.translation
    x, y, depth = xyz.unbind(-1)
    z = depth.clamp_min(0.01)
    fx, fy = camera.intrinsic[0, 0], camera.intrinsic[1, 1]
    cx, cy = camera.intrinsic[0, 2], camera.intrinsic[1, 2]
    pixels = torch.stack((fx * x / z + cx, fy * y / z + cy), dim=-1)
    zero = torch.zeros_like(z)
    jacobian = torch.stack(
        (fx / z, zero, -fx * x / z.square(), zero, fy / z, -fy * y / z.square()),
        dim=-1,
    ).reshape(-1, 2, 3)
    projected = jacobian @ camera.rotation @ factors
    floor = torch.eye(2, dtype=centers.dtype, device=centers.device) * 0.3
    covariance = projected @ projected.transpose(-1, -2) + floor
    return pixels, covariance, depth


def composite(
    pixel: torch.Tensor, alpha: torch.Tensor, colors: torch.Tensor,
    background: torch.Tensor, pixel_count: int,
) -> torch.Tensor:
    """Composite samples sorted by pixel index and then by increasing depth."""
    first = torch.ones_like(pixel, dtype=torch.bool)
    first[1:] = pixel[1:] != pixel[:-1]
    sequence = torch.arange(len(pixel), device=pixel.device)
    start = torch.where(first, sequence, 0).cummax(0).values
    log_survival = torch.log1p(-alpha).to(torch.float64)
    exclusive = log_survival.cumsum(0) - log_survival
    transmittance = (exclusive - exclusive[start]).exp().to(alpha.dtype)
    weights = alpha * transmittance
    image = colors.new_zeros((pixel_count, 3)).index_add(0, pixel, weights[:, None] * colors)
    coverage = alpha.new_zeros(pixel_count).index_add(0, pixel, weights)
    return image + (1 - coverage[:, None]) * background


def render(
    camera: Camera, centers: torch.Tensor, factors: torch.Tensor,
    rgb: torch.Tensor, opacity: torch.Tensor, background: torch.Tensor,
) -> torch.Tensor:
    pixels, covariance, depth = project(camera, centers, factors)
    height, width = camera.height, camera.width
    with torch.no_grad():
        radius = 3 * covariance.diagonal(dim1=-2, dim2=-1).sqrt()
        lower = (pixels - radius).floor().long()
        upper = (pixels + radius).ceil().long()
        lower = lower.clamp_min(0)
        upper = torch.minimum(upper, upper.new_tensor([width - 1, height - 1]))
        spans = (upper - lower + 1).clamp_min(0)
        counts = spans.prod(dim=1) * (depth > 0.01)
        total = int(counts.sum())
        if not total:
            return background.expand(height, width, 3)
        splat = torch.repeat_interleave(
            torch.arange(len(centers), device=centers.device), counts, output_size=total
        )
        offset = torch.arange(total, device=centers.device) - (counts.cumsum(0) - counts)[splat]
        span_x = spans[splat, 0]
        x = lower[splat, 0] + offset.remainder(span_x)
        y = lower[splat, 1] + torch.div(offset, span_x, rounding_mode="floor")
        pixel = y * width + x

    delta = torch.stack((x, y), dim=-1) - pixels[splat]
    dx, dy = delta.unbind(-1)
    cov_x = covariance[:, 0, 0]
    cov_y = covariance[:, 1, 1]
    cov_xy = covariance[:, 0, 1]
    determinant = (cov_x * cov_y - cov_xy.square()).clamp_min(1e-6)
    mahalanobis = (
        cov_y[splat] * dx.square() - 2 * cov_xy[splat] * dx * dy + cov_x[splat] * dy.square()
    ) / determinant[splat]
    visible = mahalanobis <= 9
    splat, pixel = splat[visible], pixel[visible]
    if not len(pixel):
        return background.expand(height, width, 3)
    alpha = (opacity[splat] * (-0.5 * mahalanobis[visible]).exp()).clamp(max=0.995)
    ranks = depth.argsort(stable=True).argsort()
    order = (pixel * len(centers) + ranks[splat]).argsort()
    image = composite(pixel[order], alpha[order], rgb[splat[order]], background, height * width)
    return image.reshape(height, width, 3)


class GaussianScene(nn.Module):
    def __init__(
        self, centers: torch.Tensor, colors: torch.Tensor, background: torch.Tensor,
        anisotropic: bool,
    ) -> None:
        super().__init__()
        distances = torch.cdist(centers, centers).topk(4, largest=False).values[:, 1:]
        scale = distances.median(dim=1).values.clamp(0.007, 0.12)
        self.centers = nn.Parameter(centers.clone())
        self.log_scales = nn.Parameter(
            scale[:, None].expand(-1, 3 if anisotropic else 1).log().clone()
        )
        self.color_logits = nn.Parameter(colors.clamp(0.02, 0.98).logit())
        self.opacity_logits = nn.Parameter(torch.zeros_like(scale))
        self.background_logits = nn.Parameter(background.clamp(0.02, 0.98).logit())
        rotation = centers.new_zeros((len(centers), 4))
        rotation[:, 0] = 1
        if anisotropic:
            self.rotation = nn.Parameter(rotation)
        else:
            self.register_buffer("rotation", rotation)

    def forward(self, camera: Camera) -> torch.Tensor:
        scale = self.log_scales.clamp(-6, -1).exp()
        factors = quaternion_rotation(self.rotation) * scale[:, None, :]
        return render(
            camera, self.centers, factors, self.color_logits.sigmoid(),
            self.opacity_logits.sigmoid(), self.background_logits.sigmoid(),
        )

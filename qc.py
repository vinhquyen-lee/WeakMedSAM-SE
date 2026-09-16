import torch
from torch import Tensor


def normalize_cam(
    cam: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    
    if cam.ndim != 4:
        raise ValueError(
            f"CAM phải có dạng [B, C, H, W], nhận được {cam.shape}"
        )

    cam = torch.relu(cam)

    min_value = cam.amin(dim=(-2, -1), keepdim=True)
    max_value = cam.amax(dim=(-2, -1), keepdim=True)

    value_range = max_value - min_value

    normalized_cam = (cam - min_value) / (value_range + eps)

    normalized_cam = torch.where(
        value_range > eps,
        normalized_cam,
        torch.zeros_like(normalized_cam),
    )

    return normalized_cam.clamp(0.0, 1.0)


def soft_iou_per_class(
    cam_before: Tensor,
    cam_after: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    
    if cam_before.shape != cam_after.shape:
        raise ValueError(
            "CAM trước và sau phải có cùng kích thước. "
            f"Nhận được {cam_before.shape} và {cam_after.shape}"
        )

    cam_before = normalize_cam(cam_before, eps)
    cam_after = normalize_cam(cam_after, eps)

    intersection = (
        cam_before * cam_after
    ).sum(dim=(-2, -1))

    union = (
        cam_before.sum(dim=(-2, -1))
        + cam_after.sum(dim=(-2, -1))
        - intersection
    )

    quality = (intersection + eps) / (union + eps)

    return quality.clamp(0.0, 1.0)


@torch.no_grad()
def calculate_pseudo_label_quality(
    cam_before: Tensor,
    cam_after: Tensor,
    class_prediction: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    
    quality_per_class = soft_iou_per_class(
        cam_before=cam_before,
        cam_after=cam_after,
        eps=eps,
    )

    if class_prediction is None:
        return quality_per_class.mean(dim=1)

    if class_prediction.ndim == 1:
        class_prediction = class_prediction.unsqueeze(1)

    if class_prediction.shape != quality_per_class.shape:
        raise ValueError(
            "class_prediction phải có dạng "
            f"{quality_per_class.shape}, "
            f"nhận được {class_prediction.shape}"
        )

    active_classes = class_prediction > 0.5
    number_of_active_classes = active_classes.sum(dim=1)

    quality_scores = (
        quality_per_class * active_classes.float()
    ).sum(dim=1)

    quality_scores = quality_scores / number_of_active_classes.clamp_min(1)

    
    quality_scores = torch.where(
        number_of_active_classes > 0,
        quality_scores,
        torch.ones_like(quality_scores),
    )

    return quality_scores.clamp(0.0, 1.0) 
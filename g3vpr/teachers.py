from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualTeacherEngine(nn.Module):
    """
    Frozen Dual-Foundation Teacher Wrapper for Training G3-VPR on Kaggle / Colab / Server GPUs.
      - 2D Universal Retrieval Teacher: 'megaloc' (CVPR 2025 via torch.hub) or 'cricavpr' (local)
      - 3D Geometric Teacher: 'depth_anything_v2' (lightweight fp16 via HuggingFace transformers,
        < 1.0 GB VRAM on Kaggle T4) or 'vggt' (CVPR 2025 facebook/VGGT-1B on Colab L4/A100).
    Never required at inference time!
    """

    def __init__(
        self,
        teacher_2d: str = "megaloc",
        teacher_3d: str = "depth_anything_v2",
        cricavpr_ckpt: Optional[str] = None,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        super().__init__()
        self.teacher_2d_name = teacher_2d.lower()
        self.teacher_3d_name = teacher_3d.lower()
        self.device_str = device
        self.use_fp16 = use_fp16 and ("cuda" in device)

        self.model_2d: Optional[nn.Module] = None
        self.model_3d: Optional[nn.Module] = None

        if self.teacher_2d_name != "none":
            self._load_2d_teacher(cricavpr_ckpt)
        if self.teacher_3d_name != "none":
            self._load_3d_teacher()

    def _load_2d_teacher(self, cricavpr_ckpt: Optional[str]) -> None:
        if self.teacher_2d_name == "megaloc":
            import os
            hub_dir = torch.hub.get_dir()
            os.makedirs(hub_dir, exist_ok=True)
            trusted_file = os.path.join(hub_dir, "trusted_list")
            existing = open(trusted_file, "r", encoding="utf-8").read() if os.path.isfile(trusted_file) else ""
            with open(trusted_file, "a", encoding="utf-8") as f:
                for repo_tag in ("gmberton_MegaLoc", "facebookresearch_dinov2"):
                    if repo_tag not in existing:
                        f.write(repo_tag + "\n")
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)
            self.model_2d = model.eval().to(self.device_str)
        elif self.teacher_2d_name == "cricavpr":
            import network
            model = network.CricaVPRNet()
            if cricavpr_ckpt is not None:
                ckpt = torch.load(cricavpr_ckpt, map_location="cpu", weights_only=False)
                sd = ckpt.get("model_state_dict", ckpt)
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                model.load_state_dict(sd, strict=False)
            self.model_2d = model.eval().to(self.device_str)
        else:
            raise ValueError(f"Unsupported 2D teacher: {self.teacher_2d_name}")

        for p in self.model_2d.parameters():
            p.requires_grad = False

    def _load_3d_teacher(self) -> None:
        if self.teacher_3d_name == "depth_anything_v2":
            from transformers import AutoModelForDepthEstimation
            model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )
            self.model_3d = model.eval().to(self.device_str)
        elif self.teacher_3d_name == "vggt":
            # Requires `pip install git+https://github.com/facebookresearch/vggt.git` on Colab L4/A100
            from vggt.models.vggt import VGGT
            model = VGGT.from_pretrained("facebook/VGGT-1B")
            self.model_3d = model.eval().to(self.device_str)
        else:
            raise ValueError(f"Unsupported 3D teacher: {self.teacher_3d_name}")

        for p in self.model_3d.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def extract_targets(self, images: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            images: ImageNet-normalized tensor [B, 3, H, W] on target device.
        Returns:
            dict with 'teacher_2d_desc' [B, D_t] and 'teacher_depth' [B, 1, H_d, W_d].
        """
        out: Dict[str, Optional[torch.Tensor]] = {
            "teacher_2d_desc": None,
            "teacher_depth": None,
        }
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16)

        with amp_ctx:
            if self.model_2d is not None:
                desc = self.model_2d(images)
                out["teacher_2d_desc"] = F.normalize(desc.float(), p=2, dim=-1).detach()

            if self.model_3d is not None:
                if self.teacher_3d_name == "depth_anything_v2":
                    # Depth-Anything-V2 expects ImageNet-normalized inputs divisible by 14
                    pred = self.model_3d(pixel_values=images).predicted_depth  # [B, H', W']
                    out["teacher_depth"] = pred.unsqueeze(1).float().detach()
                elif self.teacher_3d_name == "vggt":
                    # VGGT expects [B, S=1, 3, H, W] in [0, 1]
                    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
                    imgs_01 = (images * std + mean).clamp(0.0, 1.0).unsqueeze(1)
                    preds = self.model_3d(imgs_01)
                    depth = preds["depth"][:, 0]  # [B, H, W, 1] or [B, 1, H, W]
                    if depth.shape[-1] == 1:
                        depth = depth.permute(0, 3, 1, 2)
                    out["teacher_depth"] = (1.0 / depth.clamp(min=1e-3)).float().detach()

        return out

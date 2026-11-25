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
            import transformers.utils.import_utils as _tf_imp
            # Bypass broken/mismatched torchaudio C++ extension check inside HuggingFace transformers
            _tf_imp._torchaudio_available = False
            from transformers import AutoModelForDepthEstimation
            model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )
            self.model_3d = model.eval().to(self.device_str)
        elif self.teacher_3d_name == "vggt":
            from vggt.models.vggt import VGGT
            model = VGGT.from_pretrained("facebook/VGGT-1B")
            self.model_3d = model.eval().to(self.device_str)
        else:
            raise ValueError(f"Unsupported 3D teacher: {self.teacher_3d_name}")

        for p in self.model_3d.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def extract_targets(self, images: torch.Tensor, chunk_size: int = 64) -> Dict[str, Optional[torch.Tensor]]:
        """
        Extracts 2D relational descriptors and 3D depth maps in memory-safe micro-batches (chunk_size=64)
        so peak Teacher VRAM stays < 1.5 GB even for large student batches.
        """
        out: Dict[str, Optional[torch.Tensor]] = {
            "teacher_2d_desc": None,
            "teacher_depth": None,
        }
        B = images.shape[0]
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16)

        with amp_ctx:
            if self.model_2d is not None:
                descs = []
                for st in range(0, B, chunk_size):
                    d_chunk = self.model_2d(images[st : st + chunk_size])
                    descs.append(F.normalize(d_chunk.float(), p=2, dim=-1).detach())
                out["teacher_2d_desc"] = torch.cat(descs, dim=0)

            if self.model_3d is not None:
                depths = []
                for st in range(0, B, chunk_size):
                    img_c = images[st : st + chunk_size]
                    if self.teacher_3d_name == "depth_anything_v2":
                        # Resize to 224x224 or pass directly; handle pos_encoding interpolation across transformers versions
                        try:
                            pred = self.model_3d(pixel_values=img_c).predicted_depth
                        except Exception:
                            img_518 = F.interpolate(img_c, size=(518, 518), mode="bilinear", align_corners=False)
                            pred = self.model_3d(pixel_values=img_518).predicted_depth
                        depths.append(pred.unsqueeze(1).float().detach())
                    elif self.teacher_3d_name == "vggt":
                        mean = torch.tensor([0.485, 0.456, 0.406], device=img_c.device).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=img_c.device).view(1, 3, 1, 1)
                        imgs_01 = (img_c * std + mean).clamp(0.0, 1.0).unsqueeze(1)
                        preds = self.model_3d(imgs_01)
                        depth = preds["depth"][:, 0]
                        if depth.shape[-1] == 1:
                            depth = depth.permute(0, 3, 1, 2)
                        depths.append((1.0 / depth.clamp(min=1e-3)).float().detach())
                out["teacher_depth"] = torch.cat(depths, dim=0)

        return out

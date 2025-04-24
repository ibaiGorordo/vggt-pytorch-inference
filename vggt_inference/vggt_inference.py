import os
from dataclasses import dataclass

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy
import torch
import torch.nn as nn
import numpy as np

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from vggt_inference.preprocess import preprocess_images
from vggt_inference.utils import get_best_device



@dataclass
class InferenceResult:
    image: np.ndarray
    width: int
    height: int
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    depth_map: np.ndarray
    depth_conf: np.ndarray
    point_map: np.ndarray
    point_conf: np.ndarray
    point_map_by_unprojection: np.ndarray


class VGGTInference(nn.Module):
    def __init__(self, device=get_best_device()):

        super().__init__()
        self.device = device

        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = torch.float16
        if self.device.type == 'cuda':
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Initialize the model and load the pretrained weights.
        # This will automatically download the model weights the first time it's run, which may take a while.
        print("Loading model, downloading weights if necessary...")
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        self.model.eval()

    @torch.inference_mode
    def forward(self, input_images: list[numpy.ndarray]) -> list[InferenceResult]:
        # Load and preprocess example images (replace with your own image paths)
        images = preprocess_images(input_images).to(self.device)

        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = self.model.aggregator(images)

        # Predict Cameras
        pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # Predict Depth Maps
        depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)

        # Predict Point Maps
        point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)

        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0),
                                                                     extrinsic.squeeze(0),
                                                                     intrinsic.squeeze(0))



        # Pack results
        results = []
        for i in range(images.shape[1]):
            # Convert images back to numpy format
            image = (images.squeeze(0)[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            result = InferenceResult(
                image=image,
                width=images.shape[-1],
                height=images.shape[-2],
                extrinsic=extrinsic.squeeze(0)[i].cpu().numpy(),
                intrinsic=intrinsic.squeeze(0)[i].cpu().numpy(),
                depth_map=depth_map.squeeze(0)[i].cpu().numpy(),
                depth_conf=depth_conf.squeeze(0)[i].cpu().numpy(),
                point_map=point_map.squeeze(0)[i].cpu().numpy(),
                point_conf=point_conf.squeeze(0)[i].cpu().numpy(),
                point_map_by_unprojection=point_map_by_unprojection[i]
            )
            results.append(result)

        return results
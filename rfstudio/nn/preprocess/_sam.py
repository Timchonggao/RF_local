# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.graphics import RGBAImages, RGBImages, SegImages, SegTree
from rfstudio.graphics.math import get_bounding_box
from rfstudio.ui import console
from rfstudio.utils.download import download_model_weights
from rfstudio.utils.lazy_module import sam, torchvision

from ..module import Module

_CKPT_URLS = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
}

class _ModifiedMaskData:

    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(
                v, torch.Tensor
            ), "MaskData only supports torch tensors."
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(
            item, (dict, torch.Tensor)
        ), "MaskData only supports dict, torch tensors."
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def items(self) -> Any:
        return self._stats.items()

    def to_numpy(self) -> None:
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.detach().cpu().numpy()

    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[keep]
            elif isinstance(v, dict):
                self._stats[k] = {
                    "size": v["size"], "counts": v["counts"],
                    "starts": v["starts"][keep], "ends": v["ends"][keep]
                }
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def move_cat(self, rhs: _ModifiedMaskData) -> None:
        for k, v in rhs.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = v
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, dict):
                assert v["size"] == self._stats[k]["size"]
                self._stats[k] = {
                    "size": v["size"],
                    "counts": torch.cat((self._stats[k]["counts"], v["counts"])),
                    "starts": torch.cat((self._stats[k]["starts"], v["starts"] + self._stats[k]["counts"].shape[0])),
                    "ends": torch.cat((self._stats[k]["ends"], v["ends"] + self._stats[k]["counts"].shape[0]))
                }
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

def _rle_to_mask(rle: Dict[str, Any]) -> Iterator[torch.Tensor]:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    counts = rle["counts"].cpu()
    starts = rle["starts"].cpu()
    ends = rle["ends"].cpu()
    for i in range(len(starts)):
        mask = np.empty(h * w, dtype=bool)
        idx = 0
        parity = False
        for count in counts[starts[i]:ends[i]]:
            mask[idx : idx + count] = parity
            idx += count
            parity ^= True
        mask = mask.reshape(w, h).transpose() # Put in C order
        yield torch.from_numpy(mask).to(rle["counts"].device)

def _mask_to_rle_pytorch(tensor: torch.Tensor, *, chunk_size: int = 128) -> Dict[str, Any]:
    """
    Encodes masks to an uncompressed RLE
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape # [B, H, W]
    tensor = tensor.permute(0, 2, 1).flatten(1) # [B, W, H] -> [B, W*H]

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1] # [B, W*H-1]
    assert diff.any(-1).all()
    diff = torch.nn.functional.pad(diff, (1, 1), mode='constant', value=True)
    change_indices = diff.nonzero() # [I1+I2+...+IN, 2]

    batch_starts = []
    for i in range(0, b, chunk_size):
        real_chunk_size = min(chunk_size, b - i)
        batch_starts.append((
            change_indices[:, :1] ==
            torch.arange(i, i + real_chunk_size, device=change_indices.device, dtype=change_indices.dtype)
        ).int().argmax(0)) # [real_chunk_size]

    batch_starts = torch.cat(batch_starts) # [B]
    batch_ends = torch.cat((batch_starts[1:], torch.tensor([change_indices.shape[0]]).to(batch_starts))) # [B]
    batch_ends = (batch_ends - 1).clamp_min(0).cpu()
    batch_starts = batch_starts.cpu()
    btw_idxs = (change_indices[1:, 1] - change_indices[:-1, 1]) # [I1+I2+...+IN-1]

    # Encode run length
    out = []
    idx = 0
    starts = []
    ends = []
    zero_item = btw_idxs.new_zeros(1)
    for i, zero_first in enumerate((tensor[:, 0] == 0).cpu()):
        starts.append(idx)
        if not zero_first:
            out.append(zero_item)
            idx += 1
        out.append(btw_idxs[batch_starts[i]:batch_ends[i]])
        idx += out[-1].shape[0]
        ends.append(idx)
    counts = torch.cat(out)
    assert idx == counts.shape[0]
    return {
        "size": (h, w),
        "counts": counts,
        "starts": torch.tensor(starts).to(counts.device),
        "ends": torch.tensor(ends).to(counts.device),
    }

class _SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: sam.automatic_mask_generator.Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        nms: bool = True,
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = sam.automatic_mask_generator.build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = sam.SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.nms = nms

    @torch.no_grad()
    def generate(self, image: Float32[Tensor, "H W 3"]) -> Tensor:
        """
        Generates masks for the given image.
        """

        # Generate masks
        mask_data = self._generate_masks((image * 255).permute(2, 0, 1).contiguous().unsqueeze(0))

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        return torch.stack(tuple(_rle_to_mask(mask_data["rles"])))

    def _generate_masks(self, image: Float32[Tensor, "1 H W 3"]) -> _ModifiedMaskData:
        orig_size = image.shape[-2:]
        crop_boxes, layer_idxs = sam.automatic_mask_generator.generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = _ModifiedMaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.move_cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1 and self.nms:
            # Prefer masks from smaller crops
            scores = 1 / torchvision.ops.boxes.box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = torchvision.ops.boxes.batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        return data

    def _process_crop(
        self,
        image: Float32[Tensor, "1 3 H W"],
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> _ModifiedMaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[:, y0:y1, x0:x1]
        cropped_im_size = cropped_im.shape[-2:]
        cropped_im = self.predictor.transform.apply_image_torch(cropped_im)
        self.predictor.set_torch_image(cropped_im, cropped_im_size)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = _ModifiedMaskData()
        for (points,) in sam.automatic_mask_generator.batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.move_cat(batch_data)
        self.predictor.reset_image()

        if self.nms:
            # Remove duplicates within this crop.
            keep_by_nms = torchvision.ops.boxes.batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = sam.automatic_mask_generator.uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["crop_boxes"] = torch.tensor([
            crop_box
            for _ in range(len(data["rles"]["starts"]))
        ]).to(data["boxes"].device)

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> _ModifiedMaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = _ModifiedMaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = sam.automatic_mask_generator.calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = sam.automatic_mask_generator.batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~sam.automatic_mask_generator.is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = sam.automatic_mask_generator.uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = _mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def remove_small_regions(mask: Tensor, area_thresh: int, mode: Literal['holes', 'islands']) -> Tuple[Tensor, bool]:
        """
        Removes small disconnected regions and holes in a mask. Returns the
        mask and an indicator of if the mask has been modified.
        """
        import cv2  # type: ignore

        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask.cpu().numpy()).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[:, -1][1:]  # Row 0 is background label
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if len(small_regions) == 0:
            return mask, False
        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            # If every region is below threshold, keep largest
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]
        return torch.from_numpy(np.isin(regions, fill_labels)).to(mask), True

    @staticmethod
    def postprocess_small_regions(
        mask_data: _ModifiedMaskData, min_area: int, nms_thresh: float
    ) -> _ModifiedMaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]["starts"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for mask in _rle_to_mask(mask_data["rles"]):

            mask, changed = _SamAutomaticMaskGenerator.remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = _SamAutomaticMaskGenerator.remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = sam.automatic_mask_generator.batched_mask_to_box(masks)
        keep_by_nms = torchvision.ops.boxes.batched_nms(
            boxes.float(),
            torch.as_tensor(scores).to(boxes.device),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = _mask_to_rle_pytorch(mask_torch)
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data


@dataclass
class SamAutoMasker(Module):

    model_type: Literal['vit_h', 'vit_l', 'vit_b'] = 'vit_h'
    checkpoint: Optional[Path] = None

    points_per_side: Optional[int] = 32
    '''The number of points to be sampled
      along one side of the image. The total number of points is
      points_per_side**2. If None, 'point_grids' must provide explicit
      point sampling.'''

    points_per_batch: int = 64
    '''The number of points to be sampled
      along one side of the image. The total number of points is
      points_per_side**2. If None, 'point_grids' must provide explicit
      point sampling.'''

    pred_iou_thresh: float = 0.88
    '''A filtering threshold in [0,1], using the
      model's predicted mask quality.'''

    stability_score_thresh: float = 0.95
    '''A filtering threshold in [0,1], using
      the stability of the mask under changes to the cutoff used to binarize
      the model's mask predictions.'''

    stability_score_offset: float = 1.0
    '''The amount to shift the cutoff when
      calculated the stability score.'''

    box_nms_thresh: float = 0.7
    '''The box IoU cutoff used by non-maximal
      suppression to filter duplicate masks.'''

    crop_n_layers: int = 0
    '''If >0, mask prediction will be run again on
      crops of the image. Sets the number of layers to run, where each
      layer has 2**i_layer number of image crops.'''

    crop_nms_thresh: float = 0.7
    '''The box IoU cutoff used by non-maximal
      suppression to filter duplicate masks between different crops.'''

    crop_overlap_ratio: float = 512 / 1500
    '''Sets the degree to which crops overlap.
      In the first crop layer, crops will overlap by this fraction of
      the image length. Later layers with more crops scale down this overlap.'''

    crop_n_points_downscale_factor: int = 1
    '''The number of points-per-side
      sampled in layer n is scaled down by crop_n_points_downscale_factor**n.'''

    point_grids: Optional[List[np.ndarray]] = None
    '''A list over explicit grids
      of points used for sampling, normalized to [0,1]. The nth grid in the
      list is used in the nth crop layer. Exclusive with points_per_side.'''

    min_mask_region_area: int = 100
    '''If >0, postprocessing will be applied
      to remove disconnected regions and holes in masks with area smaller
      than min_mask_region_area. Requires opencv.'''

    def __setup__(self) -> None:
        self._basemodel = None
        self._automasker = None

    def _load(self) -> None:
        if self._basemodel is not None:
            return
        if self.checkpoint is None:
            checkpoint = download_model_weights(_CKPT_URLS[self.model_type], check_hash=True)
        else:
            checkpoint = self.checkpoint
        basemodel = sam.sam_model_registry[self.model_type](checkpoint=checkpoint).to(self.device)
        self._basemodel = basemodel
        self._automasker = _SamAutomaticMaskGenerator(
            self._basemodel,
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            stability_score_offset=self.stability_score_offset,
            box_nms_thresh=self.box_nms_thresh,
            crop_n_layers=self.crop_n_layers,
            crop_nms_thresh=self.crop_nms_thresh,
            crop_overlap_ratio=self.crop_overlap_ratio,
            crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
            point_grids=self.point_grids,
            min_mask_region_area=self.min_mask_region_area,
            nms=True,
        )

    @torch.no_grad()
    def __call__(self, images: RGBImages) -> SegImages:
        self._load()
        results = []
        for img in images:
            seg = torch.zeros_like(img[..., 0], dtype=torch.long)
            for i, mask in enumerate(self._automasker.generate(img)):
                seg[mask] = i + 1
            results.append(seg.unsqueeze(-1))
        return SegImages(results)

    def segment(self, images: RGBImages) -> SegImages:
        return self.__call__(images)

    @torch.no_grad()
    def hierarchical_segment(
        self,
        images: Union[RGBImages, RGBAImages],
        *,
        max_clusters: int = 64,
        min_cluster_area_ratio: Optional[float] = 0.0004,
        min_cluster_box_IoU: Optional[float] = 0.02,
        progress: Optional[str] = None,
    ) -> List[SegTree]:
        self._load()
        results = []
        with console.progress(desc=progress, transient=True, enabled=progress is not None) as ptrack:
            for img in ptrack(images):
                if img.shape[-1] == 3:
                    masks = self._automasker.generate(img) # [M, H, W]
                    img = torch.cat((img, torch.ones_like(img[..., :1])), dim=-1)
                else:
                    masks = self._automasker.generate(img[..., :3]) # [M, H, W]
                    image_bg = (img[..., 3] < 0.5) # [H, W]
                    bg_IoU = (
                        (masks & image_bg).sum(dim=(1, 2)) /
                        masks.sum(dim=(1, 2)).clamp_min(1)
                    ) # [M]
                    fg_IoU = (
                        masks.sum(dim=(1, 2)) /
                        (masks | ~image_bg).sum(dim=(1, 2)).clamp_min(1)
                    ) # [M]
                    masks = masks[(bg_IoU < 0.95) & (fg_IoU < 0.95)].contiguous()
                M = masks.shape[0]
                masks = masks.permute(1, 2, 0).reshape(-1, M) # [H * W, M]
                clustering: Tuple[Tensor, Tensor, Tensor] = masks.unique(
                    dim=0,
                    return_inverse=True,
                    return_counts=True,
                )
                codes, indices, counts = clustering # b[N, M], i64[H*W] \in N, i64[N]
                N = counts.shape[0]
                if N > max_clusters:
                    topk_inds = counts.topk(k=max_clusters).indices # [N'] \in N
                    counts = counts[topk_inds] # [N']
                    indices_map = (indices.unsqueeze(-1) == topk_inds) # [H*W, N']
                    indices = torch.where(indices_map.any(-1), indices_map.int().argmax(-1), -1) # [H*W] \in N'
                    codes = codes[topk_inds, :]
                    N = topk_inds.shape[0]

                # remove empty clusters
                nonempty_cluster = codes.any(1) # [N]
                if min_cluster_area_ratio is not None:
                    min_cluster_counts = int(indices.shape[0] * min_cluster_area_ratio)
                    nonempty_cluster = nonempty_cluster & (counts >= min_cluster_counts)
                if min_cluster_box_IoU is not None:
                    pixel2cluster_mask = (indices == torch.arange(codes.shape[0]).unsqueeze(-1).to(indices)) # [N, H*W]
                    bbox = get_bounding_box(pixel2cluster_mask.view(N, *img.shape[:2])) # [N, 4]
                    box_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) # [N]
                    min_cluster_counts = (box_area * min_cluster_box_IoU).long() # [N]
                    nonempty_cluster = nonempty_cluster & (counts >= min_cluster_counts) # [N]
                empty_cluster_idx = (~nonempty_cluster).nonzero().flatten() # [K]
                indices = torch.where(
                    torch.isin(indices, empty_cluster_idx),
                    -1,
                    torch.where(
                        indices < 0,
                        -1,
                        indices - (~nonempty_cluster).int().cumsum(0)[indices],
                    ),
                ) # [H*W] \in N'
                codes = codes[nonempty_cluster] # [N', M]

                # remove unrelated masks
                related_masks = codes.any(0) # [M]
                codes = codes[:, related_masks] # [N, M']
                masks = masks[:, related_masks].contiguous() # [H*W, M']

                codes_l = codes.long()
                cluster_correlation = (codes_l.unsqueeze(-1) * codes_l.T).sum(-2) # [N, N]
                cluster_idx = indices.view(*img.shape[:2]) # [H, W]
                cluster_masks = codes.contiguous() # [N, M]
                results.append(
                    SegTree(
                        cluster_correlation=cluster_correlation,
                        pixel2cluster=cluster_idx,
                        cluster2mask=cluster_masks,
                        masks=masks.view(*img.shape[:2], -1),
                        image=img,
                    )
                )
        return results

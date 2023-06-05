#pylint: disable-all

"Model Definition of NVIDIA SSD300 in Pytorch"

import torch
import torchvision
from gi.repository import Gst  # pylint: disable=no-name-in-module
from torchvision import transforms
from utils.ssd import Encoder, dboxes300_coco
from typing import *


class SSD300(torch.nn.Module):
    def __init__(self, model_precision = "fp32", conf_thresh = 0.4, nms_thresh = 0.45, preprocess_transforms = None):
        super().__init__()
        self.model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", model_math=model_precision)
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.model_precision = model_precision
        
        self.preprocess_transforms = transforms.Compose(
                        [
                            transforms.CenterCrop(300),
                            transforms.Normalize(127.5, 127.5),
                        ]
                     ) if preprocess_transforms is None else preprocess_transforms 
    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """Preprocess image according to NVIDIA SSD - 
        https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/dle/inference.py
        
        The input image must be of uint8 with values [0, 255]"""
        return self.preprocess_transforms(img)

    def _postprocess(self, ploc: torch.Tensor, pconf: torch.Tensor) -> List[Tuple[List,List,List]]:
        """
        Input - Tuple [ ploc , pconf ]
            ploc shape - [bsz, 4, num_boxes]
            pconf shape - [bsz, 81, num_boxes]

        There are 81 classes and 8732 num_boxes

        Returns - List[Tuple[bbox, label, score]]
        """
        encoder = Encoder(dboxes300_coco())
        bboxes, probs = encoder.scale_back_batch(ploc, pconf)
        # bboxes = [bsz, num_boxes, 4] , probs = [bsz, num_boxes, 81]

        (bsz, num_boxes, num_classes) = probs.shape
        (score, lbl) = torch.max(probs, dim=-1)
        # score = [bsz, num_boxes] , lbl = [bsz, num_boxes]

        image_idx = torch.arange(
            bsz, dtype=probs.dtype, device=probs.device
        ).repeat_interleave(num_boxes)
        # [0, 0, 1, 1, 2, 2, 3, 3] assume num_boxes = 2 and bsz = 4 and num_classes = 81

        offset = image_idx * num_classes
        # [0, 0, 81, 81, 162, 162, 243, 243]

        flat_lbl = lbl.view(-1)
        flat_score = score.view(-1)
        # lbl = [0, 80, 80, 80, 0, 0, 80, 0] assume num_classes = 81
        flat_bbox = bboxes.reshape(-1, 4)

        encode_lbl = flat_lbl + offset
        # [0, 80, 161, 161, 162, 162, 323, 243]
        # for decoding
        # lbl (% num_classes) [0, 80, 80, 80, 0, 0, 80, 0]
        # batch (//num_classes) [0, 1, 1, 1, 2, 2, 3, 3]

        conf_mask = (flat_score > self.conf_thresh) & (flat_lbl > 0)
        flat_bbox, flat_score, encode_lbl = (
            flat_bbox[conf_mask, :],
            flat_score[conf_mask],
            encode_lbl[conf_mask],
        )

        nms_mask = torchvision.ops.batched_nms(
            flat_bbox, flat_score, encode_lbl, self.nms_thresh
        )

        flat_bbox, encode_lbl, flat_score = (
            flat_bbox[nms_mask, :].cpu(),
            encode_lbl[nms_mask].cpu(),
            flat_score[nms_mask].cpu(),
        )

        # output = [[[], [], []] for _ in range(bsz)]
        # for bbox, lbl, score in zip(flat_bbox, encode_lbl, flat_score):
        #     decode_lbl = int(lbl) % num_classes
        #     decode_img_id = int(lbl) // num_classes
        #     output[decode_img_id][0].append(bbox)
        #     output[decode_img_id][1].append(decode_lbl)
        #     output[decode_img_id][2].append(score)

        return flat_bbox, encode_lbl, flat_score
    
    def forward(self, input_tensor : torch.Tensor) -> List[Tuple[List, List, List]]:
        """Input
           -----
           input_tensor : torch.Tensor of shape [B,C,H,W]

           Output
           -----
           List[Tuple[bbox, label, score]]
        """
        input_tensor = self._preprocess(input_tensor)
        ploc, pconf = self.model(input_tensor)
        output = self._postprocess(ploc, pconf)
        return output


    

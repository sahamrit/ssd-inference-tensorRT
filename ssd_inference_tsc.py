#pylint: disable-all
import torch
import os
import numpy as np
from torchvision import transforms
from ssd_model import SSD300
from utils.util import plt_results

LOGS_DIR = "logs"
ENABLE_PLOTTING = False

uris = [
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
]

MODEL_PRECISION = "fp32" 
MODEL_DTYPE = torch.float16 if MODEL_PRECISION == "fp16" else torch.float32

ssd_model = SSD300(model_precision=MODEL_PRECISION,preprocess_transforms=transforms.Compose([]),)
ssd_utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils"
)

def preprocess_img(img: np.array) -> np.array:
    """Preprocess image according to NVIDIA SSD -
    https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/dle/inference.py
    """
    img = ssd_utils.rescale(img, 300, 300)
    img = ssd_utils.crop_center(img, 300, 300)
    img = ssd_utils.normalize(img)

    return img

inputs = [ssd_utils.prepare_input(uri) for uri in uris]
tensor = ssd_utils.prepare_tensor(inputs)

device = "cuda" if torch.cuda.is_available() else "cpu"
input_tensor = tensor.to(device = device, dtype = MODEL_DTYPE)
ssd_model.eval()
ssd_model.to(device = device, dtype = MODEL_DTYPE)

with torch.no_grad():
    detections_batch = ssd_model(input_tensor)

# if ENABLE_PLOTTING:
#     plt_results([detections_batch[-1]],
#                 [inputs[-1]],
#                 os.path.join(LOGS_DIR, f"ssd_infer_pytorch.png"),
#                 ssd_utils,
#                 )

# Convert model to torchscript and infer

ssd_model_1 = SSD300(preprocess_transforms=transforms.Compose([]),model_precision= MODEL_PRECISION)
ssd_model_1.eval()
ssd_model_1.to("cpu")
ssd_model_traced = torch.jit.trace(ssd_model_1, torch.randn(tensor.size()))

print(ssd_model_traced.code)
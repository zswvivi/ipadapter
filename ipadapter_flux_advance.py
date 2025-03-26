import torch
import os
import logging
import folder_paths
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
import numpy as np
from .attention_processor import IPAFluxAttnProcessor2_0
from .utils import is_model_patched, FluxUpdateModules
from .flux.model import inject_flux
from .flux.layers import inject_blocks, inject_ipadapter_blocks
#from optimum.quanto import freeze, qfloat8, quantize, qint4, qfloat8_e5m2, qfloat8_e4m3fn
import gc

MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter-flux")
if "ipadapter-flux" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter-flux"]
folder_paths.folder_names_and_paths["ipadapter-flux"] = (current_paths, folder_paths.supported_pt_extensions)
google_siglip_model_dir = os.path.join(folder_paths.models_dir, 'siglip-so400m-patch14-384')
folder_paths.add_model_folder_path('siglip-so400m-patch14-384', google_siglip_model_dir)


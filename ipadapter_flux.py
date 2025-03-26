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

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class IPAdapterClip:
    def __init__(self, image_encoder_path, device):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.image_encoder = SiglipVisionModel.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16)
        self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
    
    @torch.inference_mode()
    def get_clip_embedding(self, pil_image=None, clip_image_embeds=None):
        with torch.no_grad():
            if pil_image is not None:
                if isinstance(pil_image, Image.Image):
                    pil_image = [pil_image]
                clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
                #clip_image_embeds = self.image_encoder(clip_image.to(dtype=self.image_encoder.dtype)).pooler_output
                clip_image_embeds = clip_image_embeds.to(dtype=torch.float16)
            else:
                clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        return clip_image_embeds

class FluxInstantXImageProjModel:
    def __init__(self, ip_ckpt, device, num_tokens = 4):
        self.device = device
        #self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.joint_attention_dim = 4096
        self.hidden_size = 3072
    
    def init_proj(self):
        self.image_proj_model = MLPProjModel(
            cross_attention_dim=self.joint_attention_dim, # 4096
            id_embeddings_dim=1152, 
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        self.image_proj_model.load_state_dict(torch.load(os.path.join(MODELS_DIR,self.ip_ckpt), map_location="cpu"), strict=True)

    @torch.inference_mode()
    def get_image_embeds(self, clip_image_embeds = None):
        with torch.no_grad():
            if clip_image_embeds is not None:
                clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
                image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            else:
                image_prompt_embeds = None
        return image_prompt_embeds


class FluxInstantXIPAdapterModel:
    def __init__(self, ip_ckpt, device, num_tokens = 4):
        self.device = device
        #self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.joint_attention_dim = 4096
        self.hidden_size = 3072

    def set_ip_adapter(self, weight, timestep_percent_range=(0.0, 1.0)):

        timestep_range = (1.0, 0.0)
        ip_attn_procs = {} # 19+38=57
        dsb_count = 19
        for i in range(dsb_count):
            name = f"double_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale = weight,
                    timestep_range = timestep_range
                ).to(self.device, dtype=torch.float16)
        ssb_count = 38
        for i in range(ssb_count):
            name = f"single_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale = weight,
                    timestep_range = timestep_range
                ).to(self.device, dtype=torch.float16)
        return ip_attn_procs

    def load_ip_attn(self, weight):
        ip_attn_procs = self.set_ip_adapter(weight)
        ip_layers = torch.nn.ModuleList(ip_attn_procs.values())
        ip_layers.load_state_dict(torch.load(os.path.join(MODELS_DIR,self.ip_ckpt), map_location="cpu"), strict=True)
        return ip_attn_procs
    

class InstantXFluxIPAdapterModel:
    #def __init__(self, image_encoder_path, ip_ckpt, device, num_tokens=4):
    def __init__(self, ip_ckpt, device, num_tokens = 4):
        self.device = device
        #self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        # load image encoder
        #self.image_encoder = SiglipVisionModel.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16)
        #self.image_encoder = SiglipVisionModel.from_pretrained(self.image_encoder_path).to(dtype=torch.float16)
        #self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
        # state_dict
        self.state_dict = torch.load(os.path.join(MODELS_DIR,self.ip_ckpt), map_location="cpu")
        self.joint_attention_dim = 4096
        self.hidden_size = 3072
        #self.ip_attn_procs = {}

    def init_proj(self):
        self.image_proj_model = MLPProjModel(
            cross_attention_dim=self.joint_attention_dim, # 4096
            id_embeddings_dim=1152, 
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        '''
        self.image_proj_model = MLPProjModel(
            cross_attention_dim=self.joint_attention_dim, # 4096
            id_embeddings_dim=1152, 
            num_tokens=self.num_tokens,
        ).to(dtype=torch.float16)
        '''

    def set_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
        #timestep_range =  (1.0, 0.0)
        ip_attn_procs = {} # 19+38=57
        dsb_count = len(flux_model.diffusion_model.double_blocks)
        for i in range(dsb_count):
            name = f"double_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale = weight,
                    timestep_range = timestep_range
                ).to(self.device, dtype=torch.float16)
        ssb_count = len(flux_model.diffusion_model.single_blocks)
        print("-->>|||||| timestep_range: ", timestep_range, dsb_count, ssb_count)
        for i in range(ssb_count):
            name = f"single_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale = weight,
                    timestep_range = timestep_range
                ).to(self.device, dtype=torch.float16)
        return ip_attn_procs
    
    def load_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        self.image_proj_model.load_state_dict(self.state_dict["image_proj"], strict=True)
        '''
        quantize(self.image_proj_model, qfloat8)
        freeze(self.image_proj_model)
        '''
        ip_attn_procs = self.set_ip_adapter(flux_model, weight, timestep_percent_range)
        ip_layers = torch.nn.ModuleList(ip_attn_procs.values())
        ip_layers.load_state_dict(self.state_dict["ip_adapter"], strict=True)

        return ip_attn_procs

    def load_ip_attn(self, flux_model, weight, timestep_percent_range = (0.0, 1.0)):
        ip_attn_procs = self.set_ip_adapter(flux_model, weight, timestep_percent_range)
        ip_layers = torch.nn.ModuleList(ip_attn_procs.values())
        ip_layers.load_state_dict(self.state_dict["ip_adapter"], strict=True)
        return ip_attn_procs

    def load_ip_img_proj(self):
        self.image_proj_model.load_state_dict(self.state_dict["image_proj"], strict=True)
        #self.image_proj_model.to('cpu')

    @torch.inference_mode()
    def get_image_embeds(self, clip_image_embeds = None):
        with torch.no_grad():
            if clip_image_embeds is not None:
                clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
                image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            else:
                image_prompt_embeds = None
        return image_prompt_embeds
    '''
    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        with torch.no_grad():
            if pil_image is not None:
                if isinstance(pil_image, Image.Image):
                    pil_image = [pil_image]
                clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
                #clip_image_embeds = self.image_encoder(clip_image.to(dtype=self.image_encoder.dtype)).pooler_output
                clip_image_embeds = clip_image_embeds.to(dtype=torch.float16)
            else:
                clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
            
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds
    '''


class IPAdapterFluxImageProjLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "image_proj": (folder_paths.get_filename_list("ipadapter-flux"),),
                "provider":  ("STRING", {"default": "cuda"}),
            }
        }
    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("image_proj_model",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, image_proj, provider):
        image_proj_model = FluxInstantXImageProjModel(ip_ckpt = image_proj, device = provider, num_tokens = 128)
        image_proj_model.init_proj()
        return (image_proj_model, )

class IPAdapterFluxAdapterLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "ip_adapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                "provider":  ("STRING", {"default": "cuda"}),
            },
            "optional": {
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
            },
        }
    RETURN_TYPES = ("IP_ATTN_PROCS",)
    RETURN_NAMES = ("ip_attn_procs",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, ip_adapter, provider, weight):
        flux_ipadapter_model = FluxInstantXIPAdapterModel(ip_ckpt = ip_adapter, device = provider, num_tokens = 128)
        ip_attn_procs = flux_ipadapter_model.load_ip_attn(weight)
        return (ip_attn_procs, )

class IPAdapterFluxLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                #"clip_vision": (["google/siglip-so400m-patch14-384"],),
                #"clip_vision": (folder_paths.get_folder_paths('siglip-so400m-patch14-384'),),
                #"provider": (["cuda", "cpu", "mps"],),
                "provider":  ("STRING", {"default": "cuda"}),
            }
        }
    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"
    '''
    def load_model(self, ipadapter, clip_vision, provider):
        logging.info(f"Loading InstantX IPAdapter Flux model. clip_vision_model_path: {clip_vision}")
        if clip_vision and not clip_vision.startswith("/"):
            clip_vision = os.path.join(folder_paths.models_dir, clip_vision)
        model = InstantXFluxIPAdapterModel(image_encoder_path=clip_vision, ip_ckpt=ipadapter, device=provider, num_tokens=128)
        return (model,)
    '''

    def load_model(self, ipadapter, provider):
        model = InstantXFluxIPAdapterModel(ip_ckpt = ipadapter, device = provider, num_tokens = 128)
        return (model, )

class IpadapterClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                #"ipadapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                #"clip_vision": (["google/siglip-so400m-patch14-384"],),
                "clip_vision": (folder_paths.get_folder_paths('siglip-so400m-patch14-384'),),
                #"provider": (["cuda", "cpu", "mps"],),
                "provider":  ("STRING", {"default": "cuda"}),
            }
        }
    RETURN_TYPES = ("IPAdapterClip",)
    RETURN_NAMES = ("ipadapterclip",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, clip_vision, provider):
        logging.info(f"Loading InstantX IPAdapter Flux model. clip_vision_model_path: {clip_vision}")
        if clip_vision and not clip_vision.startswith("/"):
            clip_vision = os.path.join(folder_paths.models_dir, clip_vision)
        ipadapter_clip = IPAdapterClip(clip_vision, provider)
        return (ipadapter_clip, )


class LoadIPAdapterModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
        }

    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX", "IP_ATTN_PROCS",)
    FUNCTION = "load_ipadapter_model"
    CATEGORY = "InstantXNodes"

    def load_ipadapter_model(self, model, ipadapter_flux, weight, start_percent, end_percent):
        ip_attn_procs = ipadapter_flux.load_ip_attn(model.model, weight, (start_percent, end_percent))
        return (ipadapter_flux, ip_attn_procs, )

class LoadIPAdapterImageProj:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX", ),
            },
        }

    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    FUNCTION = "load_ipadapter_image_proj"
    CATEGORY = "InstantXNodes"

    def load_ipadapter_image_proj(self, ipadapter_flux,):
        ipadapter_flux.init_proj()
        ipadapter_flux.load_ip_img_proj()
        return (ipadapter_flux, )

class GetIPAadapterEmb:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX", ),
                "ipadapter_clip": ("IPAdapterClip", ),
                "image": ("IMAGE", ),
            },
            "optional": {
                "precal_pulid_cond": ("CLIP_EMBEDS",),
            },
        }

    RETURN_TYPES = ("IMAGE_EMBED",)
    FUNCTION = "apply_ipadapter_image_emb"
    CATEGORY = "InstantXNodes"

    def _tensor_to_pil(self, tensor):
        tensor = tensor.squeeze(0)
        tensor = tensor.cpu().detach().numpy()
        tensor = (tensor * 255).astype(np.uint8)
        img = Image.fromarray(tensor)
        return img

    def apply_ipadapter_image_emb(self, ipadapter_flux, ipadapter_clip, image, precal_pulid_cond=None):
        if precal_pulid_cond is not None:
            #print("--->>>ipadapter_img_emb+11: ", precal_pulid_cond["ipa"])
            return (precal_pulid_cond["ipa"], )
        # convert image to pillow
        image_prompt_embed_list = []
        #ipadapter_flux.image_proj_model.to('cuda')
        if image is not None:
            for img in image:
                pil_image = self._tensor_to_pil(img)
                #pil_image = image.numpy()[0] * 255.0
                #pil_image = Image.fromarray(pil_image.astype(np.uint8))
                clip_image_embeds = ipadapter_clip.get_clip_embedding(pil_image=pil_image, clip_image_embeds=None)
                image_prompt_embeds = ipadapter_flux.get_image_embeds(
                    #pil_image=pil_image, clip_image_embeds=None
                    clip_image_embeds = clip_image_embeds
                )
                if image_prompt_embeds is not None:
                    image_prompt_embed_list.append(image_prompt_embeds)
        if len(image_prompt_embed_list) > 0:
            image_prompt_embeds_avg = torch.mean(torch.cat(image_prompt_embed_list, dim =0), dim =0).unsqueeze(0)
        else:
            image_prompt_embeds_avg = None
        #ipadapter_flux.image_proj_model.to('cpu')
        #print("--->>>ipadapter_img_emb+22: ", image_prompt_embeds_avg)
        return (image_prompt_embeds_avg, )

class ConfigureModifiedFlux_V1_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ip_attn_procs": ("IP_ATTN_PROCS", ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "conf_modify_flux_v1"
    CATEGORY = "InstantXNodes"

    def conf_modify_flux_v1(self, model, ip_attn_procs):
        model_clone = model.clone()
        #inject_flux(model.model.diffusion_model)
        #inject_ipadapter_blocks(model, ip_attn_procs)
        #inject_flux(model_clone.model.diffusion_model, enable_cache, rel_l1_thresh, steps)
        inject_flux(model_clone.model.diffusion_model)
        inject_ipadapter_blocks(model_clone, ip_attn_procs)   
        del ip_attn_procs
        gc.collect()
        torch.cuda.empty_cache()
        #return (model, )
        return (model_clone, )

class ConfigureModifiedFlux_V2_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "conf_modify_flux_v2_node"
    CATEGORY = "InstantXNodes"

    def conf_modify_flux_v2_node(self, model):
        model_clone = model.clone()
        #inject_flux(model.model.diffusion_model)
        #inject_ipadapter_blocks(model, ip_attn_procs)
        inject_flux(model_clone.model.diffusion_model)
        inject_blocks(model_clone.model.diffusion_model)
        return (model_clone, )

class ApplyIPAdapterFluxV1:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                #"image_emb": ("IMAGE_EMBED", )
            },
            "optional": {
                "image_emb": ("IMAGE_EMBED", ),
                "ip_scale": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "rp_attrs": ("RP_ATTRS", {}),
                "image_emb_2": ("IMAGE_EMBED", ),
                #"ip_mask_method": (["bbox", "split_area", "all"],),  
                "ip_mask_method": ("STRING", {"default": "split_area"}), 
                "do_lora_1": ("BOOLEAN", {"default": False}),
                "do_lora_2": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter_flux_v1"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter_flux_v1(
        self, 
        model, 
        image_emb, 
        ip_scale, 
        start_percent, 
        end_percent, 
        rp_attrs = {}, 
        image_emb_2 = None,
        ip_mask_method = "bbox",
        do_lora_1 = False,
        do_lora_2 = False
    ):
        rp_num = rp_attrs.get('rp_num', 0)
        if image_emb is None and image_emb_2 is None:
            #print("---->>>>>scene image generator !!!!")
            return (model, )
            
        latent_w = int(rp_attrs.get('width', 1280)//16)
        latent_h = int(rp_attrs.get('height', 720)//16)
        rp_split_ratio = rp_attrs.get('rp_split_ratio', 0.5)

        img_emb_device = image_emb.device if image_emb is not None else image_emb_2.device
        img_emb_dtype = image_emb.dtype if image_emb is not None else image_emb_2.dtype

        sr_masks = []
        if ip_mask_method == "split_area":
            
            split_cols = int(latent_w * rp_split_ratio)
            left_mask = torch.zeros((latent_h, latent_w), device = img_emb_device, dtype = img_emb_dtype)
            left_mask[:, 0:split_cols] = 1.0

            right_mask = torch.zeros((latent_h, latent_w), device = img_emb_device, dtype = img_emb_dtype)
            right_mask[:, split_cols: latent_w] = 1.0
            sr_masks.append(left_mask)
            sr_masks.append(right_mask)
        elif ip_mask_method == "bbox":
            sr_bbox = rp_attrs.get('rp_bbox', [])
            if len(sr_bbox) > 0:
                sr_bbox = sorted(sr_bbox, key = lambda item: item[0])
            for sr_box in sr_bbox:
                x_center, y_center, w, h = sr_box
                x1 = int((x_center - w/2)* latent_w)
                y1 = int((y_center - h/2)* latent_h)
                x2 = int((x_center + w/2)* latent_w)
                y2 = int((y_center + h/2)* latent_h)
                mask = torch.zeros((latent_h, latent_w), device = img_emb_device, dtype = img_emb_dtype)
                mask[y1:y2, x1:x2] = 1.0
                sr_masks.append(mask)
        else:
            sr_masks.append(None)
            sr_masks.append(None)
        ip_embeds_mask= []
        if image_emb is not None:
            '''
            if rp_num == 1:
                full_mask = torch.ones((latent_h, latent_w), device = img_emb_device, dtype = img_emb_dtype)
                ip_embeds_mask.append([image_emb, full_mask])
            else:
                ip_embeds_mask.append([image_emb, sr_masks[0]])
            '''
            if do_lora_1:
                image_emb = None
                print("--->>do_lora_1: ", do_lora_1)
            ip_embeds_mask.append([image_emb, sr_masks[0]])
        if image_emb_2 is not None:
            '''
            if rp_num == 1:
                full_mask = torch.ones((latent_h, latent_w), device = img_emb_device, dtype = img_emb_dtype)
                ip_embeds_mask.append([image_emb, full_mask])
            else:
                ip_embeds_mask.append([image_emb_2, sr_masks[1]])
            '''
            if do_lora_2:
                image_emb_2 = None
                print("--->>do_lora_2: ", do_lora_1)
            ip_embeds_mask.append([image_emb_2, sr_masks[1]])
        '''
        latent_size = latent_w * latent_h
        mask_rps = rp_attrs.get('mask_rp', {})

        mask_rp = mask_rps.get(latent_size, [])
        ip_embeds_mask= []
        if image_emb is not None:
            if ip_mask_method == "split_area":
                ip_embeds_mask.append([image_emb, sr_masks[0]])
            else:
                ip_embeds_mask.append([image_emb, mask_rp[1]])
        if image_emb_2 is not None:
            if ip_mask_method == "split_area":
                ip_embeds_mask.append([image_emb_2, sr_masks[1]])
            else:
                ip_embeds_mask.append([image_emb_2, mask_rp[2]])
        '''

        #print("--->>ip_embeds_mask num: ", len(ip_embeds_mask))
        timestep_percent_range = (start_percent, end_percent)
        s = model.model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
        transformer_options = model.model_options.get('transformer_options', {})
        transformer_options = { **transformer_options }
        #rp_attrs.get()

        #transformer_options['image_emb'] = [image_emb]
        transformer_options['image_emb'] = ip_embeds_mask
        transformer_options['ip_scale'] = ip_scale
        transformer_options['timestep_range'] = timestep_range
        transformer_options['use_ipadapter'] = True
        model.model_options['transformer_options'] = transformer_options
        #print("--->>>t1: ", t1-t0, ", t2: ", t2-t1, ", t3: ", t3-t2, ", t4: ", t4-t3)
        return (model,)


class ApplyIPAdapterFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX", ),
                "image": ("IMAGE", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter_flux"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent):
        # convert image to pillow
        pil_image = image.numpy()[0] * 255.0
        pil_image = Image.fromarray(pil_image.astype(np.uint8))
        # initialize ipadapter
        import time
        t0 = time.time()
        ipadapter_flux.init_proj()
        ip_attn_procs = ipadapter_flux.load_ip_adapter(model.model, weight, (start_percent, end_percent))
        t1 = time.time()
        # process control image 
        image_prompt_embeds = ipadapter_flux.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=None
        )
        t2 = time.time()
        # set model
        is_patched = is_model_patched(model.model)
        t3 = time.time()
        bi = model.clone()
        FluxUpdateModules(bi, ip_attn_procs, image_prompt_embeds, is_patched)
        t4 = time.time()

        #print("--->>>t1: ", t1-t0, ", t2: ", t2-t1, ", t3: ", t3-t2, ", t4: ", t4-t3)
        return (bi,)



NODE_CLASS_MAPPINGS = {
    "IPAdapterFluxLoader": IPAdapterFluxLoader,
    "ApplyIPAdapterFlux": ApplyIPAdapterFlux,
    "ConfigureModifiedFlux_V1_Node": ConfigureModifiedFlux_V1_Node,
    "LoadIPAdapterModel": LoadIPAdapterModel,
    "LoadIPAdapterImageProj": LoadIPAdapterImageProj,
    "ApplyIPAdapterFluxV1": ApplyIPAdapterFluxV1,
    "GetIPAadapterEmb": GetIPAadapterEmb,
    "ConfigureModifiedFlux_V2_Node": ConfigureModifiedFlux_V2_Node,
    "IpadapterClipLoader": IpadapterClipLoader,
    "IPAdapterFluxImageProjLoader": IPAdapterFluxImageProjLoader,
    "IPAdapterFluxAdapterLoader": IPAdapterFluxAdapterLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterFluxLoader": "Load IPAdapter Flux Model",
    "ApplyIPAdapterFlux": "Apply IPAdapter Flux Model",
    "ConfigureModifiedFlux_V1_Node": "Configure Modified Flux V1Node",
    "LoadIPAdapterModel": "Load IPAdapter Model",
    "LoadIPAdapterImageProj": "Load IPAdapter Image Proj",
    "ApplyIPAdapterFluxV1": "Apply IPAdapter Flux V1",
    "GetIPAadapterEmb": "Get IPAadapter Emb",
    "ConfigureModifiedFlux_V2_Node": "Configure Modified Flux V2Node",
    "IpadapterClipLoader": "Ipadapter Clip Loader",
    "IPAdapterFluxImageProjLoader": "IPAdapter Flux Image Proj Loader",
    "IPAdapterFluxAdapterLoader": "IPAdapter Flux Adapter Loader"
}

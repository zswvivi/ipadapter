import os
import logging
import math
import time
import copy
from copy import deepcopy

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import folder_paths
from PIL import Image

from .pulid.pulid_pipeline import PuLIDPipeline

pulid_model_path = os.path.join(folder_paths.models_dir, "pulid")
eval_clip_model_path = os.path.join(folder_paths.models_dir, "eva-clip")
facexlib_path = os.path.join(folder_paths.models_dir, 'pulid/facexlib')
face_anti_path = os.path.join(folder_paths.models_dir, "pulid/antelopev2")

folder_paths.add_model_folder_path("pulid", pulid_model_path)
folder_paths.add_model_folder_path("eva-clip", eval_clip_model_path)
folder_paths.add_model_folder_path("pulid/facexlib", facexlib_path)
folder_paths.add_model_folder_path("pulid/antelopev2", face_anti_path)


class PuLIDLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pulid_model_folder": (folder_paths.get_folder_paths("pulid"),),
                #"eva_clip_folder": (folder_paths.get_folder_paths('eva-clip'),),   
            },
            "optional": {
                #"device": (['cuda', 'cpu'], ),
                "device": ("STRING", {"default": "cuda"}),  
            }
        }
    RETURN_TYPES = ("PuLIDModel", "PuLIDModelAttr", )
    FUNCTION = "pulid_loader"
    CATEGORY = "pulid/pulid_loader"

    def pulid_loader(self, pulid_model_folder, device = 'cuda'):
        
        if pulid_model_folder and not pulid_model_folder.startswith("/"):
            pulid_model_folder = os.path.join(folder_paths.models_dir, pulid_model_folder)
        pulid_model_file = os.path.join(pulid_model_path, "pulid_flux_v0.9.1.safetensors")
        print("pulid_model_file: ", pulid_model_file)
        pulid_pipeline = PuLIDPipeline(device = device, dtype = torch.bfloat16)
        pulid_pipeline.load_pretrain(pulid_model_file)

        pulid_model_attr = {
            "dtype": pulid_pipeline.dtype,
            "device": pulid_pipeline.device,
            "pulid_double_interval": pulid_pipeline.pulid_double_interval,
            "pulid_single_interval": pulid_pipeline.pulid_single_interval
        }

        return (pulid_pipeline, pulid_model_attr, )

class PuLIDSigLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parse_model_file": (folder_paths.get_filename_list('pulid/facexlib'),),
                "eva_clip_folder": (folder_paths.get_folder_paths('eva-clip'),), 
                "face_anti_folder": (folder_paths.get_folder_paths('pulid/antelopev2'),), 
            },
            "optional": {
                #"device": (['cuda', 'cpu'], ),
                "device": ("STRING", {"default": "cuda"}), 
            }
        }
    RETURN_TYPES = ("PuLIDSigModel", )
    FUNCTION = "pulid_loader"
    CATEGORY = "pulid/pulid_loader"

    def pulid_loader(self, parse_model_file= '', eva_clip_folder ='', face_anti_folder = '' , device = 'cuda'):
        if eva_clip_folder and not eva_clip_folder.startswith("/"):
            eva_clip_folder = os.path.join(folder_paths.models_dir, eva_clip_folder)

        parse_model_path = os.path.join(facexlib_path, parse_model_file)
        eva_clip_checkpoint_path = os.path.join(eva_clip_folder, "EVA02_CLIP_L_336_psz14_s6B.pt")
        pulid_pipeline = PuLIDPipeline(device = device, dtype = torch.bfloat16)
        if face_anti_folder and not face_anti_folder.startswith("/"):
            face_anti_folder = os.path.join(folder_paths.models_dir, face_anti_folder)
        pulid_pipeline.load_pulid_models(parse_model_path, eva_clip_checkpoint_path, face_anti_folder)
        return (pulid_pipeline, )


class PuLIDCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pulid_sig_model" : ("PuLIDSigModel", ),
            },
            "optional": {
                "images": ("IMAGES", {}),
                "id_idx": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "id_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.05 }),
                "prev_id_cond_pair": ("IDCond", ),
                "precal_pulid_cond": ("CLIP_EMBEDS",),
            }
        }
    RETURN_TYPES = ("IDCond", )
    FUNCTION = "get_pulid_cond"
    CATEGORY = "pulid/get_pulid_cond"

    def _tensor_to_pil(self, tensor):
        tensor = tensor.squeeze(0)
        tensor = tensor.cpu().detach().numpy()
        tensor = (tensor * 255).astype(np.uint8)
        img = Image.fromarray(tensor)
        return img
    '''
    def _tensor_to_pil2(self, tensor_image):
        i = 255.0 * tensor_image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img
    '''
    
    def get_pulid_cond(
        self, 
        pulid_sig_model,  
        images,  
        id_idx,  
        id_weight,  
        prev_id_cond_pair=None,
        precal_pulid_cond=None,
    ):
        prev_id_cond_pair = prev_id_cond_pair or []
        id_cond_pair = [*prev_id_cond_pair]
        if precal_pulid_cond is not None:
            '''
            id_cond, id_vit_hidden, _, _ = precal_pulid_cond['pulid_cond']
            id_cond_pair.append([id_cond, id_vit_hidden, id_idx, id_weight])
            return (id_cond_pair, )
            '''
            if 'pulid_cond' in precal_pulid_cond:
                for each in precal_pulid_cond['pulid_cond']:
                    id_cond0, id_vit_hidden0, _, _ = each
                    id_cond_pair.append([id_cond0, id_vit_hidden0, id_idx, id_weight])
            else:
                #for each in precal_pulid_cond['pulid_encoder']:
                if len(precal_pulid_cond.get('pulid_encoder', [])) > 0:
                    id_cond_pair.append([ None, None, id_idx, id_weight, precal_pulid_cond['pulid_encoder']])
                else:
                    print('--->>>extract id_cond && pulid_encoder is empty !!!!')
                    for img in images:
                        pil_img = self._tensor_to_pil(img)
                        id_cond, id_vit_hidden =  pulid_sig_model.get_id_cond(pil_img, cal_uncond=False)
                        if id_cond is None:
                            print("--->>extra img faceid cond is empty!!!!")
                            continue
                        id_cond_pair.append([id_cond, id_vit_hidden, id_idx, id_weight])
        else:
            for img in images:
                pil_img = self._tensor_to_pil(img)
                id_cond, id_vit_hidden =  pulid_sig_model.get_id_cond(pil_img, cal_uncond=False)
                #print("--->>id_cond: {} ".format(id_cond))
                if id_cond is None:
                    continue
                id_cond_pair.append([id_cond, id_vit_hidden, id_idx, id_weight])
        return (id_cond_pair, )

class PuLIDEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pulid_encoder_model" : ("PuLIDModel", ),
                "id_cond_pairs": ("IDCond", ),
            }
        }
    RETURN_TYPES = ("IDEmebdding", )
    FUNCTION = "get_id_embedding"
    CATEGORY = "pulid/get_id_embedding"
    
    def get_id_embedding(self, pulid_encoder_model, id_cond_pairs):
        #id_embeds = []
        final_id_embedding_pair = []
        id_embed_dict = {}
        cnt = 0
        for id_cond_pair in id_cond_pairs:
            _id_cond = id_cond_pair[0]
            _id_vit_hidden = id_cond_pair[1]
            _id_idx = id_cond_pair[2]
            _id_weight = id_cond_pair[3]
            #torch.set_printoptions(threshold=torch.inf)
            #print('--->>id_cond: ', _id_vit_hidden, cnt)
            #torch.save(_id_cond, f'id_cond_{cnt}.pt')
            if _id_cond is not None:
                cnt += 1
                _id_embedding, _ = pulid_encoder_model.pulid_id_encoder(_id_cond, _id_vit_hidden,cal_uncond = False)
            else:
                precal_id_embedding = id_cond_pair[4]
                if len(precal_id_embedding) >0:
                    _id_embedding = precal_id_embedding[0][0]
                else:
                    print('offline feature exception !!!')
            print("--->>id_embedding: ", _id_embedding.shape)
                #print("_id_embedding: ", _id_embedding)
            #print("--->>_id_embedding: ",_id_embedding, cnt)
            if _id_idx not in id_embed_dict:
                id_embed_dict[_id_idx] = [[_id_embedding, _id_idx, _id_weight]]
            else:
                id_embed_dict[_id_idx].append([_id_embedding, _id_idx, _id_weight])
        #print("id_embed_dict: ", id_embed_dict)
        for id_idx, id_embedding_pairs in id_embed_dict.items():
            id_embeds = []
            id_weight = 1.0
            for id_embedding_pair in id_embedding_pairs:
                id_embeds.append(id_embedding_pair[0])
                id_weight = id_embedding_pair[2]
            
            if len(id_embeds) > 0:
                id_embedding = torch.mean(torch.cat(id_embeds, dim =0), dim = 0).unsqueeze(0)
                final_id_embedding_pair.append([id_embedding, id_idx, id_weight])

        print('--->>final_id_embedding_pair: ', final_id_embedding_pair)
        return (final_id_embedding_pair, )

class GetIdEmbedByIdx:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id_embeds": ("IDEmebdding", ),
            },
            "optional": {
                "id_idx": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "id_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            }
        }
    RETURN_TYPES = ("IDEmebdding", )
    FUNCTION = "get_pulid_cond_byidx"
    CATEGORY = "pulid/get_pulid_cond_byidx"

    def get_pulid_cond_byidx(self, id_embeds, id_idx, id_weight):
        id_embed_byidx = []
        for id_embed in id_embeds:
            if len(id_embed) ==3:
                if id_embed[1] == id_idx:
                    #id_embed_byidx.append(id_embed)
                    id_embed_byidx.append([id_embed[0], id_embed[1], id_weight])
        return (id_embed_byidx, )

class ConfigClear:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
            },
            "optional": {
                "ipadapter": ("BOOLEAN", {"default": True}),
                "pulid": ("BOOLEAN", {"default": True}),
                "noise": ("NOISE",),
            }
        }
    RETURN_TYPES = ("MODEL", )
    FUNCTION = "execute"
    CATEGORY = "pulid/config_clear"
    '''
    @classmethod
    def IS_CHANGED(*args, **kwargs):
        return float("NaN")
    '''

    def execute(self, model, ipadapter = True, pulid = True, noise = None):
        transformer_options = model.model_options.get('transformer_options', {})
        transformer_options = { **transformer_options }
        if ipadapter:
            transformer_options['image_emb'] = []
            transformer_options['timestep_range'] = None
            transformer_options['use_ipadapter'] = False
        if pulid:
            transformer_options['id_embedding_pair'] = []
            transformer_options['pulid_timestep_range'] = None
            transformer_options['use_ipadapter'] = False
        model.model_options['transformer_options'] = transformer_options
        #print("!!!!!!!!!!!-->>transformer_options: ",transformer_options, ipadapter, pulid, " <<<<<<<--------!!!!!!")
        return (model, )

class PulidInjectFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "pulid_model_attr": ("PuLIDModelAttr", )
            },
            "optional": {
                "rp_attrs": ("RP_ATTRS", {}),
                "id_embedding_pair": ("IDEmebdding", ),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "id_mask_method": (["bbox", "split_area", "all"],),
                "use_ipadapter": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL", )
    FUNCTION = "pulid_inject_flux"
    CATEGORY = "pulid/pulid_inject_flux"

    def pulid_inject_flux(self, model ,pulid_model_attr, rp_attrs, id_embedding_pair = [], start_percent = 0.0 , end_percent = 1.0, id_mask_method = 'bbox', use_ipadapter = True):
        if len(id_embedding_pair) == 0:
            transformer_options = model.model_options.get('transformer_options', {})
            transformer_options = { **transformer_options }
            transformer_options['id_embedding_pair'] = []
            transformer_options['use_ipadapter'] = use_ipadapter
            model.model_options['transformer_options'] = transformer_options
            return (model, )
        timestep_percent_range = (start_percent, end_percent)
        s = model.model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))

        latent_w = int(rp_attrs.get('width', 1280)//16)
        latent_h = int(rp_attrs.get('height', 720)//16)
        rp_split_ratio = rp_attrs.get('rp_split_ratio', 0.5)

        #prepare sr mask
        sr_masks = []
        if id_mask_method == "split_area":
            sr_masks = []
            split_cols = int(latent_w * rp_split_ratio)
            left_mask = torch.zeros((latent_h, latent_w), device = pulid_model_attr['device'], dtype = pulid_model_attr['dtype'])
            left_mask[:, 0:split_cols] = 1.0

            right_mask = torch.zeros((latent_h, latent_w), device = pulid_model_attr['device'], dtype = pulid_model_attr['dtype'])
            right_mask[:, split_cols: latent_w] = 1.0
            sr_masks.append(left_mask)
            sr_masks.append(right_mask)
        elif id_mask_method == "bbox":
            sr_bbox = rp_attrs.get('rp_bbox', [])
            if len(sr_bbox) > 0:
                sr_bbox = sorted(sr_bbox, key = lambda item: item[0])
            for sr_box in sr_bbox:
                x_center, y_center, w, h = sr_box
                x1 = int((x_center - w/2)* latent_w)
                y1 = int((y_center - h/2)* latent_h)
                x2 = int((x_center + w/2)* latent_w)
                y2 = int((y_center + h/2)* latent_h)
                mask = torch.zeros((latent_h, latent_w), device = pulid_model_attr['device'], dtype = pulid_model_attr['dtype'])
                mask[y1:y2, x1:x2] = 1.0
                sr_masks.append(mask)
        else:
            for id in range(10):
                sr_masks.append(None)
        id_emb_struct_data = []
        for _id_embedding in id_embedding_pair:
            id_embedding = _id_embedding[0]
            id_idx = _id_embedding[1]
            id_weight = _id_embedding[2]
            id_mask = sr_masks[id_idx]
            id_emb_struct_data.append([id_embedding, id_idx, id_weight,  id_mask])
        
        transformer_options = model.model_options.get('transformer_options', {})
        transformer_options = { **transformer_options }
        transformer_options['id_embedding_pair'] = id_emb_struct_data
        #transformer_options['pulid_ca'] = pulid_model.pulid_ca
        transformer_options['pulid_double_interval'] = pulid_model_attr['pulid_double_interval']
        transformer_options['pulid_single_interval'] = pulid_model_attr['pulid_single_interval']
        transformer_options['pulid_timestep_range'] = timestep_range
        transformer_options['use_ipadapter'] = use_ipadapter
        model.model_options['transformer_options'] = transformer_options
        return (model,)

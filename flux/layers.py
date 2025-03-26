import torch
from torch import Tensor, nn

from .math import attention, rp_attention, rp_attention_xformers
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from ..attention_processor import IPAFluxAttnProcessor2_0
import comfy.model_management
from types import MethodType
from typing import List, Dict, Optional, Tuple


from comfy.ldm.flux import math as flux_math
from xformers.ops import memory_efficient_attention as xattention
from einops import rearrange


def xformers_atten(q: Tensor, k: Tensor, v: Tensor, pe: Tensor,
                            attn_mask: Optional[Tensor] = None,
                            mask: Optional[Tensor] = None) -> Tensor:
    q, k = flux_math.apply_rope(q, k, pe)
    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")
    
    # Use attn_mask if provided, otherwise use the mask parameter
    attention_bias = attn_mask if attn_mask is not None else mask
    
    if attention_bias is not None:
        x = xattention(q, k, v, attn_bias=attention_bias)
    else:
        x = xattention(q, k, v)
        
    x = rearrange(x, "B L H D -> B L (H D)")
    return x

class DoubleStreamBlockIPA(nn.Module):
    def __init__(self, original_block: DoubleStreamBlock, ip_adapter: list[IPAFluxAttnProcessor2_0], image_emb):
        super().__init__()

        mlp_hidden_dim  = original_block.img_mlp[0].out_features
        mlp_ratio = mlp_hidden_dim / original_block.hidden_size
        mlp_hidden_dim = int(original_block.hidden_size * mlp_ratio)
        self.num_heads = original_block.num_heads
        self.hidden_size = original_block.hidden_size
        self.img_mod = original_block.img_mod
        self.img_norm1 = original_block.img_norm1
        self.img_attn = original_block.img_attn

        self.img_norm2 = original_block.img_norm2
        self.img_mlp = original_block.img_mlp

        self.txt_mod = original_block.txt_mod
        self.txt_norm1 = original_block.txt_norm1
        self.txt_attn = original_block.txt_attn

        self.txt_norm2 = original_block.txt_norm2
        self.txt_mlp = original_block.txt_mlp

        self.ip_adapter = ip_adapter
        self.image_emb = image_emb
        self.device = comfy.model_management.get_torch_device()

    def add_adapter(self, ip_adapter: IPAFluxAttnProcessor2_0, image_emb):
        self.ip_adapter.append(ip_adapter)
        self.image_emb.append(image_emb)
    
    #def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, t: Tensor, transformer_options: dict = {}):
    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, timestep: Tensor, transformer_options: dict = {}):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        image_emb = transformer_options.get('image_emb', [])
        if len(image_emb) == 0:
            image_emb = self.image_emb
        ip_scale = transformer_options.get('ip_scale', 0.0)
        timestep_range = transformer_options.get('timestep_range', None)
        use_ipadapter = transformer_options.get('use_ipadapter', False)
        is_base = transformer_options.get('is_base', False)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        '''
        attn = attention(torch.cat((txt_q, img_q), dim=2),
                         torch.cat((txt_k, img_k), dim=2),
                         torch.cat((txt_v, img_v), dim=2), pe=pe)
        '''
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        mask_fn = transformer_options.get('patches_replace', {}).get(f'double', {}).get(('mask_fn', self.idx), None) 
        mask = None
        if mask_fn is not None:
            mask = mask_fn(q, transformer_options, txt.shape[1])

        if is_base:
            attn = rp_attention(q, k, v, pe=pe)
        else:
            attn = rp_attention(q, k, v, pe=pe, mask = mask)

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        if use_ipadapter:
            if len(self.ip_adapter) == len(image_emb):
                #for adapter, image in zip(self.ip_adapter, self.image_emb):
                for adapter, image in zip(self.ip_adapter, image_emb):
                    # this does a separate attention for each adapter
                    if isinstance(image, list):
                        _img_emb = image[0]
                        _img_mask = image[1]
                        if _img_emb is None:
                            continue
                        adapter.set_ip_scale(ip_scale, timestep_range)
                        ip_hidden_states = adapter(self.num_heads, img_q, _img_emb, timestep)
                        img_attn_device = img_attn.device
                        if ip_hidden_states is not None:
                            #ip_hidden_states = ip_hidden_states.to(self.device)
                            ip_hidden_states = ip_hidden_states.to(img_attn_device)
                            img_mask = _img_mask.reshape(1, ip_hidden_states.shape[1]).unsqueeze(-1).repeat(1,1, ip_hidden_states.shape[2]).to(device=img_attn_device, dtype = ip_hidden_states.dtype)
                            #print(f"--->>double ip_hidden_states+++111: {ip_hidden_states.shape}, img_attn: {img_attn.shape}, img_mask: {_img_mask.shape}, {img_mask.shape}")
                            img_attn = img_attn + ip_hidden_states * img_mask
                    else:
                        if image is not None:
                            adapter.set_ip_scale(ip_scale, timestep_range)
                            ip_hidden_states = adapter(self.num_heads, img_q, image, timestep)
                            img_attn_device = img_attn.device
                            if ip_hidden_states is not None:
                                ip_hidden_states = ip_hidden_states.to(img_attn_device)
                                img_attn = img_attn + ip_hidden_states
            else:
                for img_emb in image_emb:
                    # this does a separate attention for each adapter
                    _img_emb = img_emb[0]
                    _img_mask = img_emb[1]
                    if _img_emb is None:
                        continue
                    if _img_mask is not None:
                        self.ip_adapter[0].set_ip_scale(ip_scale, timestep_range)
                        ip_hidden_states = self.ip_adapter[0](self.num_heads, img_q, _img_emb, timestep)
                        img_attn_device = img_attn.device
                        if ip_hidden_states is not None:
                            #ip_hidden_states = ip_hidden_states.to(self.device)
                            ip_hidden_states = ip_hidden_states.to(img_attn_device)
                            img_mask = _img_mask.reshape(1, ip_hidden_states.shape[1]).unsqueeze(-1).repeat(1,1, ip_hidden_states.shape[2]).to(device=img_attn_device, dtype = ip_hidden_states.dtype)
                            #print(f"--->>double ip_hidden_states+++222: {ip_hidden_states.shape}, img_attn: {img_attn.shape}, img_mask: {_img_mask.shape}, {img_mask.shape}")
                            img_attn = img_attn + ip_hidden_states * img_mask

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt

class SingleStreamBlockIPA(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, original_block: SingleStreamBlock, ip_adapter: list[IPAFluxAttnProcessor2_0], image_emb):
        super().__init__()
        self.hidden_dim = original_block.hidden_size
        self.num_heads = original_block.num_heads
        self.scale = original_block.scale

        self.mlp_hidden_dim = original_block.mlp_hidden_dim
        # qkv and mlp_in
        self.linear1 = original_block.linear1
        # proj and mlp_out
        self.linear2 = original_block.linear2

        self.norm = original_block.norm

        self.hidden_size = original_block.hidden_size
        self.pre_norm = original_block.pre_norm

        self.mlp_act = original_block.mlp_act
        self.modulation = original_block.modulation

        self.ip_adapter = ip_adapter
        self.image_emb = image_emb
        self.device = comfy.model_management.get_torch_device()

    def add_adapter(self, ip_adapter: IPAFluxAttnProcessor2_0, image_emb):
        self.ip_adapter.append(ip_adapter)
        self.image_emb.append(image_emb)

    #def forward(self, x: Tensor, vec: Tensor, pe: Tensor, t:Tensor, transformer_options: dict = {}) -> Tensor:
    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, timestep:Tensor, transformer_options: dict = {}) -> Tensor:
        image_emb = transformer_options.get('image_emb', [])
        ip_scale = transformer_options.get('ip_scale', 0.0)
        timestep_range = transformer_options.get('timestep_range', None)
        use_ipadapter = transformer_options.get('use_ipadapter', False)
        is_base = transformer_options.get('is_base', False)

        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        mask_fn = transformer_options.get('patches_replace', {}).get(f'single', {}).get(('mask_fn', self.idx), None) 
        mask = None
        
        if mask_fn is not None:
            mask = mask_fn(q, transformer_options, transformer_options['txt_size'])
        
        if is_base:
            attn = rp_attention(q, k, v, pe=pe)
        else:
            attn = rp_attention(q, k, v, pe=pe, mask = mask)

        if len(image_emb) == 0:
            image_emb = self.image_emb
                
        if use_ipadapter:
            if len(self.ip_adapter) == len(image_emb):
                #for adapter, image in zip(self.ip_adapter, self.image_emb):
                for adapter, image in zip(self.ip_adapter, image_emb):
                    # this does a separate attention for each adapter
                    # maybe we want a single joint attention call for all adapters?
                    if isinstance(image, list):
                        _img_emb = image[0]
                        _img_mask = image[1]
                        if _img_emb is None:
                            continue
                        latent_h, latent_w = _img_mask.shape
                        adapter.set_ip_scale(ip_scale, timestep_range)
                        ip_hidden_states = adapter(self.num_heads, q, _img_emb, timestep)
                        img_attn_device = attn.device
                        if ip_hidden_states is not None:
                            ip_hidden_states = ip_hidden_states.to(img_attn_device)
                            '''
                            img_mask = _img_mask.reshape(ip_hidden_states.shape[0], latent_h* latent_w).unsqueeze(-1).repeat(1,1, ip_hidden_states.shape[2]).to(device=ip_hidden_states.device, dtype = ip_hidden_states.dtype)
                            ip_hidden_states[:, ip_hidden_states.shape[1]-latent_h*latent_w:, :] = ip_hidden_states[:, ip_hidden_states.shape[1]-latent_h*latent_w:, :]* img_mask
                            '''
                            empty_mask = torch.zeros((ip_hidden_states.shape[0],ip_hidden_states.shape[1]-latent_h*latent_w), dtype = _img_mask.dtype, device= _img_mask.device)
                            img_mask = torch.cat([
                                empty_mask,
                                _img_mask.reshape(ip_hidden_states.shape[0], latent_h* latent_w)],
                                dim=1
                                ).unsqueeze(-1).repeat(1,1, ip_hidden_states.shape[2]).to(device=img_attn_device, dtype = ip_hidden_states.dtype)
                            ip_hidden_states = ip_hidden_states* img_mask
                            
                            attn = attn + ip_hidden_states
                    else:
                        if image is not None:
                            adapter.set_ip_scale(ip_scale, timestep_range)
                            ip_hidden_states = adapter(self.num_heads, q, image, timestep)
                            img_attn_device = attn.device
                            if ip_hidden_states is not None:
                                ip_hidden_states = ip_hidden_states.to(img_attn_device)
                                attn = attn + ip_hidden_states
            else:
                for img_emb in image_emb:
                    _img_emb = img_emb[0]
                    _img_mask = img_emb[1]
                    if _img_emb is None:
                        continue
                    if _img_mask is not None:
                        latent_h, latent_w = _img_mask.shape
                        # this does a separate attention for each adapter
                        # maybe we want a single joint attention call for all adapters?
                        self.ip_adapter[0].set_ip_scale(ip_scale, timestep_range)

                        ip_hidden_states = self.ip_adapter[0](self.num_heads, q, _img_emb, timestep)
                        img_attn_device = attn.device
                        if ip_hidden_states is not None:
                            ip_hidden_states = ip_hidden_states.to(img_attn_device)
                            '''
                            img_mask = _img_mask.reshape(ip_hidden_states.shape[0], latent_h* latent_w).unsqueeze(-1).repeat(1,1, ip_hidden_states.shape[2]).to(device=ip_hidden_states.device, dtype = ip_hidden_states.dtype)
                            ip_hidden_states[:, ip_hidden_states.shape[1]-latent_h*latent_w:, :] = ip_hidden_states[:, ip_hidden_states.shape[1]-latent_h*latent_w:, :]* img_mask
                            '''
                            empty_mask = torch.zeros((ip_hidden_states.shape[0],ip_hidden_states.shape[1]-latent_h*latent_w), dtype = _img_mask.dtype, device= _img_mask.device)
                            img_mask = torch.cat([
                                empty_mask,
                                _img_mask.reshape(ip_hidden_states.shape[0], latent_h* latent_w)],
                                dim=1
                                ).unsqueeze(-1).repeat(1,1, ip_hidden_states.shape[2]).to(device=img_attn_device, dtype = ip_hidden_states.dtype)
                            ip_hidden_states = ip_hidden_states* img_mask
                            
                            attn = attn + ip_hidden_states
        
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x

def inject_blocks(diffusion_model):
    for i, block in enumerate(diffusion_model.double_blocks):
        block.__class__ = DoubleStreamBlockIPA
        block.idx = i

    for i, block in enumerate(diffusion_model.single_blocks):
        block.__class__ = SingleStreamBlockIPA
        block.idx = i
    return diffusion_model

def inject_ipadapter_blocks(model, ip_attn_procs):
    flux_model = model.model
    #model.add_object_patch(f"diffusion_model.forward_orig", MethodType(flux_model.diffusion_model.forward_orig_ipa, flux_model.diffusion_model))
    for i, original in enumerate(flux_model.diffusion_model.double_blocks):
        patch_name = f"double_blocks.{i}"
        maybe_patched_layer = model.get_model_object(f"diffusion_model.{patch_name}")
        procs = [ip_attn_procs[patch_name]]
        embs = [None]
        #print("patched_layer: ", type(maybe_patched_layer), maybe_patched_layer)
        if isinstance(maybe_patched_layer, DoubleStreamBlockIPA):
            procs = maybe_patched_layer.ip_adapter + procs
            embs = maybe_patched_layer.image_emb + embs
        
        new_layer = DoubleStreamBlockIPA(original, procs, embs)
        new_layer.idx = i
        model.add_object_patch(f"diffusion_model.{patch_name}", new_layer)
    
    for i, original in enumerate(flux_model.diffusion_model.single_blocks):
        patch_name = f"single_blocks.{i}"
        maybe_patched_layer = model.get_model_object(f"diffusion_model.{patch_name}")
        procs = [ip_attn_procs[patch_name]]
        embs = [None]
        #print("patched_layer: ", type(maybe_patched_layer), maybe_patched_layer)
        if isinstance(maybe_patched_layer, SingleStreamBlockIPA):
            procs = maybe_patched_layer.ip_adapter + procs
            embs = maybe_patched_layer.image_emb + embs
        
        new_layer = SingleStreamBlockIPA(original, procs, embs)
        new_layer.idx = i
        model.add_object_patch(f"diffusion_model.{patch_name}", new_layer)

    return model
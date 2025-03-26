#Original code can be found on: https://github.com/black-forest-labs/flux
from typing import Any, Dict, List
import torch
from torch import Tensor, nn

from comfy.ldm.flux.layers import timestep_embedding, DoubleStreamBlock, SingleStreamBlock
from comfy.ldm.flux.model import Flux as OriginalFlux
from .layers import DoubleStreamBlockIPA, SingleStreamBlockIPA
import numpy as np
from einops import rearrange, repeat
import comfy.ldm.common_dit
import time

class Flux(OriginalFlux):
    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control=None,
        transformer_options = {},
        #ref_config: Dict[Any, Any] | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for i, block in enumerate(self.double_blocks):
            
            #img, txt = block(img=img, txt=txt, vec=vec, pe=pe, ref_config=ref_config, timestep=timesteps, transformer_options=transformer_options)
            #img, txt = block(img=img, txt=txt, vec=vec, pe=pe, timestep=timesteps, transformer_options=transformer_options)
            if isinstance(block, DoubleStreamBlock):
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            else:
                #print("transformer_options: {}".format(transformer_options.keys()))
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, timestep = timesteps, transformer_options=transformer_options)
                #img, txt = block(img=img, txt=txt, vec=vec, pe=pe, timestep = timesteps)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:1] += add
                        if ref_pes is not None:
                            img[1:] += add
                        #     img[-1:, txt.shape[1] :txt.shape[1]+4096, ...] += add
                        #     img[:, 4096*1:4096*2] += add
                            # img[:, 4096*2:4096*3] += add

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            #img = block(img, vec=vec, pe=pe, ref_config=None, timestep=timesteps, transformer_options=transformer_options)
            #img = block(img, vec=vec, pe=pe, timestep=timesteps, transformer_options=transformer_options)
            if isinstance(block, SingleStreamBlock):
                img = block(img, vec=vec, pe=pe)
            else:
                #print("transformer_options: {}".format(transformer_options.keys()))
                img = block(img, vec=vec, pe=pe, timestep=timesteps, transformer_options=transformer_options)
                #img = block(img, vec=vec, pe=pe, timestep=timesteps)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:1, txt.shape[1] :, ...] += add
                        # img[:1, txt.shape[1] :txt.shape[1]+4096, ...] += add
                        if ref_pes is not None:
                            img[-1:, txt.shape[1] :, ...] += add
                        #     img[:, txt.shape[1]+4096*1 :txt.shape[1]+4096*2, ...] += add
                            # img[:, txt.shape[1]+4096*2 :txt.shape[1]+4096*3, ...] += add
                        
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
    
    def _get_img_ids(self, x, bs, h_len, w_len, h_start, h_end, w_start, w_end):
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(h_start, h_end - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(w_start, w_end - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        return img_ids

    def forward_orig_ipa(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        #region_txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor|None = None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        
        #pulid_ca = transformer_options.get('pulid_ca', None)
        pulid_double_interval = transformer_options.get('pulid_double_interval', 2)
        pulid_single_interval = transformer_options.get('pulid_single_interval', 4)
        pulid_timestep_range = transformer_options.get('pulid_timestep_range', None)
        id_embedding_pair = transformer_options.get('id_embedding_pair', [])

        ca_idx = 0
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})

        # enable teacache
        if self.enable_cache:
            inp = img.clone()
            vec_ = vec.clone()
            img_mod1, _ = self.double_blocks[0].img_mod(vec_)
            modulated_inp = self.double_blocks[0].img_norm1(inp)
            modulated_inp = (1 + img_mod1.scale) * modulated_inp + img_mod1.shift

            if self.cnt == 0 or self.cnt == self.steps - 1:
                should_calc = True
                self.accumulated_rel_l1_distance  = 0
            else:
                #print("--->>modulated_inp: {}, self.previous_modulated_input: {}".format(modulated_inp.shape, self.previous_modulated_input.shape))
                if modulated_inp.shape  ==  self.previous_modulated_input.shape:
                    coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
                    rescale_func = np.poly1d(coefficients)
                    self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_rel_l1_distance = 0
                else:
                    should_calc = True
                    self.cnt = self.steps -1
            
            self.previous_modulated_input = modulated_inp
            self.cnt += 1
            if self.cnt == self.steps:
                self.cnt = 0
            #print("--->>self.enable_cache: ", self.enable_cache, ", should_calc: ", should_calc, ", self.cnt: ", self.cnt, ", timesteps: ", timesteps[0])
        else:
            should_calc = True

        #print("--->>id_embedding_pair: ", len(id_embedding_pair), ", pulid_timestep_range: ", pulid_timestep_range, ", should_calc: ", should_calc, ", self.cnt: ", self.cnt, ", timesteps: ", timesteps[0], ", self.rel_l1_thres: ", self.rel_l1_thresh)
        t0 = float(timesteps[0])

        start_time = time.time()
        if not should_calc:
            img += self.previous_residual
        else:
            ori_img = img.clone()
            for i, block in enumerate(self.double_blocks):
                t01 = time.time()
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        if isinstance(block, DoubleStreamBlockIPA): # ipadaper 
                            out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], timestep=args["timesteps"], transformer_options = args["transformer_options"])
                        else:
                            out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"])
                        return out
                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "timesteps": timesteps, "transformer_options": transformer_options}, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    if isinstance(block, DoubleStreamBlockIPA): # ipadaper 
                        img, txt = block(img=img, txt=txt, vec=vec, pe=pe, timestep=timesteps, transformer_options = transformer_options)
                        #img, txt = block(img=img, txt=txt, vec=vec, pe=pe, timestep=timesteps)
                    else:
                        img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
                
                t02 = time.time()
                #t0 = float(timesteps[0])
                t021 = time.time()
                if (self.pulid_ca is not None) and (pulid_timestep_range is not None) and (t0 <= pulid_timestep_range[0] and t0 >= pulid_timestep_range[1]):
                    if len(id_embedding_pair) >0 and i % pulid_double_interval == 0:
                        #print("---->>>pulid_double")
                        device = img.device
                        for _id_embedding in id_embedding_pair:
                            id_emb = _id_embedding[0]
                            pulid_ca_device = id_emb.device
                            img = img.to(pulid_ca_device)
                            id_weight = _id_embedding[2]
                            id_mask = _id_embedding[3]
                            if id_mask is not None:
                                _id_mask = id_mask.reshape(1, img.shape[1]).unsqueeze(-1).repeat(1,1, img.shape[2]).to(device= pulid_ca_device, dtype = img.dtype)
                                img  = img + self.pulid_ca[ca_idx](id_emb, img) * _id_mask * id_weight
                            else:
                                #print('double id_emb: ', id_emb, ", id_weight: ", id_weight)
                                img  = img + self.pulid_ca[ca_idx](id_emb, img)  * id_weight
                        img = img.to(device)
                        ca_idx += 1
                t03 = time.time()
                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add
                t04 = time.time()

            img = torch.cat((txt, img), 1)

            for i, block in enumerate(self.single_blocks):
                ts01 = time.time()
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        if isinstance(block, SingleStreamBlockIPA): # ipadaper
                            out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], timestep=args["timesteps"], transformer_options = args["transformer_options"])
                        else:
                            out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"])
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "timesteps": timesteps, "transformer_options": transformer_options}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    if isinstance(block, SingleStreamBlockIPA): # ipadaper
                        img = block(img, vec=vec, pe=pe, timestep=timesteps, transformer_options = transformer_options)
                        #img = block(img, vec=vec, pe=pe, timestep=timesteps)
                    else:
                        img = block(img, vec=vec, pe=pe)
                
                ts02 = time.time()
                #t0 = float(timesteps[0])
                if (self.pulid_ca is not None) and (pulid_timestep_range is not None) and (t0 <= pulid_timestep_range[0] and t0 >= pulid_timestep_range[1]):
                    if len(id_embedding_pair) >0 and i % pulid_single_interval == 0:
                        txt, img = img[:, : txt.shape[1], ...], img[:, txt.shape[1] :, ...]
                        device = img.device
                        for _id_embedding in id_embedding_pair:
                            id_emb = _id_embedding[0]
                            pulid_ca_device = id_emb.device
                            img = img.to(pulid_ca_device)
                            id_weight = _id_embedding[2]
                            id_mask = _id_embedding[3]
                            if id_mask is not None:
                                _id_mask = id_mask.reshape(1, img.shape[1]).unsqueeze(-1).repeat(1,1, img.shape[2]).to(device=pulid_ca_device, dtype = img.dtype)
                                img = img + self.pulid_ca[ca_idx](id_emb, img)* _id_mask * id_weight
                            else:
                                #print('single id_emb: ', id_emb, ", id_weight: ", id_weight)
                                img = img + self.pulid_ca[ca_idx](id_emb, img) * id_weight
                        img = img.to(device)
                        ca_idx += 1
                        img = torch.cat([txt, img], dim = 1)
                ts03 = time.time()
                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add
                ts04 = time.time()

            img = img[:, txt.shape[1] :, ...]
            if self.enable_cache:
                self.previous_residual = img - ori_img

        t1 = time.time()
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        transformer_options['original_shape'] = x.shape
        patch_size = 2
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))
        transformer_options['patch_size'] = patch_size
        rp_weight = transformer_options.get('rp_weight', -1.0)
        #transformer_options.pop('empty_mask')

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        region_context = None

        regional_conditioning = transformer_options.get('patches', {}).get('regional_conditioning', None)
        if regional_conditioning is not None:
            region_cond = regional_conditioning[0](transformer_options)
            if region_cond is not None:
                if rp_weight <0.0:
                    context = torch.cat([context, region_cond.to(context.dtype)], dim=1)
                else:
                    region_context = region_cond.to(context.dtype)
        
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        img_ids_orig = self._get_img_ids(x, bs, h_len, w_len, 0, h_len, 0, w_len)

        transformer_options['txt_size'] = context.shape[1]
        '''
        out = self.forward_orig(
            img, 
            img_ids_orig, 
            context, 
            txt_ids, 
            timestep, 
            y, 
            guidance, 
            control, 
            transformer_options=transformer_options
        )
        '''
        base_transformer_options = transformer_options
        if rp_weight >= 0.0:
            base_transformer_options['is_base'] = True
            base_transformer_options['txt_size'] = transformer_options['txt_size']
            base_transformer_options['original_shape'] = transformer_options['original_shape']
            base_transformer_options['patch_size'] = transformer_options['patch_size']

        out = self.forward_orig_ipa(
            img, 
            img_ids_orig, 
            context, 
            txt_ids, 
            timestep, 
            y, 
            guidance, 
            control, 
            transformer_options = base_transformer_options, 
        )

        #print("-->>timestemp: {}, out: {}".format(timestep, out))
        if region_context is not None:
            transformer_options['is_base'] = False
            regional_txt_ids = torch.zeros((bs, region_context.shape[1], 3), device=x.device, dtype=x.dtype)
            out_rp = self.forward_orig_ipa(
                img, 
                img_ids_orig, 
                #context, 
                region_context,
                #txt_ids, 
                regional_txt_ids,
                timestep, 
                y, 
                guidance, 
                None, 
                transformer_options=transformer_options
            )

            out = out* (1-rp_weight)+ out_rp* rp_weight
        
        #out = out[:-1]
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]


def inject_flux(diffusion_model: OriginalFlux):
    diffusion_model.__class__ = Flux
    '''
    diffusion_model.__class__.enable_cache = enable_cache
    if enable_cache:
        diffusion_model.__class__.cnt = 0
        diffusion_model.__class__.rel_l1_thresh = rel_l1_thresh
        diffusion_model.__class__.steps = steps
    '''
    # diffusion_model.is_ref = True
    return diffusion_model
'''

def inject_flux(diffusion_model, pulid_model):
    diffusion_model.__class__ = Flux
    diffusion_model.pulid_ca = pulid_model.pulid_ca
    return diffusion_model
'''
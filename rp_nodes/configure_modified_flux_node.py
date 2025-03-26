from ..flux.model import inject_flux
from ..flux.layers import inject_blocks, inject_ipadapter_blocks
import gc
import torch

class ConfigureModifiedFluxNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
        }}
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "fluxtapoz"
    FUNCTION = "apply"

    def apply(self, model):
        inject_flux(model.model.diffusion_model)
        inject_blocks(model.model.diffusion_model)
        return (model,)

class CacheConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL",),
            },
            "optional": {
                "cache_enable": ("BOOLEAN", {"default": False}),
                "cache_steps": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "rel_l1_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "fluxtapoz/flux_cache"
    FUNCTION = "apply"
    '''
    @classmethod
    def IS_CHANGED(*args, **kwargs):
        return float("NaN")
    '''

    def apply(self, model, cache_enable = False, cache_steps = 20, rel_l1_thresh = 0.4):
        model.model.diffusion_model.__class__.enable_cache = cache_enable
        model.model.diffusion_model.__class__.cnt = 0
        model.model.diffusion_model.__class__.rel_l1_thresh = rel_l1_thresh
        model.model.diffusion_model.__class__.steps = cache_steps
            
        #print("--->>cache_enable: ", cache_enable, ", cnt: ", model.model.diffusion_model.__class__.cnt)
        return (model, )


class InjectFluxIPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL",),
                "ip_attn_procs": ("IP_ATTN_PROCS", ),
            }
        }
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "flux/flux_ipadapter"
    FUNCTION = "apply"

    def apply(self, model, ip_attn_procs):
        '''
        transformer_options = model.model_options.get('transformer_options', {})
        transformer_options = {** transformer_options }
        transformer_options['use_ipadapter'] = use_ipadapter
        model.model_options['transformer_options'] = transformer_options
        '''

        model_clone = model.clone()
        inject_flux(model_clone.model.diffusion_model)
        inject_ipadapter_blocks(model_clone, ip_attn_procs)   

        '''
        transformer_options = model_clone.model_options.get('transformer_options', {})
        transformer_options = {** transformer_options }
        transformer_options['use_ipadapter'] = use_ipadapter
        model_clone.model_options['transformer_options'] = transformer_options
        '''
        
        del ip_attn_procs
        gc.collect()
        torch.cuda.empty_cache()
        
        return (model_clone, )


class InjectFluxPulidAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL",),
            },
            "optional": {
                "pulid_model": ("PuLIDModel",),
                "enable_inject_pulid": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "flux/flux_pulid"
    FUNCTION = "apply"

    def apply(self, model, pulid_model = None, enable_inject_pulid = False):
        if enable_inject_pulid:
            model.model.diffusion_model.__class__.pulid_ca = pulid_model.pulid_ca
        else:
            model.model.diffusion_model.__class__.pulid_ca = None

        return (model, )


class DoLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "lora_model": ("STRING", {"default": ""}), 
            }
        }
    RETURN_TYPES = ("BOOLEAN",)

    CATEGORY = "flux/lora_ref"
    FUNCTION = "apply"

    def apply(self, lora_model):
        flag = False
        if lora_model != '':
            flag = True
        #print("-->>islora: ", flag)
        return (flag, )

class RPParam:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ip_adapter_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "ip_adapter_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "region_prompt_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("RP_Param", )
    RETURN_NAMES = ("rp_param",  )

    CATEGORY = "flux/rp_param"
    FUNCTION = "apply"
    def apply(self, ip_adapter_scale = 0.5, ip_adapter_ratio = 0.1, region_prompt_ratio = 0.2):
        rp_param = {
            "ip_adapter_scale": ip_adapter_scale,
            "ip_adapter_ratio": ip_adapter_ratio,
            "region_prompt_ratio": region_prompt_ratio
        }
        return (rp_param, )

class ModifyConfigParam:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "is_lora_1": ("BOOLEAN", {"default": False}),
                "is_lora_2": ("BOOLEAN", {"default": False}),
                "rp_num": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            },
            "optional": {
                "rp_param": ("RP_Param", ),
                "modify_rp_param": ("RP_Param", ),
            }
        }
    RETURN_TYPES = ("BOOLEAN","FLOAT", "FLOAT", "FLOAT", )
    RETURN_NAMES = ("is_modify_param", "ip_adapter_scale", "ip_adapter_ratio", "region_prompt_ratio", )

    CATEGORY = "flux/modify_config_param"
    FUNCTION = "apply"

    def apply(self, is_lora_1, is_lora_2, rp_num, rp_param = {}, modify_rp_param = {}):
        is_modify_param = False
        is_ref_plus_lora = (((is_lora_1 == False) and is_lora_2) or (is_lora_1 and (is_lora_2 == False)))
        is_double_lora = (is_lora_1 and is_lora_2)
        print(f"--->>rp_num: {rp_num}, is_lora_1: {is_lora_1}, is_lora_2: {is_lora_2}, is_ref_plus_lora: {is_ref_plus_lora}, is_double_lora: {is_double_lora}")
        if rp_num == 2 and is_ref_plus_lora:
            is_modify_param = True

        _ip_adapter_scale = rp_param.get('ip_adapter_scale', 0.5)
        _ip_adapter_ratio = rp_param.get('ip_adapter_ratio', 0.1)
        _region_prompt_ratio = rp_param.get('region_prompt_ratio', 0.2)
        if is_modify_param:
            _ip_adapter_scale = modify_rp_param.get('ip_adapter_scale', 0.5)
            _ip_adapter_ratio = modify_rp_param.get('ip_adapter_ratio', 0.5)
            _region_prompt_ratio = modify_rp_param.get('region_prompt_ratio', 0.5)
        if is_double_lora:
            _region_prompt_ratio = modify_rp_param.get('region_prompt_ratio', 0.5)

        return (is_modify_param, _ip_adapter_scale, _ip_adapter_ratio, _region_prompt_ratio, )
            



    
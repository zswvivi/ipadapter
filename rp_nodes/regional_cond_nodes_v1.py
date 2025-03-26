import torch

import comfy.sd
import comfy.model_sampling
import node_helpers
from .regional_cond_nodes import RegionalMask, RegionalConditioning

DEFAULT_REGIONAL_ATTN = {
    'double': [i for i in range(1, 19, 2)],
    'single': [i for i in range(1, 38, 2)]
}

'''
class RegionalMask(torch.nn.Module):
    def __init__(self, mask: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer('mask', mask)
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, q, transformer_options, *args, **kwargs):
        if self.start_percent <= 1 - transformer_options['sigmas'][0] < self.end_percent:
            return self.mask
        
        return None
    

class RegionalConditioning(torch.nn.Module):
    def __init__(self, region_cond: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer('region_cond', region_cond)
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, transformer_options, *args,  **kwargs):
        if self.start_percent <= 1 - transformer_options['sigmas'][0] < self.end_percent:
            return self.region_cond
        return None

class CreateRegionalCondNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "cond": ("CONDITIONING",),
            "mask": ("MASK",),
        }, "optional": {
            "prev_regions": ("REGION_COND",),
        }}

    RETURN_TYPES = ("REGION_COND",)
    FUNCTION = "create"

    CATEGORY = "flux-regional"

    def create(self, cond, mask, prev_regions=[]):
        prev_regions = [*prev_regions]
        print("--->>mask: ", mask.shape)
        prev_regions.append({
            'mask': mask,
            'cond': cond[0][0]
        })

        return (prev_regions,)
'''

class CreateRegionalCondNodeV2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
            "prompt": ("LIST", {"default": []}),
            "sr_prompt": ("LIST", {"default": []}),
            "background_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
            "rp_attrs": ("RP_ATTRS", {"default": {}}),
            #"sr_bbox":  ("LIST", {"default": []}),
            #"rp_split_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                #"guidance": ("FLOAT", {"default": 3.5, "min": -100.0, "max": 100.0, "step": 0.1}),
                #"method": (["merge_background", "merge_base_prompt", "normal"],),
                "switch_method":  (["sr_bbox", "sr_area_lr", "sr_area_box"],),
                #"method": ("STRING", {"default": "normal"}),
                #"switch_method":  ("STRING", {"default": "sr_area"})
            }
        }

    RETURN_TYPES = ("REGION_COND", "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("region_cond", "base_cond", "background_cond", )
    FUNCTION = "create_regionals_cond"

    CATEGORY = "flux-regional"
    '''
    def create(self, prompt, sr_prompt, sr_bbox):
        prev_regions = [*prev_regions]
        prev_regions.append({
            'mask': mask,
            'cond': cond[0][0]
        })

        return (prev_regions,)
    '''

    def prepare_mask(self, latent_h, latent_w, rp_split_ratio):
        sr_masks = []
        
        split_cols = int(latent_w * rp_split_ratio)
        left_mask = torch.zeros(latent_h, latent_w)
        left_mask[:, 0:split_cols] = 1.0

        right_mask = torch.zeros(latent_h, latent_w)
        right_mask[:, split_cols: latent_w] = 1.0
        sr_masks.append(left_mask)
        sr_masks.append(right_mask)
        return sr_masks

    def box_norm(self, box, img_size):
        latent_h, latent_w = img_size
        x_center, y_center, w, h = box
        x1 = int((x_center - w/2)* latent_w)
        y1 = int((y_center - h/2)* latent_h)
        x2 = int((x_center + w/2)* latent_w)
        y2 = int((y_center + h/2)* latent_h)
        return [x1, y1, x2, y2]
    
    def create_regionals_cond(self, clip, prompt, sr_prompt, background_prompt, rp_attrs, switch_method):
        rp_attrs.get('')
        tokens = clip.tokenize(background_prompt)
        output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
        cond = output.pop("cond")
        background_condition = [[cond, output]]
        rp_num = rp_attrs.get('rp_num', 0)

        if len(prompt) == 1 or rp_num == 0:
            tokens = clip.tokenize(prompt[0])
            output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
            cond = output.pop("cond")
            base_condition = [[cond, output]]
            return ([], base_condition, background_condition, )
        else:
            regions = []
            latent_w = int(rp_attrs.get('width', 1280)//16)
            latent_h = int(rp_attrs.get('height', 720)//16)
            latent_size = latent_w * latent_h
            #mask_rp = rp_attrs.get('mask_rp', {}).get(latent_size, [])
            sr_bbox = rp_attrs.get('rp_bbox', [])
            rp_split_ratio = rp_attrs.get('rp_split_ratio', 0.5)
            if len(sr_bbox) > 0:
                sr_bbox = sorted(sr_bbox, key = lambda item: item[0])
            tokens = clip.tokenize(prompt[0])
            output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
            cond = output.pop("cond")
            base_condition = [[cond, output]]

            background_mask = torch.ones(latent_h, latent_w)
            if switch_method == 'sr_bbox':
                for _prompt, _sr_box in zip(prompt[1:], sr_bbox, ):
                    tokens = clip.tokenize(_prompt)
                    output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
                    cond = output.pop("cond")
                    region_prompt_cond = [[cond, output]]

                    full_mask = torch.zeros(latent_h, latent_w)
                    sr_box_axis = self.box_norm(_sr_box, [latent_h, latent_w])
                    x1, y1, x2, y2 = sr_box_axis
                    full_mask[y1:y2, x1:x2] = 1.0

                    background_mask -= full_mask

                    regions.append({
                        "cond": region_prompt_cond[0][0],
                        "mask": full_mask.unsqueeze(0),
                    })
            elif switch_method == 'sr_area_lr' or switch_method == 'sr_area_box':
                sr_masks =self.prepare_mask(latent_h, latent_w, rp_split_ratio)
                for _sr_prompt, _sr_box, _sr_mask in zip(sr_prompt, sr_bbox, sr_masks):
                    tokens = clip.tokenize(_sr_prompt)
                    output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
                    cond = output.pop("cond")
                    sr_prompt_cond = [[cond, output]]

                    if switch_method == 'sr_area_lr':
                        background_mask -= _sr_mask

                        regions.append({
                            "cond": sr_prompt_cond[0][0],
                            "mask": _sr_mask.unsqueeze(0)
                        })
                    elif switch_method == 'sr_area_box':
                        full_mask = torch.zeros(latent_h, latent_w)
                        sr_box_axis = self.box_norm(_sr_box, [latent_h, latent_w])
                        x1, y1, x2, y2 = sr_box_axis
                        full_mask[y1:y2, x1:x2] = 1.0
                        background_mask -= full_mask

                        regions.append({
                            "cond": sr_prompt_cond[0][0],
                            "mask": full_mask.unsqueeze(0)
                        })
            if background_mask.sum() > 0:
                regions.insert(0, {
                    "cond": background_condition[0][0],
                    "mask": background_mask.unsqueeze(0),
                })

            
            return (regions, base_condition, background_condition, )


class ApplyRegionalCondsNodeV1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "region_conds": ("REGION_COND",),
            "latent": ("LATENT",),
            "start_percent": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
        }, "optional": {
            "attn_override": ("ATTN_OVERRIDE",),
            "base_cond": ("CONDITIONING", )
            #"background_cond": ("CONDITIONING",),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "flux-regional"

    def patch(self, model, region_conds, latent, start_percent, end_percent, attn_override=DEFAULT_REGIONAL_ATTN, base_cond = []):
        model = model.clone()

        if len(region_conds) == 0:
            return (model,)

        debug = True

        latent = latent['samples']
        b, c, h, w = latent.shape
        h //=2
        w //=2

        img_len = h*w

        regional_conditioning = torch.cat([region_cond['cond'] for region_cond in region_conds], dim=1)
        if len(base_cond) == 0:
            emb_len = 256
            text_len = emb_len + regional_conditioning.shape[1]
        else:
            emb_len = base_cond[0][0].shape[1]
            text_len = emb_len + regional_conditioning.shape[1]

        regional_mask = torch.zeros((text_len + img_len, text_len + img_len), dtype=torch.bool, device = 'cuda')

        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        main_prompt_start = 0
        main_prompt_end = emb_len

        start_subprompt_pos = emb_len
        end_subprompt_pos = 0
        for i in range(len(region_conds)):
            region_cond = region_conds[i]
            sub_cond = region_cond['cond']
            sub_mask = region_cond['mask'][0]

            end_subprompt_pos = start_subprompt_pos + sub_cond.shape[1]

            

            sub_mask_pil = to_pil(sub_mask.to(torch.float32))
            sub_mask_pil.save(f'aaaaa_cr_{i}.jpg')

            sub_mask_flatten = torch.nn.functional.interpolate(sub_mask[None, None, :, :], (h, w), mode='nearest-exact').flatten()
            print("-->>>sub_mask_flatten: ", sub_mask_flatten.shape, emb_len, sub_cond.shape, sub_mask_flatten.device)
            m_with_tokens = torch.cat([torch.ones(text_len), sub_mask_flatten])
            mb = m_with_tokens > 0.0
            print("--->>mb: ", mb.shape)
            regional_mask[~mb, start_subprompt_pos: end_subprompt_pos] = 1
            regional_mask[start_subprompt_pos: end_subprompt_pos, ~mb] = 1

            positions_idx = (sub_mask_flatten >0.0).nonzero(as_tuple=True)[0] + text_len
            regional_mask[positions_idx[:, None], main_prompt_start: main_prompt_end] = 1
            regional_mask[main_prompt_start: main_prompt_end, positions_idx[:, None]] = 1

            other_subprompt_start =  emb_len
            other_subprompt_end = 0
            for j in range(len(region_conds)):
                other_subprompt_end = other_subprompt_start + region_conds[j]['cond'].shape[1]
                print('!!!!!!!--->>i : ', i, ", j: ", j, ", other_subprompt_start: ", other_subprompt_start, ", other_subprompt_end: ", other_subprompt_end)
                if i != j:
                    regional_mask[start_subprompt_pos: end_subprompt_pos, other_subprompt_start: other_subprompt_end ] = 1
                    regional_mask[other_subprompt_start: other_subprompt_end, start_subprompt_pos: end_subprompt_pos] = 1
                other_subprompt_start = other_subprompt_start + region_conds[j]['cond'].shape[1]

            start_subprompt_pos = start_subprompt_pos + sub_cond.shape[1]


            # Block attention between positions not in mask and subprompt
            #regional_mask[~mb, i* :sp_end] = 1
            #regional_mask[sp_start:sp_end, ~mb] = 1
        regional_mask.fill_diagonal_(0)

        to_pil = ToPILImage()
        pil_mask = to_pil(regional_mask.to(torch.float32))
        pil_mask.save(f'xxxxxx_regional_mask.jpg')

        #attn_mask_bool = regional_mask > 0.5
        #regional_mask.masked_fill_(attn_mask_bool, float('-inf'))
        '''
        self_attend_masks = torch.zeros((img_len, img_len), dtype=torch.bool, device = 'cuda')
        union_masks = torch.zeros((img_len, img_len), dtype=torch.bool, device = 'cuda')
        '''



        '''
        if len(base_cond) == 0:
            region_conds = [
                { 
                    'mask': torch.ones((1, h, w), dtype=torch.float16),
                    #'mask': torch.zeros((1, h, w), dtype=torch.float16, device = 'cuda'),
                    'cond': torch.ones((1, 256, 4096), dtype=torch.float16)
                },
                *region_conds
            ]
        else:
            region_conds = [
                { 
                    'mask': torch.ones((1, h, w), dtype=torch.float16),
                    #'mask': torch.zeros((1, h, w), dtype=torch.float16, device = 'cuda'),
                    'cond': torch.ones((1, base_cond[0][0].shape[1], 4096), dtype=torch.float16)
                },
                *region_conds
            ]
        
        print("region_conds: ", len(region_conds))
        if debug:
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()

        current_seq_len = 0
        idx = 0
        for region_cond_dict in region_conds:
            region_cond = region_cond_dict['cond']
            #region_mask = 1 - region_cond_dict['mask'][0]
            region_mask = region_cond_dict['mask'][0]
            print("--->>region_mask: ", region_mask.shape)

            if debug:
                pil_img = to_pil(region_mask.to(torch.float32))
                pil_img.save(f'mask_region_{idx}.jpg')
            region_mask = torch.nn.functional.interpolate(region_mask[None, None, :, :], (h, w), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, region_cond.size(1)).to('cuda')
            next_seq_len = current_seq_len + region_cond.shape[1]

            # txt attends to itself
            regional_mask[current_seq_len:next_seq_len, current_seq_len:next_seq_len] = True

            # txt attends to corresponding regional img
            regional_mask[current_seq_len:next_seq_len, text_len:] = region_mask.transpose(-1, -2)

            # regional img attends to corresponding txt
            regional_mask[text_len:, current_seq_len:next_seq_len] = region_mask
            if debug:
                pil_img = to_pil(regional_mask.to(torch.float32))
                pil_img.save(f'regional_mask_{idx}.jpg')
            # regional img attends to corresponding regional img
            img_size_masks = region_mask[:, :1].repeat(1, img_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks, 
                                                    torch.logical_and(img_size_masks, img_size_masks_transpose))

            # update union
            union_masks = torch.logical_or(union_masks, 
                                            torch.logical_or(img_size_masks, img_size_masks_transpose))
            if debug:
                pil_img = to_pil(self_attend_masks.to(torch.float32))
                pil_img.save(f'self_attend_masks_{idx}.jpg')

                pil_img = to_pil(union_masks.to(torch.float32))
                pil_img.save(f'union_masks_{idx}.jpg')
            current_seq_len = next_seq_len
            idx += 1

        background_masks = torch.logical_not(union_masks)

        if debug:
            pil_img = to_pil(background_masks.to(torch.float32))
            pil_img.save(f'background_masks.jpg')

        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

        if debug:
            pil_img = to_pil(background_and_self_attend_masks.to(torch.float32))
            pil_img.save(f'background_and_self_attend_masks.jpg')

        regional_mask[text_len:, text_len:] = background_and_self_attend_masks

        if debug:
            pil_img = to_pil(regional_mask.to(torch.float32))
            pil_img.save(f'regional_mask_final.jpg')
        
        #regional_mask.masked_fill_(~regional_mask, float('-inf'))
        '''

        # Patch
        regional_mask = RegionalMask(regional_mask, start_percent, end_percent)
        regional_conditioning = RegionalConditioning(regional_conditioning, start_percent, end_percent)

        model.set_model_patch(regional_conditioning, 'regional_conditioning')

        for block_idx in attn_override['double']:
            model.set_model_patch_replace(regional_mask, f"double", "mask_fn", int(block_idx))

        for block_idx in attn_override['single']:
            model.set_model_patch_replace(regional_mask, f"single", "mask_fn", int(block_idx))

        return (model,)
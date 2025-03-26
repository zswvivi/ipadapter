import torch

import comfy.sd
import comfy.model_sampling
import node_helpers

DEFAULT_REGIONAL_ATTN = {
    'double': [i for i in range(1, 19, 2)],
    'single': [i for i in range(1, 38, 2)]
}


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

class CreateRegionalCondNodeV1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
            "prompt": ("LIST", {"default": []}),
            "sr_prompt": ("LIST", {"default": []}),
            "background_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
            #"sr_bbox":  ("LIST", {"default": []}),
            #"rp_split_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                #"guidance": ("FLOAT", {"default": 3.5, "min": -100.0, "max": 100.0, "step": 0.1}),
                #"method": (["merge_background", "merge_base_prompt", "normal"],),
                #"switch_method":  (["sr_bbox", "sr_area"],),
                "rp_attrs": ("RP_ATTRS", {"default": {}}),
                "face_prompt_one": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "face_prompt_two": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "method": ("STRING", {"default": "normal"}),
                "switch_method":  ("STRING", {"default": "sr_area"})
            }
        }

    RETURN_TYPES = ("REGION_COND", "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("region_cond", "base_cond", "background_cond", "face_prompt_one_cond", "face_prompt_two_cond",)
    FUNCTION = "create_regionals_cond"

    CATEGORY = "flux-regional"

    def box_norm(self, box, img_size):
        latent_h, latent_w = img_size
        x_center, y_center, w, h = box
        x1 = int((x_center - w/2)* latent_w)
        y1 = int((y_center - h/2)* latent_h)
        x2 = int((x_center + w/2)* latent_w)
        y2 = int((y_center + h/2)* latent_h)
        return [x1, y1, x2, y2]
    
    def create_regionals_cond(self, clip, prompt, sr_prompt, background_prompt, rp_attrs, face_prompt_one, face_prompt_two, method, switch_method):
        rp_attrs.get('')
        tokens = clip.tokenize(background_prompt)
        output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
        cond = output.pop("cond")
        background_condition = [[cond, output]]
        rp_num = rp_attrs.get('rp_num', 0)
        face_prompt_one_cond = None
        face_prompt_two_cond = None
        '''
        if (face_prompt_one != '') or (face_prompt_one is not None):
            tokens = clip.tokenize(face_prompt_one)
            output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
            cond = output.pop("cond")
            face_prompt_one_cond = [[cond, output]]
        else:
            face_prompt_one_cond = None
        
        if (face_prompt_two != '') or (face_prompt_two is not None):
            tokens = clip.tokenize(face_prompt_two)
            output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
            cond = output.pop("cond")
            face_prompt_two_cond = [[cond, output]]
        else:
            face_prompt_two_cond = None
        '''

        if len(prompt) == 1 or rp_num == 0:
            tokens = clip.tokenize(prompt[0])
            output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
            cond = output.pop("cond")
            base_condition = [[cond, output]]
            return ([], base_condition, base_condition, face_prompt_one_cond, face_prompt_two_cond, )
        else:
            regions = []
            latent_w = int(rp_attrs.get('width', 1280)//16)
            latent_h = int(rp_attrs.get('height', 720)//16)
            latent_size = latent_w * latent_h
            mask_rp = rp_attrs.get('mask_rp', {}).get(latent_size, [])
            sr_bbox = rp_attrs.get('rp_bbox', [])
            rp_split_ratio = rp_attrs.get('rp_split_ratio', 0.5)
            if len(sr_bbox) > 0:
                sr_bbox = sorted(sr_bbox, key = lambda item: item[0])
            tokens = clip.tokenize(prompt[0])
            output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
            cond = output.pop("cond")
            base_condition = [[cond, output]]
            background_mask = torch.ones(latent_h, latent_w)
            sr_masks = []
            if switch_method == 'sr_area':
                split_cols = int(latent_w * rp_split_ratio)
                left_mask = torch.zeros(latent_h, latent_w)
                left_mask[:, 0:split_cols] = 1.0
        
                right_mask = torch.zeros(latent_h, latent_w)
                right_mask[:, split_cols: latent_w] = 1.0
                sr_masks.append(left_mask)
                sr_masks.append(right_mask)
            else:
                for sr_box in sr_bbox:
                    x_center, y_center, w, h = sr_box
                    x1 = int((x_center - w/2)* latent_w)
                    y1 = int((y_center - h/2)* latent_h)
                    x2 = int((x_center + w/2)* latent_w)
                    y2 = int((y_center + h/2)* latent_h)
                    mask = torch.zeros(latent_h, latent_w)
                    mask[y1:y2, x1:x2] = 1.0
                    sr_masks.append(mask)
            cnt = 0
            for _sr_prompt, _sr_mask in zip(sr_prompt, sr_masks):
                tokens = clip.tokenize(_sr_prompt)
                output = clip.encode_from_tokens(tokens, return_pooled = True, return_dict = True)
                cond = output.pop("cond")
                sr_prompt_cond = [[cond, output]]

                if cnt == 0:
                    face_prompt_one_cond = [[cond, output]]
                elif cnt ==1:
                    face_prompt_two_cond = [[cond, output]]
                cnt +=1

                background_mask -= _sr_mask
                regions.append({
                    "cond": sr_prompt_cond[0][0],
                    "mask": _sr_mask.unsqueeze(0)
                })

            cond_mask_pair = {
                "mask": None,
                "cond": None,
            }
            if method == 'merge_background':
                cond_mask_pair['cond'] = background_condition[0][0]
                cond_mask_pair['mask'] = background_mask.unsqueeze(0)
                regions.insert(0, cond_mask_pair)
            elif method == 'merge_base_prompt':
                cond_mask_pair['cond'] = base_condition[0][0]
                #cond_mask_pair['mask'] = background_mask.unsqueeze(0)
                cond_mask_pair['mask'] = torch.ones((1, latent_h, latent_w))
                regions.insert(0, cond_mask_pair)
            elif method == "both":
                if background_mask.sum() >0:
                    regions.insert(0, {
                        "cond": background_condition[0][0],
                        "mask": background_mask.unsqueeze(0)
                    })
                regions.insert(0, {
                    "cond": base_condition[0][0],
                    "mask": torch.ones((1, latent_h, latent_w))
                    })
            
            return (regions, base_condition, background_condition, face_prompt_one_cond, face_prompt_two_cond, )


class ApplyRegionalCondsNode:
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
            "base_cond": ("CONDITIONING", ),
            "rp_weight": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "rp_num": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            #"background_cond": ("CONDITIONING",),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "flux-regional"

    def patch(self, model, region_conds, latent, start_percent, end_percent, attn_override=DEFAULT_REGIONAL_ATTN, base_cond = [], rp_weight = 0.8, rp_num = 2):
        model = model.clone()

        if len(region_conds) == 0:
            return (model,)

        debug = True

        latent = latent['samples']
        b, c, h, w = latent.shape
        h //=2
        w //=2

        img_len = h*w

        #region_conds = [region_cond['cond'].to(model.load_device) for region_cond in region_conds]

        regional_conditioning = torch.cat([region_cond['cond'].to(model.load_device) for region_cond in region_conds], dim=1)
        if rp_weight < 0.0:
            if len(base_cond) == 0:
                text_len = 256 + regional_conditioning.shape[1]
            else:
                text_len = base_cond[0][0].shape[1] + regional_conditioning.shape[1]
        else:
            text_len = regional_conditioning.shape[1]

        
        regional_mask = torch.zeros((text_len + img_len, text_len + img_len), dtype=torch.bool, device = 'cuda')

        #print("--->>text_len: {}, img_len: {}, regional_mask: {}".format(text_len, img_len, regional_mask.shape))

        self_attend_masks = torch.zeros((img_len, img_len), dtype=torch.bool, device = 'cuda')
        union_masks = torch.zeros((img_len, img_len), dtype=torch.bool, device = 'cuda')
        
        if rp_weight <0.0:
            if len(base_cond) == 0:
                region_conds = [
                    { 
                        #'mask': torch.ones((1, h, w), dtype=torch.float16),
                        #'mask': torch.ones((1, h, w), dtype=torch.float16, device = 'cuda'),
                        'mask': torch.zeros((1, h, w), dtype=torch.float16, device = 'cuda'),
                        'cond': torch.ones((1, 256, 4096), dtype=torch.float16)
                    },
                    *region_conds
                ]
            else:
                region_conds = [
                    { 
                        #'mask': torch.ones((1, h, w), dtype=torch.float16),
                        #'mask': torch.ones((1, h, w), dtype=torch.float16, device = 'cuda'),
                        'mask': torch.zeros((1, h, w), dtype=torch.float16, device = 'cuda'),
                        'cond': torch.ones((1, base_cond[0][0].shape[1], 4096), dtype=torch.float16)
                    },
                    *region_conds
                ]
        #print("region_conds: ", len(region_conds))
        if debug:
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()

        current_seq_len = 0
        idx = 0
        for region_cond_dict in region_conds:
            region_cond = region_cond_dict['cond']
            #region_mask = 1 - region_cond_dict['mask'][0]
            region_mask = region_cond_dict['mask'][0]

            if debug:
                pil_img = to_pil(region_mask.to(torch.float32))
                pil_img.save(f'mask_region_{idx}.jpg')
            print("-->>region_mask: ", region_mask.shape)
            region_mask = torch.nn.functional.interpolate(region_mask[None, None, :, :], (h, w), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, region_cond.size(1))
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
                                                    torch.logical_and(img_size_masks, img_size_masks_transpose).to(self_attend_masks.device))

            # update union
            union_masks = torch.logical_or(union_masks, 
                                            torch.logical_or(img_size_masks, img_size_masks_transpose).to(union_masks.device))
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

        if rp_num == 1:
            end_percent = 0.0
        # Patch
        regional_mask = RegionalMask(regional_mask, start_percent, end_percent)
        regional_conditioning = RegionalConditioning(regional_conditioning, start_percent, end_percent)

        model.set_model_patch(regional_conditioning, 'regional_conditioning')
        if rp_weight >=0.0:
            model.model_options['transformer_options']['rp_weight'] = rp_weight

        for block_idx in attn_override['double']:
            model.set_model_patch_replace(regional_mask, f"double", "mask_fn", int(block_idx))

        for block_idx in attn_override['single']:
            model.set_model_patch_replace(regional_mask, f"single", "mask_fn", int(block_idx))

        return (model,)
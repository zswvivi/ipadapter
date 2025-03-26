import comfy.sd
import folder_paths

class CheckpointLoaderSimpleAdavance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            },
            "optional": {
                "unet": ("BOOLEAN", {"default": True}),
                "clip": ("BOOLEAN", {"default": True}),
                "vae": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.", 
                       "The CLIP model used for encoding text prompts.", 
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_name, unet, clip, vae):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=vae, output_clip=clip, output_model = unet, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]
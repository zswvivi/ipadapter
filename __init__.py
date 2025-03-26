from .ipadapter_flux import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .rp_nodes.attn_override_node import FluxAttnOverrideNode, RFSingleBlocksOverrideNode, RFDoubleBlocksOverrideNode
from .rp_nodes.configure_modified_flux_node import ConfigureModifiedFluxNode, CacheConfig, InjectFluxIPAdapter, InjectFluxPulidAdapter, DoLora, RPParam, ModifyConfigParam
from .rp_nodes.flux_deguidance_node import FluxDeGuidance
from .rp_nodes.regional_cond_nodes import CreateRegionalCondNode, CreateRegionalCondNodeV1, ApplyRegionalCondsNode
from .pulid_flux import PuLIDLoader, PulidInjectFlux, PuLIDSigLoader, PuLIDCond, PuLIDEncoder, GetIdEmbedByIdx, ConfigClear
#from .rp_nodes.load_diffusion_model import CheckpointLoaderSimpleAdavance
from .rp_nodes.regional_cond_nodes_v1 import CreateRegionalCondNodeV2, ApplyRegionalCondsNodeV1
from .rp_nodes.flux_custom_sampler import SamplerCustomXAdvanced

NODE_CLASS_MAPPINGS["FluxAttnOverrideNode"] = FluxAttnOverrideNode
NODE_CLASS_MAPPINGS["RFSingleBlocksOverrideNode"] = RFSingleBlocksOverrideNode
NODE_CLASS_MAPPINGS["RFDoubleBlocksOverrideNode"] = RFDoubleBlocksOverrideNode
NODE_CLASS_MAPPINGS["ConfigureModifiedFluxNode"] = ConfigureModifiedFluxNode
NODE_CLASS_MAPPINGS["FluxDeGuidance"] = FluxDeGuidance
NODE_CLASS_MAPPINGS["CreateRegionalCondNode"] = CreateRegionalCondNode
NODE_CLASS_MAPPINGS["CreateRegionalCondNodeV1"] = CreateRegionalCondNodeV1
NODE_CLASS_MAPPINGS["ApplyRegionalCondsNode"] = ApplyRegionalCondsNode
NODE_CLASS_MAPPINGS["PuLIDLoader"] = PuLIDLoader
NODE_CLASS_MAPPINGS["PuLIDSigLoader"] = PuLIDSigLoader
NODE_CLASS_MAPPINGS["PulidInjectFlux"] = PulidInjectFlux
NODE_CLASS_MAPPINGS["PuLIDCond"] = PuLIDCond
NODE_CLASS_MAPPINGS["PuLIDEncoder"] = PuLIDEncoder
NODE_CLASS_MAPPINGS["CreateRegionalCondNodeV2"] = CreateRegionalCondNodeV2
NODE_CLASS_MAPPINGS["ApplyRegionalCondsNodeV1"] = ApplyRegionalCondsNodeV1
NODE_CLASS_MAPPINGS["GetIdEmbedByIdx"] = GetIdEmbedByIdx
NODE_CLASS_MAPPINGS["SamplerCustomXAdvanced"] = SamplerCustomXAdvanced
NODE_CLASS_MAPPINGS["CacheConfig"] = CacheConfig
NODE_CLASS_MAPPINGS['ConfigClear'] = ConfigClear
NODE_CLASS_MAPPINGS['InjectFluxIPAdapter'] = InjectFluxIPAdapter
NODE_CLASS_MAPPINGS['InjectFluxPulidAdapter'] = InjectFluxPulidAdapter
NODE_CLASS_MAPPINGS['DoLora'] = DoLora
NODE_CLASS_MAPPINGS['RPParam'] = RPParam
NODE_CLASS_MAPPINGS['ModifyConfigParam'] = ModifyConfigParam


NODE_DISPLAY_NAME_MAPPINGS["FluxAttnOverrideNode"] = "Flux Attention Override"
NODE_DISPLAY_NAME_MAPPINGS["RFSingleBlocksOverrideNode"] = "RF-Edit Single Layers Override"
NODE_DISPLAY_NAME_MAPPINGS["RFDoubleBlocksOverrideNode"] = "RF-Edit Double Layers Override"
NODE_DISPLAY_NAME_MAPPINGS["ConfigureModifiedFluxNode"] = "Configure Modified Flux"
NODE_DISPLAY_NAME_MAPPINGS["FluxDeGuidance"] = "Flux DeGuidance"
NODE_DISPLAY_NAME_MAPPINGS["CreateRegionalCondNode"] = "Create Flux Regional Cond"
NODE_DISPLAY_NAME_MAPPINGS["CreateRegionalCondNodeV1"] = "Create Regional Cond Node V1"
NODE_DISPLAY_NAME_MAPPINGS["ApplyRegionalCondsNode"] = "Apply Flux Regional Conds"
NODE_DISPLAY_NAME_MAPPINGS["PuLIDLoader"] = "Loader Pulid"
NODE_DISPLAY_NAME_MAPPINGS["PuLIDSigLoader"] = "Loader Siglip Model"
NODE_DISPLAY_NAME_MAPPINGS["PulidInjectFlux"] = "Pulid Inject Flux"
NODE_DISPLAY_NAME_MAPPINGS["PuLIDCond"] = "Pulid ID Condition"
NODE_DISPLAY_NAME_MAPPINGS["PuLIDEncoder"] = "Pulid ID Encoder"
NODE_DISPLAY_NAME_MAPPINGS["CreateRegionalCondNodeV2"] = "Create Regional Cond Node V2"
NODE_DISPLAY_NAME_MAPPINGS["ApplyRegionalCondsNodeV1"] = "Apply Regional Conds Node V1"
NODE_DISPLAY_NAME_MAPPINGS["GetIdEmbedByIdx"] = "Get IdEmbed ByIdx"
NODE_DISPLAY_NAME_MAPPINGS["SamplerCustomXAdvanced"] = "Sampler Custom X Advanced"
NODE_DISPLAY_NAME_MAPPINGS["CacheConfig"] = "Cache Config"
NODE_DISPLAY_NAME_MAPPINGS['ConfigClear'] = "Config Clear"
NODE_DISPLAY_NAME_MAPPINGS["InjectFluxIPAdapter"] = "Inject Flux IPAdapter"
NODE_DISPLAY_NAME_MAPPINGS['InjectFluxPulidAdapter'] = "Inject Flux PulidAdapter"
NODE_DISPLAY_NAME_MAPPINGS['DoLora'] = "Do Lora"
NODE_DISPLAY_NAME_MAPPINGS['RPParam'] = "RP Param"
NODE_DISPLAY_NAME_MAPPINGS['ModifyConfigParam'] = "Modify Config Param"




__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
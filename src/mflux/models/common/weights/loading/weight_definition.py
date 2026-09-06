from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, TypeAlias

import mlx.core as mx

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget

if TYPE_CHECKING:
    from mflux.models.common.tokenizer.tokenizer import BaseTokenizer
    from mflux.models.depth_pro.weights.depth_pro_weight_definition import DepthProWeightDefinition
    from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition
    from mflux.models.fibo_vlm.weights.fibo_vlm_weight_definition import FIBOVLMWeightDefinition
    from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition
    from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
    from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition
    from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

    WeightDefinitionType: TypeAlias = type[
        FluxWeightDefinition
        | FIBOWeightDefinition
        | FIBOVLMWeightDefinition
        | QwenWeightDefinition
        | ZImageWeightDefinition
        | SeedVR2WeightDefinition
        | DepthProWeightDefinition
        | Ideogram4WeightDefinition
    ]


@dataclass
class ComponentDefinition:
    name: str
    hf_subdir: str
    mapping_getter: Callable[[], List[WeightTarget]] | None = None
    model_attr: str | None = None
    num_blocks: int | None = None
    num_layers: int | None = None
    loading_mode: str = "mlx_native"
    precision: mx.Dtype | None = None
    skip_quantization: bool = False
    bulk_transform: Callable[[mx.array], mx.array] | None = None
    weight_subkey: str | None = None
    download_url: str | None = None
    weight_prefix_filters: List[str] | None = None
    weight_files: List[str] | None = None  # Specific files to load (if None, loads all *.safetensors)
    key_transform: Callable[[str], str | None] | None = None
    weight_transform: Callable[[str, mx.array], mx.array] | None = None
    # Picks a concrete component definition from what is on disk at the resolved root
    # path. Lets a single component support more than one storage layout (e.g. a native
    # single-file checkpoint vs a diffusers sharded directory with different keys).
    variant_selector: Callable[[Path], "ComponentDefinition"] | None = None

    @staticmethod
    def save_subdirs(components: "List[ComponentDefinition]") -> dict:
        # Where ModelSaver writes each component and where the loader probes for an
        # mflux-saved one. Normally the hf_subdir; but two independent components that share
        # one (SeedVR2's transformer and vae both sit flat at repo root, told apart on load
        # by weight_files) would overwrite each other's shards and index there. Give an
        # independent component its own <hf_subdir>/<name> directory whenever anything else
        # shares its hf_subdir.
        #
        # A component that carves a subset out of a shared source with weight_prefix_filters
        # (FIBO VLM's decoder and visual read the same files and split them by prefix) is a
        # deliberate shared-source split, so it keeps its hf_subdir; the count includes it,
        # so an independent component next to it is still moved out of the way. Every model
        # with unique subdirs is returned unchanged.
        #
        # Computed from the static definition, matching ModelSaver, which writes to that same
        # hf_subdir and never runs variant_selector, so the loader's probe here agrees with
        # where the checkpoint was written. Keyed on the raw hf_subdir string: no shipped
        # model spells one directory two ways, and normalizing "" to "." would wrongly merge
        # FIBO VLM's decoder/visual ("") with its fibo_vlm (".") component.
        counts: dict = {}
        for c in components:
            counts[c.hf_subdir] = counts.get(c.hf_subdir, 0) + 1
        return {
            c.name: (
                str(Path(c.hf_subdir) / c.name)
                if counts[c.hf_subdir] > 1 and c.weight_prefix_filters is None
                else c.hf_subdir
            )
            for c in components
        }


@dataclass
class TokenizerDefinition:
    name: str
    hf_subdir: str
    tokenizer_class: str = "AutoTokenizer"
    fallback_subdirs: List[str] | None = None
    download_patterns: List[str] | None = None
    encoder_class: type["BaseTokenizer"] | None = None
    max_length: int = 512
    padding: str = "max_length"
    template: str | None = None
    use_chat_template: bool = False
    chat_template_kwargs: dict | None = field(default_factory=dict)
    add_special_tokens: bool = True
    processor_class: type | None = None
    image_token: str = "<|image_pad|>"
    chat_template: str | None = None  # Jinja2 template for apply_chat_template

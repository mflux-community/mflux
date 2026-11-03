from collections import defaultdict
from pathlib import Path

import pytest
from huggingface_hub import HfApi

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
from mflux.models.flux3.weights.flux3_lora_mapping import Flux3LoRAMapping

# Tensor names and shapes read from the safetensors header of LoRAs people publish (no weights), one file per
# format the model's users train with. Captured 2026-11-03 from:
#   flux3_klein_ai_toolkit  ausboss/FLUX-3-Image-Klein-Outpaint-LoRA @ b12111c044be
#                           flux-3-image-klein-outpaint.safetensors (ai-toolkit, ComfyUI names, lora_A/B, no alpha)
#   flux3_klein_kohya       Muinez/flux3-klein-distil-4step @ 56905ed03885
#                           flux3-klein-distil-4step.safetensors (lora_down/up with an alpha per layer)
#   flux3_klein_diffusers   diffusers-modular/flux3-tarot-lora @ 96fe1071462e
#                           pytorch_lora_weights.safetensors (diffusers training script, transformer. prefix)
# All three were trained on Klein Base. To regenerate one, write "<name> <shape joined by x>" for each tensor of
# the header, sorted by name; a scalar has shape 1.

PUBLISHED = [
    pytest.param(
        "flux3_klein_ai_toolkit",
        "ausboss/FLUX-3-Image-Klein-Outpaint-LoRA",
        "flux-3-image-klein-outpaint.safetensors",
        Flux3LoRAMapping,
        lambda: Flux3Transformer(flow_map=False),
        id="flux3_klein_ai_toolkit",
    ),
    pytest.param(
        "flux3_klein_kohya",
        "Muinez/flux3-klein-distil-4step",
        "flux3-klein-distil-4step.safetensors",
        Flux3LoRAMapping,
        lambda: Flux3Transformer(flow_map=False),
        id="flux3_klein_kohya",
    ),
    pytest.param(
        "flux3_klein_diffusers",
        "diffusers-modular/flux3-tarot-lora",
        "pytorch_lora_weights.safetensors",
        Flux3LoRAMapping,
        lambda: Flux3Transformer(flow_map=False),
        id="flux3_klein_diffusers",
    ),
]


class _PublishedLoRA:
    DIR = Path(__file__).parent.parent / "resources" / "lora_keys"

    @staticmethod
    def shapes(fixture: str) -> dict[str, tuple[int, ...]]:
        lines = (_PublishedLoRA.DIR / f"{fixture}.txt").read_text().splitlines()
        return {name: tuple(int(d) for d in shape.split("x")) for name, shape in (line.split() for line in lines)}

    @staticmethod
    def hub_shapes(repo_id: str, filename: str) -> dict[str, tuple[int, ...]]:
        metadata = HfApi().parse_safetensors_file_metadata(repo_id, filename)
        return {name: tuple(info.shape) or (1,) for name, info in metadata.tensors.items()}

    @staticmethod
    def problems(shapes: dict[str, tuple[int, ...]], mapping, transformer) -> list[str]:
        # The loader's own pattern match, so a name it would skip or send to the wrong layer shows up here.
        # Patterns are bucketed by the text after their last "{block}" (the whole pattern if none): a name can only
        # match a pattern it ends with, with or without ".weight". The match itself stays the loader's.
        buckets = defaultdict(list)
        for match in LoRALoader._build_pattern_mappings(mapping.get_mapping()):
            buckets[match.source_pattern.rsplit("{block}", 1)[-1].removesuffix(".weight")].append(match)
        problems, by_target, sources = [], defaultdict(dict), {}
        for name, shape in shapes.items():
            stem = name.removesuffix(".weight")
            candidates = [match for tail, matches in buckets.items() if stem.endswith(tail) for match in matches]
            hits = {
                (match.target_path.format(block=block), match.matrix_name)
                for match in candidates
                if (block := LoRALoader._match_pattern(name, match.source_pattern)) is not None
            }
            if len(hits) != 1:
                problems.append(f"{name}: matches {sorted(hits) or 'no layer'}")
                continue
            target, matrix = hits.pop()
            # Two names for one slot (a BFL and a diffusers alias, say): the loader rejects the file.
            if (first := sources.setdefault((target, matrix), name)) != name:
                problems.append(f"{name}: fills {target}.{matrix}, which {first} fills already")
                continue
            by_target[target][matrix] = shape
        for target, matrices in by_target.items():
            problems += _PublishedLoRA._layer_problems(target, matrices, transformer)
        return problems

    @staticmethod
    def _layer_problems(target: str, matrices: dict[str, tuple[int, ...]], transformer) -> list[str]:
        try:
            module = LoRALoader._get_target_module(transformer, target)
        except (AttributeError, IndexError, KeyError):
            return [f"{target}: no such layer"]
        if not hasattr(module, "weight"):
            return [f"{target}: not a linear layer"]
        if set(matrices) - {"alpha"} != {"lora_A", "lora_B"}:
            return [f"{target}: has {sorted(matrices)}, needs lora_A and lora_B"]
        out_dim, in_dim = module.weight.shape
        rank = matrices["lora_A"][0]
        problems = []
        if matrices["lora_A"] != (rank, in_dim):
            problems.append(f"{target}: down is {matrices['lora_A']}, the layer takes {in_dim} inputs")
        if matrices["lora_B"] != (out_dim, rank):
            problems.append(f"{target}: up is {matrices['lora_B']}, the layer gives {out_dim} outputs at rank {rank}")
        if matrices.get("alpha", (1,)) != (1,):
            problems.append(f"{target}: alpha is not a scalar")
        return problems


@pytest.mark.fast
@pytest.mark.parametrize(("fixture", "repo_id", "filename", "mapping", "build"), PUBLISHED)
def test_a_published_lora_reaches_a_layer_with_every_tensor(fixture, repo_id, filename, mapping, build):
    # Building the full-size transformer only reads shapes: MLX allocates its weights on first evaluation.
    assert _PublishedLoRA.problems(_PublishedLoRA.shapes(fixture), mapping, build()) == []


@pytest.mark.fast
def test_two_names_for_one_matrix_are_a_problem():
    shapes = {
        "diffusion_model.single_blocks.0.q_proj.lora_A.weight": (4, 3072),
        "diffusion_model.single_blocks.0.q_proj.lora_B.weight": (3072, 4),
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": (4, 3072),
    }

    problems = _PublishedLoRA.problems(shapes, Flux3LoRAMapping, Flux3Transformer(flow_map=False))

    assert problems == [
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight: fills "
        "single_transformer_blocks.0.attn.to_q.lora_A, which diffusion_model.single_blocks.0.q_proj.lora_A.weight "
        "fills already"
    ]


@pytest.mark.slow
@pytest.mark.parametrize(("fixture", "repo_id", "filename", "mapping", "build"), PUBLISHED)
def test_the_published_file_on_the_hub_still_loads(fixture, repo_id, filename, mapping, build):
    # Reads only the header of the current upload, so a re-upload in a layout mflux does not read fails here.
    assert _PublishedLoRA.problems(_PublishedLoRA.hub_shapes(repo_id, filename), mapping, build()) == []

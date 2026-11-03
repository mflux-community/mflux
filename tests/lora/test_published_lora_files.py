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
#   flux3_klein_distill_extract  WAS/flux3_klein @ 1e5b75f47f85
#                           flux3_image_klein_distill_r32.safetensors (the distilled model minus Base: lora_down/up
#                           without alpha, and .diff for the q and k norm scales)
# The first three were trained on Klein Base, and the extract goes on Base. To regenerate one, write
# "<name> <shape joined by x>" for each tensor of the header, sorted by name; a scalar has shape 1.

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
    pytest.param(
        "flux3_klein_distill_extract",
        "WAS/flux3_klein",
        "flux3_image_klein_distill_r32.safetensors",
        Flux3LoRAMapping,
        lambda: Flux3Transformer(flow_map=False),
        id="flux3_klein_distill_extract",
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
            return [f"{target}: has no weight"]
        # A .diff (.diff_b) is added to the weight (bias) vector of the layer itself, such as a norm scale.
        problems = []
        for name, attribute in (("diff", "weight"), ("diff_b", "bias")):
            if name in matrices:
                vector = getattr(module, attribute, None)
                if vector is None or vector.ndim != 1 or matrices[name] != tuple(vector.shape):
                    problems.append(
                        f"{target}: {name} is {matrices[name]}, the layer has no {attribute} vector of that shape"
                    )
        if set(matrices) <= {"diff", "diff_b"}:
            return problems
        if set(matrices) - {"alpha", "diff", "diff_b"} != {"lora_A", "lora_B"}:
            return [f"{target}: has {sorted(matrices)}, needs lora_A and lora_B"]
        out_dim, in_dim = module.weight.shape
        rank = matrices["lora_A"][0]
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


@pytest.mark.fast
def test_a_diff_must_fit_the_vector_it_patches():
    shapes = {
        "diffusion_model.single_blocks.0.norm.query_norm.diff": (128,),
        "diffusion_model.single_blocks.0.norm.key_norm.diff": (3072,),
    }

    problems = _PublishedLoRA.problems(shapes, Flux3LoRAMapping, Flux3Transformer(flow_map=False))

    assert problems == [
        "single_transformer_blocks.0.attn.norm_k: diff is (3072,), the layer has no weight vector of that shape"
    ]


@pytest.mark.slow
@pytest.mark.parametrize(("fixture", "repo_id", "filename", "mapping", "build"), PUBLISHED)
def test_the_published_file_on_the_hub_still_loads(fixture, repo_id, filename, mapping, build):
    # Reads only the header of the current upload, so a re-upload in a layout mflux does not read fails here.
    assert _PublishedLoRA.problems(_PublishedLoRA.hub_shapes(repo_id, filename), mapping, build()) == []

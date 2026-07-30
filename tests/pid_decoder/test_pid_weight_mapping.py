import mlx.core as mx

from mflux.models.common.pid_decoder.gemma2.gemma2_config import Gemma2Config
from mflux.models.common.pid_decoder.gemma2.gemma2_model import Gemma2Model
from mflux.models.common.pid_decoder.pid_weight_mapping import PID_WEIGHT_MAPPING
from mflux.models.common.pid_decoder.pixdit.pixdit_network import PidNet


def _resolve(modules: dict, dotted_path: str) -> mx.array:
    """Walk a dotted attribute path (e.g. "pid_net.patch_blocks.0.attn.qkv_x.weight")
    against `modules`, indexing into plain Python lists (patch_blocks, gate_modules,
    the adaLN_modulation[0] Sequential-stand-in, ...) by int where attribute lookup
    doesn't apply."""
    parts = dotted_path.split(".")
    obj = modules[parts[0]]
    for part in parts[1:]:
        if isinstance(obj, (list, tuple)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _tiny_net() -> PidNet:
    # List-length-determining params (patch_depth, pixel_depth, lq_num_res_blocks,
    # lq_interval, pit_lq_inject) match the real checkpoint's counts (14 patch blocks,
    # 2 pixel blocks, 7 lq gate/output heads, 4 latent res blocks, pit head+gate present)
    # so every index PID_WEIGHT_MAPPING references resolves. Tensor widths
    # (hidden_size, pixel_hidden_size, ...) are shrunk for a fast test -- PID_WEIGHT_MAPPING
    # only cares that the attribute *exists*, not its shape.
    return PidNet(
        hidden_size=32,
        pixel_hidden_size=8,
        pixel_attn_hidden_size=None,
        patch_depth=14,
        pixel_depth=2,
        num_groups=4,
        patch_size=4,
        txt_embed_dim=16,
        txt_max_length=8,
        rope_ref_h=64,
        rope_ref_w=64,
        lq_latent_channels=4,
        lq_hidden_dim=8,
        lq_num_res_blocks=4,
        pit_lq_inject=True,
        sr_scale=4,
        latent_spatial_down_factor=8,
    )


def test_every_mapped_target_resolves_to_a_real_parameter():
    net = _tiny_net()
    gemma = Gemma2Model(
        Gemma2Config(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    )
    modules = {"pid_net": net, "gemma2": gemma}

    missing = []
    for target_path in PID_WEIGHT_MAPPING.values():
        try:
            value = _resolve(modules, target_path)
        except (AttributeError, IndexError, KeyError, TypeError) as e:
            missing.append(f"{target_path}: {e!r}")
            continue
        if not isinstance(value, mx.array):
            missing.append(f"{target_path}: resolved to {type(value)}, not mx.array")

    assert not missing, f"Unresolved weight-mapping targets ({len(missing)}): {missing[:20]}"


def test_mapping_has_no_duplicate_targets():
    targets = list(PID_WEIGHT_MAPPING.values())
    assert len(targets) == len(set(targets)), "PID_WEIGHT_MAPPING has colliding target paths"

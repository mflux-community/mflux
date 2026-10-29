import random
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.common.training import runner as runner_module
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.dataset.batch import DataItem
from mflux.models.common.training.trainer import TrainingTrainer
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.ernie_image.training_adapter.ernie_training_adapter import ErnieTrainingAdapter
from mflux.models.flux2.training_adapter.flux2_base_training_adapter import Flux2BaseTrainingAdapter
from mflux.models.krea2.training_adapter.krea2_training_adapter import Krea2TrainingAdapter
from mflux.models.z_image.training_adapter.z_image_training_adapter import ZImageTrainingAdapter

pytestmark = pytest.mark.fast

RANK, ALPHA = 4, 8.0
MAPPING = [
    LoRATarget(
        model_path="proj",
        possible_up_patterns=["transformer.proj.lora_B.weight"],
        possible_down_patterns=["transformer.proj.lora_A.weight"],
        possible_alpha_patterns=["transformer.proj.alpha"],
    )
]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)


def _trained_lora(scale: float) -> LoRALinear:
    mx.random.seed(0)
    lora = LoRALinear.from_linear(nn.Linear(8, 8), r=RANK, scale=scale)
    lora._mflux_lora_role = "train"
    lora.lora_B = 0.1 * mx.random.normal(lora.lora_B.shape)
    return lora


def _saved(tmp_path, lora: LoRALinear):
    path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(str(path), TrainingUtil.lora_weights("transformer.proj", lora))
    return str(path)


class TestLoRAScale:
    def test_a_scale_other_than_one_is_saved_as_alpha(self, tmp_path):
        assert float(mx.load(_saved(tmp_path, _trained_lora(ALPHA / RANK)))["transformer.proj.alpha"]) == ALPHA
        assert "transformer.proj.alpha" not in mx.load(_saved(tmp_path, _trained_lora(1.0)))

    def test_generation_loads_the_lora_at_the_strength_it_trained_at(self, tmp_path):
        trained = _trained_lora(ALPHA / RANK)
        x = mx.random.normal((1, 8))
        model = _Model()
        expected = trained.linear(x) + ALPHA / RANK * (x @ trained.lora_A @ trained.lora_B)
        LoRALoader.load_and_apply_lora(MAPPING, model, [_saved(tmp_path, trained)], [1.0], bake_lora=False)
        model.proj.linear = trained.linear
        assert mx.allclose(model.proj(x), expected, atol=1e-5).item()

    def test_resuming_keeps_alpha_as_the_layer_scale(self, tmp_path):
        # Folded into B the output would match, but B and the restored optimizer state would not
        trained = _trained_lora(ALPHA / RANK)
        model = _Model()
        LoRALoader.load_and_apply_lora(
            MAPPING, model, [_saved(tmp_path, trained)], [1.0], role="train", bake_lora=False
        )
        assert model.proj.scale == pytest.approx(ALPHA / RANK)
        assert mx.array_equal(model.proj.lora_B, trained.lora_B).item()

    @pytest.mark.parametrize(
        ("append", "prefix"),
        [
            (lambda w, t, p: ZImageTrainingAdapter._append_train_lora_weights(w, t, p), "diffusion_model"),
            (lambda w, t, p: Flux2BaseTrainingAdapter._append_train_lora_weights(w, t, p), "transformer"),
            (lambda w, t, p: ErnieTrainingAdapter._append_lora_weights(w, t, p), "transformer"),
            (
                lambda w, t, p: Krea2TrainingAdapter._append_train_lora_weights(
                    SimpleNamespace(_krea2=SimpleNamespace(transformer=t)), w, p
                ),
                "transformer",
            ),
        ],
    )
    def test_every_trainer_saves_alpha(self, monkeypatch, append, prefix):
        trained = _trained_lora(ALPHA / RANK)
        monkeypatch.setattr(TrainingUtil, "get_train_lora", staticmethod(lambda transformer, module_path: trained))
        weights = {}
        append(weights, None, "layers.0.proj")
        assert float(weights[f"{prefix}.layers.0.proj.alpha"]) == ALPHA


class _GridAdapter(TrainingAdapter):
    # Draws from the scheduler's grid (no sample_sigma), with a schedule that depends on the image size
    def __init__(self):
        self.sigmas = None

    def create_config(self, training_spec, *, width: int, height: int):
        return _Config(width, height)

    def predict_noise(self, *, t, latents_t, sigmas, cond, config):  # noqa: ARG002
        self.sigmas = sigmas
        return mx.zeros_like(latents_t)


class _Config:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.num_inference_steps, self.precision, self.image_seq_len = 2, mx.float32, width * height // 256
        self._scheduler = None

    @property
    def scheduler(self):
        if self._scheduler is None:
            self._scheduler = SimpleNamespace(sigmas=mx.array([self.width / 1024, 0.0]))
        return self._scheduler


class TestNoiseSchedule:
    def test_an_item_of_another_size_gets_its_own_schedule(self):
        adapter = _GridAdapter()
        spec = SimpleNamespace(training_loop=SimpleNamespace(timestep_low=0, timestep_high=None, timestep_type=None))
        base_config = adapter.create_config(spec, width=1024, height=1024)
        _ = base_config.scheduler
        for width, first_sigma in ((512, 0.5), (1024, 1.0)):
            item = DataItem(data_id=0, prompt="", image_path="x.png", clean_latents=mx.zeros((1, 4)), cond=None, width=width, height=width)  # fmt: off
            TrainingTrainer._single_example_loss(adapter, spec, base_config, item, random.Random(0))
            assert float(adapter.sigmas[0]) == first_sigma

    def test_the_hooks_default_to_doing_nothing(self):
        adapter = _GridAdapter()
        assert adapter.sample_sigma(rng=random.Random(0), image_seq_len=4096) is None
        assert adapter.release_text_encoder(SimpleNamespace()) is False


class TestTimestepOptions:
    @pytest.mark.parametrize("option", [{"timestep_type": "sigmoid"}, {"timestep_low": 2}, {"timestep_high": 10}])
    def test_they_are_refused_when_the_adapter_draws_its_own_noise_level(self, monkeypatch, option):
        class _ContinuousAdapter(_GridAdapter):
            def sample_sigma(self, *, rng, image_seq_len):  # noqa: ARG002
                return 0.5

        loop = SimpleNamespace(**{"timestep_low": 0, "timestep_high": None, "timestep_type": None, **option})
        spec = SimpleNamespace(model="fibo2", training_loop=loop)
        with pytest.raises(ValueError, match="remove timestep_low, timestep_high and timestep_type"):
            runner_module.TrainingRunner._check_timestep_options(_ContinuousAdapter, spec)
        # A grid adapter takes them
        runner_module.TrainingRunner._check_timestep_options(_GridAdapter, spec)

    def test_a_bad_config_stops_before_the_model_loads(self, monkeypatch, tmp_path):
        from mflux.models.fibo2.training_adapter.fibo2_training_adapter import Fibo2TrainingAdapter

        built = []
        monkeypatch.setattr(Fibo2TrainingAdapter, "__init__", lambda self, **kwargs: built.append(kwargs))
        loop = SimpleNamespace(timestep_low=0, timestep_high=None, timestep_type="sigmoid")
        spec = SimpleNamespace(model="fibo2", training_loop=loop, cache_limit_gb=None, seed=0, is_edit=False)
        monkeypatch.setattr(runner_module.TrainingSpec, "resolve", staticmethod(lambda **kwargs: spec))
        with pytest.raises(ValueError, match="remove timestep_low"):
            runner_module.TrainingRunner.train(config_path=str(tmp_path / "train.json"), resume_path=None)
        assert built == []


@pytest.mark.parametrize("alpha", [0, -4, float("nan"), float("inf")])
def test_a_target_alpha_must_be_positive(tmp_path, alpha):
    from mflux.models.common.training.state.training_spec import TrainingSpec

    (tmp_path / "01.txt").write_text("a fox", encoding="utf-8")
    (tmp_path / "01.jpeg").write_bytes(b"")
    conf = {
        "model": "dev", "seed": 42, "steps": 20, "data": str(tmp_path),
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "lora_layers": {"targets": [{"module_path": "proj", "rank": 4, "alpha": alpha}]},
    }  # fmt: off
    with pytest.raises(ValueError, match="'alpha' must be a number > 0"):
        TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)


class _ZImageGlobals(nn.Module):
    # The Z-Image global targets, under the transformer's module paths
    def __init__(self):
        super().__init__()
        self.all_x_embedder = {"2-1": nn.Linear(8, 8)}
        self.all_final_layer = {"2-1": _Model()}
        self.all_final_layer["2-1"].linear = nn.Linear(8, 8)
        self.cap_embedder = [nn.Identity(), nn.Linear(8, 8)]


@pytest.mark.parametrize("module_path", ["all_x_embedder.2-1", "all_final_layer.2-1.linear", "cap_embedder.1"])
def test_z_image_global_targets_load_at_the_trained_alpha(tmp_path, module_path):
    # The trainer saves .alpha for these too; without an alpha pattern they loaded at alpha = rank
    from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping

    trained = _trained_lora(ALPHA / RANK)
    path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(str(path), TrainingUtil.lora_weights(f"diffusion_model.{module_path}", trained))
    x = mx.random.normal((1, 8))
    for role in (None, "train"):
        model = _ZImageGlobals()
        LoRALoader.load_and_apply_lora(ZImageLoRAMapping.get_mapping(), model, [str(path)], [1.0], role=role, bake_lora=False)  # fmt: off
        layer = LoRALoader._get_target_module(model, module_path)
        expected = layer.linear(x) + ALPHA / RANK * (x @ trained.lora_A @ trained.lora_B)
        assert mx.allclose(layer(x), expected, atol=1e-5).item(), role

from types import SimpleNamespace

from mflux.models.common.training.trainer import TrainingTrainer


class _DummyOptimizer:
    def __init__(self, state):
        self.optimizer = SimpleNamespace(state=state)
        self.saved_paths = []

    def save(self, path):
        self.saved_paths.append(path)


class TestTrainingTrainer:
    def test_generate_previews_with_optimizer_offload_low_ram(self, monkeypatch):
        dummy_optimizer = _DummyOptimizer(state=["original_state"])
        training_state = SimpleNamespace(optimizer=dummy_optimizer)
        training_spec = SimpleNamespace(low_ram=True)
        adapter = object()

        preview_state_snapshots = []
        clear_cache_calls = []
        gc_calls = []

        def fake_generate_previews(_adapter, _training_spec, _training_state):
            preview_state_snapshots.append(_training_state.optimizer.optimizer.state)

        monkeypatch.setattr(TrainingTrainer, "_generate_previews", fake_generate_previews)
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.clear_cache", lambda: clear_cache_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.gc.collect", lambda: gc_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.load", lambda _path: {"k": "v"})
        monkeypatch.setattr(
            "mflux.models.common.training.trainer.tree_unflatten",
            lambda items: ["restored", items],
        )

        TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)

        assert len(dummy_optimizer.saved_paths) == 1
        assert dummy_optimizer.saved_paths[0].name == "optimizer_offload.safetensors"
        assert preview_state_snapshots == [[]]
        assert dummy_optimizer.optimizer.state == ["restored", [("k", "v")]]
        assert len(clear_cache_calls) == 2
        assert len(gc_calls) == 2

    def test_generate_previews_with_optimizer_offload_non_low_ram(self, monkeypatch):
        dummy_optimizer = _DummyOptimizer(state=["original_state"])
        training_state = SimpleNamespace(optimizer=dummy_optimizer)
        training_spec = SimpleNamespace(low_ram=False)
        adapter = object()

        preview_state_snapshots = []
        clear_cache_calls = []
        gc_calls = []

        def fake_generate_previews(_adapter, _training_spec, _training_state):
            preview_state_snapshots.append(_training_state.optimizer.optimizer.state)

        monkeypatch.setattr(TrainingTrainer, "_generate_previews", fake_generate_previews)
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.clear_cache", lambda: clear_cache_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.gc.collect", lambda: gc_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.load", lambda _path: {"k": "v"})
        monkeypatch.setattr(
            "mflux.models.common.training.trainer.tree_unflatten",
            lambda items: ["restored", items],
        )

        TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)

        assert len(dummy_optimizer.saved_paths) == 1
        assert dummy_optimizer.saved_paths[0].name == "optimizer_offload.safetensors"
        assert preview_state_snapshots == [[]]
        assert dummy_optimizer.optimizer.state == ["restored", [("k", "v")]]
        assert len(clear_cache_calls) == 2
        assert len(gc_calls) == 2


class TestAccumulationWindow:
    """The window has to close on valid micro-batches, not on iterations."""

    @staticmethod
    def _run(pattern, accum_steps):
        """Feed a pattern of ok/skip micro-batches; return the gradient of each optimizer step."""
        import mlx.core as mx

        accumulated, count, steps = None, 0, []
        for ok in pattern:
            if not ok:
                accumulated, count = None, 0
                continue
            grads, count, at_boundary = TrainingTrainer._fold_into_window(
                {"w": mx.ones((1,))}, accumulated, accum_steps, count
            )
            accumulated = None if at_boundary else grads
            if at_boundary:
                steps.append(float(grads["w"][0]))
        return steps

    def test_a_clean_window_averages_every_micro_batch(self):
        assert self._run([True] * 8, accum_steps=4) == [1.0, 1.0]

    def test_a_skip_extends_the_window_instead_of_shrinking_the_step(self):
        # Micro-batch 2 is non-finite, so the window restarts and needs four more valid
        # ones. Counting iterations would instead close it at the fourth micro-batch with
        # two gradients still divided by four, a half-sized update.
        steps = self._run([True, False, True, True, True, True], accum_steps=4)
        assert steps == [1.0]

    def test_a_window_that_never_completes_never_steps(self):
        assert self._run([True, True, False, True], accum_steps=4) == []

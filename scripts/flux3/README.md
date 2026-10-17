# FLUX 3 transformer parity reference

`tests/image_generation/test_flux3_transformer_parity.py` compares the MLX transformer with fixed Diffusers outputs. The fixture uses small random models, exact weights and exact input tensors. It needs no Hugging Face login or model download. Running the tests needs no Diffusers installation.

## Coverage

The 23 fast tests cover:

- Distilled and Base transformers, with their target-timestep contracts.
- Text-to-image, one reference, and two references with distinct negative layer IDs.
- A final distilled step whose target time is zero.
- Three numerical profiles: `fp32`, `bf16_timestep`, and `bf16`.

`fp32` compares unmodified Diffusers FP32 with MLX FP32, including FP32 timestep embeddings. This test overrides the timestep dtype locally.

`bf16_timestep` keeps model weights and streams in FP32, but rounds the timestep and sinusoidal embedding to BF16. This matches the timestep convention described by the author of [#878](https://github.com/mflux-community/mflux/pull/878#issuecomment-6207137899). Diffusers receives an explicit adapter around `time_proj`: recover `t` from `t * 1000`, round to BF16 before multiplying, then round the sinusoidal output to BF16 and return it in FP32. This is a Diffusers reference with a documented numerical adapter, not a run of the BFL implementation. The MLX test uses the production timestep default.

`bf16` compares BF16 weights and streams with unmodified Diffusers BF16 execution. Both sides keep Q/K normalisation weights in FP32. The MLX timestep default stays active.

The test timestep `0.7137` and target `0.3713` expose BF16 rounding. The FP32 profiles use `rtol=2e-5`, `atol=2e-6`, and relative L2 error below `1e-5`. BF16 uses `rtol=0.02`, `atol=0.02`, and relative L2 error below `0.02` to allow differences between Metal kernels. The tests also reject non-finite output.

This fixture tests transformer maths. It does not test trained-model image quality, quantised weights, prompt encoding, VAE decoding, or a full denoising schedule. Base guidance combines transformer calls outside this scope.

## Reproduce

Run the existing fixture:

```sh
uv run python -m pytest tests/image_generation/test_flux3_transformer_parity.py
```

Regenerate on an Apple Silicon Mac with MPS, using an isolated reference environment:

```sh
uv run --no-project --python 3.11 --with 'torch==2.14.1' --with 'numpy==2.4.6' --with 'transformers==5.20.0' --with 'diffusers @ https://github.com/huggingface/diffusers/archive/7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941.zip' python scripts/flux3/export_parity_reference.py tests/resources/flux3_transformer_parity.npz
```

The exporter verifies the Diffusers revision in its installation metadata. It records the reference source hash, package versions, device and seeds inside the NPZ. Synthetic weights and inputs are generated on CPU with fixed seeds, then every reference forward pass runs on MPS. Arrays use numeric or string dtypes, so tests load them with `allow_pickle=False`.

The checked-in fixture was generated with Python 3.11.13, PyTorch 2.14.1, NumPy 2.4.6 and macOS 14.6. The Diffusers transformer source SHA256 is `0776af11d42fef95712a3d28e3b2c9cd7b13a499cbe8775619d97e301418acea`. The archive contains only synthetic weights and tensors. Re-exporting on another device or runtime can change rounding; review tensor differences before replacing it.

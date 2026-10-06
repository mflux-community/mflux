# Schedulers - User Manual

## What is a scheduler?

A scheduler decides how the model removes noise at each step. Most users do not need to change it. The default for each model gives good results.

## How to use it

Add `--scheduler` to a generate command:

```sh
mflux-generate-krea2 --prompt "a chrome teapot" --steps 8 --scheduler euler -q 8
```

## Which values you can use

| Model | Values | Default |
| --- | --- | --- |
| FLUX.1 | `linear`, `flow_match_euler_discrete` | `linear` |
| Qwen Image, Qwen Image Edit | `linear`, `flow_match_euler_discrete` | `linear` |
| Qwen Image 2.1 (generate) | `linear`, `flow_match_euler_discrete`, `viggle_turbo` | `linear` |
| Qwen Image 2.1 (edit) | `linear`, `viggle_turbo` only | `linear` |
| Z-Image (base command, including `--model z-image-turbo`) | `linear`, `flow_match_euler_discrete` | `flow_match_euler_discrete` |
| Z-Image Turbo (dedicated command) | `linear`, `flow_match_euler_discrete` | `linear` |
| ERNIE-Image | `linear`, `flow_match_euler_discrete` | `linear` |
| Krea 2 | `er_sde`, `euler` | `er_sde` |

Other models set the scheduler for you: FLUX.2, FIBO, Ideogram 4, SeedVR2, Lens, Ming-Image and Boogu Image. Lens, Ming-Image and Boogu Image ignore `--scheduler`. Lens and Ming-Image show a warning. Boogu Image shows no warning. FLUX.2 and FIBO commands also use their own scheduler.

Ideogram 4 has `--preset`. It selects the step count and noise schedule. Ideogram 4 does not use the `linear` scheduler.

## What the values mean

- `linear`: the standard schedule. Use it unless you have a reason not to.
- `flow_match_euler_discrete`: a different noise schedule for flow-match models.
- `er_sde`, `euler`: two samplers for Krea 2. `er_sde` is the default.
- `viggle_turbo`: for the Viggle turbo LoRA on Qwen Image 2.1. Use `--steps 6` and `--lora-paths` with the turbo LoRA. Other step counts give an error.

## Use your own scheduler

A developer can write a scheduler class. You then pass its full Python path:

```sh
mflux-generate --prompt "a lighthouse" --scheduler my_package.my_module.MyScheduler
```

The package must be installed in the same Python environment as MFLUX.

## Errors

- `The scheduler 'x' is not implemented by mflux.`: the name is wrong. Check the table above.
- `viggle_turbo ... --steps`: the step count is not 6. Set `--steps 6`.
- The help text for `--scheduler` says "linear only for now". That text is out of date.

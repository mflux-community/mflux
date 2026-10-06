# Schedulers - Technical Overview

## What a scheduler does

A scheduler gives the denoise loop a list of noise levels (sigmas). Most schedulers also do the update at each step. A scheduler can instead leave the update to a separate sampler. `Krea2FlowScheduler` does this. Its `step()` raises `NotImplementedError`.

Every scheduler inherits from `BaseScheduler` in `src/mflux/models/common/schedulers/base_scheduler.py`. The class needs two members:

- `sigmas` is a property. It returns the noise levels as an `mx.array`, with a final `0`.
- `step(noise, timestep, latents, **kwargs)` returns the new latents. A scheduler that delegates the update to a sampler can raise `NotImplementedError`.

`scale_model_input(latents, t)` is optional. It returns the latents unchanged by default.

## How a scheduler is chosen

`Config` (`src/mflux/models/common/config/config.py`) holds the scheduler name as a string. It builds the scheduler on the first read of `config.scheduler`. It looks for the name in this order:

1. `"linear"` builds `LinearScheduler`.
2. A name in `SCHEDULER_REGISTRY` builds that class.
3. A name with a dot, such as `my_pkg.my_module.MyScheduler`, is imported. The class must inherit from `BaseScheduler`.
4. Any other name raises `NotImplementedError`.

The command line flag is `--scheduler`. It is defined in `src/mflux/cli/parser/parsers.py`. The default is `linear`.

## Schedulers that exist today

| Name | Class | Where | Notes |
| --- | --- | --- | --- |
| `linear` | `LinearScheduler` | common | Straight sigma line from 1.0 down. Applies a resolution-based shift when the model needs it. Plain Euler step. |
| `flow_match_euler_discrete` | `FlowMatchEulerDiscreteScheduler` | common | Flow-match Euler with a dynamic shift. Used by FLUX.2, FIBO and the Z-Image base model. |
| `seedvr2_euler` | `SeedVR2EulerScheduler` | common | For SeedVR2 only. SeedVR2 sets it in code. |
| `euler`, `er_sde` | `Krea2FlowScheduler` | `krea2/model/krea2_scheduler` | Krea 2 only. The class builds the sigmas. The Krea 2 sampler does the stepping. |
| `viggle_turbo` | `ViggleTurboScheduler` | `qwen21/model/qwen21_scheduler.py` | Qwen Image 2.1 only. Fixed 6-step schedule for the Viggle turbo LoRA. Needs `--steps 6`. |

The common schedulers register with the names `linear`, `flow_match_euler_discrete` and `seedvr2_euler`. The class names also work.

## Which models have scheduler options

### The user can choose the scheduler

| Model | Choices | Default |
| --- | --- | --- |
| FLUX.1 (all commands) | any registered name or external class | `linear` |
| Qwen Image and Qwen Image Edit | any registered name or external class | `linear` |
| Qwen Image 2.1 (generate) | any registered name or external class | `linear` |
| Qwen Image 2.1 (edit) | `linear`, `viggle_turbo` only | `linear` |
| Z-Image (base command, including `--model z-image-turbo`) | any registered name or external class | `flow_match_euler_discrete` |
| Z-Image Turbo and Turbo ControlNet (dedicated commands) | any registered name or external class | `linear` |
| ERNIE-Image and ERNIE-Image Turbo | any registered name or external class | `linear` |
| Krea 2 | `er_sde`, `euler` | `er_sde` (`linear` maps to `er_sde`) |

### The code fixes the scheduler

The user cannot change the scheduler for these models.

| Model | Scheduler |
| --- | --- |
| FLUX.2 and FLUX.2 Edit | `flow_match_euler_discrete` (the CLI sets it). |
| FIBO and FIBO Edit | `flow_match_euler_discrete` (the CLI sets it). |
| Ideogram 4 | `Config` names `linear`, but the denoise loop does not use it. `Ideogram4Scheduler.make_timesteps` builds the timesteps. Use `--preset` to select the step count and noise schedule. |
| SeedVR2 | `seedvr2_euler` |
| Lens, Ming-Image, Boogu Image | The model uses the schedule it was trained on. The CLI accepts `--scheduler` and ignores it. Lens and Ming warn (they list it in `IGNORED_OPTIONS`). Boogu does not warn. |

Python callers can pass `scheduler=` to the FLUX.2 and FIBO model classes. The CLI does not.

## How to add a new scheduler

1. Create a class that inherits from `BaseScheduler`.
2. Make the constructor take one argument, the `Config`. Read `config.num_inference_steps`, `config.width`, `config.height` and `config.model_config` from it.
3. Add the `sigmas` property. Return `num_steps + 1` values. The last value is `0`.
4. Add `step()`. For a plain Euler update, copy `LinearScheduler.step`:
   `latents + noise * (sigmas[t + 1] - sigmas[t])`.
5. Add a `timesteps` property if the model reads it. FLUX.2 and Qwen Image Edit do.
6. Optional: add `set_image_seq_len(n)`. `Config` calls it when the model needs a sigma shift.
7. Optional: add a static `check_args(parser, args)`. The CLI can call it before the model loads, to fail fast on bad flags. `ViggleTurboScheduler` shows how.
8. Register the class. Choose one way:
   - **Built in:** call `register_contrib(MyScheduler, "my_name")` in a module. Import that module from the model package or initializer, so the registration runs. Krea 2 and Qwen 2.1 do this.
   - **External:** do not register. The user passes `--scheduler my_pkg.my_module.MyScheduler`. The package must be importable.
9. Make the model accept the name. Most FLUX.1, Qwen and ERNIE commands pass `args.scheduler` straight through. Some commands limit the names (Qwen 2.1 edit, Krea 2). Change those checks if you need to.
10. Add a test. Use the tiny, hermetic tests as a pattern. Do not run image generation to test a new scheduler.

### Points to check

- The denoise loops call `config.scheduler.sigmas`, `scale_model_input` and `step`. The FLUX.1 transformer reads `sigmas[t] * num_train_steps` as its timestep.
- A model that has its own stepper (Krea 2) uses the scheduler for sigmas only. Its `step()` can raise `NotImplementedError`.
- Image-to-image starts the loop at `config.init_time_step`. Your sigmas must work from that index.
- The `--scheduler` help text in `parsers.py` still says "linear only for now". It is out of date.
- The registry is global. A name registers only after its module is imported.

## Key files

- `src/mflux/models/common/schedulers/__init__.py`: registry, `register_contrib`, external import.
- `src/mflux/models/common/schedulers/base_scheduler.py`: base class.
- `src/mflux/models/common/config/config.py`: scheduler lookup.
- `src/mflux/cli/parser/parsers.py`: `--scheduler` flag.

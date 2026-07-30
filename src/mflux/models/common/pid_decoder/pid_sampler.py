import mlx.core as mx
from tqdm import tqdm

from mflux.models.common.pid_decoder.pixdit.pixdit_network import PidNet

# Source: pid_distill_model.py::_student_sample_loop / _student_sample_1step /
# _velocity_to_x0 (fetched and cross-checked 2026-07-24 against
# nv-tlabs/PiD@main:pid/_src/models/pid_distill_model.py).
#
# Confirmed for the target checkpoint: prediction_type="velocity",
# student_sample_type="sde", fm_timescale=1000.0.
STUDENT_T_LIST = [0.999, 0.866, 0.634, 0.342, 0.0]


def _velocity_to_x0(x_t: mx.array, v: mx.array, t: mx.array) -> mx.array:
    # prediction_type="velocity" (documented default): x0 = x_t - t * net_output.
    # pid_distill_model.py::_net_output_to_x0, line 743-757.
    t_bcast = t.reshape(-1, 1, 1, 1)
    return x_t - t_bcast * v


def sample(
    net: PidNet,
    caption_embs: mx.array,
    lq_latent: mx.array,
    sigma: mx.array,
    target_h: int,
    target_w: int,
    seed: int,
    num_steps: int = 4,
    timescale: float = 1000.0,
) -> mx.array:
    """4-step distilled SDE sampling loop: noise -> x0 in [-1, 1].

    Matches _student_sample_loop's control flow: for a t_list of length N+1
    (N=num_steps), runs N-1 intermediate re-noise steps then one final step
    that emits clean x0 at t_list[-2] (the last non-zero timestep). Only the
    validated 4-step / 5-element STUDENT_T_LIST schedule is supported.
    """
    if num_steps != 4:
        raise NotImplementedError("Only the validated 4-step schedule (STUDENT_T_LIST) is supported this phase.")

    B = lq_latent.shape[0]
    mx.random.seed(seed)
    x = mx.random.normal((B, 3, target_h, target_w))
    # Each tick lands after the step's mx.eval below, so it reflects real elapsed work
    # rather than the lazy graph being queued.
    progress = tqdm(total=num_steps, desc="PiD decode")

    # Intermediate steps: t_list has 5 entries -> 3 iterations here, then 1 final step = 4 net calls.
    for i in range(len(STUDENT_T_LIST) - 2):
        t_cur = mx.full((B,), STUDENT_T_LIST[i])
        t_next = STUDENT_T_LIST[i + 1]
        v_pred = net(x, t_cur * timescale, caption_embs, lq_latent, sigma)
        x0_pred = _velocity_to_x0(x, v_pred, t_cur)
        eps = mx.random.normal(x0_pred.shape)
        x = (1.0 - t_next) * x0_pred + t_next * eps
        # Force evaluation per step -- matches every other sampling loop in this codebase
        # (e.g. ZImage.generate_image's mx.eval(latents) per diffusion step). Without this,
        # MLX defers all 4 PidNet forward passes (14+2 depth transformer at full SR
        # resolution) into one unified lazy graph, which can produce a single Metal command
        # buffer large enough to trip the OS's GPU watchdog (observed: "[METAL] Command
        # buffer execution failed: Caused GPU Timeout Error" when pid-decode ran immediately
        # after a live generation in the same process, Task 12 validation, 2026-07-24).
        mx.eval(x)
        progress.update()

    # Final step: t_next == 0 implicitly, uses t_list[-2] (last non-zero timestep, not t_list[-1] == 0.0).
    t_final = mx.full((B,), STUDENT_T_LIST[-2])
    v_pred = net(x, t_final * timescale, caption_embs, lq_latent, sigma)
    x0 = _velocity_to_x0(x, v_pred, t_final)
    result = mx.clip(x0, -1.0, 1.0)
    mx.eval(result)
    progress.update()
    progress.close()
    return result

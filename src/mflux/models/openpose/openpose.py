"""OpenPose body-pose preprocessor, native in MLX.

Ports Hzzone/pytorch-openpose (the body model controlnet_aux uses) to mlx.nn: a VGG-19 backbone feeds a
two-branch, six-stage network that emits part-affinity fields (PAF, 38ch) and keypoint heatmaps (19ch).
The network runs in MLX; peak finding, greedy limb assembly and skeleton drawing are numpy/cv2, matching
the reference. Only weight loading touches torch (a .pth), like DepthPro and the HED preprocessor. The
body_pose_model weights are downloaded at runtime from lllyasviel/Annotators, the same source controlnet_aux uses.
"""

from __future__ import annotations

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import PIL.Image

_HF_REPO = "lllyasviel/Annotators"
_HF_FILE = "body_pose_model.pth"

_VGG = [
    "conv1_1",
    "conv1_2",
    "P",
    "conv2_1",
    "conv2_2",
    "P",
    "conv3_1",
    "conv3_2",
    "conv3_3",
    "conv3_4",
    "P",
    "conv4_1",
    "conv4_2",
    "conv4_3_CPM",
    "conv4_4_CPM",
]
_S1 = ["conv5_1_CPM_L1", "conv5_2_CPM_L1", "conv5_3_CPM_L1", "conv5_4_CPM_L1", "conv5_5_CPM_L1"]

# COCO-18 skeleton: limb keypoint pairs (1-indexed), the PAF channels per limb, and per-keypoint colors.
_LIMB_SEQ = [
    [2, 3],
    [2, 6],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [2, 9],
    [9, 10],
    [10, 11],
    [2, 12],
    [12, 13],
    [13, 14],
    [2, 1],
    [1, 15],
    [15, 17],
    [1, 16],
    [16, 18],
    [3, 17],
    [6, 18],
]
_MAP_IDX = [
    [31, 32],
    [39, 40],
    [33, 34],
    [35, 36],
    [41, 42],
    [43, 44],
    [19, 20],
    [21, 22],
    [23, 24],
    [25, 26],
    [27, 28],
    [29, 30],
    [47, 48],
    [49, 50],
    [53, 54],
    [51, 52],
    [55, 56],
    [37, 38],
    [45, 46],
]
_COLORS = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]


class _OpenPoseBodyNet(nn.Module):
    def __init__(self, state: dict[str, mx.array]):
        super().__init__()
        # torch Conv2d weight is OIHW; MLX conv2d weight is OHWI. Preload once as (weight, bias, pad).
        self._w = {}
        for name, weight in state.items():
            if not name.endswith(".weight"):
                continue
            key = name[: -len(".weight")]
            w = weight.transpose(0, 2, 3, 1)
            self._w[key] = (w, state[f"{key}.bias"], w.shape[1] // 2)

    def _seq(self, h: mx.array, names: list[str], last_no_relu: bool = True) -> mx.array:
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for i, name in enumerate(names):
            if name == "P":
                h = pool(h)
                continue
            w, b, pad = self._w[name]
            h = mx.conv2d(h, w, padding=pad) + b
            if not (last_no_relu and i == len(names) - 1):
                h = mx.maximum(h, 0.0)
        return h

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        feat = self._seq(x, _VGG, last_no_relu=False)
        paf = self._seq(feat, _S1)
        heat = self._seq(feat, [k.replace("L1", "L2") for k in _S1])
        for stage in range(2, 7):
            inp = mx.concatenate([paf, heat, feat], axis=3)
            paf = self._seq(inp, [f"Mconv{i}_stage{stage}_L1" for i in range(1, 8)])
            heat = self._seq(inp, [f"Mconv{i}_stage{stage}_L2" for i in range(1, 8)])
        return paf, heat


class OpenPoseBody:
    def __init__(self):
        import torch  # weights only; the forward pass is MLX
        from huggingface_hub import hf_hub_download

        raw = torch.load(hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE), map_location="cpu", weights_only=True)
        self._net = _OpenPoseBodyNet({k: mx.array(v.float().numpy()) for k, v in raw.items()})

    def pose_map(self, image: PIL.Image.Image, detect_resolution: int = 512) -> PIL.Image.Image:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        out_h, out_w = bgr.shape[:2]
        paf, heat = self._infer(bgr, detect_resolution, (out_h, out_w))
        candidate, subset = self._assemble(paf, heat)
        return self._draw((out_h, out_w), candidate, subset)

    def _infer(self, bgr: np.ndarray, detect_resolution: int, out_hw: tuple[int, int]):
        out_h, out_w = out_hw
        scale = detect_resolution / max(out_h, out_w)
        img = cv2.resize(bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
        padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        x = mx.array(padded[None].astype(np.float32) / 256.0 - 0.5)  # NHWC
        paf, heat = self._net(x)
        mx.eval(paf, heat)
        paf, heat = np.array(paf)[0], np.array(heat)[0]

        def to_full(m):
            m = cv2.resize(m, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            m = m[: padded.shape[0] - pad_h, : padded.shape[1] - pad_w]
            return cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        return to_full(paf), to_full(heat)

    @staticmethod
    def _assemble(paf: np.ndarray, heat: np.ndarray, thre1: float = 0.1, thre2: float = 0.05):
        all_peaks, counter = [], 0
        for part in range(18):
            m = cv2.GaussianBlur(heat[:, :, part], (0, 0), sigmaX=3)
            left, right, up, down = (np.zeros_like(m) for _ in range(4))
            left[1:, :], right[:-1, :] = m[:-1, :], m[1:, :]
            up[:, 1:], down[:, :-1] = m[:, :-1], m[:, 1:]
            binary = (m >= left) & (m >= right) & (m >= up) & (m >= down) & (m > thre1)
            peaks = list(zip(*np.nonzero(binary)[::-1]))
            with_score = [p + (heat[p[1], p[0], part], counter + i) for i, p in enumerate(peaks)]
            all_peaks.append(with_score)
            counter += len(peaks)

        connection_all, special_k, mid_num = [], [], 10
        for k in range(len(_MAP_IDX)):
            score_mid = paf[:, :, [x - 19 for x in _MAP_IDX[k]]]
            cand_a, cand_b = all_peaks[_LIMB_SEQ[k][0] - 1], all_peaks[_LIMB_SEQ[k][1] - 1]
            if not cand_a or not cand_b:
                special_k.append(k)
                connection_all.append([])
                continue
            candidates = []
            for i, a in enumerate(cand_a):
                for j, b in enumerate(cand_b):
                    vec = np.subtract(b[:2], a[:2])
                    norm = max(0.001, float(np.hypot(*vec)))
                    unit = vec / norm
                    line = list(zip(np.linspace(a[0], b[0], mid_num), np.linspace(a[1], b[1], mid_num)))
                    vx = np.array([score_mid[int(round(y)), int(round(x)), 0] for x, y in line])
                    vy = np.array([score_mid[int(round(y)), int(round(x)), 1] for x, y in line])
                    scores = vx * unit[0] + vy * unit[1]
                    prior = min(0.5 * heat.shape[0] / norm - 1, 0)
                    total = float(scores.mean()) + prior
                    if np.count_nonzero(scores > thre2) > 0.8 * len(scores) and total > 0:
                        candidates.append([i, j, total])
            candidates.sort(key=lambda z: z[2], reverse=True)
            connection = np.zeros((0, 5))
            for i, j, s in candidates:
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                    if len(connection) >= min(len(cand_a), len(cand_b)):
                        break
            connection_all.append(connection)

        subset = -1 * np.ones((0, 20))
        candidate = np.array([p for peaks in all_peaks for p in peaks])
        for k in range(len(_MAP_IDX)):
            if k in special_k:
                continue
            part_as, part_bs = connection_all[k][:, 0], connection_all[k][:, 1]
            index_a, index_b = np.array(_LIMB_SEQ[k]) - 1
            for i in range(len(connection_all[k])):
                found, rows = 0, [-1, -1]
                for j in range(len(subset)):
                    if subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]:
                        rows[found] = j
                        found += 1
                        if found == 2:
                            break
                if found == 1:
                    j = rows[0]
                    if subset[j][index_b] != part_bs[i]:
                        subset[j][index_b] = part_bs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:
                    j1, j2 = rows
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:
                        subset[j1][:-2] += subset[j2][:-2] + 1
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:
                        subset[j1][index_b] = part_bs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[index_a], row[index_b] = part_as[i], part_bs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

        drop = [i for i in range(len(subset)) if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4]
        return candidate, np.delete(subset, drop, axis=0)

    @staticmethod
    def _draw(shape: tuple[int, int], candidate: np.ndarray, subset: np.ndarray) -> PIL.Image.Image:
        canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        for i in range(17):
            for person in subset:
                index = person[np.array(_LIMB_SEQ[i]) - 1]
                if -1 in index:
                    continue
                pts = candidate[index.astype(int), :2]
                mx_, my_ = pts[:, 0], pts[:, 1]
                length = float(np.hypot(my_[0] - my_[1], mx_[0] - mx_[1]))
                angle = float(np.degrees(np.arctan2(my_[0] - my_[1], mx_[0] - mx_[1])))
                poly = cv2.ellipse2Poly((int(mx_.mean()), int(my_.mean())), (int(length / 2), 4), int(angle), 0, 360, 1)
                # Reference (controlnet_aux) blends each limb into the canvas at 0.6 so overlapping
                # limbs mix instead of the later one replacing the earlier one.
                cur_canvas = canvas.copy()
                cv2.fillConvexPoly(cur_canvas, poly, _COLORS[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        for i in range(18):
            for person in subset:
                idx = int(person[i])
                if idx == -1:
                    continue
                x, y = candidate[idx][:2]
                cv2.circle(canvas, (int(x), int(y)), 4, _COLORS[i], thickness=-1)
        return PIL.Image.fromarray(canvas)

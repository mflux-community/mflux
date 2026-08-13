import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np


@dataclass(frozen=True, slots=True)
class SafetensorsTensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


class SafetensorsReader:
    _DTYPES: dict[str, tuple[np.dtype, mx.Dtype | None]] = {
        "F8_E4M3": (np.dtype(np.uint8), mx.uint8),
        "BF16": (np.dtype(np.uint16), mx.bfloat16),
        "F32": (np.dtype(np.float32), mx.float32),
        "F16": (np.dtype(np.float16), mx.float16),
        "I64": (np.dtype(np.int64), mx.int64),
        "I32": (np.dtype(np.int32), mx.int32),
        "U8": (np.dtype(np.uint8), mx.uint8),
        "BOOL": (np.dtype(np.bool_), mx.bool_),
    }

    @staticmethod
    def read_file(path: str | Path) -> dict[str, mx.array]:
        file_path = Path(path)
        with file_path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))

        data_start = 8 + header_len
        tensors: dict[str, mx.array] = {}
        for name, raw_info in header.items():
            if name == "__metadata__" or not isinstance(raw_info, dict):
                continue
            info = SafetensorsTensorInfo(
                name=name,
                dtype=str(raw_info["dtype"]),
                shape=tuple(int(dim) for dim in raw_info["shape"]),
                data_offsets=(
                    int(raw_info["data_offsets"][0]),
                    int(raw_info["data_offsets"][1]),
                ),
            )
            tensors[name] = SafetensorsReader._read_tensor(file_path, data_start, info)
        return tensors

    @staticmethod
    def read_directory(directory: str | Path) -> dict[str, mx.array]:
        root = Path(directory).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Missing weight directory: {root}")
        files = sorted(p for p in root.glob("*.safetensors") if not p.name.startswith("._"))
        if not files:
            raise FileNotFoundError(f"No safetensors files found under {root}")
        weights: dict[str, mx.array] = {}
        for file in files:
            weights.update(SafetensorsReader.read_file(file))
        return weights

    @staticmethod
    def _read_tensor(path: Path, data_start: int, info: SafetensorsTensorInfo) -> mx.array:
        try:
            np_dtype, mx_view_dtype = SafetensorsReader._DTYPES[info.dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported safetensors dtype {info.dtype!r} in {path}") from exc

        start, end = info.data_offsets
        expected = math.prod(info.shape) if info.shape else 1
        byte_count = end - start
        if byte_count != expected * np_dtype.itemsize:
            raise ValueError(
                f"Tensor {info.name!r} in {path} has {byte_count} bytes, expected {expected * np_dtype.itemsize}"
            )

        array = np.memmap(
            path,
            mode="r",
            offset=data_start + start,
            dtype=np_dtype,
            shape=info.shape or (),
        )
        tensor = mx.array(array)
        if info.dtype == "BF16":
            return mx.view(tensor, mx.bfloat16)
        if mx_view_dtype is not None and tensor.dtype != mx_view_dtype:
            return tensor.astype(mx_view_dtype)
        return tensor

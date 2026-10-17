import shutil
import sys
from pathlib import Path

import mlx.core as mx

from mflux.models.moss_image.weights.moss_image_weight_mapping import MOSSImageWeightMapping


def convert_moss_image(
    src_dir: str | Path,
    out_dir: str | Path,
):
    src = Path(src_dir)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print("🚀 开始将 MOSS-Image 转换为 MLX 原生格式:")
    print(f"   源路径: {src}")
    print(f"   目标路径: {dest}")

    # 1. 转换 Transformer
    t_src = src / "transformer"
    t_dest = dest / "transformer"
    t_dest.mkdir(parents=True, exist_ok=True)

    mapping = MOSSImageWeightMapping.get_transformer_mapping()
    target_map = {src_p: t.to_pattern for t in mapping for src_p in t.from_pattern}

    print("\n📦 正在转换 Transformer 权重 ...")
    t_files = sorted(list(t_src.glob("*.safetensors")))
    assert t_files, f"No transformer safetensors found in {t_src}"

    converted_tensors = {}
    for tf in t_files:
        print(f"   读取分片: {tf.name} ...")
        loaded = mx.load(str(tf))
        for k, v in loaded.items():
            mapped_k = target_map.get(k, k)
            converted_tensors[mapped_k] = v

    print(f"   共转换 {len(converted_tensors)} 个 Transformer 权重。保存中 ...")
    # 分片保存，每个分片约 4.5GB，或者直接保存为 0, 1 分片
    shard0, shard1 = {}, {}
    for i, (k, v) in enumerate(converted_tensors.items()):
        if i < len(converted_tensors) // 2:
            shard0[k] = v
        else:
            shard1[k] = v

    print(f"   写入分片 0.safetensors ({len(shard0)} keys) ...")
    mx.save_safetensors(str(t_dest / "0.safetensors"), shard0)
    print(f"   写入分片 1.safetensors ({len(shard1)} keys) ...")
    mx.save_safetensors(str(t_dest / "1.safetensors"), shard1)

    # 复制 transformer/config.json
    if (t_src / "config.json").exists():
        shutil.copy2(t_src / "config.json", t_dest / "config.json")

    # 2. 复制/链接 VAE
    v_src = src / "vae"
    v_dest = dest / "vae"
    v_dest.mkdir(parents=True, exist_ok=True)
    for vf in v_src.glob("*"):
        if vf.is_file():
            shutil.copy2(vf, v_dest / vf.name)
    print("✅ VAE 转移完成")

    # 3. 复制/链接 text_encoder 和 tokenizer
    for comp in ["text_encoder", "tokenizer"]:
        c_src = src / comp
        c_dest = dest / comp
        c_dest.mkdir(parents=True, exist_ok=True)
        for cf in c_src.glob("*"):
            if cf.is_file():
                shutil.copy2(cf, c_dest / cf.name)
        print(f"✅ {comp} 转移完成")

    # 4. 复制根目录元信息
    for root_file in ["model_index.json", "README.md"]:
        if (src / root_file).exists():
            shutil.copy2(src / root_file, dest / root_file)

    print(f"\n🎉 MOSS-Image-SFT MLX 原生模型转换成功！已落盘至: {dest}")


if __name__ == "__main__":
    src_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/vanch/models/hf_raw/MOSS-Image-SFT"
    dest_path = sys.argv[2] if len(sys.argv) > 2 else "/Users/vanch/models/mlx/MOSS-Image-SFT-mlx-bf16"
    convert_moss_image(src_path, dest_path)

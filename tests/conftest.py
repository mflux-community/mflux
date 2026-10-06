import os

# MLX 0.32 runs float32 matmuls at reduced precision on hardware that supports it (TF32 on an M5
# GPU, about 8e-4 of the result) unless MLX_ENABLE_TF32=0. The tests compare float32 results with
# each other and with torch references, so they always get full-precision matmuls (#812), even when
# the environment turns TF32 on. MLX reads the variable at the first matmul, after this file loads.
os.environ["MLX_ENABLE_TF32"] = "0"

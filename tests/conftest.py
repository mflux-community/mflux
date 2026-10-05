import os

# MLX 0.32 runs float32 matmuls at reduced precision on hardware that supports it (TF32 on an M5
# GPU, about 8e-4 of the result) unless MLX_ENABLE_TF32=0. The tests compare float32 results with
# each other and with torch references, so they get full-precision matmuls on every machine (#812).
# MLX reads the variable at the first matmul, which is after this file loads.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

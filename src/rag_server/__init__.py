import os
os.environ["OMP_NUM_THREADS"] = "1"  # Fix for sklearn OpenMP TLS issue
os.environ["LD_PRELOAD"] = "/home/ias/satya/robotdog_guide/.venv/lib/python3.10/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0"
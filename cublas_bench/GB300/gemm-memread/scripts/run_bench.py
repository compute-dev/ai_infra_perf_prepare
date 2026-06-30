import subprocess
import argparse
import json
import os
import sys
import platform

parser = argparse.ArgumentParser()
parser.add_argument('--dtype', default="fp8", choices=["nvfp4-out-e2m1", "nvfp4-out-e5m2", "mxfp4-out-e2m1","mxfp4-out-e5m2", "fp8", "fp16", "bf16", "fp32"], help="GEMM datatype, default = fp8")
parser.add_argument('--duty_cycle', type=str, help="Duty cycle of the GEMM")
parser.add_argument('--gemm_dim', required=True, choices=["square", "no-wave-quant", "min-dram"], help="GEMM dimensions: square , no-wave-quant, min-dram")
parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to run on, default = 1")
parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (only used if --gpus is 1), default = 0")
args, unknown_args = parser.parse_known_args()

# Detect host type
hosttype = platform.machine().lower()

GEMM_DIR = os.getenv("GEMM_DIR", default="")
CUDA_HOME = os.getenv("CUDA_HOME", default="")
config_path = GEMM_DIR + "/configs/commandlines.json"

with open(config_path, 'r') as f:
    commandlines = json.loads(f.read())

assert args.duty_cycle in commandlines.keys(), "Duty cycle not found in commandlines.json"

# Select key by tag only (commandlines.json uses tags exclusively)
def _select_dtype_key(commandlines_obj: dict, duty_cycle: str, dtype: str, tag: str) -> str:
    keys = list(commandlines_obj[duty_cycle].keys())
    for key in keys:
        if key.startswith(f"{dtype}_") and (f"_{tag}" in key):
            return key
    available = [k for k in keys if k.startswith(f"{dtype}_")]
    raise KeyError(f"No matching key for dtype={dtype}, tag={tag}. Available: {available[:10]}...")

# Generate dtype key
tag = args.gemm_dim
dtype_key = _select_dtype_key(commandlines, args.duty_cycle, args.dtype, tag)

assert dtype_key in commandlines[args.duty_cycle].keys(), "GEMM dtype not found in commandlines.json"

env = os.environ.copy()
env['GEMM_DIR'] = GEMM_DIR
env['CUDA_HOME'] = CUDA_HOME
gpus = [str(i) for i in range(args.gpus)] if args.gpus > 1 else [str(args.gpu)]
for gpu in gpus:
    cmd = 0
    if args.duty_cycle == "100":
        cmd = commandlines[args.duty_cycle][dtype_key]
    else:
        cmd = commandlines[args.duty_cycle][dtype_key][gpu]
    
    # Expand environment variables in the command
    cmd = os.path.expandvars(cmd)
    
    out = subprocess.check_output(f"{cmd} --device {gpu}", shell=True, env=env)
    out_str = out.decode("utf-8", errors="replace")
    print(out_str)

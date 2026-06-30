import subprocess
import re
import os
import json
import sys
import argparse
import math
import platform

pat = re.compile('GEMM duty cycle (\d+.\d+)')
GEMM_DIR = os.getenv("GEMM_DIR", default="")
CUDA_HOME = os.getenv("CUDA_HOME", default="")

def linear_fit(point1, point2):
    """Calculates the linear fit (slope and intercept) from two points."""
    x1, y1 = point1
    x2, y2 = point2

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    return slope, intercept

def controlled_system(streamiters, gemm_iters, gpu_id):
    cmd = template_cmd.replace("{STREAMITERS}", str(streamiters)).replace("{GEMM_ITERS}", str(gemm_iters)).replace("{WARMUPS}", str(int(10000/gemm_iters))).replace("{ITERS}", str(int(5000/gemm_iters))) + f" --device {gpu_id}"
    
    # Expand environment variables in the command
    cmd = os.path.expandvars(cmd)
    
    env = os.environ.copy()
    env['GEMM_DIR'] = GEMM_DIR
    env['CUDA_HOME'] = CUDA_HOME
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    out = subprocess.check_output(cmd, shell=True, env=env)
    out_str = out.decode("utf-8")
    return out_str

def get_gemm_duty_cycle(streamiters, gemm_iters, gpu_id):
    out_str = controlled_system(streamiters, gemm_iters, gpu_id)
    gemm_duty_cycle = float(re.findall(pat, out_str)[0])
    return gemm_duty_cycle * 100

parser = argparse.ArgumentParser()
parser.add_argument('--dtype', default="fp8", choices=["nvfp4-out-e2m1", "nvfp4-out-e5m2","mxfp4-out-e2m1","mxfp4-out-e5m2", "fp8", "fp16", "bf16", "fp32"], help="GEMM datatype, default = fp8")
parser.add_argument('--gemm_dim', required=True, choices=["square", "no-wave-quant", "min-dram"], help="GEMM dimensions: square, no-wave-quant, min-dram")
parser.add_argument('--lower', default=-1, type=int, help="Lower bound for streamiters in the successive approximation method")
parser.add_argument('--upper', default=-1, type=int, help="Upper bound for streamiters in the successive approximation method")
parser.add_argument('--target_duty_cycle', default=65, type=int, help="Target duty cycle for GEMM (pct), default = 65")
parser.add_argument('--tolerance', default=1, type=float, help="Duty cycle variation tolerance, default = 1")
parser.add_argument('--norun', default=False, action='store_true', help="Disable running the duty cycled GEMM commandline, default = False")
parser.add_argument('--nocache', default=False, action='store_true', help="Disable caching the duty cycled GEMM commandline in /benchmarks/configs/commandlines.json, default = False")
parser.add_argument('--gpus', default=1, type=int, help="Number of GPUs to tune")
parser.add_argument('--max_steps', default=20, type=int, help="Maximum number of steps for the successive approximation method, default = 20")
args, unknown_args = parser.parse_known_args()

# Detect host type
hosttype = platform.machine().lower()
config_path = GEMM_DIR + "/configs/commandlines.json"


with open(config_path, 'r') as f:
    commandlines = json.loads(f.read())

# Select key by tag only (commandlines.json uses tags exclusively)
def _select_dtype_key(commandlines_obj: dict, dtype: str, tag: str) -> str:
    keys = list(commandlines_obj['100'].keys())
    for key in keys:
        if key.startswith(f"{dtype}_") and (f"_{tag}" in key):
            return key
    available = [k for k in keys if k.startswith(f"{dtype}_")]
    raise KeyError(f"No matching key for dtype={dtype}, tag={tag}. Available: {available[:10]}...")

# Generate dtype key
dtype_key = _select_dtype_key(commandlines, args.dtype, args.gemm_dim)

target_duty_cycle = args.target_duty_cycle
template_cmd = f"{commandlines['100'][dtype_key]} " + " -w {WARMUPS} -i {ITERS} --gemm_iters {GEMM_ITERS} --streamiters {STREAMITERS}"
lower = 5
upper = 60


for gpu in range(args.gpus):
    gemmi = 1
    step = 0
    print(f"\nWorking on GPU: {gpu}")
    print(f"Getting GEMM duty cycle for boundary values")
    boundary_values = [(lower, get_gemm_duty_cycle(lower, gemmi, gpu)), (upper, get_gemm_duty_cycle(upper, gemmi, gpu))]
    
    # make sure we get a valid upper bound where duty cycle is less than target_duty_cycle
    while boundary_values[1][1] > target_duty_cycle:
        print("Adjusting initial conditions to correctly bound the target duty cycle...")
        new_streamiters = boundary_values[1][0] * 2
        boundary_values[1] = (new_streamiters, get_gemm_duty_cycle(new_streamiters, gemmi, gpu))

    # make sure we get a valid lower bound where duty cycle is greater than target_duty_cycle
    while boundary_values[0][1] < target_duty_cycle:
        print("Adjusting initial conditions to correctly bound the target duty cycle...")
        # if streamiters is 1, we need to double gemm_iters to get a valid lower bound
        if boundary_values[0][0] == 1:
            gemmi *= 2
            # adjust the upper bound to keep it valid
            # doubling gemm and streamiters should keep the duty cycle the same
            boundary_values[1] = (boundary_values[1][0] * 2, boundary_values[1][1])
            # reprofile the lower bound
            boundary_values[0] = (boundary_values[0][0], get_gemm_duty_cycle(1, gemmi, gpu))
        else:
            new_streamiters = max(boundary_values[0][0] // 2, 1)
            boundary_values[0] = (new_streamiters, get_gemm_duty_cycle(new_streamiters, gemmi, gpu))

    # upper bound of the duty cycle is the lower bound of the streamiters. Print "upper bound" for clarity.
    print(f"GEMM duty cycle at upper bound: {boundary_values[0][1]:.2f}")
    print(f"GEMM duty cycle at lower bound: {boundary_values[1][1]:.2f}")
    streamiters = upper
    current_duty_cycle = boundary_values[1][1]
    
    replace_index = 0
    while(abs(target_duty_cycle - current_duty_cycle) >  args.tolerance):
        print(f"Step {step}: Current GEMM duty cycle = {current_duty_cycle:.2f}, Boundary values = {boundary_values}")
        
        # if our lower and upper bounds are within 1 of each other or the calculated streamiters match a previous boundary value, we need finer granularity
        # double gemm_iters and streamiters and try again
        slope, intercept = linear_fit(boundary_values[0], boundary_values[1])
        streamiters = abs(int((target_duty_cycle - intercept)/slope))
        if (boundary_values[0][0] >= boundary_values[1][0] - 1) or (streamiters in [x[0] for x in boundary_values]):
            gemmi *= 2
            boundary_values[0] = (boundary_values[0][0] * 2, boundary_values[0][1])
            boundary_values[1] = (boundary_values[1][0] * 2, boundary_values[1][1])
            slope, intercept = linear_fit(boundary_values[0], boundary_values[1])
            streamiters = abs(int((target_duty_cycle - intercept)/slope))
        current_duty_cycle = get_gemm_duty_cycle(streamiters, gemmi, gpu)
        if current_duty_cycle < target_duty_cycle:
            replace_index = 1
        else:
            replace_index = 0
        boundary_values[replace_index] = (streamiters, current_duty_cycle)
        if step >= args.max_steps - 1:
            print(f"Maximum number of steps ({args.max_steps}) reached, using current duty cycle = {current_duty_cycle:.2f}")
            break
        step += 1

    print(f"Target achieved in step {step}: Current GEMM duty cycle = {current_duty_cycle:.2f}")
    final_cmd = template_cmd.replace("{STREAMITERS}", str(streamiters)).replace("{GEMM_ITERS}", str(gemmi)).replace("{WARMUPS}", str(int(10000/gemmi))).replace("{ITERS}", str(int(5000/gemmi)))
    print(f"Commandline: {final_cmd} --device {gpu}")

    if not args.nocache:
        target_duty_cycle_str = str(target_duty_cycle)
        if target_duty_cycle_str not in commandlines:
            commandlines[target_duty_cycle_str] = {}
        if dtype_key not in commandlines[target_duty_cycle_str]:
            commandlines[target_duty_cycle_str][dtype_key] = {}
        commandlines[target_duty_cycle_str][dtype_key][str(gpu)] = template_cmd.replace("{STREAMITERS}", str(streamiters)).replace("{GEMM_ITERS}", str(gemmi)).replace("{WARMUPS}", str(int(10000/gemmi))).replace("{ITERS}", str(int(5000/gemmi)))
        json_str = json.dumps(commandlines, indent=4)
        with open(config_path, 'w') as f:
            f.write(json_str)

    if not args.norun:
        print(controlled_system(streamiters, gemmi, gpu))


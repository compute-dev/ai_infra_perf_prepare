## Environment Setup
Install the latest CUDA v13.0 

Set the CUDA device order to be consistent with PCI bus order
$ export CUDA_DEVICE_ORDER=PCI_BUS_ID

Set the env var `GEMM_DIR` to benchmark directory and `CUDA_HOME` to CUDA v13.0 
$ export GEMM_DIR=<path to gemm_memread directory>
$ export CUDA_HOME=<path to cuda 13.0 toolkit>

## Running GEMM-MemRead
'--gpu' : <gpu_id>, GPU to run on (only used if --gpus is 1). Default = 0
'--gpus' : <gpu_num>, number of GPUs on the system
'--dtype': <mxfp4-out-e2m1> <fp8> <fp16> or <bf16>, datatype
'--gemm_dim': <square>, <min-dram>, <no-wave-quant>, gemm sizes

$ cd ./scripts/

## Run Basic Linear Algebra
FP4: $ python run_bench.py --dtype mxfp4-out-e2m1 --gemm_dim min-dram --duty_cycle 100 --gpu 0

## Run GEMM-MemRead
1. Tune the duty cycle for the GEMM-MemRead benchmark. Tuning must be done for each GPU and should be re-run for each configuration change
FP4: $ python duty_cycle_controller_v2.5.py --dtype mxfp4-out-e2m1 --gemm_dim min-dram --gpus 4 --tolerance 1 --target_duty_cycle 65
FP8: $ python duty_cycle_controller_v2.5.py --dtype fp8 --gemm_dim min-dram --gpus 4 --tolerance 1 --target_duty_cycle 65 
FP16: $ python duty_cycle_controller_v2.5.py --dtype fp16 --gemm_dim min-dram --gpus 4 --tolerance 1 --target_duty_cycle 65 
BF16: $ python duty_cycle_controller_v2.5.py --dtype bf16 --gemm_dim min-dram --gpus 4 --tolerance 1 --target_duty_cycle 65 

2. The duty_cycle_controller script will cache the tuned duty cycle settings. You can re-run the cached GEMM-MemRead benchmark settings by running the following commands
## Run on multiple GPUs sequentially
$ python run_bench.py --dtype <dtype> --gemm_dim <gemm_dim> --gpus 4 --duty_cycle 65 
FP4: $ python run_bench.py --dtype mxfp4-out-e2m1 --gemm_dim min-dram --gpus 4 --duty_cycle 65 
FP8: $ python run_bench.py --dtype fp8 --gemm_dim min-dram --gpus 4 --duty_cycle 65 
FP16: $ python run_bench.py --dtype fp16 --gemm_dim min-dram --gpus 4 --duty_cycle 65 
BF16: $ python run_bench.py --dtype bf16 --gemm_dim min-dram --gpus 4 --duty_cycle 65 

## Note for GEMM-MemRead
By default, the duty cycle controller script tries to get the duty cycle within +/- 1%
The '--tolerance' setting allows the duty cycle controller to use any duty cycle within +-tolerance%
In case script goes into an infinite loop, increase the tolerance (default is 1%) by the smallest value and try again
For example, --tolerance can be <1.5>, <2>

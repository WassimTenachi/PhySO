# About computational performances

The main performance bottleneck of `physo` is free constant optimization, therefore, in non-parallel execution mode, performances are almost linearly dependent on the number of free constant optimization steps and on the number of trial expressions per epoch (ie. the batch size).

In addition, it should be noted that generating monitoring plots takes ~3s, therefore we suggest making monitoring plots every >10 epochs for low time / epoch cases.

## Expected computational performances

Summary of expected performances with `physo` (in parallel mode):

| Time / epoch | Batch size | # free const | free const <br>opti steps | Example                             | Device                                    |
|--------------|------------|--------------|---------------------------|-------------------------------------|-------------------------------------------|
| ~5s          | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | CPU: Intel W-2155 10c/20t <br>RAM: 128 Go |
| ~10s         | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | CPU: Mac M1 <br>RAM: 16 Go                |
| ~30s         | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | CPU: Intel i7 4770 <br>RAM: 16 Go         |
| ~250s        | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | GPU: Nvidia GV100 <br>VRAM : 32 Go        |
| ~1s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | CPU: Intel W-2155 10c/20t <br>RAM: 128 Go |
| ~6s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | CPU: Mac M1 <br>RAM: 16 Go                |
| ~30s         | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | CPU: Intel i7 4770 <br>RAM: 16 Go         |
| ~4s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | GPU: Nvidia GV100 <br>VRAM : 32 Go        |

Please note that using a CPU typically results in higher performances than when using a GPU.

## Parallel mode

### Parallel free constant optimization

Parallel constant optimization is enabled if and only if :
- The system is compatible (checked by `physo.physym.execute.ParallelExeAvailability`).
- `parallel_mode = True` in the reward computation configuration.
- `physo.physym.reward.USE_PARALLEL_OPTI_CONST = True`.

By default, both of these are true as parallel mode is typically faster for this task.
However, if you are using a batch size <10k, due to communication overhead it might be worth it to disable it for this task via:
```
physo.physym.reward.USE_PARALLEL_OPTI_CONST = False
```

### Parallel reward computation

Parallel reward computation is enabled if and only if :
- The system is compatible (checked by `physo.physym.execute.ParallelExeAvailability`).
- `parallel_mode = True` in the reward computation configuration.
- `physo.physym.reward.USE_PARALLEL_EXE = True`.

By default, `physo.physym.reward.USE_PARALLEL_EXE = False`, i.e. parallelization is not used for this task due to communication overhead making it typically slower for such individually inexpensive tasks.
However, if you are using $>10^6$ data points it tends to be faster, so we recommend enabling it by setting:
```
physo.physym.reward.USE_PARALLEL_EXE = True
```

### Miscellaneous

- Efficiency curves (nb. of CPUs vs individual task time) are produced by `execute_UnitTest.py` in a realistic toy case with batch size = 10k and $10^3$ data points.
- Parallel mode is not available from jupyter notebooks on spawn multiprocessing systems (typically MACs/Windows), run .py scripts on those.
- The use of `parallel_mode` can be managed in the configuration of the reward which can itself be managed through a hyperparameter config file (see `config` folder) which is handy for running a benchmark on an HPC with a predetermined number of CPUs.
- Disabling parallel mode entirely via `USE_PARALLEL_EXE=False` `USE_PARALLEL_OPTI_CONST=False` is recommended before running `physo` in a debugger.

### Efficiency curve in a realistic case

![parallel_performances](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/docs/assets/physo_parallel_efficiency_padded.png)

Computational time optimizing free constants $\{a, b \}$ in $y = a \sin (b.x) + e^{-x}$ over 20 iterations using $10^3$ data points when running this task $10\ 000$ times in parallel on an Apple M1 CPU (a typically fast single core CPU) and an Intel Xeon W-2155 CPU (a typically high core count CPU).


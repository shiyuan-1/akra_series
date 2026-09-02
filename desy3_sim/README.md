# DES Y3 simulation workflows

> [!IMPORTANT]
> The absolute paths, cluster node names, queue settings, modules, and environment names in this directory are examples from the author's computing environment. They will not work unchanged on another computer or cluster. Review and replace them before running any script or notebook.

## Paths and settings to update

| File | Machine-specific values |
| --- | --- |
| `massmapping_100realizations_withNoise.py` | `AKRA_DIR`, `SKYMAP_DIR`, `DATA_FILE`, and optionally `--output-dir` |
| `test_1realization_withNoise.ipynb` | AKRA `utils/` and `core/` paths, `skymap_dir`, and output locations |
| `test_1realization_woNoise.ipynb` | AKRA `utils/` and `core/` paths, `skymap_dir`, and the comparison-figure output path |
| `test_1realization_speed.ipynb` | AKRA `utils/` and `core/` paths and any optional saved-data paths |
| `analyze_power_spectra_withNoise.ipynb` | `INPUT_DIR`, `OUTPUT_DIR`, and the optional local plotting-style path |
| `run_massmapping_100realizations_withNoise*.pbs` | PBS nodes, queue, wall time, module/Conda environment, site helper script, working directory, MPI layout, and output paths |

For example, replace the Python configuration with paths on your own machine:

```python
# Keep the trailing slash because the current scripts concatenate path strings.
AKRA_DIR = "/path/to/your/akra_series/"
SKYMAP_DIR = "/path/to/your/desy3_data/"
DATA_FILE = "your_des_y3_shear_map.hdf5"
```

For batch simulations, an explicit output directory can be supplied instead of using the default location under `SKYMAP_DIR`:

```bash
mpirun -np <number-of-ranks> python -u massmapping_100realizations_withNoise.py \
  --n-realizations 100 \
  --output-dir /path/to/your/output
```

## Cluster-specific configuration

The PBS launchers were written for one particular cluster. Before using `qsub`, review every `#PBS` resource line and update:

- node names and processor counts;
- queue and wall-time limits;
- environment modules and Conda environment;
- `/opt/sharing/cpus-per-task.sh`, which is a site-local helper;
- the `cd` command pointing to `desy3_sim/`;
- MPI process/thread settings.

If your system uses Slurm, LSF, or another scheduler, the PBS launchers must be translated to that scheduler rather than used directly.

## Notebook outputs

Saved notebook outputs may still display paths from the author's previous runs. Those output strings are historical records, not portable configuration. Change the path variables in the executable cells before rerunning the notebooks.

Large DES Y3 input maps and generated HDF5 realizations are not included in this repository.

# AKRA Series

Accurate Kappa Reconstruction Algorithm (AKRA) is a collection of research tools for weak-lensing mass mapping on flat and curved skies. The repository includes the original explicit-matrix curved-sky implementation, a matrix-free conjugate-gradient prototype for AKRA 3.0, example notebooks, and DES Y3 simulation workflows.

> This is active research software. Interfaces, documentation, and numerical workflows may change as the project develops.
>
> This project is currently developed and maintained solely by me. The relevant documentation and instructions are still being prepared and will be updated gradually. If you encounter any issues or have suggestions, please feel free to contact me at: shiyuan0929@gmail.com
>
> Thank you for your interest and support!

## Highlights

- **Flat- and curved-sky reconstruction:** examples cover flat-field tests and full/partial-sky HEALPix maps.
- **Mask-aware harmonic treatment:** the curved-sky implementation keeps the real and imaginary spin-2 components required to retain E/B-mode information in the presence of a mask.
- **Matrix-free AKRA 3.0 prototype:** `core/akra_full_cg.py` applies the forward and adjoint operators through spherical-harmonic transforms and solves the regularized system with conjugate gradients, avoiding construction of the full coupling matrix.
- **DES Y3 workflows:** notebooks and MPI/PBS scripts support noiseless and noisy reconstruction experiments, batch realizations, and power-spectrum analysis.

## Repository layout

| Path | Description |
| --- | --- |
| `core/akra_full.py` | Curved-sky shear/convergence transforms and the explicit-matrix `KappaRec_sphere` solver. |
| `core/akra_full_cg.py` | Matrix-free coupling operator and `KappaRec_sphere_fast` conjugate-gradient solver. |
| `core/sphere_ks.py` | Spherical Kaiser-Squires utilities. |
| `utils/` | Map generation, power-spectrum, plotting, and HEALPix helper functions. |
| `akra_spere/test_akra_sphere.ipynb` | Curved-sky AKRA demonstration notebook. |
| `AKRA_HSC/test_akra.ipynb` | Flat-sky/HSC-style reconstruction demonstration. |
| `memory_refined.ipynb` | AKRA 3.0 memory and wall-time estimates. |
| `test_mask.ipynb` | Mask construction and validation experiments. |
| `desy3_sim/` | DES Y3 single-realization tests, batch simulations, power-spectrum analysis, PBS launchers, and comparison output. |

## Getting started

Clone the repository and create a Python environment:

```bash
git clone https://github.com/shiyuan-1/akra_series.git
cd akra_series
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy healpy matplotlib astropy tqdm h5py jupyter
```

Some notebooks and the DES Y3 workflow additionally use `pyccl`, `pandas`, `seaborn`, and `mpi4py`. Dependency versions are not yet pinned, so record the versions used for production analyses.

The repository is not packaged yet. Run from the repository root and add the source directories to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/core:$PWD/utils${PYTHONPATH:+:$PYTHONPATH}"
```

### Matrix-free curved-sky reconstruction

Given HEALPix shear maps `gamma1` and `gamma2` and a survey `mask` with the same pixelization:

```python
import healpy as hp
from akra_full_cg import KappaRec_sphere_fast

nside = hp.get_nside(gamma1)
reconstructor = KappaRec_sphere_fast(
    gamma1,
    gamma2,
    mask=mask,
    nside_out=nside,
    lmax=2 * nside,
    neff=neff,          # optional effective-number-density map
)

kappa = reconstructor.sphere_AKRA(
    lam=1e-3,
    maxiter=150,
)
```

Set `neff=None` when no inverse-noise weighting is required. The regularization strength and convergence settings should be validated for each survey geometry and resolution.

## DES Y3 simulations

`desy3_sim/massmapping_100realizations_withNoise.py` distributes noisy realizations across MPI ranks and writes one HDF5 file per realization. The accompanying notebooks compare Kaiser-Squires and AKRA reconstructions and measure their power spectra.

These files currently contain site-specific paths and PBS resource requests. Before running them, update:

- `AKRA_DIR`, `SKYMAP_DIR`, and `DATA_FILE` in the Python workflow;
- the working directory, queue, nodes, and environment in the PBS scripts;
- the input and output directories in the analysis notebooks.

The DES Y3 input maps and generated HDF5 realizations are not included in this repository.

## Development snapshot

- **2025-11-26:** added the curved-sky demonstration notebook.
- **2025-12-30:** added the flat-sky/HSC reconstruction example.
- **2026-09-02:** added the AKRA 3.0 matrix-free solver, resource-estimate notebook, and DES Y3 simulation/analysis workflows.

## Research landscape

The following four works show how the project connects methodological development with data applications across large-scale surveys and galaxy-cluster scales.

| Physical scale ↓ / contribution → | Method / theory | Observation / data application |
| --- | --- | --- |
| **Large-scale structure and surveys** | [**AKRA 3.0**](https://arxiv.org/abs/2606.06175) — matrix-free inversion for high-resolution mass mapping | [**The first AKRA mass map reconstruction from HSC Y1 data**](https://doi.org/10.1088/1475-7516/2026/02/085) — first AKRA application to real survey data |
| **Galaxy clusters** | [**Nonlinear weak lensing reconstruction for galaxy clusters**](https://doi.org/10.1103/hn39-9hyy) — nonlinear reduced-shear reconstruction near cluster cores | [**Lambda as a Probe of Lensing Consistency**](https://arxiv.org/abs/2607.08286) — an observation-facing consistency framework for joint strong- and weak-lensing reconstruction, currently validated with simulated clusters |

The placement reflects each paper's primary emphasis. AKRA 3.0 also includes an application to DES Y3 data, while the Lambda framework is designed for observational use but is currently demonstrated with simulations.

### Related publications

1. **Lambda as a Probe of Lensing Consistency**<br>
   Yuan Shi, Li Cui, and Carlo Giocoli<br>
   [arXiv:2607.08286](https://arxiv.org/abs/2607.08286) · [PDF](https://arxiv.org/pdf/2607.08286.pdf)

2. **AKRA 3.0: A Matrix-Free Inversion Framework for Weak Lensing Mass Mapping and Its Application to DES Y3 Data**<br>
   Yuan Shi, Pengjie Zhang, Li Cui, Jian Qin, and Ji Yao<br>
   [arXiv:2606.06175](https://arxiv.org/abs/2606.06175) · [PDF](https://arxiv.org/pdf/2606.06175.pdf)

3. **Nonlinear Weak Lensing Reconstruction for Galaxy Clusters**<br>
   Yuan Shi and Li Cui<br>
   *Physical Review D* **113**, 103514 (2026) · [Published article](https://doi.org/10.1103/hn39-9hyy)

4. **The First AKRA Mass Map Reconstruction from HSC Y1 Data**<br>
   Yuan Shi, Pengjie Zhang, Zhao Chen, Jian Qin, Li Cui, Furen Deng, and Ji Yao<br>
   *Journal of Cosmology and Astroparticle Physics* **2026** (02), 085 · [Published article](https://doi.org/10.1088/1475-7516/2026/02/085)

## Foundational AKRA citations

If AKRA contributes to your research, please cite the corresponding papers:

- Y. Shi et al., “AKRA 2.0: Accurate Kappa Reconstruction Algorithm for masked shear catalog,” *Journal of Cosmology and Astroparticle Physics* **2025** (07), 038. [https://doi.org/10.1088/1475-7516/2025/07/038](https://doi.org/10.1088/1475-7516/2025/07/038)
- Y. Shi et al., “Accurate kappa reconstruction algorithm for masked shear catalog,” *Physical Review D* **109**, 123530 (2024). [https://doi.org/10.1103/PhysRevD.109.123530](https://doi.org/10.1103/PhysRevD.109.123530)

```bibtex
@article{Shi2025AKRA2,
  author  = {Shi, Yuan and Zhang, Pengjie and Deng, Furen and Zhou, Shuren and Cai, Hongbo and Yao, Ji and Sun, Zeyang},
  title   = {{AKRA 2.0}: Accurate Kappa Reconstruction Algorithm for masked shear catalog},
  journal = {Journal of Cosmology and Astroparticle Physics},
  year    = {2025},
  volume  = {2025},
  number  = {07},
  pages   = {038},
  doi     = {10.1088/1475-7516/2025/07/038}
}

@article{Shi2024AKRA,
  author  = {Shi, Yuan and Zhang, Pengjie and Sun, Zeyang and Wang, Yihe},
  title   = {Accurate kappa reconstruction algorithm for masked shear catalog},
  journal = {Physical Review D},
  year    = {2024},
  volume  = {109},
  pages   = {123530},
  doi     = {10.1103/PhysRevD.109.123530}
}
```

## Contact

The project is currently developed and maintained by Yuan Shi. For questions or suggestions, contact [shiyuan0929@gmail.com](mailto:shiyuan0929@gmail.com).

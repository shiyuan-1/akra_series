# akra_series
Accurate Kappa Reconstruction Algorithm (AKRA): a series of open-source tools for weak-lensing mass mapping on flat and curved sky.

2025-11-26: you can test test_akra_sphere.ipynb in akra_sphere

2025-12-30: AKRA flat: you can generate convergence in akra_hsc/ test
* we first add the test_akra.ipynb to test the convergence and B mode map 
* TBD: HSC series data product


**AKRA-sphere** employs *spin-weighted spherical harmonic transforms* to reconstruct the convergence field ($\kappa$) on the full or curved sky.

This release extends the AKRA framework to a full, curved sky formulation that naturally supports HEALPix pixelization and realistic survey masks.

Highlights: 

- **Curved-sky system matrix ($A$-matrix):** AKRA-sphere explicitly constructs the linear operator that couples shear and convergence in harmonic space, taking into account the spin-2 nature of the shear field and the spherical geometry.
- **Complex harmonic treatment:** while real-valued maps (e.g., temperature fields) can be represented by the real part of their spherical harmonics, weak-lensing mass mapping requires both **real and imaginary components** of the spin-2 field to preserve the full E/B-mode information when mask exists.
    
    (See Appendix A.2 of *Shi et al., AKRA 2.0*, for a detailed derivation.)
    

### 📅 Coming Soon

- ~~ **AKRA-flat** — the flat-sky version optimized for flat-field reconstructions ~~
- **AKRA 2.0** — scale-splitting strategy; a unified, high-performance release that integrates spherical and flat modules with.
- **AKRA 3.0** - largely speed up

Stay tuned — **AKRA-flat** and **AKRA 2.0** will be released **very soon**.

### 📄 Citation

- If AKRA series contributes to your research, please cite:
    
    @article{Shi_AKRA-2,
    author = {Shi, Yuan and Zhang, Pengjie and Deng, Furen and Zhou, Shuren and Cai, Hongbo and Yao, Ji and Sun, Zeyang},
    title = {AKRA 2.0: Accurate Kappa Reconstruction Algorithm for masked shear catalog},
    journal = {Journal of Cosmology and Astroparticle Physics},
    volume = {2025},
    pages = {038},
    keywords = {weak gravitational lensing
    gravitational lensing
    power spectrum
    Astrophysics - Instrumentation and Methods for Astrophysics
    Astrophysics - Cosmology and Nongalactic Astrophysics},
    ISSN = {1475-7516},
    DOI = {10.1088/1475-7516/2025/07/038},
    url = {https://ui.adsabs.harvard.edu/abs/2025JCAP...07..038Shttps://iopscience.iop.org/article/10.1088/1475-7516/2025/07/038},
    year = {2025},
    type = {Journal Article}
    }
    
    @article{Shi_AKRA-flat,
    author = {Shi, Yuan and Zhang, Pengjie and Sun, Zeyang and Wang, Yihe},
    title = {Accurate kappa reconstruction algorithm for masked shear catalog},
    journal = {Physical Review D},
    volume = {109},
    pages = {123530},
    keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics
    Astrophysics -
    Instrumentation and Methods for Astrophysics},
    ISSN = {1550-79980556-2821},
    DOI = {10.1103/PhysRevD.109.123530},
    url = {https://ui.adsabs.harvard.edu/abs/2024PhRvD.109l3530Shttps://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.123530},
    year = {2024},
    type = {Journal Article}
    }

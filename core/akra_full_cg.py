"""
CORRECT Matrix-Free Implementation following get_wk_pol exactly

The key insight:
1. Extract half_alm (m≥0) from the full x
2. Extract half_alm_mneg (m<0 contribution) using: 1j * (x[ii] - half_alm[ii])
3. Apply TWO SHT operations:
   - One for the m≥0 part (M1)
   - One for the m<0 part (M2)
4. Combine the results
"""

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
import healpy as hp

def degrade(map_in, nside_out, lmax=None):
    if lmax is None:
        lmax = 3*nside_out-1
    map_alm = hp.map2alm(map_in, lmax=lmax)
    map_out = hp.alm2map(map_alm, nside=nside_out)
    return map_out
def upgrade(map_in, nside_out, lmax=None):
    if lmax is None:
        lmax = 3*hp.npix2nside(len(map_in))-1
    map_alm = hp.map2alm(map_in, lmax=lmax)
    map_out = hp.alm2map(map_alm, nside=nside_out)
    return map_out


def get_spinalm(map1, map2, lmax):
    alm_E, alm_B = hp.map2alm_spin([map1, map2], spin=2, lmax=lmax)
    p2lm = -1*alm_E - 1j*alm_B
    m2lm = -1*alm_E + 1j*alm_B
    return p2lm, m2lm


def get_fullalm(alm_half, lmax=None):
    if lmax is None:
        lmax = hp.Alm.getlmax(len(alm_half))
    ells, emms = hp.Alm.getlm(lmax)
    return np.concatenate([alm_half, (-1) ** emms[lmax+1:] * alm_half[lmax+1:].conj()])


def get_halfalm(alm_full, lmax):
    lmax = int(lmax)
    lm_size = int(hp.Alm.getsize(lmax))
    ells, ems = hp.Alm.getlm(lmax)
    neg_m_indices = np.arange(lm_size) + lm_size - lmax - 1
    sign = (-1) ** ems
    # Vectorized computation
    alm_half = np.where(
        ems == 0,
        alm_full[:lm_size],
        (alm_full[:lm_size] + sign * alm_full[neg_m_indices].conj()) / 2.0
    )

    return alm_half



class MatrixFreeCoupling:
    """
    Matrix-free mask coupling following get_wk_pol exactly.
    """
    def __init__(self, mask, lmax, nside, neff_weight=None, sigma_e=0.26):
        self.mask = mask
        self.lmax = int(lmax)
        self.nside = nside

        self.lm_size = int(hp.Alm.getsize(self.lmax))
        self.n_m_pos = self.lm_size - (self.lmax + 1)

        self.n_cols = self.lm_size + self.n_m_pos  # full_size
        self.full_size = self.n_cols
        self.n_rows = 2 * self.full_size

        self._lm_cache = [hp.Alm.getlm(self.lmax, ii) for ii in range(self.lm_size)]
        self._ell_values, self._m_values = hp.Alm.getlm(self.lmax)
        self._m_nonzero_mask = self._m_values != 0

        self.sigma_e = sigma_e
        self._setup_noise_weight(neff_weight)


    def _setup_noise_weight(self, neff_weight):
        """Set up noise weighting from n_eff map."""
        if neff_weight is None:
            self.inv_noise_var = None
            # self.weighted_fsky = np.mean(self.mask**2)
        else:
            neff_safe = np.maximum(neff_weight, 1e-10)
            self.inv_noise_var = neff_safe / (self.sigma_e ** 2)
            self.inv_noise_var *= self.mask
        # self.weighted_fsky = np.mean(self.mask**2)

    def apply_A(self, x):
        """Forward operation: x → (a2lm_full, am2lm_full)"""
        half_alm = get_halfalm(x, self.lmax)
        half_alm_mneg = np.zeros_like(half_alm)
        x_half = x[:self.lm_size]              # size: lm_size
        half_alm_mneg[self._m_nonzero_mask] = 1.0j * (x_half[self._m_nonzero_mask] - half_alm[self._m_nonzero_mask])
        a_Elm = -1 * half_alm
        a_Blm = -1 * half_alm_mneg

        map_Q, map_U = hp.alm2map_spin([a_Elm, a_Blm], self.nside, spin=2, lmax=self.lmax)
        map_Q *= self.mask
        map_U *= self.mask

        a2lm, am2lm = get_spinalm(map_Q, map_U, self.lmax)

        return np.concatenate([get_fullalm(a2lm, self.lmax), get_fullalm(am2lm, self.lmax)])


    def apply_AH(self, y):

        y1_full = y[:self.full_size]
        y2_full = y[self.full_size:]

        a2lm_half = get_halfalm(y1_full, self.lmax)
        am2lm_half = get_halfalm(y2_full, self.lmax)

        # Step 2: Adjoint of get_spinalm
        alm_E_adj = -(a2lm_half + am2lm_half) / 2
        alm_B_adj = 1J * (a2lm_half - am2lm_half) / (2)

        map_Q, map_U = hp.alm2map_spin([alm_E_adj, alm_B_adj], self.nside, spin=2, lmax=self.lmax)
        map_Q *= np.conj(self.mask)
        map_U *= np.conj(self.mask)

        if self.inv_noise_var is not None:
            map_Q = map_Q * self.inv_noise_var
            map_U = map_U * self.inv_noise_var

        a2lm_adj, am2lm_adj = get_spinalm(map_Q, map_U, self.lmax)
        a2lm = get_fullalm(a2lm_adj, self.lmax)
        am2lm_adj = get_fullalm(am2lm_adj, self.lmax)

        return a2lm+am2lm_adj


    def apply_gram(self, x, lam=1e-3):
        """Apply (A^H @ A + λI) @ x"""
        # if self.inv_noise_var is not None:
        #     lam = lam * np.mean(self.inv_noise_var[self.mask > 0])
        return self.apply_AH(self.apply_A(x)) + lam * x



def solve_cg_matrixfree(op, aT_y, lam=1e-3, tol=1e-2, maxiter=150, verbose=True, x0=None):
    n = op.n_cols
    # Scale lambda consistently with apply_gram
    if op.inv_noise_var is not None:
        lam_effective = lam * np.mean(op.inv_noise_var[op.mask > 0])
        if verbose: print("lam scaled by mean inv_noise_var: ", lam_effective)
    else:
        lam_effective = lam
        if verbose: print("lam used without scaling: ", lam_effective)

    def matvec(x):
        return op.apply_gram(x, lam_effective)  # This internally scales lam

    gram_op = LinearOperator((n, n), matvec=matvec, dtype=complex)

    # Preconditioner must use the SAME effective lambda
    if op.inv_noise_var is not None:
        f_sky = np.mean((op.mask**2) * op.inv_noise_var)
    else:
        f_sky = np.mean(op.mask**2)

    diag_approx = f_sky + lam_effective

    def precond(x):
        return x / diag_approx

    precond_op = LinearOperator((n, n), matvec=precond, dtype=complex)

    if x0 is not None:
        x0 = x0.astype(np.complex128)
    else:
        x0 = np.zeros(n, dtype=np.complex128)

    # Solve
    if verbose:
        print("Solving with matrix-free CG...")
        iter_count = [0]
        def callback(xk):
            iter_count[0] += 1
            if iter_count[0] % 20 == 0:
                print(f"  Iteration {iter_count[0]}")
        x, info = cg(gram_op, aT_y, M=precond_op, rtol=tol, maxiter=maxiter, callback=callback, x0=x0)
        print(f"CG finished: {iter_count[0]} iterations, info={info}")
    else:
        x, info = cg(gram_op, aT_y, M=precond_op, rtol=tol, maxiter=maxiter, x0=x0)

    return x, info


class KappaRec_sphere_fast:

    def __init__(self, gamma1, gamma2, mask=None, nside_out=None, lmax=None, verbose=True, nosh=True, neff=None, sigma_e=0.26):
        self.complex_type = np.complex128
        self.gamma1 = gamma1 * mask
        self.gamma2 = gamma2 * mask
        self.mask = mask
        self.verbose = verbose
        self.x0 = None

        if lmax is None:
            self.lmax = int(2*nside_out)
        else:
            self.lmax = int(lmax)

        if nside_out is None:
            self.nside_out = hp.get_nside(gamma1)
        else:
            self.nside_out = nside_out

        # Create matrix-free operator
        self.op = MatrixFreeCoupling(self.mask, self.lmax, self.nside_out, neff_weight=neff, sigma_e=sigma_e)

        # if self.op.inv_noise_var is not None:
        #     self.gamma1 = self.gamma1 * self.op.inv_noise_var
        #     self.gamma2 = self.gamma2 * self.op.inv_noise_var

        # Compute observed alm (same as original)
        self.a2lm_mask, self.am2lm_mask = get_spinalm(self.gamma1, self.gamma2, lmax=self.lmax)

        if nosh:
            ell, emm = hp.Alm.getlm(lmax=self.lmax)
            factor = (((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1.))) ** 0.5)
            factor[ell == 0] = 0.0
            factor[ell == 1] = 0.0
            self.a2lm_mask = self.a2lm_mask * factor
            self.am2lm_mask = self.am2lm_mask * factor

        self.yy = np.concatenate([
            get_fullalm(self.a2lm_mask, self.lmax),
            get_fullalm(self.am2lm_mask, self.lmax)
            ]).astype(self.complex_type)



        if self.verbose:
            print(f"KappaRec_sphere_fast initialized:")
            print(f"  lmax = {self.lmax}")
            print(f"  nside = {self.nside_out}")
            print(f"  n_cols = {self.op.n_cols}")
            print(f"  Observed yy shape: {self.yy.shape}")
            matrix_size_gb = (self.op.n_rows * self.op.n_cols * 16) / 1e9
            print(f"  Explicit matrix would be: {matrix_size_gb:.2f} GB")

    def sphere_AKRA(self, lam=1e-3, cgtol=2e-3, maxiter=150):
        """
        Matrix-free AKRA solver.

        This replaces:
        1. get_wk_pol + matrix_a formation
        2. psf = A^H @ A computation
        3. solve_cg with explicit matrix

        With matrix-free operations via SHT.
        """
        import time

        if self.verbose:
            print("Computing A^H @ y via SHT...")
        start = time.time()

        # Compute A^H @ y (matrix-free)
        aT_y = self.op.apply_AH(self.yy)

        if self.verbose:
            print(f"  Time: {time.time() - start:.2f}s")
            print("Solving (A^H A + λI) x = A^H y with matrix-free CG...")
        start = time.time()

        # Solve with matrix-free CG
        xx, info = solve_cg_matrixfree(
            self.op, aT_y,
            lam=lam, tol=cgtol, maxiter=maxiter,
            verbose=self.verbose, x0 = self.x0
        )

        if self.verbose:
            print(f"  Total solve time: {time.time() - start:.2f}s")

        # Extract kappa from solution (same as original)
        self._extract_kappa(xx)

        return self.kappa_akra

    def _extract_kappa(self, xx):
        """
        Extract kappa map from solution vector xx.
        Same logic as your original code.
        """
        lm_size = int(hp.Alm.getsize(self.lmax))

        # Convert xx to half alm format
        alm_full = np.zeros(self.op.full_size, dtype=np.complex128)
        alm_full[:len(xx)] = xx  # Pad if needed

        alm_kappar = get_halfalm(alm_full, self.lmax)
        alm_kappai = np.zeros_like(alm_kappar)

        for ii in range(lm_size):
            thisl, thism = hp.Alm.getlm(self.lmax, ii)
            if thism == 0:
                continue
            else:
                alm_kappai[ii] = 1.0j * (xx[ii] - alm_kappar[ii])
        ell, emm = hp.Alm.getlm(lmax=self.lmax)
        alm_kappar[ell == 0] = 0.0
        alm_kappai[ell == 0] = 0.0
        alm_kappar[ell == 1] = 0.0
        alm_kappai[ell == 1] = 0.0

        kappa_r = hp.alm2map(- 1 * alm_kappar, nside=self.nside_out, pol=False)
        kappa_i = hp.alm2map(- 1 * alm_kappai, nside=self.nside_out, pol=False)

        self.kappa_akra = kappa_r + 1.0j * kappa_i
        self.kappa_akra = self.kappa_akra.real


    def verify_adjoint(self):
        """Check ⟨Ax, y⟩ = ⟨x, A†y⟩"""
        np.random.seed(42)
        self.n_cols = self.op.n_cols
        self.n_rows = self.op.n_rows
        x = np.random.randn(self.n_cols) + 1j * np.random.randn(self.n_cols)
        y = np.random.randn(self.n_rows) + 1j * np.random.randn(self.n_rows)

        Ax = self.op.apply_A(x)
        AHy = self.op.apply_AH(y)

        lhs = np.vdot(Ax, y)  # ⟨Ax, y⟩
        rhs = np.vdot(x, AHy)  # ⟨x, A†y⟩

        print(f"⟨Ax, y⟩ = {lhs}")
        print(f"⟨x, A†y⟩ = {rhs}")
        print(f"Ratio: {np.abs(lhs/rhs)}")
        print(f"Should be ~1.0 for correct adjoint")
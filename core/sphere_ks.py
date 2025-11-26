from pathlib import Path
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pyccl as ccl
import scipy.stats as stats
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_map_cl(map1, map2, ratio=False, label1="map1", label2="map2", xrange=None, yrange=None, **kwargs):
    cl1 = hp.anafast(map1, **kwargs)
    cl2 = hp.anafast(map2, **kwargs)
    ell = np.arange(len(cl1))
    plt.figure(figsize=(4,3), dpi=100)
    if ratio:
        plt.plot(ell, cl1/cl2, c="black", label=label1+"/"+label2, ls="solid")
    else:
        plt.loglog(ell, cl1, c="black", label=label1, ls="solid")
        plt.loglog(ell, cl2, c="red", label=label2, ls="dashed")
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell^{KK}$')
    if xrange is not None:
        plt.xlim(xrange)
    if yrange is not None:
        plt.ylim(yrange)
    plt.legend()
    plt.show()


class sphere:
    @classmethod
    def dn_dz(cls, z):
        """redshift distribution
        dn/dz = n_gal * p(z)
        z: redshift, numpy 1D array
        """
        Mag_lim_iband = 26.0  # Limiting i-band magnitude
        density_gal = (
            46.0 * 100.31 * (Mag_lim_iband - 25.0)
        )  # Normalisation, galaxies/arcmin^2
        z0 = 0.0417 * Mag_lim_iband - 0.744
        pz = (
            1.0 / (2.0 * z0) * (z / z0) ** 2.0 * np.exp(-z / z0)
        )  # pdf of redshift: p(z)
        dndz = density_gal * pz  # Number density distribution
        return dndz

    @classmethod
    def C_ell_kk(
        cls,
        z,
        nside,
        cosmo=ccl.Cosmology(
            Omega_c=0.26447, Omega_b=0.0493, h=0.6736, sigma8=0.8111, n_s=0.9649
        ),
    ):
        """pyccl calculate C_ell of \kappa \kappa
        z: redshift, numpy 1D array
        nside: nside of healpy map
        cosmo: Plank18 as default
        """
        dndz = cls.dn_dz(z)  # Number density distribution
        tracers = {
            "lens": ccl.WeakLensingTracer(cosmo, dndz=(z, dndz), has_shear=True),
        }
        lmax = max(
            1000, 3 * nside - 1
        )  # https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.synfast.html#healpy-sphtfunc-synfast

        ells = np.arange(lmax + 1)  # 0, 1, 2, ..., lmax

        C_ell = ccl.angular_cl(cosmo, tracers["lens"], tracers["lens"], ells)
        return ells, C_ell

    @classmethod
    def kappa_map(
        cls,
        z,
        nside,
        cosmo=ccl.Cosmology(
            Omega_c=0.26447, Omega_b=0.0493, h=0.6736, sigma8=0.8111, n_s=0.9649
        ),
    ):
        """generate \kappa map
        z: redshift, numpy 1D array
        nside: nside of healpy map
        cosmo: Plank18 as default
        """
        lmax = int(2 * nside)  # follow DES-3 paper
        ells, C_ell = cls.C_ell_kk(z, nside, cosmo)
        kappa, alm_k = hp.synfast(C_ell, nside=nside, alm=True, lmax=lmax)
        return {"kappa_map": kappa, "ells": ells, "C_ell_kk": C_ell}

    @classmethod
    def gamma2kappa_KS(cls, gamma1, gamma2, mask=None, nest=False, nosh=True) -> np.ndarray:
        """KS test for gamma1, gamma2
        Input:
            gamma1, gamma2: np.ndarray
                gamma1, gamma2 map
        Output:
            kappa: np.ndarray
                kappa map
        """
        # check input
        # if not np.ndarray, convert them
        if not isinstance(gamma1, np.ndarray) or not isinstance(gamma2, np.ndarray):
            gamma1 = np.array(gamma1)
            gamma2 = np.array(gamma2)
        # check if input batch?
        if len(gamma1.shape) == 2:
            B, npix = gamma1.shape
            res = []
            for i in range(B):
                res.append(cls.gamma2kappa_KS(gamma1[i], gamma2[i], mask, nest))
            return np.array(res)

        if hp.isnpixok(len(gamma1)) and len(gamma1) == len(gamma2):
            nside = hp.npix2nside(len(gamma1))
        else:
            raise ValueError("npix is not valid")
        if mask is not None:
            gamma1[mask == 1] = 0
            gamma2[mask == 1] = 0
        if nest:
            # reorder to RING
            gamma1 = hp.reorder(gamma1, n2r=True)
            gamma2 = hp.reorder(gamma2, n2r=True)

        # map -> alm:  input (T,Q,U), output (T,E,B)
        map_T = np.zeros_like(gamma1)
        _, alm_E, _ = hp.map2alm([map_T, gamma1, gamma2], pol=True)
        lmax = hp.Alm.getlmax(len(alm_E)) 
        ell, emm = hp.Alm.getlm(lmax=lmax)

        if nosh:
            alm_E = 1.*alm_E*((ell*(ell+1.))/((ell+2.)*(ell-1)))**0.5
        else:
            alm_E = alm_E*1.

        alm_kappa = alm_E
        alm_kappa[ell==0] = 0.0
        alm_kappa[ell==1] = 0.0

        # alm -> map
        kappa = hp.alm2map(alm_kappa, nside, pol=False)

        return hp.reorder(kappa, r2n=True) if nest else kappa

    @classmethod
    def g2k(cls, gamma1, gamma2, mask=None, nest=False) -> np.ndarray:
        return cls.gamma2kappa_KS(gamma1, gamma2, mask=mask, nest=nest)

    @classmethod
    def kappa2gamma_KS(cls, kappa, mask=None, nest=False) -> (np.ndarray, np.ndarray):
        """KS93 method for kappa -> gamma1, gamma2
        Input:
            kappa: np.ndarray
                kappa map
        Output:
            gamma1, gamma2: np.ndarray
                gamma1, gamma2 map
        """
        # check input
        # if kappa is not np.ndarray, convert to np.ndarray
        if not isinstance(kappa, np.ndarray):
            kappa = np.array(kappa)
        if hp.isnpixok(len(kappa)):
            nside = hp.npix2nside(len(kappa))
        else:
            raise ValueError("npix is not valid")
        if mask is not None:
            kappa[mask == 1] = 0
        if nest:
            # reorder to RING
            kappa = hp.reorder(kappa, n2r=True)

        # map -> alm
        alm_kappa = hp.map2alm(kappa)

        lmax = hp.Alm.getlmax(len(alm_kappa))
        ells, emms = hp.Alm.getlm(lmax)

        alm_E = alm_kappa
        # *((ells*(ells+1.))/((ells+2.)*(ells-1)))**0.5

        # alm -> map: (T,E,B) -> (T,Q,U)
        alm_T = alm_B = np.zeros_like(alm_kappa, dtype=complex)
        _, map_g1, map_g2 = hp.alm2map([alm_T, alm_E, alm_B], nside, pol=True)

        return (
            (hp.reorder(map_g1, r2n=True), hp.reorder(map_g2, r2n=True))
            if nest
            else (map_g1, map_g2)
        )

    @classmethod
    def k2g(cls, kappa, mask=None, nest=False) -> (np.ndarray, np.ndarray):
        return cls.kappa2gamma_KS(kappa, mask=mask, nest=nest)

    @classmethod
    def cal_Cells(cls, map1, map2=None, mask=None, lmax=None):
        """
        Compute the power spectrum
        Parameters
        ----------
        map1 : np.ndarray
            1D array of the first map
        map2 : np.ndarray
            1D array of the second map
        mask : np.ndarray
            1D array of the mask
            1: masked, 0: visible
        lmax: int
            maximum multipole
        Returns
        -------
        ells : np.ndarray
            1D array of the multipoles
        cells : np.ndarray
            1D or 4D array of the power spectrum
        """
        import pymaster as nmt

        if map2 is not None:
            assert len(map1) == len(
                map2
            ), f"map1 and map2 should have the same length, but got {len(map1)} and {len(map2)}"
        assert len(map1.shape) == 1, f"map1 should be 1 dim, but got {map1.shape}"

        npix = len(map1)
        nside = hp.npix2nside(npix)
        resol = hp.nside2resol(nside, arcmin=True)  # degree^2
        if mask is None:
            mask = np.zeros_like(map1)  # all visible
        if np.sum(mask) == len(mask):
            raise ValueError("All pixels are masked!")
        if lmax is None:
            lmax = int(2 * nside)  # default lmax = 3*nside - 1

        mask = 1 - mask  # reverse mask, 1: visible, 0: masked

        # Initialize binning scheme with 4 ells per bandpower
        b = nmt.NmtBin.from_lmax_linear(lmax, 4)
        ells = b.get_effective_ells()

        if map2 is None:
            f_0 = nmt.NmtField(mask, [map1])
            cl = nmt.compute_full_master(f_0, f_0, b)
            return ells, cl[0]
        else:
            f_2 = nmt.NmtField(mask, [map1, map2])
            cl = nmt.compute_full_master(f_2, f_2, b)
            return ells, cl

    @classmethod
    def power_spectrum(cls, map1, map2=None, mask=None, lmax=None):
        return cls.cal_Cells(map1, map2, mask, lmax)
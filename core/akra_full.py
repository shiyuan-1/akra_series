import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
from tqdm import tqdm
# import pymaster as nmt
from astropy import units as u
import sys, os
from scipy.linalg import cholesky, inv, cho_factor, cho_solve
sys.path.append("./utils/")
from akra_utils import degrade, upgrade
from scipy.spatial.transform import Rotation

def sph_kappa2gamma(kappa_map, nside_out, lmax=None):
    if lmax is None:
        lmax = 3*nside_out-1
    alm_kappa = hp.map2alm(kappa_map, lmax=lmax)
    ell, emms = hp.Alm.getlm(lmax)
    kalmsE = alm_kappa / (((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
    kalmsE[ell == 0] = 0.0
    kalmsE[ell == 1] = 0.0
    _, gamma1_map, gamma2_map = hp.alm2map([np.zeros_like(alm_kappa), kalmsE, np.zeros_like(alm_kappa)], nside=nside_out, pol=True)

    return gamma1_map, gamma2_map

def sph_gamma2kappa(gamma1_map, gamma2_map, nside_out, lmax=None):
    if lmax is None:
        lmax = 3*nside_out-1
    map_T = np.zeros_like(gamma1_map)
    _, alm_E, _ = hp.map2alm([map_T, gamma1_map, gamma2_map], pol=True, lmax=lmax)
    lmax = hp.Alm.getlmax(len(alm_E)) 
    ell, emm = hp.Alm.getlm(lmax=lmax)
    alm_E = 1.*alm_E*((ell*(ell+1.))/((ell+2.)*(ell-1)))**0.5
    alm_kappa = alm_E
    alm_kappa[ell==0] = 0.0
    alm_kappa[ell==1] = 0.0
    kappa_map = hp.alm2map(alm_kappa, nside_out, pol=False)
    return kappa_map

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


def full_mask(nside_out, random_seed=1234, mask_nside=32, small_per=None):
    np.random.seed(random_seed)
    idx_mask1 = hp.query_strip(mask_nside, np.pi/5.0, np.pi/3.0)
    idx_mask2 = []
    for ii in range(10):
        th = np.random.rand()*np.pi
        phi = np.random.rand()*2*np.pi
        vec = hp.ang2vec(th, phi)
        radius = np.random.uniform(10*hp.nside2resol(mask_nside), 20*hp.nside2resol(mask_nside))
        idx_mask2.append(hp.query_disc(mask_nside, vec, radius))
    idx_mask = np.concatenate(idx_mask2+[idx_mask1])
    idx_mask = np.unique(idx_mask)
# NOTE: mask==0 is unseen, according to the convention for lots of package for pseudo-cls
    mask = np.ones(hp.nside2npix(mask_nside), dtype=np.float64)
    mask[idx_mask] = 0.0
    mask = hp.ud_grade(mask, nside_out=nside_out)
    if small_per is not None:
        unseen = np.random.choice(np.arange(len(mask)), size=int(len(mask)*small_per))
        mask[unseen] = 0
    return mask

def smooth_boundary_mask(mask, aposcale=None, **kwargs):
    import pymaster as nmt
    if aposcale is None:
        reso = hp.get_nside(mask)
        aposcale = reso.to_value(u.deg)*5
    mask = nmt.mask_apodization(mask, aposcale, apotype="C2", **kwargs)
    return mask

def smooth_transit(mask_smooth, th, width, verbose=True):
    x = (mask_smooth - th - width)/(width/3.0)
    if verbose:
        plt.figure(figsize=(5, 3), dpi=100)
        x = np.linspace(0, 1, 1001)
        plt.plot(x, smooth_transit(x, 0.5, 0.1))
        plt.show()
    return (1+np.tanh(x))/2.0


def get_spinalm(map1, map2, lmax=None):
    if lmax is None:
        lmax = 2*nside
    alm_E, alm_B = hp.map2alm_spin([map1, map2], spin=2, lmax=lmax)
    ell, emms = hp.Alm.getlm(lmax)
    # alm_E[ell == 0] = 0.0
    # alm_E[ell == 1] = 0.0
    # alm_B[ell == 0] = 0.0
    # alm_B[ell == 1] = 0.0
    p2lm = -1*alm_E - 1j*alm_B
    m2lm = -1*alm_E + 1j*alm_B
    return p2lm, m2lm

def get_fullalm(alm_half, lmax=None):
    if lmax is None:
        lmax = hp.Alm.getlmax(len(alm_half))
    ells, emms = hp.Alm.getlm(lmax)
    return np.concatenate([alm_half, (-1) ** emms[lmax+1:] * alm_half[lmax+1:].conj()])

def get_halfalm(alm_full, lmax):
    lm_size = int(hp.Alm.getsize(lmax))
    print(lm_size)
    lmax = int(lmax)
    alm_half = np.zeros(lm_size, dtype=np.complex128)
    for ii in tqdm(range(lm_size)):
        thisl, thism = hp.Alm.getlm(lmax, ii)
        if thism == 0:
            alm_half[ii] = alm_full[ii]
            continue
        else:
            alm_half[ii] = (alm_full[ii]+(-1)**thism*alm_full[ii+lm_size-lmax-1].conj())/2.0
    
    return alm_half

def get_wk_fromM12(half_M1, half_M2):
    lm_size = half_M1.shape[-1]
    lmax = hp.Alm.getlmax(lm_size)
    M1 = np.zeros([2, lm_size+lm_size-lmax-1, lm_size], dtype=np.complex64)
    M2 = np.zeros([2, lm_size+lm_size-lmax-1, lm_size-lmax-1], dtype=np.complex64)
    for ii in tqdm(range(lm_size)):
        a2lm_M1 = get_fullalm(half_M1[0,ii], lmax=lmax)
        am2lm_M1 = get_fullalm(half_M1[1,ii], lmax=lmax)
        thisl, thism = hp.Alm.getlm(lmax, ii)

        if thism == 0:
            M1[0, :, ii] = a2lm_M1
            M1[1, :, ii] = am2lm_M1
            continue
        
        a2lm_M2 = get_fullalm(half_M2[0,ii], lmax)
        am2lm_M2 = get_fullalm(half_M2[1,ii], lmax)

        M1[0, :, ii] = (a2lm_M1+a2lm_M2/(1.J))/2.0
        M2[0, :, ii-lmax-1] = (a2lm_M1-a2lm_M2/(1.J))/2.0/(-1)**thism
        
        M1[1, :, ii] = (am2lm_M1+am2lm_M2/(1.J))/2.0
        M2[1, :, ii-lmax-1] = (am2lm_M1-am2lm_M2/(1.J))/2.0/(-1)**thism
    a2lm_M = np.concatenate([M1[0,:,:], M2[0,:,:]], axis=1)
    am2lm_M = np.concatenate([M1[1,:,:], M2[1,:,:]], axis=1)
    return a2lm_M, am2lm_M


def get_wk_pol(mask, lmax, niter=3, nside_out=None):
    if nside_out is None:
        nside_out = hp.get_nside(mask)
    else:
        if nside_out != hp.get_nside(mask):
            mask = degrade(mask, nside_out)
    # lm_size = hp.Alm.getsize(lmax)
    lm_size = int(hp.Alm.getsize(lmax))
    lmax = int(lmax)
    M1 = np.zeros([2, lm_size+lm_size-lmax-1, lm_size], dtype=np.complex128)
    M2 = np.zeros([2, lm_size+lm_size-lmax-1, lm_size-lmax-1], dtype=np.complex128)
    a_2lm = np.zeros(lm_size, dtype=np.complex128)
    for ii in tqdm(range(lm_size)):
        a_2lm[:] = 0.0
        a_2lm[ii] = 1.0
        a_m2lm = a_2lm
        a_Elm = -1*(a_2lm + a_m2lm)/2
        a_Blm = 1.J*(a_2lm - a_m2lm)/2
        thisl, thism = hp.Alm.getlm(lmax, ii)
        thisQ, thisU = hp.alm2map_spin([a_Elm, a_Blm], nside_out, spin=2, lmax=lmax) 
        # thisQ, thisU = hp.alm2map([np.zeros_like(a_Blm), a_Elm, a_Blm], nside_out, pol=True, lmax=lmax)
        a2lm_M1, am2lm_M1 = get_spinalm(thisQ*mask, thisU*mask, lmax=lmax)
        a2lm_M1 = get_fullalm(a2lm_M1, lmax)
        am2lm_M1 = get_fullalm(am2lm_M1, lmax)

        if thism == 0:
            M1[0, :, ii] = a2lm_M1
            M1[1, :, ii] = am2lm_M1
            continue
        a_2lm[:] = 0.0
        a_2lm[ii] = 1.J
        a_m2lm = a_2lm
        a_Elm = -1*(a_2lm + a_m2lm)/2
        a_Blm = 1.J*(a_2lm - a_m2lm)/2
        thisQ, thisU = hp.alm2map_spin([a_Elm, a_Blm], nside_out, spin=2, lmax=lmax) 
        a2lm_M2, am2lm_M2 = get_spinalm(thisQ*mask, thisU*mask, lmax=lmax)
        a2lm_M2 = get_fullalm(a2lm_M2, lmax)
        am2lm_M2 = get_fullalm(am2lm_M2, lmax)
        M1[0, :, ii] = (a2lm_M1+a2lm_M2/(1.J))/2.0
        M2[0, :, ii-lmax-1] = (a2lm_M1-a2lm_M2/(1.J))/2.0/(-1)**thism
        
        M1[1, :, ii] = (am2lm_M1+am2lm_M2/(1.J))/2.0
        M2[1, :, ii-lmax-1] = (am2lm_M1-am2lm_M2/(1.J))/2.0/(-1)**thism
    a2lm_M = np.concatenate([M1[0,:,:], M2[0,:,:]], axis=1)
    am2lm_M = np.concatenate([M1[1,:,:], M2[1,:,:]], axis=1)
    return a2lm_M, am2lm_M



class KappaRec_sphere():
    def __init__(self, gamma1, gamma2, mask=None,nside_out=None, choSolve=True, b_inv=None, lmax=None, verbose=True):
        self.complex_type = "complex64"
        self.gamma1 = gamma1*mask
        self.gamma2 = gamma2*mask
        self.mask = mask
        self.shp = self.gamma1.shape
        self.choSolve = choSolve
        if lmax is None:
            self.lmax = 120
        else:
            self.lmax = int(lmax)
        if nside_out is None:
            self.nside_out = hp.get_nside(gamma1)
        else:
            self.nside_out = nside_out
        self.b_inv = b_inv
        self.verbose = verbose
        self.a2lm_mask, self.am2lm_mask = get_spinalm(self.gamma1, self.gamma2, lmax=self.lmax)
        self.yy = np.vstack((get_fullalm(self.a2lm_mask), get_fullalm(self.am2lm_mask))).astype(self.complex_type)
        if self.verbose:
            print("Observed yy shape: ", self.yy.shape)


    def sphere_KS(self, gamma1=None, gamma2=None, nosh=False):
        if gamma1 is None:
            gamma1 = self.gamma1
        if gamma2 is None:
            gamma2 = self.gamma2
        import sphere_ks as sph
        self.kappa_ks = sph.sphere.gamma2kappa_KS(gamma1, gamma2, nosh=nosh) 
    
    def sphere_AKRA(self, nosh=False, psf=None, matrix_a=None, use_cg=False, cgtol=1e-3, **kwargs):
        if psf is None or matrix_a is None:
            self.get_psf()
        else:
            print("Using the provided psf and matrix_a ... \n")
            self.psf = psf
            self.matrix_a = matrix_a
        aT_y = np.dot(self.matrix_a.conj().T, self.yy.flatten())
        if use_cg:
            from scipy.sparse.linalg import cg
            del self.matrix_a
            xx, _ = cg(self.psf, aT_y, tol=cgtol)
        else:
            print("Calculating the inverse of the psf: ... \n")  
            print("psf dtype: ", self.psf.dtype)
            if self.choSolve:
                try:
                    self.b_inv = self.get_Binv_cho(self.psf, **kwargs)  
                except:
                    print("Cholesky decomposition failed. Calculating the inverse of the psf: ... \n")
                    self.b_inv = self.get_Binv(self.psf, **kwargs)        
            else:
                self.b_inv = self.get_Binv(self.psf, **kwargs)
            xx = np.dot(self.b_inv, aT_y) 

        ells, emms = hp.Alm.getlm(self.lmax)
        alm_kappar = get_halfalm(xx, self.lmax)
        alm_kappai = np.zeros_like(alm_kappar)
        lm_size = int(hp.Alm.getsize(self.lmax))
        for ii in range(lm_size):
            thisl, thism = hp.Alm.getlm(self.lmax, ii)
            if thism == 0:
                continue
            else:
                alm_kappai[ii] = 1.0J * (xx[ii]-alm_kappar[ii])
        ell, emm = hp.Alm.getlm(lmax=self.lmax)
        if nosh:
            alm_kappar = 1*alm_kappar*((ell*(ell+1.))/((ell+2.)*(ell-1)))**0.5
            alm_kappai = 1*alm_kappai*((ell*(ell+1.))/((ell+2.)*(ell-1)))**0.5
        alm_kappar[ell==0] = 0.0
        alm_kappai[ell==0] = 0.0
        alm_kappar[ell==1] = 0.0
        alm_kappai[ell==1] = 0.0

        self.kappa_akra = hp.alm2map(-1*alm_kappar, nside=self.nside_out, pol=False) + 1.0J*hp.alm2map(-1*alm_kappai, nside=self.nside_out, pol=False)
        self.kappa_akra = self.kappa_akra.real

        
    def get_psf(self, nside_M=None):
        if nside_M is None:
            nside_M = self.nside_out
        if self.verbose:
            print("Calculating A^T A...")
        import time
        start_time = time.time()
        M_a2lm, M_am2lm = get_wk_pol(self.mask, self.lmax, nside_out=nside_M)
        self.matrix_a = np.vstack([M_a2lm, M_am2lm]).astype(self.complex_type)

        over_time = time.time()
        print("Time used: ", over_time-start_time, "s")

        del M_a2lm, M_am2lm
        print("The shape of A is: ", self.matrix_a.shape)
        # import psutil
        # process = psutil.Process()
        # print("Memory used: ", process.memory_info().rss/1024/1024, "MB")
        # print("CPU used: ", len(os.sched_getaffinity(0)), 'cores')
        self.psf = np.matmul(self.matrix_a.conj().T, self.matrix_a).astype(self.complex_type)
    
    def get_pinv_svd(self, eigenvalues, eigenvectors, cond=None):
        if cond is None:
            t = eigenvalues.dtype.char.lower()
            factor = {'f': 1E3, 'd': 1E6} 
            cond = factor[t] * np.finfo(t).eps
        else:
            cond = cond
        valid = eigenvalues > cond * eigenvalues.max()
        print("    Calculating pinv_svd ... (cond = %.2e) and rank=%s"% (cond, np.count_nonzero(valid)))
        eigenvalues_inv = np.zeros_like(eigenvalues)
        eigenvalues_inv[valid] = 1 / eigenvalues[valid]
        pseudo_inverse = np.matmul((eigenvectors * eigenvalues_inv), eigenvectors.conj().T)
        # pseudo_inverse = np.dot(eigenvectors, np.dot(np.diag(eigenvalues_inv), eigenvectors.conj().T)) 
        return pseudo_inverse 


    def get_Binv(self, psf=None, var=1e-4, cond=1e-4):
        if psf is None:
            psf = self.psf + var * np.eye(self.psf.shape[0])
        print(" .   Calculating the eigenvalues and eigenvectors of the psf ... \n")
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(psf)
        print(" .   Calculating the pseudo inverse of the psf ... \n")
        self.pinv_svd = self.get_pinv_svd(self.eigenvalues, self.eigenvectors, cond=cond)
        return self.pinv_svd
    

    def get_Binv_cho(self, psf=None, var=1e-4):
        if psf is None:
            psf = self.psf
        psf += var * np.eye(psf.shape[0])
        psf = psf.astype(self.complex_type)
        print(" .   Calculating the cholesky decomposition of the psf ... \n")
        import time
        st = time.time()
        c, low = cho_factor(psf)
        self.inv_cho = cho_solve((c, low), np.eye(psf.shape[0]))
        print("    Time taken for cholesky decomposition: %.2f s"%(time.time()-st))
        return self.inv_cho
    

def get_grid(ra0, dec0, dx, nx, dy, ny):
    '''
    Make grid in flat sky approximation at arbitrary center (ra0, dec0)\n
    all quantity in deg\n
    return\n
        ra: ra for each point in deg\n
        dec: dec for each point in deg\n
        (x, y): relative position for each point with respect to (ra0, dec0) in deg\n
    '''
    x = np.arange(nx)*dx
    x = x-x.mean()
    y = np.arange(ny)*dy
    y = y-y.mean()
    xm, ym = np.meshgrid(x, y, indexing='ij')
    shp = xm.shape
    xm = np.deg2rad(xm.reshape(-1))
    ym = np.deg2rad(ym.reshape(-1))
    zm = np.sqrt(1-xm**2-ym**2)
    pos = np.asfortranarray([zm, xm, ym]).T # nx*ny, 3
    rot = Rotation.from_euler('yz', [-dec0, ra0], degrees=True)
    pos = rot.apply(pos)
    ra, dec = hp.vec2ang(pos, lonlat=True)
    ra = ra.reshape(shp)
    dec = dec.reshape(shp)
    return ra, dec, (x, y)


def gpad(gmap):
    """
    Zero pad the input map
    :return: zero-padded convergence map
    """
    def padwithzeros(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector
    return np.lib.pad(gmap, 2*gmap.shape[0], padwithzeros)

def mapcrop(inmap,n):
    """
    Crop the map by removing the region added for zero padding
    :param inmap: input map to be cropped (e.g. the lensing potential
    :return: outmap - cropped map
    """
    xmin=int(inmap.shape[0]/2-n/2)
    ymin=int(inmap.shape[1]/2-n/2)
    xmax=int(xmin+n)
    ymax=int(ymin+n)
    outmap=inmap[xmin:xmax,ymin:ymax]
    return(outmap)
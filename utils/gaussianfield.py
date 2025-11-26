#%%
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy as sc
import scipy.fft as fftengine
import pyccl as ccl

#%%
# Define a simple binned galaxy number density curve as a function of redshift
# Generate power spectrum from ccl
def cosmo_dafault():
    H0 = 67.36;ombh2 = 0.02237;omch2 = 0.1200
    mnu = 0.06;omk = 0.;tau = 0.0544
    ns = 0.9649;r = 0.;As = 2.100/1e9
    Om = (ombh2+omch2)/(H0/100.)**2.
    sigma8 = 0.812;A_IA = 1 #Plank 2018宇宙学
    cosmo = ccl.Cosmology(Omega_c=omch2/(H0/100.)**2, Omega_b=ombh2/(H0/100.)**2, 
                      h=H0/100., n_s=ns, sigma8=sigma8,  
                      transfer_function='boltzmann_camb', 
                      matter_power_spectrum='halofit')
    return cosmo

def get_cl(cosmo=cosmo_dafault(), z_n=np.linspace(0., 1., 200), ell=np.arange(2, 1000)):
    n = np.ones(z_n.shape)
    tracer_k = ccl.WeakLensingTracer(cosmo,dndz=(z_n, n)) #指定kappa的红移分布
    cl_kk = ccl.angular_cl(cosmo, tracer_k, tracer_k, ell) #kappa自相关
    return cl_kk
#%%
# nside = 512
# ell = np.arange(2, 3*nside)
# cl_kk = get_cl(cosmo_dafault(), ell=ell)
# #Plot cl_kk
# plt.figure()
# plt.loglog(ell, cl_kk)
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$C_\ell^{KK}$')
# plt.show()

#%%
# convert cl_kk in curved sky to 2d_cl in flat sky
def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.
    output:
    lx, ly
    """
    nx, ny, dx, dx = flatskymapparams
    dx = np.radians(dx/60.)
    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ) )
    lx *= 2* np.pi
    ly *= 2* np.pi
    return lx, ly

def cl_to_cl2d(el, cl, flatskymapparams):
    """
    converts 1d_cl to 2d_cl
    inputs:
    el = el values over which cl is defined
    cl = power spectra - cl
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.
    output:
    2d_cl
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape) 
    return cl2d

################################################################################################################
# Gaussian field generation from 2d_cl
def cl2map(flatskymapparams, cl, el = None):

    """
    cl2map module - creates a flat sky map based on the flatskymap parameters and the input power spectra
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.
    cl: 1d (T-only) or nd (TP) vector of cl: temp / pol. power spectra
    el: if None, then computed here.
    output:
    flatskymap with the given map specifications
    """
    print("dx, dy are in arcminutes.")

    if el is None:
        el = np.arange(len(cl))

    nx, ny, dx, dx = flatskymapparams

    #get 2D cl
    cl2d = cl_to_cl2d(el, cl, flatskymapparams) 

    #pixel area normalisation
    dx_rad = np.radians(dx/60.)
    pix_area_norm = np.sqrt(1./ (dx_rad**2.))
    cl2d_sqrt_normed = np.sqrt(cl2d) * pix_area_norm

    #make a random Gaussian realisation now
    gauss_reals = np.random.randn(nx,ny)
    
    #convolve with the power spectra
    flatskymap = np.fft.ifft2( np.fft.fft2(gauss_reals) * cl2d_sqrt_normed).real
    flatskymap = flatskymap - np.mean(flatskymap)

    return flatskymap 


def map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, binsize = None, maxbin=1000):

    """
    map2cl module - get the power spectra of map/maps
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.
    flatskymap1: map1 with dimensions (ny, nx)
    flatskymap2: provide map2 with dimensions (ny, nx) cross-spectra
    binsize: el bins. computed automatically if None
    cross_power: if set, then compute the cross power between flatskymap1 and flatskymap2
    output:
    auto/cross power spectra: [el, cl, cl_err]
    """
    nx, ny, dx, dx = flatskymapparams
    dx_rad = np.radians(dx/60.)
    lx, ly = get_lxly(flatskymapparams)
    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]
    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * dx_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * dx_rad * np.conj( np.fft.fft2(flatskymap2) ) * dx_rad / (nx * ny)
        flatskymap_psd = np.abs(flatskymap_psd)

    rad_prf = radial_profile(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = 0.1, maxbin = maxbin, to_arcmins = 0)
    el, cl = rad_prf[:,0], rad_prf[:,1]

    return el, cl


    
def radial_profile(z, xy = None, bin_size = 1., minbin = 0., maxbin = 10., to_arcmins = 0, bin_arr = None):

    """
    get the radial profile of an image (both real and fourier space)
    """

    z = np.asarray(z)
    if xy is None:
        x, y = np.indices(image.shape)
    else:
        x, y = xy
    #radius = np.hypot(X,Y) * 60.
    radius = (x**2. + y**2.) ** 0.5
    if to_arcmins: radius *= 60.
    binarr=np.arange(minbin,maxbin,bin_size)
    # radprf=np.zeros((len(binarr),3))
    radprf=np.zeros((len(binarr),3), dtype=z.dtype)
    hit_count=[]
    for b,bin in enumerate(binarr):
        ind=np.where((radius>=bin) & (radius<bin+bin_size))
        radprf[b,0]=(bin+bin_size/2.)
        hits = len(np.where(abs(z[ind])>0.)[0])
        if hits>0:
            radprf[b,1]=np.sum(z[ind])/hits
            radprf[b,2]=np.std(z[ind])
        hit_count.append(hits)
    # # print(bin_size)
    # print(radprf[0:5,0])

    hit_count=np.asarray(hit_count)
    std_mean=np.sum(radprf[:,2]*hit_count)/np.sum(hit_count)
    errval=std_mean/(hit_count)**0.5
    radprf[:,2]=errval
    return radprf



#%%
# nside = 512
# ell = np.arange(2, 3*nside)
# cl_kk = get_cl(ell=ell)
# resolution = hp.nside2resol(nside, arcmin=True)
# flatskymapparams = [100, 100, resolution, resolution]
# lx, ly, dx, dy = flatskymapparams
# kappa_map = cl2map(flatskymapparams, cl_kk, el = ell)
# cl2 = map2cl(flatskymapparams, kappa_map)

# plt.loglog(ell, cl_kk, c="black", label="Plank 2018", ls="dashed")
# plt.loglog(cl2[0], cl2[1], c="black", label="Gaussian", ls="solid")
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$C_\ell^{KK}$')
# plt.legend()
# plt.show()












# #%%
# # resolution = hp.nside2resol(nside, arcmin=True)
# # flatskymapparams = [100, 100, resolution, resolution]
# # flatskymap = cl2map(flatskymapparams, cl_kk, el = ell)

# # cl2 = map2cl(flatskymapparams, flatskymap)
# # plt.loglog(ell, cl_kk, c="black", label="true", ls="dashed")
# # plt.loglog(cl2[0], cl2[1], c="black", label="true", ls="solid")
# # plt.legend()
# #%%
# # import sys
# # sys.path.append("/home/yshi/rec_kappa2/")
# # from power import compute_PS
# # nx, ny, dx, dy = flatskymapparams
# # dx_rad = np.radians(dx/60.)
# # fieldSize = nx * dx_rad
# # lk, Plkappa = compute_PS(flatskymap, FieldSize=fieldSize)
# # plt.loglog(lk, Plkappa, c="black", label="true", ls="solid")
# # plt.loglog(ell, cl_kk, c="black", label="true", ls="dashed")
# # plt.legend()
# # plt.show()
# # # %%
# # from power import fft2d
# # x = np.arange(nx) * dx_rad
# # y = np.arange(ny) * dx_rad
# # flatskymap = flatskymap.reshape([1, nx, ny])
# # pk_dict = fft2d(flatskymap, x=x, y=y)
# # k1 = pk_dict['k1d']
# # psd1 = pk_dict['psd1d'].mean(axis=0)
# # plt.loglog(k1, psd1, c="blue", label="Gaussian field", ls="solid")
# # plt.loglog(ell, cl_kk, c="black", label="true", ls="dashed")
# # plt.show()
# # # %%

# # # %%

# %%

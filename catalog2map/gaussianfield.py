#%%
import numpy as np
import matplotlib.pyplot as plt
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
    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ), indexing='ij')
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
    # print(cl2d_sqrt_normed.shape)

    #make a random Gaussian realisation now
    gauss_reals = np.random.randn(nx,ny)
    
    #convolve with the power spectra
    flatskymap = np.fft.ifft2( np.fft.fft2(gauss_reals) * cl2d_sqrt_normed).real
    flatskymap = flatskymap - np.mean(flatskymap)

    return flatskymap 


def map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, binsize = None, maxbin=None, minbin=0.1, binarr=None):

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
        binsize = (ly.ravel()[1] -ly.ravel()[0])
    if maxbin == None:
        maxbin = int(np.pi/dx_rad) +1 

    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * dx_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * dx_rad * np.conj( np.fft.fft2(flatskymap2) ) * dx_rad / (nx * ny)
        flatskymap_psd = np.real(flatskymap_psd)

    rad_prf = radial_profile(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = minbin, maxbin = maxbin, to_arcmins = 0, binarr=binarr)
    el, cl = rad_prf[:,0], rad_prf[:,1]

    return el, cl


    
def radial_profile(z, xy = None, bin_size = 1., minbin = 0., maxbin = 10., to_arcmins = 0, binarr = None):

    """
    get the radial profile of an image (both real and fourier space)
    """

    z = np.asarray(z)
    x, y = xy
    #radius = np.hypot(X,Y) * 60.
    radius = (x**2. + y**2.) ** 0.5
    if to_arcmins: radius *= 60.
    # print(minbin, maxbin, bin_size)
    if binarr is None: 
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
    # print(bin_size)
    # print(radprf[0:5,0])
    hit_count=np.asarray(hit_count)
    std_mean=np.sum(radprf[:,2]*hit_count)/np.sum(hit_count)
    errval=std_mean/(hit_count)**0.5
    radprf[:,2]=errval
    return radprf

# #%%
# dx = 0.05 # degree
# nx = 100 
# ny = 100
# lmax = np.pi/np.deg2rad(dx)
# ell = np.arange(lmax)
# z_n = np.linspace(0., 3.5, 200)
# cls = get_cl(ell=ell, z_n=z_n )
# resolution = dx * 60 # arcmin
# flatskymapparams = [nx, ny, resolution, resolution]
# kappa_map = cl2map(flatskymapparams, cls, el=ell)
# # %%
# noise_map = np.random.randn(nx, ny) * 0.017
# el, cl_x = map2cl(flatskymapparams, kappa_map, noise_map)
# cl1 = map2cl(flatskymapparams, kappa_map)[1]
# cl2 = map2cl(flatskymapparams, noise_map)[1]

# # %%
# plt.plot(el, cl1)
# plt.plot(el, cl2)
# plt.plot(ell, cls)
# plt.yscale('log')
# #%%
# import matplotlib.pyplot as plt
# plt.plot(el, cl_x/np.sqrt(cl1*cl2))
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$r_{\ell}$')
# plt.show()
# %%

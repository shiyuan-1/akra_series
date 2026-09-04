#%%
import numpy as np
import healpy as hp
from tqdm import tqdm
from astropy import units as u
from scipy.spatial.transform import Rotation
import scipy.fft as fftengine
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import sys, os
sys.path.append(os.path.split(os.path.realpath(__file__))[0]+'/utils/')
# sys.path.append(os.getcwd()+"/utils/")
from gaussianfield import get_cl
import plot


#%%
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

def get_bound(ra0, dec0, dx, nx, dy, ny):
    '''
    TODO: this function cannot handle the case where dec cross dec=180 deg\n
    '''
    ra, dec, _ = get_grid(ra0, dec0, dx, nx, dy, ny)
#    ra = np.unwrap(ra[::-1], axis=0, period=360)
#    ra = ra[::-1]
    ra[ra>180] = ra[ra>180] - 360
    phi = []
    phi.append(ra[:,0])
    phi.append(ra[-1,:])
    phi.append(ra[::-1,-1])
    phi.append(ra[0,::-1])
    phi = np.deg2rad(np.concatenate(phi))
    th = []
    th.append(90-dec[:,0])
    th.append(90-dec[-1,:])
    th.append(90-dec[::-1,-1])
    th.append(90-dec[0,::-1])
    th = np.deg2rad(np.concatenate(th))
    return th, phi

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

def gpad2(gmap, pad_width=None):
    if pad_width is None:
        pad_width = gmap.shape[0]*1
    return np.pad(gmap, pad_width, mode='constant', constant_values=0.)

def mapcrop(inmap,pad_width=None):
    if pad_width is None:
        pad_width = int(inmap.shape[0])
    return inmap[pad_width:-pad_width, pad_width:-pad_width]
#%%

# Define the colors in the colormap
# background-image: from https://mycolor.space/gradient3?ori=to+right+top&hex=%23D16BA5&hex2=%2386E7CF&hex3=%23FAFB5F&submit=submit
colors = ['#d16ba5', '#b589d7', '#84a7f4', '#4ec0fa', '#43d4ed', '#48e0e1', '#66e9cf', '#8df1bb', '#9ff7a7', '#b8fa8f', '#d7fc77', '#fafb5f'][::-1]
positions = np.linspace(0, 1, len(colors))
# Create the colormap using LinearSegmentedColormap
cmap2 = mcolors.LinearSegmentedColormap.from_list("my_cmap", list(zip(positions, colors)))

from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_flat_map(ax, flat_map, extent, plot_cbar=True, title=None, cmap=None, vmax=None, vmin=None, set_bad=True, **kwargs):
    if cmap is None:
        cmap = cmap2
    if set_bad:
        cmap.set_bad(color='gray')

    if vmax is None:
        vmax = np.percentile(flat_map, 98)
        # vmax = round(vmax, 2)
    if vmin is None:
        vmin = np.percentile(flat_map, 2)
        # vmin = round(vmin, 2)
    im = ax.imshow(flat_map[::-1].T, extent=extent, origin="lower", aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin, **kwargs)
    if plot_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='%d%%'%(0.05*100), pad=0.05)
        cbar=plt.colorbar(im, cax=cax)
    if title is not None:
        ax.set_title(title)
    return im

##### KS method
def getCosSin_phi(map):
    """
    Get cos(2phi) and sin(2phi) from a map
    """
    lx = fftengine.fftfreq(map.shape[0])
    ly = fftengine.fftfreq(map.shape[1])
    f1, f2 = np.meshgrid(lx, ly, indexing='ij')
    p1 = f1*f1 - f2*f2
    p2 = 2*f1*f2
    f2 = f1*f1 + f2*f2
    f2[0,0] = 1
    sin_2_phi = p2 / f2
    cos_2_phi = p1 / f2
    return cos_2_phi, sin_2_phi

def gamma2kappa(gamma10, gamma20, return_B=False):
    """
    Convert shear to convergence
    Input: gamma1, gamma2 map
    Output: kappa map
    """
    ft_g1 = fftengine.fft2(gamma10)
    ft_g2 = fftengine.fft2(gamma20)
    cos_2_phi, sin_2_phi = getCosSin_phi(gamma10)
    #Compute E and B components
    ft_E = cos_2_phi * ft_g1 + sin_2_phi * ft_g2
    ft_B = -sin_2_phi * ft_g1 + cos_2_phi * ft_g2
    conv_ks = fftengine.ifft2(ft_E).real
    if return_B:
        return conv_ks, fftengine.ifft2(ft_B).real
    else:
        return conv_ks

def kappa2gamma(kappa):
    """
    Generate shear map from kappa map
    Input: kappa map (real)
    Output: gamma1, gamma2 map (complex)
    """
    cos_2_phi, sin_2_phi = getCosSin_phi(kappa)
    kappa_l = fftengine.fft2(kappa)
    gamma1_l = kappa_l * cos_2_phi 
    gamma2_l = kappa_l * sin_2_phi
    gamma1 = fftengine.ifft2(gamma1_l).real
    gamma2 = fftengine.ifft2(gamma2_l).real
    return gamma1, gamma2

# %%
if __name__ == "__main__":
    nside = 512
    nside_out = nside
    lmax = int(1.*nside-1)
    l = np.arange(lmax)
    cls = np.ones_like(l, dtype=float)*1e-9
    cls = get_cl(ell=l)
    # cls[1:] = l[1:]**-0.8
    cls[0] = 0.0
    cls[1] = 0.0
    alm_skymap = hp.synalm(cls)
    kappa_full = hp.alm2map(alm_skymap, nside=nside_out)

    hp.mollview(kappa_full, sub=121)

    # %%
    alm_kappa = hp.map2alm(kappa_full, lmax=lmax)
    ell, emms = hp.Alm.getlm(lmax)

    # kalmsE = - alm_kappa / (((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
    kalmsE = -1. * alm_kappa
    kalmsE[ell == 0] = 0.0
    kalmsE[ell == 1] = 0.0

    ## E, B --> gamma1, gamma2 full sky
    # gamma1, gamma2 = hp.alm2map_spin([kalmsE, np.zeros_like(alm_kappa)], nside=nside_out, lmax=lmax, spin=2)
    _, gamma1, gamma2 = hp.alm2map([np.zeros_like(alm_kappa), kalmsE, np.zeros_like(alm_kappa)], nside=nside_out, pol=True)
    gamma1 = gamma1 * -1.
    hp.mollview(gamma1, sub=121, title="gamma1")
    hp.mollview(gamma2, sub=122, title="gamma2")
    # %%
    ## Use gamma1, gamma2 in sphere to get gamma1, gamma2 in flat sky
    resolution = hp.nside2resol(hp.get_nside(gamma1), arcmin=True)/60
    ra0 = 0.0
    dec0 = 0.0
    dx = resolution
    dy = resolution
    nx = 100
    ny = 100
    ra, dec, (xx, yy) = get_grid(ra0, dec0, dx, nx, dy, ny)
    th_bound, phi_bound = get_bound(ra0, dec0, dx, nx, dy, ny)
    extent = [ra0+(nx+1)*dx/2, ra0-(nx+1)*dx/2, dec0-(ny+1)*dy/2, dec0+(ny+1)*dy/2]
    flat_gamma1 = hp.get_interp_val(gamma1, ra, dec, lonlat=True)
    flat_gamma2 = hp.get_interp_val(gamma2, ra, dec, lonlat=True)
    flat_kappa = hp.get_interp_val(kappa_full, ra, dec, lonlat=True)
    pad_width = int(flat_gamma1.shape[0]/4)
    # pad_width = 10
    # flat_gamma1 = gpad2(flat_gamma1, pad_width)
    # flat_gamma2 = gpad2(flat_gamma2, pad_width)
    print(flat_gamma1.shape)
    print(flat_gamma2.shape)
    print("RA range: ", ra.min(), ra.max())
    print("DEC range: ", dec.min(), dec.max())
    print("width:", nx*dx, ny*dy)
    print("Theta range: ", np.rad2deg(th_bound.min()), np.rad2deg(th_bound.max()))
    print("Phi range: ", np.rad2deg(phi_bound.min()), np.rad2deg(phi_bound.max()))


    #%%
    # Use KS method to get gamma from kappa
    flat_gamma11, flat_gamma21 = kappa2gamma(flat_kappa)
    nx2 = flat_gamma1.shape[0]
    extent2 = [ra0+(nx2+1)*dx/2, ra0-(nx2+1)*dx/2, dec0-(nx2+1)*dy/2, dec0+(nx2+1)*dy/2]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=300)
    plot_flat_map(axs[0], flat_kappa, extent, plot_cbar=True, title="kappa")
    plot_flat_map(axs[1], flat_gamma1, extent2, plot_cbar=True, title="gamma1")
    plot_flat_map(axs[2], flat_gamma2, extent2, plot_cbar=True, title="gamma2")
    plt.tight_layout()

    # %%
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(flat_gamma1 - flat_gamma11, cmap="jet")
    plt.colorbar()
    plt.title("gamma1")
    plt.subplot(122)
    plt.imshow(flat_gamma2 - flat_gamma21, cmap="jet")
    plt.colorbar()
    plt.title("gamma2")
    plt.show()

    # %%
    # conv_ks, conv_B = gamma2kappa(flat_gamma1.T, flat_gamma2.T, return_B=True)
    # conv_ks = conv_ks.T


    conv_ks, conv_B = gamma2kappa(flat_gamma1, flat_gamma2, return_B=True)

    conv_ks = conv_ks


    # conv_ks = mapcrop(conv_ks, pad_width)

    ll = flat_kappa.shape[0]
    ll = int(ll/4)

    extent = [ra0+(nx+1)*dx/2, ra0-(nx+1)*dx/2, dec0-(ny+1)*dy/2, dec0+(ny+1)*dy/2]

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=300)
    plot_flat_map(axs[0], flat_kappa, extent, plot_cbar=True, title="input")
    plot_flat_map(axs[1], conv_ks.real, extent, plot_cbar=True, title="recovered")
    plot_flat_map(axs[2], conv_ks.real-flat_kappa, extent, plot_cbar=True, title="residual")
    plt.tight_layout()
    # print("RA range: %.2f - %.2f deg, DEC range: %.2f - %.2f deg"%(ra.min(), ra.max(), dec.min(), dec.max()))

    print("RA0, DEC0: ", ra0, dec0)
    print("width: ", nx*dx, ny*dy, "deg")

    # plt.figure(figsize=(10, 5))
    # plt.subplot(131)
    # plt.imshow(flat_kappa[::-1].T, cmap="jet")
    # plt.title("input")
    # plt.colorbar()
    # plt.subplot(132)
    # plt.imshow(conv_ks.real, cmap="jet")
    # plt.title("recovered")
    # plt.colorbar()
    # plt.subplot(133)
    # plt.imshow(conv_ks.real-flat_kappa, cmap="jet")
    # plt.title("residual")
    # plt.colorbar()
    # plt.show()

    plt.figure(figsize=(5, 5), dpi=100)
    plt.subplot(111)
    plot.plot_hist2d(flat_kappa[ll:-ll, ll:-ll], conv_ks.real[ll:-ll, ll:-ll], bins=300, cmap="jet")
    # %%
    ## compare power spectrum
    import gaussianfield
    resolution =  hp.nside2resol(nside, arcmin=True) 
    flatskymapparams = [flat_kappa[ll:-ll, ll:-ll].shape[0], flat_kappa[ll:-ll, ll:-ll].shape[1], resolution, resolution]
    el, cl_kappa = gaussianfield.map2cl(flatskymapparams, flat_kappa[ll:-ll, ll:-ll], maxbin=lmax, binsize=100)
    el, cl_ks = gaussianfield.map2cl(flatskymapparams, conv_ks.real[ll:-ll, ll:-ll], maxbin=lmax, binsize=100)
    _, cl_ks_true = gaussianfield.map2cl(flatskymapparams, flat_kappa[ll:-ll, ll:-ll], conv_ks.real[ll:-ll, ll:-ll], maxbin=lmax, binsize=100)
    plt.figure(figsize=(6, 3), dpi=100)
    plt.subplot(121)
    plt.loglog(el, cl_kappa, label="True")
    # plt.plot(el, cl_kappa/cl_ks)
    plt.loglog(el, cl_ks, label="KS")
    # plt.loglog(el, cl_inv, label="AKRA", ls="--")
    plt.legend()
    plt.ylabel(r"$C_{\ell}$")
    plt.xlabel(r"$\ell$")
    plt.subplot(122)
    plt.plot(el, np.abs(cl_ks_true)/np.sqrt(cl_kappa*cl_ks), label="True")
    plt.ylabel(r"$r_{\ell}$")
    plt.xlabel(r"$\ell$")
    # plt.ylim(0.99, 1.01)
    plt.xlim(2, lmax)
    plt.tight_layout()
    plt.show()
#%%







#%%


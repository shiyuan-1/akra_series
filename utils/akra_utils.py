#%%
import numpy as np
import healpy as hp
from tqdm import tqdm
from matplotlib import pyplot as plt
from astropy import units as u
from hpplot import mollview
import sys
sys.path.append("/home/yshi/py_style")
# sys.path.append("/gravity/home/yshi/py_style")
import plt_style
# %%
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


#%%
linestyle_tuple = [
     ('solid', 'solid'),   
     ('loosely dotted',        (0, (1, 3))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 2))),

     ('loosely dashed',        (0, (2, 4))),
     ('dashed',                (0, (2, 3))),
     ('densely dashed',        (0, (2, 1))),

     ('loosely dashdotted',    (0, (1, 3, 1, 3))),
     ('dashdotted',            (0, (1, 2, 1, 2))),
     ('densely dashdotted',    (0, (1, 1, 1, 1))),

     ('dashdotdotted',         (0, (1, 3, 1, 3, 1, 3))),
     ('loosely dashdotdotted', (0, (1, 3, 1, 3, 1, 3))),
     ('densely dashdotdotted', (0, (2, 1, 1, 1, 1, 1)))]


def show_map_cl(maps_in, labels=None, linestyles=None, **kwargs):
    np.random.seed(951121)
    if labels is None:
        labels = [f'map {i}' for i in range(len(maps_in))]
    if linestyles is None:
        lt_index = np.arange(len(linestyle_tuple))
        lt_in = np.random.choice(lt_index[1:], size=len(maps_in), replace=False)
        linestyles = [linestyle_tuple[0][1]]  + [linestyle_tuple[i][1] for i in lt_in]
        # print(linestyles)
    for i, (map_in, label, linestyle) in enumerate(zip(maps_in, labels, linestyles)):
        cl = hp.anafast(map_in)
        plt.loglog(cl, label=label, ls=linestyle, **kwargs)

    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell$')
    plt.legend()
    plt.tight_layout()


def show_alm_cl(alms_in, labels=None, linestyle=None, **kwargs):
    np.random.seed(100)
    if labels is None:
        labels = [f'map {i}' for i in range(len(alms_in))]
    if linestyle is None:
        lt_index = np.arange(len(linestyle_tuple))
        lt_in = np.random.choice(lt_index[1:], size=len(alms_in), replace=False)
        linestyles = [linestyle_tuple[0][1]]  + [linestyle_tuple[i][1] for i in lt_in]
    for i, (alm_in, label, linestyle) in enumerate(zip(alms_in, labels, linestyles)):
        cl = hp.alm2cl(alm_in)
        plt.loglog(cl, label=label, ls=linestyle, **kwargs)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell$')
    plt.legend()
    plt.tight_layout()


# %%
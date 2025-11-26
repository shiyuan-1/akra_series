import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from mpl_toolkits.axes_grid1 import make_axes_locatable



def round(x, n):
    fct = 10**np.floor(np.log10(np.abs(x)))
    y = np.around(x/fct, n)
    return y*fct

def mollview(skymap, n_max=99, n_min=1, vmax=None, vmin=None, cmap="RdBu_r", unseen=False, **kwargs):
    if vmax is None:
        vmax = np.percentile(skymap, n_max)
        vmax = round(vmax, 1)
    if vmin is None:
        vmin = np.percentile(skymap, n_min)
        vmin = round(vmin, 1)
    if unseen:
        skymap_ma = skymap.copy()
        skymap_ma[skymap_ma==0] = hp.UNSEEN
        hp.projview(hp.ma(skymap_ma), max=vmax, min=vmin, cmap=cmap, **kwargs)
    else:
        hp.projview(skymap, max=vmax, min=vmin, cmap=cmap, **kwargs)

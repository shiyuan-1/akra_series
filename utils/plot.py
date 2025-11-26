#%%
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import scipy as sc
import scipy.fft as fftengine
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors

# %matplotlib qt
font = {'family' : 'times',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 16,
        }
# %%

# Define the colors in the colormap
# background-image: from https://mycolor.space/gradient3?ori=to+right+top&hex=%23D16BA5&hex2=%2386E7CF&hex3=%23FAFB5F&submit=submit
colors = ['#d16ba5', '#b589d7', '#84a7f4', '#4ec0fa', '#43d4ed', '#48e0e1', '#66e9cf', '#8df1bb', '#9ff7a7', '#b8fa8f', '#d7fc77', '#fafb5f'][::-1]
positions = np.linspace(0, 1, len(colors))
# Create the colormap using LinearSegmentedColormap
cmap2 = mcolors.LinearSegmentedColormap.from_list("my_cmap", list(zip(positions, colors)))

#%%
import matplotlib as mpl
def plot_hist2d(map1, map2, bins=100, cmap="jet", norm=mpl.colors.LogNorm(), xlabel=None, ylabel=None, **kwargs):
    plt.hist2d(map1.ravel(), map2.ravel(), bins=bins, cmap=cmap, norm=norm, **kwargs)
    plt.plot([map1.min(), map1.max()], [map1.min(), map1.max()], 'k--', lw=2)
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel("True")
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("Recovered")


# from mpl_toolkits.axes_grid1 import make_axes_locatable
# def plot_flat_map(ax, flat_map, extent, plot_cbar=True, title=None, cmap=None, vmax=None, vmin=None, **kwargs):
#     if cmap is None:
#         cmap = cmap2
#     if vmax is None:
#         vmax = np.percentile(flat_map, 98)
#         # vmax = round(vmax, 2)
#     if vmin is None:
#         vmin = np.percentile(flat_map, 2)
#         # vmin = round(vmin, 2)
#     im = ax.imshow(flat_map[::-1].T, extent=extent, origin="lower", aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin, **kwargs)
#     if plot_cbar:
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes('right', size='%d%%'%(0.05*100), pad=0.05)
#         cbar=plt.colorbar(im, cax=cax)
#     if title is not None:
#         ax.set_title(title)
#     return im


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
    im = ax.imshow(flat_map, extent=extent, origin="lower", aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin, **kwargs)
    if plot_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='%d%%'%(0.05*100), pad=0.05)
        cbar=plt.colorbar(im, cax=cax)
    if title is not None:
        ax.set_title(title)
    return im


## Plot matrix
def plot_mat(ax, matrix0, title=" ", cmap="RdYlBu_r", label_x=None, label_y=None, **kwargs):
    """ 
    Plot a matrix map with a colorbar
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    im = ax.imshow(matrix0.real, cmap=cmap, **kwargs)
    plt.title(title)
    if label_x is not None:
        plt.xlabel(label_x)
    if label_y is not None:
        plt.ylabel(label_y)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
# %%
def plot_mat2(ax, matrix0, title=None, set_colorbar=None, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    im = ax.imshow(matrix0.real, origin='lower',**kwargs)
    ax.set_xlabel(r'RA (deg)')
    ax.set_ylabel(r'Dec (deg)')
    if title is not None:
        ax.set_title(title)

    if set_colorbar is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)



def plot_kde2(ax, matrixA, matrixB, title=" ", cmap="rocket_r", text_str=None):
    xx = [-0.03, 0.03]
    ax.plot(xx, xx, 'k', lw=3)
    df = pd.DataFrame({'kappa': matrixA.ravel(), 'conv_ks': matrixB.ravel()})
    sns.regplot(x='kappa', y='conv_ks', data=df, ax=ax, scatter_kws={'s': 10})
    ax.set(xlim=(-0.02, 0.02), ylim=(-0.02, 0.02), title=title, xlabel=r'$\kappa^{\rm true}$', 
           ylabel=r'$\kappa^{\rm rec}$')
    ax.grid(linestyle='--', linewidth=2, alpha=0.9)
    if text_str is not None:
        ax.text(0.2, 0.9, text_str, ha='center', transform=ax.transAxes, fontsize=10)

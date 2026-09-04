import numpy as np
# import pyccl as ccl
import healpy as hp
from flat_sky_tools import interp_grid, get_grid, get_rot_inv
from numba import jit
from scipy.spatial import cKDTree
import h5py

def get_interp_val(map_in, lon, lat, lonlat=True, return_index=False):
    nside_out = hp.get_nside(map_in)
    if return_index:
        ind = hp.ang2pix(nside_out, lon, lat, lonlat=lonlat)
        return map_in[ind], ind
    else:
        ind = hp.ang2pix(nside_out, lon, lat, lonlat=lonlat)
        return map_in[ind]

def save2hdf5(file_name, save_dict, attr_dict=None, replace_file=True):
    if replace_file:
        with h5py.File(file_name, 'w') as f:
            for key in save_dict.keys():
                f.create_dataset(key, data=save_dict[key])
            if attr_dict is not None:
                for key in attr_dict.keys():
                    f.attrs[key] = attr_dict[key]
    else:
        with h5py.File(file_name, 'a') as f:
            for key in save_dict.keys():
                if key in f:
                    del f[key]
                f.create_dataset(key, data=save_dict[key])
            if attr_dict is not None:
                for key in attr_dict.keys():
                    f.attrs[key] = attr_dict[key]


def lonlat2hpmap(lon0, lat0, x, y, field, lon=None, lat=None, nside=None, return_ipix=False, remove_pixs=10):
    if remove_pixs is not None:
        pix_i = int(remove_pixs)
    if lon is None or lat is None:
        print("Lon and lat are not provided, please provide them.") 
    else:
        lon = lon[pix_i:-pix_i, pix_i:-pix_i]
        lat = lat[pix_i:-pix_i, pix_i:-pix_i]
    npix = hp.nside2npix(nside)
    vec = hp.ang2vec(lon, lat, lonlat=True)
    ipix = hp.query_polygon(nside, [vec[0,0], vec[-1,0], vec[-1, -1], vec[0, -1]])
    lon_pix, lat_pix = hp.pix2ang(nside, ipix, lonlat=True)
    hpmap = np.zeros(npix)
    hpmap[ipix] = interp_grid(lon0, lat0, x, y, field, lon_pix, lat_pix, kx=3, ky=3)
    if return_ipix:
        return hpmap, ipix
    else:
        return hpmap


def get_lonRange(lon_array):
    lon_values = lon_array.copy()
    lon_values = lon_values%360
    lon_values[lon_values>180] -= 360
    lonLowRange = np.abs(lon_values.T[0,0] - lon_values.T[0,-1])
    lonHighRange = np.abs(lon_values.T[-1,0] - lon_values.T[-1,-1])
    if lonLowRange > lonHighRange:
        return lonHighRange, lonLowRange
    else:
        return lonLowRange, lonHighRange


def set_zero(map_input, ll_shape):
    map_output = np.copy(map_input)
    map_output[0:ll_shape, :] = 0
    map_output[-ll_shape:, :] = 0
    map_output[:, 0:ll_shape] = 0
    map_output[:, -ll_shape:] = 0
    return map_output


def get_interp_pix(kdt, lon, lat, npix_use, lonlat=True):
    vec_flat = hp.ang2vec(lon.reshape(-1), lat.reshape(-1), lonlat=lonlat)
    vec_flat = np.asfortranarray(vec_flat)
    dist, ipix = kdt.query(vec_flat, npix_use, workers=100)
    return dist, ipix

from numba import jit
_NUMBA_CACHE=True
@jit(nopython=True, parallel=True, cache=_NUMBA_CACHE)
def exp_numba(dist, sigma):    
    return np.exp(-dist**2/2./sigma**2)


def choose_valid(pix_gal, lon0, lat0, nside, radius_deg):
    vec0 = hp.ang2vec(lon0, lat0, lonlat=True)
    pix_valid = hp.query_disc(nside, vec0, np.deg2rad(radius_deg), inclusive=True)
    valid = np.where(np.isin(pix_gal, pix_valid))[0]
    return valid

def make_healpix_map(pixels, weights, nside=None, minlength=None):
    if minlength is not None:
        minlength = minlength
        nside = None
    if nside is not None:
        minlength=hp.nside2npix(nside) 
    # bincount = np.bincount(pixels, minlength=minlength)
    bincount_weighted = np.bincount(pixels, weights=weights, minlength=minlength)
    return bincount_weighted

def create_skymap(kdtree, vec, npix_use, sigma, skyobs, npix, nx, ny):
    dist, ipix = kdtree.query(vec, npix_use, workers=-1)
    w = exp_numba(dist, sigma)
    skymap_unnorm = np.zeros(npix, dtype=np.float64)
    weights = np.zeros(npix, dtype=np.float64)
    skymap_unnorm = make_healpix_map(ipix.reshape(-1), (w * skyobs[:, None]).reshape(-1), minlength=npix)
    weights = make_healpix_map(ipix.reshape(-1), w.reshape(-1), minlength=npix)
    mask0 = weights > 0
    skymap = np.zeros_like(skymap_unnorm)
    skymap[mask0] = skymap_unnorm[mask0] / weights[mask0]
    skymap = skymap.reshape(nx, ny)
    return skymap


_NUMBA_CACHE=True
@jit(nopython=True, parallel=True, cache=_NUMBA_CACHE)
def interp_arrays(skymaps, dist_array, ipix_array, sigma, pix_area, nx=100, ny=100):
    if len(skymaps.shape) == 1:
        skymaps = skymaps.reshape(1, -1)
    fields = np.zeros((len(dist_array), skymaps.shape[0], ipix_array[0].shape[0]))  # Convert ny to integer
    for i in range(len(dist_array)):
        dist = dist_array[i]
        ipix = ipix_array[i]
        w = np.exp(-dist**2/2.0/sigma**2)/(2*np.pi*sigma**2)*pix_area
        temp = np.zeros((skymaps.shape[0], ipix.shape[0]))
        for j in range(skymaps.shape[0]):
            for k in range(ipix.shape[0]):
                for l in range(ipix.shape[-1]):
                    temp[j, k] += skymaps[j, ipix[k, l]] * w[k,l]
        fields[i] = temp  
    return fields



from numba import jit
# from numba import prange
_NUMBA_CACHE=True
@jit(nopython=True, parallel=True, cache=_NUMBA_CACHE)
def interp_flatmap(skymap_obs, x_idx, y_idx, nx, ny):
    skymap_flat = np.zeros((nx, ny))
    weight_flat = np.zeros((nx, ny))
    for i in range(skymap_obs.shape[0]):
        if x_idx[i] >= nx or y_idx[i] >= ny:
            continue
        if x_idx[i] < 0 or y_idx[i] < 0:
            continue
        skymap_flat[x_idx[i], y_idx[i]] += skymap_obs[i]
        weight_flat[x_idx[i], y_idx[i]] += 1
    return skymap_flat, weight_flat

def interp_cic_flatmap(skymap_obs, x_new, y_new, nx, ny, x1, y1, dx, dy):
    position = np.array([x_new-x1[0], y_new-y1[0]]).T
    indices = np.array(position/dx, dtype=int)
    dr = (position/dx-(indices)).T
    dr = np.abs(dr)
    tr = 1-dr
    skymap_flat = np.zeros((nx, ny))
    weight_flat = np.zeros((nx, ny))

    for i in range(skymap_obs.shape[0]):
        if 0 <= indices[i,0] < nx and 0 <= indices[i,1] < ny:
            indx = indices[i,:].T
            skymap_flat[tuple(indx)] += np.prod(tr[:,i])*skymap_obs[i]
            weight_flat[tuple(indx)] += np.prod(tr[:,i])

            skymap_flat[tuple((indx+np.array([1,0]))%[nx,ny])] += dr[0,i]*tr[1,i]*skymap_obs[i]
            weight_flat[tuple((indx+np.array([1,0]))%[nx,ny])] += dr[0,i]*tr[1,i]

            skymap_flat[tuple((indx+np.array([0,1]))%[nx,ny])] += tr[0,i]*dr[1,i]*skymap_obs[i]
            weight_flat[tuple((indx+np.array([0,1]))%[nx,ny])] += tr[0,i]*dr[1,i]

            skymap_flat[tuple((indx+np.array([1,1]))%[nx,ny])] += dr[0,i]*dr[1,i]*skymap_obs[i]
            weight_flat[tuple((indx+np.array([1,1]))%[nx,ny])] += dr[0,i]*dr[1,i]
    return skymap_flat, weight_flat

def choose_valid2(pix, lon0, lat0, nside, radius_deg):
    vec = hp.ang2vec(lon0, lat0, lonlat=True)
    pix0 = hp.query_disc(nside, vec, np.deg2rad(radius_deg))
    valid = np.in1d(pix, pix0)
    return valid

def catalog2map(nside_cat, lon_cat, lat_cat, skymap_cat, lon0, lat0, x, y, dx, nx, dy, ny, choose_valid_catalog=True, window='cic'):
    pix_cat = hp.ang2pix(nside_cat, lon_cat, lat_cat, lonlat=True)
    # valid = np.ones_like(pix_cat, dtype=bool)
    if choose_valid_catalog:
        valid = choose_valid2(pix_cat, lon0, lat0, nside_cat, nx*dx*1.5)
    else:
        valid = np.ones_like(pix_cat, dtype=bool)
    skymap_obs = skymap_cat[valid]
    vec_in = hp.ang2vec(lon_cat[valid], lat_cat[valid], lonlat=True)
    rot_inv = get_rot_inv(lon0, lat0)
    vec_out = rot_inv.apply(vec_in)
    x_new = np.rad2deg(vec_out[...,1])
    y_new = np.rad2deg(vec_out[...,2])
    # z_new = np.rad2deg(vec_out[...,0])
    # valid2 = z_new > 0
    if window=='ngp':
        x_idx = np.around( (x_new-x[0])/dx).astype(int)
        y_idx = np.around( (y_new-y[0])/dy).astype(int)
        skymap_flat, weight_flat = interp_flatmap(skymap_obs, x_idx, y_idx, nx, ny)
    elif window=='cic':
        skymap_flat, weight_flat = interp_cic_flatmap(skymap_obs, x_new, y_new, nx, ny, x, y, dx, dy)

    # skymap_flat, weight_flat = interp_flatmap(skymap_obs[valid2], x_idx[valid2], y_idx[valid2], nx, ny) 
    return skymap_flat, weight_flat


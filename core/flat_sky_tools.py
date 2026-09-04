#%%
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import RectBivariateSpline
import healpy as hp

def get_rot_inv(lon0, lat0):
    return Rotation.from_euler('zy', [-lon0, lat0], degrees=True)

def get_rot(lon0, lat0):
    return Rotation.from_euler('yz', [-lat0, lon0], degrees=True)

def get_grid(lon0, lat0, dx, nx, dy, ny):
    '''
    Make grid in flat sky approximation at arbitrary center (lon0, lat0)\n
    all quantity in deg\n
    return\n
        ra: ra for each point in deg\n
        dec: dec for each point in deg\n
        (x, y): relative position for each point with respect to (lon0, lat0) in deg\n
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
    #rot = Rotation.from_euler('yz', [-lat0, lon0], degrees=True)
    rot = get_rot(lon0, lat0)
    pos = rot.apply(pos)
    ra, dec = hp.vec2ang(pos, lonlat=True)
    ra = ra.reshape(shp)
    dec = dec.reshape(shp)
    return ra, dec, (x, y)

def interp_grid(lon0, lat0, x, y, field, lon_out, lat_out, bbox=[None]*4, kx=3, ky=3, s=0, full=False):
    lon_out, lat_out = np.broadcast_arrays(lon_out, lat_out)
    shp = lon_out.shape
    lon_out = lon_out.reshape(-1)
    lat_out = lat_out.reshape(-1)
    rot_inv = get_rot_inv(lon0, lat0)
    vec_out = hp.ang2vec(lon_out, lat_out, lonlat=True)
    vec_new = rot_inv.apply(vec_out)
    x_new = np.rad2deg(vec_new[...,1].reshape(shp))
    y_new = np.rad2deg(vec_new[...,2].reshape(shp))
    intf = RectBivariateSpline(x, y, field, bbox=bbox, kx=kx, ky=ky, s=s)
    field_new = intf(x_new, y_new, grid=False)
    if full:
        return field_new, (x_new, y_new, intf)
    else:
        return field_new

def rot_grid(x, y, field, th_rot, x_out=None, y_out=None, bbox=[None]*4, kx=3, ky=3, s=0):
    if x_out is None or y_out is None:
        x_out, y_out = np.meshgrid(x, y, indexing='ij')
    x_itp = x_out*np.cos(-th_rot) - y_out*np.sin(-th_rot)
    y_itp = x_out*np.sin(-th_rot) + y_out*np.cos(-th_rot)
    intf = RectBivariateSpline(x, y, field, bbox=bbox, kx=kx, ky=ky, s=s)
    field_new = intf(x_itp, y_itp, grid=False)
    return field_new

def get_eth_ephi(lon, lat):
    lon, lat = np.broadcast_arrays(lon, lat)
    eth = np.zeros(lon.shape+(3,), dtype=np.float64)
    ephi = np.zeros(lon.shape+(3,), dtype=np.float64)
    th = np.deg2rad(90 - lat)
    phi = np.deg2rad(lon)
    eth[...,0] = np.cos(th)*np.cos(phi)
    eth[...,1] = np.cos(th)*np.sin(phi)
    eth[...,2] = -np.sin(th)
    ephi[...,0] = -np.sin(phi)
    ephi[...,1] = np.cos(phi)
    return eth, ephi

def get_coord_rot(lon0, lat0, lon1):
    th0 = np.deg2rad(90 - lat0)
    phi0 = np.deg2rad(lon0)
    phi1 = np.deg2rad(lon1)
    costh = np.sin(phi0) * np.sin(phi1) + np.cos(phi0)*np.cos(phi1)
    sinth = -np.sin(phi1)*np.cos(th0)*np.cos(phi0) + np.cos(phi1)*np.cos(th0)*np.sin(phi0)
    return np.arctan2(-sinth, costh) # as theta increase, y decrease and therefore there is a minus before sinth
#%%
if __name__ == '__main__':
    ## test for rotate field
    from matplotlib import pyplot as plt
    x = np.linspace(-2, 2, 101)*3
    y = np.linspace(-1, 1, 201)*3
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    extent = [x.min()-dx/2, x.max()+dx/2, y.min()-dy/2, y.max()+dy/2]
    field = np.exp(-x[:,None]**2/2.0-y**2)
    plt.figure()
    plt.subplot(121)
    plt.imshow(field.T, origin='lower', extent=extent)
    th_rot = np.pi/5
    field_new = rot_grid(x, y, field, th_rot)
    plt.subplot(122)
    plt.imshow(field_new.T, origin='lower', extent=extent)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(x, np.tan(th_rot)*x, 'r')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()
#%%
#%%
if __name__ == '__main__':
    ## test for interpolate grid
    from matplotlib import pyplot as plt
    nside = 256
    cls = np.zeros(2*nside, dtype=np.float64)
    cls[1:] = np.arange(1.0, cls.shape[0])**-3
    skymap = hp.synfast(cls, nside=nside)
    hp.mollview(skymap)
    lon0 = 70
    lat0 = 85
    dx = 0.2
    dy = 0.1
    nx = 128
    ny = 256
    lon, lat, (x, y) = get_grid(lon0, lat0, dx, nx, dy, ny)
    vec0 = hp.ang2vec(lon0, lat0, lonlat=True)
    vec = hp.ang2vec(lon.reshape(-1), lat.reshape(-1), lonlat=True)
    vec = vec.reshape(lon.shape+(3,))
    #ipix = hp.query_disc(nside, vec0, np.deg2rad(10.0))
    ipix = hp.query_polygon(nside, [vec[32,64], vec[96,64], vec[96, 128], vec[32, 128]])
    lon_pix, lat_pix = hp.pix2ang(nside, ipix, lonlat=True)
    skymap_new0 = np.zeros_like(skymap)
    skymap_new0[ipix] = skymap[ipix]
    field = hp.get_interp_val(skymap, lon, lat, lonlat=True)
    plt.figure()
    plt.imshow(field)
    skymap_new1 = np.zeros_like(skymap)
    skymap_new1[ipix] = interp_grid(lon0, lat0, x, y, field, lon_pix, lat_pix)
    plt.figure()
    hp.mollview(skymap_new0, sub=131)
    hp.mollview(skymap_new1, sub=132)
    hp.mollview((skymap_new0-skymap_new1)/skymap_new0[ipix].std(), sub=133)
    field0 = hp.get_interp_val(skymap_new0, lon, lat, lonlat=True)
    field1 = hp.get_interp_val(skymap_new1, lon, lat, lonlat=True)
    plt.figure(figsize=[5, 15])
    plt.subplot(311)
    plt.imshow(field0.T, origin='lower')
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(field1.T, origin='lower')
    plt.colorbar()
    plt.subplot(313)
    plt.imshow(field0.T-field1.T, origin='lower')
    plt.colorbar()

#%%
if __name__ == '__main__':
    ## test for rotate coordinate
    # from make_grid import get_grid
    from matplotlib import pyplot as plt
    lon0 = 70
    lat0 = 85
    #lat0 = 35
    dx = 0.05
    dy = 0.05
    nx = 256
    ny = 256
    lon, lat, (x, y) = get_grid(lon0, lat0, dx, nx, dy, ny)
    #%%
    extent = [lon0-nx*dx/2.0, lon0+nx*dx/2.0, lat0-ny*dy/2.0, lat0+ny*dy/2.0]
    plt.subplot(211)
    plt.imshow(lon.T, origin='lower', extent=extent)
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(lat.T, origin='lower', extent=extent)
    plt.colorbar()
    #%%
    th_rot = get_coord_rot(lon0, lat0, lon)
    plt.figure()
    plt.imshow(lon.T, origin='lower', extent=[x.min()-dx/2.0, x.max()+dx/2.0, y.min()-dy/2.0, y.max()+dy/2.0])
    for ii in range(20, lon.shape[0]-20, 10):
        for jj in range(20, lon.shape[1]-20, 10):
            plt.arrow(x[ii], y[jj], np.cos(th_rot[ii,jj]), np.sin(th_rot[ii,jj]), head_width=0.2)
    plt.figure()
    plt.imshow(lat.T, origin='lower', extent=[x.min()-dx/2.0, x.max()+dx/2.0, y.min()-dy/2.0, y.max()+dy/2.0])
    for ii in range(20, lon.shape[0]-20, 10):
        for jj in range(20, lon.shape[1]-20, 10):
            plt.arrow(x[ii], y[jj], -np.sin(th_rot[ii,jj]), np.cos(th_rot[ii,jj]), head_width=0.2)
    #%%
    grad_lon_x = np.gradient(lon, x, axis=0)
    grad_lon_y = np.gradient(lon, y, axis=1)
    th_rot1 = np.arctan2(grad_lon_y, grad_lon_x)
    #%%
    plt.figure()
    plt.subplot(311)
    plt.imshow(th_rot1)
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(th_rot)
    plt.colorbar()
    plt.subplot(313)
    plt.imshow(th_rot-th_rot1, vmax=0.02, vmin=-0.02)
    plt.colorbar()
    #%%
    #%%
    #%%
    lon0 = 50
    lat0 = 85
    lon1 = 60
    lat1 = 85
    eth0, ephi0 = get_eth_ephi(lon0, lat0)
    eth1, ephi1 = get_eth_ephi(lon1, lat1)
    #%%
    print(eth0@eth0, ephi0@ephi0)
    print(eth0@ephi0)
    print(eth1@ephi1)
    c1 = eth1@eth0
    c2 = eth1@ephi0
    print(c1, c2, np.sqrt(c1**2 + c2**2))
    c1 = ephi1@eth0
    c2 = ephi1@ephi0
    print(c1, c2, np.sqrt(c1**2 + c2**2))
    th_rot = get_coord_rot(lon0, lat0, lon1)
    print(-np.sin(th_rot), np.cos(th_rot))
    #%%
    #%%
    #%%
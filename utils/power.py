import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fftengine
import scipy.stats as stats

def compute_PS(input_map, input_map2=None, FieldSize=10):
    """ 
    Compute the angular power spectrum of input_map.
    :param input_map: input map (n x n numpy array) :param FieldSize: the side-length of the input map in degrees
    :return: l, Pl - the power-spectrum at l 
    """ 
    # set the number of pixels and the unit conversion factor 
    npix = input_map.shape[0] 
    factor = 2.0*np.pi * np.radians(FieldSize/npix)

    # take the Fourier transform of the input map: 
    fourier_map = fftengine.fft2(input_map)/npix**2

    # compute the Fourier amplitudes 
    if input_map2 is not None:
        fourier_map2 = fftengine.fft2(input_map2)/npix**2
        fourier_amplitudes = np.real(fourier_map*fourier_map2.conj())
    else:
        fourier_amplitudes = np.abs(fourier_map)**2
    fourier_amplitudes = fourier_amplitudes.flatten()

    # compute the wave vectors 
    kfreq = fftengine.fftfreq(input_map.shape[0])*input_map.shape[0] 
    kfreq2D = np.meshgrid(kfreq, kfreq)

    # take the norm of the wave vectors 
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2) 
    knrm = knrm.flatten()

    # set up k bins. The PS will be evaluated in these bins 
    half = npix/2 
    rbins = int(np.sqrt(2*half**2))+1 
    kbins = np.linspace(0.0,rbins,(rbins+1))

    # use the middle points in each bin to define the values of k 
    # where the PS is evaluated 
    kvals = 0.5 * (kbins[1:] + kbins[:-1])*factor

    # now compute the PS: calculate the mean of the 
    # Fourier amplitudes in each kbin 
    Pbins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic = "mean", bins = kbins)
    # return kvals and PS 
    l=kvals[1:] 
    Pl=Pbins[1:]/factor**2
    return l, Pl


#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
from scipy.stats import multivariate_normal
from astropy.cosmology import FlatLambdaCDM
import matplotlib
matplotlib.rcParams["font.size"] = 18
#matplotlib.rc_file('matplotlibrc')
import os
import six
import scipy as sc

def herminian_1d(arr):
    larr = len(arr)
    if larr%2 == 0:
        arr[0] = np.abs(arr[0])
        arr[int(larr/2)] = np.abs(arr[int(larr/2)])
        arr[1:int(larr/2)] = np.flip( arr[int(larr/2)+1:larr].conj() )
    else:
        arr[0] = np.abs(arr[0])
        arr[1:int(larr/2)+1] = np.flip( arr[int(larr/2)+1:larr].conj() )
    return arr

def generate_field2d(statistic, power_spectrum, shape, unit_length=1,
                   fft=sc.fft, fft_args=dict()):

    '''
    statistic: function that generate random values for real and imaginary part\n
    power_spectrum: power_spectrum(kx, ky) returns the 1d power spectrum\n
    shape: shape of the final 2d field, length must be 2, shape[1] must be even\n
    unit_length: unit_length of the final field\n
    fft: fft class\n
    fft_args: args for fft.irfftn\n
    return: ndarray, the 2d field
    '''
    assert len(shape)==2, 'Only support 2D field presently!'
    if np.ndim(unit_length) == 0:
        unit_length = [unit_length]*len(shape)
    assert len(unit_length) == len(shape), 'Incompatible length of unit_length %d and shape %d!'%(len(unit_length), len(shape))
    V = [unit_length[ii]*shape[ii] for ii in range(len(shape))]
    V = np.prod(V)
    if not six.callable(statistic):
        raise Exception('`statistic` should be callable')
    if not six.callable(power_spectrum):
        raise Exception('`power_spectrum` should be callable')
    try:
        fftfreq = fft.fftfreq
        rfftfreq = fft.rfftfreq
        irfftn = fft.irfftn
        fftshift = fft.fftshift
    except NameError:
        # Fallback on numpy for the frequencies
        fftfreq = sc.fft.fftfreq
        rfftfreq = sc.fft.rfftfreq
        irfftn = sc.fft.irfftn
        fftshift = sc.fft.fftshift

    # Compute the k grid
    if shape[-1]%2 != 0:
        raise Exception('Last dimension must be even!')
    all_k = [2*np.pi*fftfreq(s, d=unit_length[ii]) for ii, s in enumerate(shape[:-1])] + \
            [2*np.pi*rfftfreq(shape[-1], d=unit_length[-1])]
    kgrid = np.meshgrid(*all_k, indexing='ij')
    fourier_shape = kgrid[0].shape
    fftfield = statistic(fourier_shape)
    #fftfield0 = fftfield.copy()
    herminian_1d(fftfield[...,0])
    herminian_1d(fftfield[...,-1])

    power_k = np.sqrt(power_spectrum(*kgrid)/1./V)
    fftfield *= power_k
    return fftshift( np.real(irfftn(fftfield, **fft_args))*np.prod(shape) )

def lognormal(x, logmu, sigma):
    x = np.asarray(x)
    valid = x>0
    logx = np.log(x[valid])
    result = np.zeros_like(x, dtype=np.float64)
    logx = np.log(x[valid])
    result[valid] = np.exp(-(logx-logmu)**2/2./sigma**2)
    return result

def gaus_distrib_ri(shape):
    #a = np.random.normal(loc=1., scale=0.05, size=shape)
    #b = np.random.uniform(low=0., high=np.pi*2, size=shape)
    #return a*np.exp(1.j * b)

    # Build a unit-distribution of complex numbers with random phase
    a = np.random.normal(loc=0, scale=2**-0.5, size=shape)
    b = np.random.normal(loc=0, scale=2**-0.5, size=shape)
    return a + 1j * b

def med(x):
    return 0.5*(np.asarray(x)[1:] + np.asarray(x)[:-1])

def fft2d(sky, x, y, sky2=None, kbins=None):
    '''
    sky: N dimensional array, fft along last two axis\n
    x, y: grid center, and the volume is corrected\n
    sky2: the same shape as sky, will be cross-correlated with sky, if None, calculate the auto correlation of sky\n
    kbins: the k bins to calculate 1d psd
    return: dict\n
        {'psd1d':psd1d, 'psd2d':psd2d, 'k1d':k1d, 'kbins':kbins, 'psd':psd, 'k':k, 'kx':kx, 'ky':ky}
        where k1d, psd1d is for 1d power spectrum density, kx ky, psd2d is for 2d power spectrum density, k and psd is the flatten version of sqrt(kx**2+ky**2) and psd2d
    '''
    axes=(-2,-1)
    sky = np.asarray(sky)
    x = np.asarray(x)
    y = np.asarray(y)
    V = (y.max() - y.min()) * (x.max() - x.min())
    V = V * x.shape[0]/(x.shape[0]-1.) * y.shape[0]/(y.shape[0]-1.)
    sky_shp = np.asarray(sky.shape, dtype=int)[list(axes)]
    sky_size = np.prod(sky_shp)
    fft = np.fft.fft2(sky, axes=axes)/sky_size
    fft = np.fft.fftshift(fft, axes=axes)
    if sky2 is not None:
        fft2 = np.fft.fft2(sky2, axes=axes)/sky_size
        fft2 = np.fft.fftshift(fft2, axes=axes)
        psd2d = np.real(fft*fft2.conj()) * V
    else:
        psd2d = np.abs(fft)**2 * V
    kx = np.fft.fftfreq(sky_shp[0], d=x[1]-x[0])
    ky = np.fft.fftfreq(sky_shp[1], d=y[1]-y[0])
    kx = np.fft.fftshift(kx)*2*np.pi
    ky = np.fft.fftshift(ky)*2*np.pi
    k = (kx[:,None]**2 + ky[None,:]**2)**0.5
    psd = psd2d.reshape( psd2d.shape[:-2]+(-1,) )
    # print(psd.shape)
    shp = psd.shape
    psd = psd.reshape(-1, shp[-1])
    k = k.reshape(-1)
    if kbins is None:
        dk = min(x.max() - x.min(), y.max() - y.min())
        dk = np.pi/dk
        #kmax = max(x[1] - x[0], y[1] - y[0])
        #kmax = np.pi/kmax
        kmax = min(kx.max(), ky.max())
        kbins = np.arange(dk*2, kmax+dk, dk)
    #print(k.shape)
    # print(psd.shape)
    psd1d = []
    for ii in range(psd.shape[0]):
        this_psd1d, _ = np.histogram(k, bins=kbins, weights=psd[ii])
        norm, kbins = np.histogram(k, bins=kbins)
        norm = np.array(norm, dtype=np.float64)
        norm[norm==0] = np.nan
        this_psd1d = this_psd1d/norm
        psd1d += [this_psd1d]
    psd1d = np.array(psd1d).reshape(shp[:-1]+(-1,))
    psd = psd.reshape(shp)
    #print(psd.shape, psd1d.shape)
    k1d = 0.5*(kbins[1:] + kbins[:-1])
    return {'psd1d':psd1d, 'psd2d':psd2d, 'k1d':k1d, 'kbins':kbins, 'psd':psd, 'k':k, 'kx':kx, 'ky':ky}




#%%
if __name__ == '__main__':
    from nbodykit.lab import *
#   test0
#   test fft2d
    lbox = 200
    nbox = 64
    fct = 1
    nroll = 0
    num = 100
    dl = lbox/nbox
    dV = (lbox/nbox)**2
    x = np.linspace(-lbox/2., lbox/2., nbox)
    x = (x[1:] + x[:-1])/2.
    y = np.linspace(-lbox/2., lbox/2., nbox)
    y = (y[1:] + y[:-1])/2.
    sig = .1

    sig2_1 = 1.
    sig2_2 = 4.
    #psd_1_func = lambda *k: np.ones_like( np.linalg.norm(k, axis=0) )*sig2_1
    #psd_2_func = lambda *k: np.ones_like( np.linalg.norm(k, axis=0) )*sig2_2
    psd_1_func = lambda *k: ( np.linalg.norm(k, axis=0) )**2*sig2_1
    psd_2_func = lambda *k: ( np.linalg.norm(k, axis=0) )**2*sig2_2
    field1 = [generate_field2d(gaus_distrib_ri, psd_1_func, (nbox, nbox), (dl, dl)) for _ in range(num)] # num, nbox, nbox
    field2 = [generate_field2d(gaus_distrib_ri, psd_2_func, (nbox, nbox), (dl, dl)) for _ in range(num)] # num, nbox, nbox
    field = np.concatenate([field1, field2], axis=0)
    print(field.shape)
    pk_dict = fft2d(field, sky2=fct*np.roll(field, nroll, axis=-1), x=x, y=y) # you may roll sky2, to give some phase to the cross correlation, this phs is linear to k
    pk1d_1 = pk_dict['psd1d'][:num].mean(axis=0) # field1
    pk1d_2 = pk_dict['psd1d'][num:].mean(axis=0) # field1
    pk2d_1 = pk_dict['psd2d'][:num].mean(axis=0) # field2
    pk2d_2 = pk_dict['psd2d'][num:].mean(axis=0) # field2
    kx = pk_dict['kx']
    ky = pk_dict['ky']

    plt.figure()
    plt.pcolormesh(kx, ky, pk2d_1.T, shading='auto')
    plt.figure()
    plt.pcolormesh(kx, ky, pk2d_2.T, shading='auto')
    plt.show()

    k1d = pk_dict['k1d']
    psd_1 = pk_dict['psd'][:num].mean(axis=0)
    psd_2 = pk_dict['psd'][num:].mean(axis=0)
    k = pk_dict['k']
    pk_nb = 0. # pk from nbodydit, average over many
    field1 = field[:num]
    field2 = fct*np.roll(field[:num], nroll, axis=-1)
    #field2 = [None]*field1.shape[0]
    for ii in range(field1.shape[0]):
        fld1 = field1[ii]
        fld2 = field2[ii]
        #print(np.abs(fld1-fld2).max())
        mesh = ArrayMesh(fld1, BoxSize=[lbox,lbox])
        mesh2 = ArrayMesh(fld2, BoxSize=[lbox,lbox])
        r = FFTPower(mesh, second=mesh2, mode='1d', dk=0.1, kmin=0.1)
        #print(r.power['power'])
        pk_nb = pk_nb + r.power['power'].real
        k_nb = r.power['k']
    pk_nb = pk_nb/len(field[:num])
    plt.figure()
    #isort = np.argsort(k)
    #plt.plot(k[isort], psd_1[isort], label='raw')
    plt.plot(k_nb, pk_nb, label='nbodykit')
    plt.plot(k1d, pk1d_1, label='fft')
    #plt.hlines(sig2_1, *plt.xlim())
    plt.plot(np.sqrt(kx**2+ky**2), psd_1_func(kx, ky), '.', label='input')
    plt.legend()
    plt.show()
    field1 = field[num:]
    field2 = fct*np.roll(field[num:], nroll, axis=-1)
    #field2 = [None]*field1.shape[0]
    for ii in range(field1.shape[0]):
        fld1 = field1[ii]
        fld2 = field2[ii]
        #print(np.abs(fld1-fld2).max())
        mesh = ArrayMesh(fld1, BoxSize=[lbox,lbox])
        mesh2 = ArrayMesh(fld2, BoxSize=[lbox,lbox])
        r = FFTPower(mesh, second=mesh2, mode='1d', dk=0.1, kmin=0.1)
        pk_nb = pk_nb + r.power['power'].real
        #print(r.power['power'])
        k_nb = r.power['k']
    # print(len(field[num:]))
    pk_nb = pk_nb/len(field[num:])
    plt.figure()
    #isort = np.argsort(k)
    #plt.plot(k[isort], psd_2[isort], label='raw')
    plt.plot(k_nb, pk_nb, label='nbodykit')
    plt.plot(k1d, pk1d_2, label='fft')
    plt.plot(np.sqrt(kx**2+ky**2), psd_2_func(kx, ky), '.', label='input')
    plt.legend()
    plt.show()

# %%

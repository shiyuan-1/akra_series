import numpy as np
import sys
# sys.path.append('./')
# from mask import get_w_k
import scipy.fft as fftengine
import matplotlib.pyplot as plt
import healpy as hp
from astropy import units as u
from scipy.linalg import cholesky, inv,cho_factor, cho_solve

def set_zero(map_input, ll_shape):
    map_output = np.copy(map_input)
    map_output[0:ll_shape, :] = 0
    map_output[-ll_shape:, :] = 0
    map_output[:, 0:ll_shape] = 0
    map_output[:, -ll_shape:] = 0
    return map_output


def apply_low_pass_filter(fft_image, cutoff_size=10):
    shifted_fft_image = fftengine.fftshift(fft_image)
    shifted_fft_image = shifted_fft_image[cutoff_size:-cutoff_size, cutoff_size:-cutoff_size]
    filtered_fft = fftengine.ifftshift(shifted_fft_image)
    return filtered_fft 

def pad_ft_map(ft_map, cutoff_size):
    ft_map = fftengine.fftshift(ft_map)
    ft_map = np.pad(ft_map, ((cutoff_size, cutoff_size), (cutoff_size, cutoff_size)))
    # print(ft_map.shape)
    ft_map = fftengine.ifftshift(ft_map)
    return ft_map


def get_w_k(mask):
    """
    Generate w_k matrix from mask
    """
    lx, ly = mask.shape
    w_k = np.zeros((lx*ly,lx*ly), dtype=np.complex64)
    NN = lx*ly
    b_k = fftengine.fft2(mask)
    for i in np.arange(NN):
        t_k = np.zeros(NN, dtype = np.float32)
        t_k[i] = 1
        t_x = fftengine.ifft2(t_k.reshape(lx,ly))
        w_k[:,i] = (fftengine.fft2(mask*t_x)).flatten()
    return w_k


def get_w_k_filtered(mask, cutoff_size=10):
    lx, ly = mask.shape
    w_k = np.zeros(((lx-2*cutoff_size)*(ly-2*cutoff_size),(lx-2*cutoff_size)*(ly-2*cutoff_size)), dtype=np.complex64)
    NN = (lx-2*cutoff_size)*(ly-2*cutoff_size)
    for i in np.arange(NN):
        t_k = np.zeros(NN, dtype = np.float32)
        t_k[i] = 1
        t_k = t_k.reshape(lx-2*cutoff_size, ly-2*cutoff_size)
        t_k = pad_ft_map(t_k, cutoff_size)
        t_x = fftengine.ifft2(t_k)
        ft = apply_low_pass_filter((fftengine.fft2(mask*t_x)), cutoff_size)
        w_k[:,i] = ft.flatten()
    return w_k


# Class to calculate the kappa map from the gamma1 and gamma2 maps
class KappaMap:
    def __init__(self, gamma1, gamma2, mask=None, w_k=None, psf=None, cho_solve=False, cutoff_size=None):
        """
        Initialize the KappaMap object.
        Parameters
        ----------
        gamma1 : 2D numpy array
            The gamma1 component of the shear field.
        gamma2 : 2D numpy array
            The gamma2 component of the shear field.
        mask : 2D numpy array, optional
            A binary mask indicating the positions of the pixels to include
            in the analysis.
        w_k : 2D numpy array, optional
            The w_k weights for the kappa map.
        psf : 2D numpy array, optional
            The point spread function of the telescope.
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.ft_g1 = fftengine.fft2(self.gamma1)
        self.ft_g2 = fftengine.fft2(self.gamma2)
        self.shp = self.gamma1.shape
        self.mask = mask
        self.w_k = w_k
        self.psf = psf
        self.get_cos_sin_phi()
        self.cutoff_size = cutoff_size
        if cho_solve:
            self.cho_solve = True
        if self.cutoff_size is not None:
            self.ft_g1 = apply_low_pass_filter(self.ft_g1, self.cutoff_size)
            self.ft_g2 = apply_low_pass_filter(self.ft_g2, self.cutoff_size)
            self.cos2phi = apply_low_pass_filter(self.cos2phi, self.cutoff_size)
            self.sin2phi = apply_low_pass_filter(self.sin2phi, self.cutoff_size)
    
    def get_cos_sin_phi(self):
        """
        Calculate the cos(2*phi) and sin(2*phi) terms of the Fourier
        transform of the shear field.

        Returns
        -------
        cos2phi : 2D numpy array
            The cos(2*phi) term of the Fourier transform.
        sin2phi : 2D numpy array
            The sin(2*phi) term of the Fourier transform.
        """
        lx = fftengine.fftfreq(self.gamma1.shape[0])
        ly = fftengine.fftfreq(self.gamma1.shape[1])
        f1, f2 = np.meshgrid(lx, ly, indexing='ij')
        p1 = f1*f1 - f2*f2
        p2 = 2*f1*f2
        f2 = f1*f1 + f2*f2
        f2[0,0] = 1
        self.cos2phi = p1 / f2
        self.sin2phi = p2 / f2
        # l_squared = lx[:,np.newaxis]**2 + ly[np.newaxis, :]**2
        # l_squared[0,0] = 1
        # sin_2_phi = 2.0 * lx[:,np.newaxis] * ly[np.newaxis,:] / l_squared
        # cos_2_phi = (lx[:,np.newaxis]**2 - ly[np.newaxis,:]**2) / l_squared
    
    def kappa_ks(self):
        """
        Calculate the convergence map by applying the Kaiser-Squires filter
        in Fourier space.

        Returns
        -------
        conv_ks : 2D numpy array
            The convergence map.
        """
        self.ft_E = self.cos2phi * self.ft_g1 + self.sin2phi * self.ft_g2
        self.ft_E[0,0] = 0
        if self.cutoff_size is not None:
            self.ft_E = pad_ft_map(self.ft_E, self.cutoff_size)

        self.conv_ks = fftengine.ifft2(self.ft_E).real
        return self.conv_ks
    
    def kappa_inv(self, b_inv = None, return_b_inv=False, **kwargs):
        """
        Calculate the inverse convergence map using the PSF.

        Returns
        -------
        conv_inv : 2D numpy array
            The inverse convergence map.
        """
        ft_g12 = np.vstack((self.ft_g1.flatten(), self.ft_g2.flatten()))
        
        if self.w_k is None or self.psf is None:
            print("Calculating the psf: ... \n")
            self.get_psf()
            print("Calculating the inverse of the psf: ... \n")  
        if b_inv is None:
            if cho_solve:
                try:
                    b_inv = self.get_Binv_cho(self.psf, **kwargs)  
                except:
                    print("Cholesky decomposition failed. Calculating the inverse of the psf: ... \n")
                    b_inv = self.get_Binv(self.psf, **kwargs)        
            else:
                b_inv = self.get_Binv(self.psf, **kwargs)
        aa = np.dot( self.w_k.conj().T, ft_g12.flatten())
        self.ft_conv = np.dot(b_inv, aa)
        if self.cutoff_size is not None:
            self.cutoff_shape = (self.shp[0]-2*self.cutoff_size, self.shp[1]-2*self.cutoff_size)
            self.ft_conv = self.ft_conv.reshape(self.cutoff_shape) 
            self.ft_conv = pad_ft_map(self.ft_conv, self.cutoff_size)
        else:
            self.ft_conv = self.ft_conv.reshape(self.shp)
        self.conv_inv = fftengine.ifft2(self.ft_conv).real
        if return_b_inv:
            return self.conv_inv, b_inv
        else:
            return self.conv_inv

    def get_psf(self):
        """
        Calculate the point spread function of the telescope.
        
        Outputs
        -------
        w_k: 2D numpy array
            The w_k weights for the kappa map.
        psf : 2D numpy array
            The point spread function of the telescope.
        """
        if self.cutoff_size is not None:
            w_k0 = get_w_k_filtered(self.mask, self.cutoff_size)
        else:
            w_k0 = get_w_k(self.mask)
        w_k1 = w_k0 * self.cos2phi.flatten()
        w_k2 = w_k0 * self.sin2phi.flatten()
        self.w_k = np.vstack((w_k1, w_k2))
        self.psf = np.matmul(self.w_k.conj().T, self.w_k)


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
        psf = psf.astype('complex64')
        print(" .   Calculating the cholesky decomposition of the psf ... \n")
        import time
        st = time.time()
        c, low = cho_factor(psf)
        self.inv_cho = cho_solve((c, low), np.eye(psf.shape[0]))
        print("    Time taken for cholesky decomposition: %.2f s"%(time.time()-st))
        return self.inv_cho

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


    def show_attr(self):
        """
        Print all the attributes of the class
        """
        for attr in self.__dict__:
            val = getattr(self, attr)
            if isinstance(val, np.ndarray):
                print(attr, val.shape)
            else:
                print(attr, val)

    # def save(self, filename):
    #     # Save all the attributes of the class 
    #     import pickle
    #     with open(filename, 'wb') as f:
    #         pickle.dump(self, f)

    def save(self, filename):
        # save all the attributes of the class in a hdf5 file 
        import h5py
        hf = h5py.File(filename, 'w')
        for key, value in self.__dict__.items():
            hf.create_dataset(key, data=value)
        # hf.create_dataset('gamma1', data=self.gamma1)
        # hf.create_dataset('gamma2', data=self.gamma2)
        # hf.create_dataset('mask', data=self.mask)
        # hf.create_dataset('psf', data=self.psf)
        # hf.create_dataset('w_k', data=self.w_k)
        # hf.create_dataset('conv_ks', data=self.conv_ks)
        # hf.create_dataset('conv_inv', data=self.conv_inv)
        hf.close()


        # import pickle
        # with open(filename, 'rb') as f:
        #     tmp_dict = pickle.load(f)
        # for key, value in tmp_dict.__dict__.items():
        #     setattr(self, key, value)
            

    @staticmethod
    def plot_mask(mask, nside=None):
        if nside is None:
            nside = 512
        resolution =  hp.nside2resol(nside, arcmin=False) *u.rad
        lx_deg = resolution.to(u.deg).value * mask.shape[0]
        lonra = [lx_deg/2*(-1),lx_deg/2]
        latra = [lx_deg/2*(-1),lx_deg/2]

        n_pixels = mask.shape[0]
        dx = (lonra[0]-lonra[1])/n_pixels/2
        plt.figure(figsize=(5,5))
        plt.imshow(mask, origin='lower',  extent=(lonra[0],lonra[1],latra[0],latra[1]), )
        x, y = np.meshgrid(np.linspace(lonra[0]-dx/2,lonra[1]+dx/2,mask.shape[0],endpoint=True), np.linspace(latra[0]-dx/2,latra[1]+dx/2,mask.shape[1], endpoint=True))
        # print(x.shape, y.shape)
        print(x.shape)
        plt.colorbar()
        plt.plot(np.ma.array(x, mask=(mask.astype(int))), y,  'x', color='black',
                alpha=0.5, markersize=2.3)
        plt.xlabel('RA [degree]')
        plt.ylabel('Dec [degree]')
        plt.show()




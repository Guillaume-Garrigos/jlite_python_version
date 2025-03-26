############################################
# This is for deblurring inverse problem
# all this code ends with the definition of the function "flou"
# could as well be imported from invprob
############################################

import numpy as np
from scipy.fftpack import dct, idct

def dct2(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def idct2(coefficient):
    return idct(idct(coefficient.T, norm='ortho').T, norm='ortho')

def create_kernel(kernel_size, kernel_std):
    x = np.concatenate((np.arange(0, kernel_size / 2), np.arange(-kernel_size / 2, 0)))
    [Y, X] = np.meshgrid(x, x)
    kernel = np.exp((-X**2 - Y**2) / (2 * kernel_std**2))
    kernel = kernel / sum(kernel.flatten())
    return kernel

def gaussian_kernel(kernel_size, kernel_std, im_shape):
    kernel = create_kernel(kernel_size, kernel_std)
    padded_kernel = np.zeros(im_shape)
    padded_kernel[0:kernel.shape[0], 0:kernel.shape[1]] = kernel
    e = np.zeros(im_shape)
    e[0,0] = 1
    i = int(np.floor(kernel.shape[0]/2)+1)
    j = int(np.floor(kernel.shape[1]/2)+1)
    center = np.array([i, j])
    m = im_shape[0]
    n = im_shape[1]
    k = min(i-1,m-i,j-1,n-j)

    PP = padded_kernel[i-k-1:i+k, j-k-1:j+k] 
    Z1 = np.diag(np.ones((k+1)),k)
    Z2 = np.diag(np.ones((k)),k+1)
    PP = Z1@PP@Z1.T + Z1@PP@Z2.T + Z2@PP@Z1.T + Z2@PP@Z2.T

    Ps = np.zeros(im_shape)
    Ps[0:2*k+1, 0:2*k+1] = PP
    kernel_fourier = dct2(Ps)/dct2(e)
    return kernel_fourier

############################################
# Code for the TP
############################################

kernel_size = 3
kernel_std = 1
im_shape = (128,128)
kernel_fourier = gaussian_kernel(kernel_size, kernel_std, im_shape)

def flou(x): # La fonction qui permet d'évaluer A@x
    return np.real(idct2(dct2(x) * kernel_fourier))

Lipschitz_flou = np.max(np.abs(kernel_fourier)) # Ceci est la constante de Lipschitz du gradient de la fonction quadratique


















############################################
# this is for doing prox with TV norm

def prox_infty(x, s=1):
    # Prox operator of the sup/infty norm
    # We use Moreau's decomposition theorem: the dual of the sup norm is the indicator of the L1 unit ball
    return x - prox.L1_ball(x, s)

def TV(x):
    return np.sum(np.abs(invprob.signal.grad(x)))

def proj_TV_ball(a, s=1, init=None, itermax=200, tol=1e-4, stepsize=0.125):
    # stepsize in ]0,1/4[
    if init is not None:
        x = init
    else:
        x = a
    u = invprob.signal.grad(x) # dual vector
    v = u
    t = 1.0
    for k in range(itermax):
        u_temp = v - stepsize * invprob.signal.grad(a - invprob.signal.div(v)) # gradient step
        u_temp = prox_infty(u_temp, stepsize*s) # prox step
        alpha = t-1
        t = (1 + np.sqrt( 1 + 4*t**2 ))/2
        alpha = alpha / t
        v = (1+alpha)*u_temp - alpha*u
        if np.linalg.norm(u-u_temp)<tol:
            print(k)
            break
        u = u_temp
    return a - invprob.signal.div(v)




from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np

def widget_TV():
    def f(b):
        plt.figure(2)
        p = proj_TV_ball(comete, b, itermax=200, tol=0.1)
        imshow(p)
        plt.show()

    interactive_plot = interactive(f, b=(1, 2000, 10))
    output = interactive_plot.children[-1]
    output.layout.height = '350px'
    interactive_plot
    
    
    

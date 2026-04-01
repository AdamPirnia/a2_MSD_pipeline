######################################################    imports    ##################
from locale import normalize
import numpy as np
from scipy import constants
from IPython.display import display, Math
from scipy import constants as scon
from scipy.optimize import root
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import statistics as st
import pathlib
from matplotlib.ticker import FormatStrFormatter as fsf
from scipy import optimize as opt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pylab import *
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# import MDAnalysis as mda
from concurrent.futures import ProcessPoolExecutor
# from MDAnalysis.lib.distances import calc_bonds as dst
# from MDAnalysis.transformations import nojump
import random
import itertools
import warnings
# import numba
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sympy import symbols, integrate, sin, cos, exp, oo

######################################################    Functions    ##################

# beta coefficient
def beta(t):
  return scon.e/(scon.k * t) # β in eV

#########################################################################################

# distributions
def gauss(x, mu, sig, amp):
   return amp/(np.sqrt(2 * np.pi * sig**2)) * np.exp(-((x - mu)**2)/(2 * sig**2))

#########################################################################################

def gauss2(x, mu, sig, amp):
   return amp * np.exp(-((x - mu)**2)/(2 * sig**2))


#########################################################################################

# importing data function
def imp_dat (dir_file):
    dat = np.genfromtxt(dir_file)
    return dat

#########################################################################################

# generating histograms function
def man_bins (dall, hist):
    dn = len (dall)
    fist = np.zeros(hist.size)
    bin_size = np.abs (hist [1] - hist [0])
    for i in tqdm(range (dn)):
        d = dall [i] - min(dall)
        pos = math.floor (d / bin_size)
        if pos >= 0 and pos < 201 :
            fist [pos] += 1
    return fist

#########################################################################################

# generating energy landscape function
def elan (histog):
    tot = np.sum (histog)
    f = - scon.k * 300/ scon.e * np.log (np.abs(histog/tot))
    return f

#########################################################################################

# mean function
def mean (arr):
    tot = np.sum(arr)
    return tot/np.size(arr)

#########################################################################################

# manual variance function
def vara (data):
     n = len(data)
     ave = sum(data) / n
     deviations = [(x - ave) ** 2 for x in data]
     return sum(deviations) / n

#########################################################################################

def ACF_set_modi(arr, delta) :
    arr = arr - np.average(arr)
    m = int(np.ceil(0.5*len(arr)))
    Cor_x = np.zeros(m)

    Cor_x[0] = np.average(arr[::delta] * arr[::delta])
    for i in np.arange(1, m):
        Cor_x[i] = np.average(arr[:-i:delta] * arr[i::delta])
    return Cor_x / np.var(arr)

#########################################################################################

# calculation of λ
def reorg (dat, t):
    return beta(t) * np.var(dat) / 2

#########################################################################################

def reorg2 (dat, t):
    return beta(t) * vara(dat) / 2

#########################################################################################

# non-parabolicity
def a_par (lst, l1, l2):
    return (2 * lst + l2)/(l1 - l2)

#########################################################################################

# ΔF0 reaction free energy
def df0 (x1, x2, a, l1):
    xm = (x1 + x2)/2
    return xm - a * l1 / (2 * (1 + a)^2)

#########################################################################################

# X0 parameter 
def x0 (df0, l1, a):
    return df0 - l1 * a^2 / (1 + a)

#########################################################################################

# normalize
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

#########################################################################################

def my_ACF (series, step_size):
    n = len(series)
    max_lag = int(np.ceil(0.5 * n))
    print(f"len = {max_lag}")
    acf_values = np.zeros(max_lag)

    mean_value = np.mean(series)
    std_value = np.std(series)

    normalized_series = (series - mean_value) / std_value
    for lag in range(1, max_lag + 1):
        numerator = np.sum(normalized_series[:-lag:step_size] * normalized_series[lag::step_size])
        denominator = n - lag
        acf_values[lag-1] = numerator / denominator

    return acf_values

#########################################################################################

# bin edges to bin averages
def binav (numbers):
    new_list = np.array([])
    for i in range(0, len(numbers) - 1):
        average = (numbers[i] + numbers[i + 1]) / 2
        new_list = np.append(new_list, average)
    return new_list

#########################################################################################

# parabolic function
def parabola (x, a, b, c):
    return a + b*x + c*x**2

#########################################################################################

# Define the function you want to fit
def tcor1(x, A1, a):
    return A1 * np.exp(-x / a)


def tcor2(x, A1, A2, a1, a2, w):
    return A1 * np.exp(-x / a1) + A2 * np.exp(-x / a2) * sin(w*x)


def tcor3(x, A1, A2, A3, a1, a2, a3, w1, w2):
    return A1 * np.exp(-x / a1) + A2 * np.exp(-x / a2) * sin(w1*x) + (1-A3) * np.exp(-x / a3) * (cos(w2*x) + (a3/w2) * sin(w2*x))

def tcor4(x, A1, A2, A3, A4, a1, a2, a3, a4):
    return A1 * np.exp(-x / a1) + A2 * np.exp(-x / a2) + A3 * np.exp(-x / a3) + A4 * np.exp(-x / a4)

#########################################################################################

# maxwell distribution
# by number
def pmax(x, σ):
    return (4 * np.pi * x**2) / ((2 * np.pi * σ**2)**(3/2)) * np.exp(-x**2 / (2 * σ**2))
# maxwell distribution
# by array
def pm(arr, σ):
    return (4 * np.pi * arr**2) / ((2 * np.pi * σ**2)**(3/2)) * np.exp(-arr**2 / (2 * σ**2))

#########################################################################################

# trace
def tr(list):
    return (np.trace(np.array(list)))

#########################################################################################

# vector rotation
def rotation (ex, ey, ez, xR, yR, zR, DeltaAlpha):
    # ex, ey, ez are vectors (1x3 arrays)
    # exR, eyR, ezR are rotation matrices (1x3 arrays)
    # DeltaAlpha is a 3x3 matrix
    
    # Compute the rotation matrix rD
    rD = np.array([
        [np.dot(ex, xR), np.dot(ex, yR), np.dot(ex, zR)],
        [np.dot(ey, xR), np.dot(ey, yR), np.dot(ey, zR)],
        [np.dot(ez, xR), np.dot(ez, yR), np.dot(ez, zR)]
    ])
    
    # Compute the result using matrix multiplication
    result = np.dot(rD, DeltaAlpha)
    result = np.dot(result, rD.T)
    
    return result

#########################################################################################

# modify energy gap by polarizability and field
def pol_x(x, anr, bnr, crs_nanb, field, exI, eyI, ezI, DeltaAlpha, const):
    nT = len(x)
    
    rD = np.array([
        rotation(anr[k], bnr[k], crs_nanb[k], exI, eyI, ezI, DeltaAlpha)
        for k in range(nT)
    ])
    
    # Compute outer products field[k] * field[k]^T for all k
    field_outer = np.einsum('ij,ik->ijk', field, field)
    
    # Compute the dot product rD * field * rD^T and sum along axes 1 and 2
    term2 = np.einsum('ijkl,jlk->il', rD, field_outer)
    
    res = x - 1 / const / 2 * np.sum(term2, axis=1)
    
    return res

#########################################################################################

# calculate the nth moment
def moment (x, n):
    return (np.mean((x - np.mean(x))**n, axis=1))

#########################################################################################

# calculate the alpha2 parameter with moment
def alpha2m (coors, nm):
    coors = coors.reshape(shape(coors)[0], nm, 3)
    delta_rs = coors - coors[0, :, :]
    rnorms = np.linalg.norm(delta_rs, axis=2)
    moms4 = moment(rnorms, 4)
    moms2 = moment(rnorms, 2)
    return (1-np.abs((3 * moms4)/(5 * moms2**2)))

#########################################################################################

# calculate the alpha2 parameter with means
def alpha2 (coors, nm):
    coors = coors.reshape(shape(coors)[0], nm, 3)
    delta_rs = coors - coors[0, :, :]
    rnorms = np.linalg.norm(delta_rs, axis=2)
    ave4 = np.mean(rnorms**4, axis=1)
    ave2 = np.mean(rnorms**2, axis=1)

    a2 = 1-np.abs((3 * ave4)/(5 * ave2**2))
    a2 = np.nan_to_num(a2, nan=0.0)
    return (a2)

#########################################################################################

# center of mass
def cmass_single_frame(n, coors, masses, mt):
    coors = np.reshape(np.array(coors), (n, 3, 3))
    com = np.sum(coors * masses[:, np.newaxis], axis=1) / mt

    return com

# Center of mass calculation for multiple frames
def cmass(n, data, masses, mt):
    """
    Calculates center of mass for a trajectory.
    
    Parameters
    ----------
    n      : int
        number of molecules
    data   : int
        the actual data with the shape of (number of frmaes, number of water molecules, 9)
    masses : list or ndarray
        list or array of atomic masses
    mt     : float
        total moass of one water molecule

    returns
    -------
    com    : ndarray
        a 3D array of the shape of (number of frames, nomber of molecules, 3), containing 
        center of mass coordinates for each molecule at each frame of a trajectory
    """
    data = np.reshape(data, (data.shape[0], n, 3, 3))
    
    # Calculate the center of mass for each water molecule in every frame
    com = np.sum(data * masses[:, np.newaxis], axis=2) / mt

    return com

#########################################################################################

def dipoleM(n, coors, charges, com):
    n_frame = coors.shape[0]
    charges = np.array(charges)
    coors = np.array(coors.reshape(n_frame, n, 3, 3))
    com = np.array(com).reshape(n_frame, n, 1, 3)
    cencor = coors - com
    
    dm = np.sum((cencor * charges[np.newaxis, :, np.newaxis]), axis=2) / 0.2081943
    dm = dm.reshape(n_frame, n, 3)
    dmag = np.linalg.norm(dm, axis=2)

    return dm, dmag

def dipoleM_test(n, coors, charges, com):
    charges = np.array(charges)

    coors = np.array(coors.reshape( n, 3, 3))
    com = np.array(com).reshape(n, 1, 3)
    cencor = coors - com
    dm = np.sum((cencor * charges[:, np.newaxis]), axis=1) / 0.2081943
    dm = dm.reshape(n, 3)
    dmag = np.linalg.norm(dm, axis=1)
    return dm, dmag

#########################################################################################

def fit_dubl_exp(t, a1, gamma1, gamma2):
    """
    Generates a function that is going to be used to fit the autocorrelation function of velocity trajectories (or displace ment at frames with the same time difference (v = Δr/Δt))m.

    This function later can be used in the calculation of diffusion constant.

    Parameters:
    t: float
        the relaxation time of the autocorrelation function (the fitting function)
    a1: float
        the coefficient determining the weight of the slow component (long tail).
    gamma1: float
        the coefficient in the exponent for the fast component.
    gamma2: float
        the coefficient in the exponent for the slow component (long tail). 
    """
    return a1 * np.exp(-gamma1 * t) + (1 - a1) * np.exp(-gamma2 * t)

#########################################################################################

def corrF(array1, array2, delta, max_lag, t1):
    """
    Calculate the normalized autocorrelation function for an array up to a specified maximum lag.

    Parameters
    ----------
    array : numpy.ndarray
        The input data array.
    delta : int
        The sampling interval.
    max_lag : int
        The maximum lag to compute the autocorrelation for.
    t1 : float
        A scaling factor for time units.

    Returns
    -------
    sf : numpy.ndarray
        An array containing the time-scaled lag values and the corresponding normalized autocorrelation values.
    """

    mean1 = np.mean(array1)
    mean2 = np.mean(array2)
    
    # Function to calculate the correlation
    def corr(arr1, arr2, mn1, mn2, delta, m):
        norm = 1 + int(len(arr1)/delta) - m
        return np.sum((arr1[:len(arr1) - m:delta] - mn1) * (arr2[m::delta] - mn2))/norm
    
    # Variance (correlation at lag 0)
    vari = corr(array1, array2, mean1, mean2, delta, 0)
    
    # Normalized correlation
    sf = np.array([[m * t1, corr(array1, array2, mean1, mean2, delta, m) / vari] for m in range(0, max_lag + 1)])
    
    return vari, sf

#########################################################################################

def corrVV_2vec_single(array1, array2, delta, max_lag, t1, coef=3):
    """
    Calculate the normalized autocorrelation function for an array up to a specified maximum lag.

    Parameters
    ----------
    array : numpy.ndarray
        The input data array.
    delta : int
        The sampling interval.
    max_lag : int
        The maximum lag to compute the autocorrelation for.
    t1 : float
        A scaling factor for time units.

    Returns
    -------
    sf : numpy.ndarray
        An array containing the time-scaled lag values and the corresponding normalized autocorrelation values.
    """
    
    # Function to calculate the correlation
    def corr(arr1, arr2, mn1, mn2, delt, mxl):
        n = arr1.shape[0]
        array_ave1 = np.array(arr1) - mn1
        array_ave2 = np.array(arr2) - mn2
        norm = 1 + int(len(arr1)/delt) - mxl

        return np.sum(np.einsum('ij,ij->i', array_ave1[:n-mxl:delt], array_ave2[mxl::delt]))/norm
    
    
    mean1 = np.mean(array1, axis=0)
    mean2 = np.mean(array2, axis=0)

    vari = corr(array1, array2, mean1, mean2, delta, 0)
      
    sf = np.array([coef/3. * corr(array1, array2, mean1, mean2, delta, m) for m in range(0, max_lag + 1)])
    ts = np.arange(max_lag+1) * t1
    return vari, np.column_stack((ts, sf/vari))


#########################################################################################

def corrVV_2vec_multiple(array1, array2, delta, max_lag, t1, coef=3):
    """
    Calculate the normalized autocorrelation function for an array up to a specified maximum lag.

    Parameters
    ----------
    array : numpy.ndarray
        The input data array.
    delta : int
        The sampling interval.
    max_lag : int
        The maximum lag to compute the autocorrelation for.
    t1 : float
        A scaling factor for time units.

    Returns
    -------
    sf : numpy.ndarray
        An array containing the time-scaled lag values and the corresponding normalized autocorrelation values.
    """
    
    # Function to calculate the correlation
    def corr(arr1, arr2, nfrm, delt, mxl):
        norm = 1 + int(len(arr1)/delt) - mxl
        res = np.sum(np.einsum('ijk,ijk->ij', arr1[:nfrm-mxl:delt,:,:], arr2[mxl::delt,:,:]), axis=0)/norm
        return res

    shp = array1.shape
    nFrames = shp[0]
    nParticles = int(shp[1]/3)

    mean1 = np.mean(array1, axis=0)
    mean2 = np.mean(array2, axis=0)
    array_ave1 = np.reshape(np.array(array1) - mean1, (nFrames, nParticles, 3))
    array_ave2 = np.reshape(np.array(array2) - mean2, (nFrames, nParticles, 3))
    
    vari = corr(array_ave1, array_ave2, nFrames, delta, 0)
    varmean = np.mean(vari)
          
    # sfmat = np.array([coef/3. * corr(array_ave1, array_ave2, nFrames, delta, m)/vari for m in range(max_lag + 1)])
    sfmat = np.array([coef/3. * corr(array_ave1, array_ave2, nFrames, delta, m) for m in range(max_lag + 1)])

    sfnorm = np.mean(sfmat, axis=1)/varmean
    sfnorm = np.column_stack((np.arange(max_lag+1)*t1, sfnorm))

    return varmean, sfnorm



#########################################################################################

def corrF_2vec(array1, array2, delta, max_lag, t1, coef=3):
    """
    Compute the velocity autocorrelation function (ACF) averaged over all particles.

    Parameters
    ----------
    array1 : numpy.ndarray
        The input velocity array of shape (nFrames, nParticles, 3).
    delta : int
        The sampling interval.
    max_lag : int
        The maximum lag to compute the autocorrelation for.
    t1 : float
        Time scaling factor for units.
    coef : float, optional
        Scaling coefficient (default is 3).

    Returns
    -------
    sf : numpy.ndarray
        A 2D array containing time-scaled lag values and corresponding normalized ACF values.
    vari : float
        The zero-lag autocorrelation value (for normalization).
    """
    
    nFrames, nParticles, _ = array1.shape  # Get dimensions
    
    # Compute mean velocity and subtract
    mean1 = np.mean(array1, axis=0)  # Shape (nParticles, 3)
    array_ave1 = array1 - mean1  # Subtract mean (broadcasting automatically applies)

    mean2 = np.mean(array2, axis=0)  # Shape (nParticles, 3)
    array_ave2 = array2 - mean2  # Subtract mean (broadcasting automatically applies)
    
    # Compute the zero-lag variance (for normalization)
    vari = np.mean(np.einsum('ijk,ijk->ij', array_ave1, array_ave2))  # (t=0)
    
    # Initialize array for autocorrelation
    sf = np.zeros((max_lag + 1, 2))



    for m in range(max_lag+1):
        slice1 = array_ave1[:nFrames-m:delta]
        slice2 = array_ave2[m::delta]
        prod   = np.einsum('ijk,ijk->ij', slice1, slice2)
        
        # mean over *all* entries → single scalar
        acf_m  = prod.mean()  

        sf[m, 0] = m * t1
        sf[m, 1] = (coef/3.) * acf_m



    # Compute ACF for each lag using vectorized einsum
    for m in range(max_lag + 1):
        num_valid_pairs = (nFrames - m) // delta  # Number of valid pairs
        if num_valid_pairs > 0:
            acf = np.mean(np.einsum('ijk,ijk->ij', array_ave1[:nFrames-m:delta], array_ave2[m::delta]), axis=1)
            sf[m, 0] = m * t1
            sf[m, 1] = coef / 3. * acf/nParticles

    return sf, vari

#########################################################################################

def diff_tensor(array1, array2):
        
    n = array1.shape[0]

    mean1 = np.mean(array1, axis=0)
    mean2 = np.mean(array2, axis=0)

    array_ave1 = np.array(array1) - mean1
    array_ave2 = np.array(array2) - mean2

    tensor = np.mean(np.einsum('ijk,ijl->ijkl', array_ave1, array_ave2), axis=0)
    
    return tensor


#########################################################################################

def nonNormal_corrF_vec(array, delta, max_lag, t1):
    """
    Calculate the normalized autocorrelation function for an array up to a specified maximum lag.

    Parameters
    ----------
    array : numpy.ndarray
        The input data array.
    delta : int
        The sampling interval.
    max_lag : int
        The maximum lag to compute the autocorrelation for.
    t1 : float
        A scaling factor for time units.

    Returns
    -------
    sf : numpy.ndarray
        An array containing the time-scaled lag values and the corresponding normalized autocorrelation values.
    """
    
    # Function to calculate the correlation
    def corr(array, mean, delta, m):
        n = array.shape[0]
        array_ave = array - mean
    
        return np.sum(np.einsum('ij,ij->i', array_ave[:n-m:delta], array_ave[m::delta]))
    
    
    mean = np.mean(array, axis=0)
  
    # Normalized correlation
    sf = np.array([[m * t1, corr(array, mean, delta, m)] for m in range(0, max_lag + 1)])
    
    return sf

#########################################################################################

def compute_diffusion_tensor(velocities, dt):
    """
    Compute the diffusion tensor D_ij from velocity autocorrelation functions.
    
    Parameters:
    velocities: numpy array, shape (n_timesteps, n_particles, 3)
                Array containing the velocities for all particles at all timesteps.
                The last dimension is 3 for the x, y, z components of velocity.
    dt: float
        The time step between successive velocity measurements.
        
    Returns:
    D_tensor: numpy array, shape (3, 3)
              The diffusion tensor D_ij.
    """
    n_timesteps, n_particles, _ = velocities.shape
    D_tensor = np.zeros((3, 3))  # Initialize the diffusion tensor (3x3 for x, y, z)

    # Compute the velocity autocorrelation functions for each component
    for i in range(3):  # x, y, z directions
        for j in range(3):  # x, y, z directions
            # Compute the velocity autocorrelation function (VACF)
            vacf = np.zeros(n_timesteps)
            for t in range(n_timesteps):
                vacf[t] = np.mean(np.sum(velocities[:n_timesteps - t, :, i] *
                                         velocities[t:, :, j], axis=1))
            
            # Integrate the VACF over time to get D_ij (using cumulative trapezoidal integration)
            D_tensor[i, j] = np.trapz(vacf, dx=dt)
    
    D_tensor *= dt  # Multiply by time step dt to scale properly
    return D_tensor

#########################################################################################

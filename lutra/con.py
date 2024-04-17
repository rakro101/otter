"""
CrossConvergenceMappingNetwork - ccmn

Credits to the original cmm module

Please cite when using
@software{Javier_causal-ccm_a_Python_2021,
author = {Javier, Prince Joseph Erneszer},
month = {6},
title = {{causal-ccm a Python implementation of Convergent Cross Mapping}},
version = {0.3.3},
year = {2021}
}


Adjusted by rakro101

"""

from __future__ import division
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import pearsonr
from ennemi import estimate_mi
import numpy as np

np.random.seed(42)


class CoOccurrence:
    """
    We're checking causality X -> Y
    Args
        X: timeseries for variable X that could cause Y
        Y: timeseries for variable Y that could be caused by X
        tau: time lag. default = 1
        E: shadow manifold embedding dimension. default = 2
        L: time period/duration to consider (longer = more data). default = length of X
    """

    def __init__(self, X, Y, L=None, num_coefficients=14):
        """
        X: timeseries for variable X
        Y: timeseries for variable Y
        L: time period/duration to consider (longer = more data)
        We're checking for X -> Y
        """
        self.X = X
        self.Y = Y
        self.num_coefficients = num_coefficients
        if L == None:
            self.L = len(X)
        else:
            self.L = L

    def add_low_amplitude_noise(self, array, amplitude=0.1):
        """
        Adds low-amplitude noise to a numpy array.

        Parameters:
        - array (numpy.ndarray): The input numpy array.
        - amplitude (float): The amplitude of the noise. Default is 0.1.
        - min_value (float): The minimum value for clipping. Default is 0.
        - max_value (float): The maximum value for clipping. Default is 1.

        Returns:
        - numpy.ndarray: The input array with added noise.
        """
        noise = amplitude * np.random.randn(array.shape[0])
        noisy_array = array + noise
        return noisy_array

    def calculate_fourier_coefficients(self, series):
        """Calculate Fourier Coefficients from the time series"""
        fourier_transform = np.fft.fft(series)
        coefficients = fourier_transform[
            1 : self.num_coefficients
        ]  # Select the desired number of coefficients
        ret = np.concatenate([np.real(coefficients), np.imag(coefficients)], axis=0)
        # return np.abs(coefficients)
        return ret

    def occurence(self):
        """
        Args:
            None
        Returns:
            (r, p, 0): pearson correlation coeff and p- value between X and Y
        """
        r, p = pearsonr(list(self.X), list(self.Y))

        return r, p, 0

    def occurence_fft_mi(self):
        """
        Args:
            None
        Returns:
            (mi, 0, 0): mutual info coeff between  X and Y
        """
        x_ = self.X.to_numpy()
        y_ = self.Y.to_numpy()
        try:
            mi = estimate_mi(x_, y_, normalize=True).item()
        except:
            x_ = self.add_low_amplitude_noise(x_, amplitude=0.1)
            y_ = self.add_low_amplitude_noise(y_, amplitude=0.1)
            mi = estimate_mi(x_, y_, normalize=True).item()
        if mi < 0:
            mi = 0

        return mi, 0, 0

    def occurence_fft(self):
        """
        Args:
            None
        Returns:
            (r, p, 0): pearson correlation coeff and p- value between X and Y
        """
        r, p = pearsonr(
            self.calculate_fourier_coefficients(list(self.X)),
            self.calculate_fourier_coefficients(list(self.Y)),
        )

        return r, p, 0

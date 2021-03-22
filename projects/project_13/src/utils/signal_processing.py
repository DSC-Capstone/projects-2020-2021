import math
import numpy as np 
from scipy import signal
import pywt


"""
Function to apply signal processing techniques for smoothening experiment

Includes Median Filter, Low Pass Filter, Haar Wavelets

Input: data
Output: processed data
"""

def apply_signal_processing(df): 

	med = signal.medfilt(df['linear_acceleration.x'], kernel_size=9)
	
	coeffs = pywt.wavedec(df['linear_acceleration.x'], wavelet='haar')
	A = pywt.waverec(coeffs[:-3] + [None] * 3, 'haar')
	
	lp = lowpass_filt(df, 75)

	return med, A, lp


"""
Function to apply Low Pass Filter

Input: data
Output: processed data
"""

def lowpass_filt(df, freq):
	
	N = df['linear_acceleration.x'].size
	k = np.linspace(0,N-1,N)

	accel_hat = np.fft.fft(df['linear_acceleration.x'])

	k_0 = freq

	# Low Pass Filter 
	h_hat = np.ones(N)
	h_hat[k_0 : N-k_0] = 0
	g_hat_filter = accel_hat*h_hat

	# Inverse DFT
	g = np.fft.ifft(g_hat_filter)

	return g 

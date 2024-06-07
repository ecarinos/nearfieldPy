import numpy as np
import re
from copy import deepcopy
from datetime import datetime
import nearfieldpy.transform as transform

def MAD(x):
    """
    Returns the median absolute deviation of a dataset.
    """
    return np.median(np.abs(x - np.median(x)))

def get_snr(signal):
    """
    Returns the signal-to-noise ratio of a signal.
    """
    return np.mean(signal)/MAD(signal)

def str_time(date_strings,date_format='%Y-%m-%d_%H%M'):
    """
    Helper function to convert an array of date strings into 
    their values in seconds, given the reference date is 
    the first entry of the array.
    """
    reference_date = datetime.strptime(date_strings[0],date_format)
    
    date_objects = np.vectorize(lambda x: datetime.strptime(x, date_format))(date_strings)
    minutes_differences = (date_objects - reference_date)
    date_seconds = np.vectorize(lambda x: x.total_seconds())(minutes_differences)
    return date_seconds

def complex_amplitude(amplitude,phase):
    """
    Returns the complex number associated with 
    the measured phase (in degrees) and the measured intensity (in dB)

    Parameters
    ----------
    amplitude : float [dB]
        Amplitude in dB of the measured signal. Note we are measuring 
        intensities and not amplitudes, hence the factor 2.
    phase : float [degrees]
        Phase in degrees of the measured signal.  

    Returns
    -------
    complex_amp : float complex [arbitrary units]
        Complex "amplitude" of the measured signal.
    """
    rad_phase = np.radians(phase)
    complex_amp = 10**(amplitude/(2*10))*np.exp(1j*rad_phase)
    return complex_amp

def get_db(complex_amp):
    """
    Helper function to get the value in db of the complex amplitude
    of the signal. Since we measure intensities, there's a factor 2 in the computation.
    """
    return 2*10*np.log10(np.abs(complex_amp))
def get_phase(complex_amp):
    """
    Helper function to get the value in degrees of the phase of the signal.
    """
    return np.angle(complex_amp,deg=True)

def get_freqlist(data):
    """
    Helper function to retrieve the frequencies (in GHz) from a measurement data file.
    """
    freqs = np.array(data.columns.values[4:],dtype='str')
    l = []
    for t in freqs:
        l.append(re.findall(r'\d+\.\d+', t))
    freqlist = np.unique(l)
    return freqlist

def probe_beam_farfield(measurement):
    """
    Helper function to get the farfield antenna pattern of the probe. Based on A. Yaghjian, 
    “Approximate formulas for the far field and gain of open-ended rectangular waveguide,” IEEE AP, 32, 4, 378, 1984. 
    (doi: 10.1109/TAP.1984.1143332)
    Equations 1, 2 and 5
    """
    a = 1.2954  #mm Dimensions of the probe used
    b = 0.6477 #mm
    coordinates = measurement.fourier_coordinates
    frequencies = measurement.freqlist_ghz
    probe_pattern = []
    for i in range(len(frequencies)):
        k = 2*np.pi*frequencies[i] /299.792458
        beta_k = np.sqrt(1 - (np.pi /(k*a))**2)
        if measurement.fourier_coordinate_system == "uv":

            u,v = np.meshgrid(coordinates[0,:,i],coordinates[1,:,i])
            theta = np.arccos(np.sqrt(1-u**2-v**2))
            phi = np.arctan2(u,v)

            E_E = ((1+beta_k*np.cos(theta))*(np.sin(k*b*np.sin(theta)/2)))/((1+beta_k)*(k*b*np.sin(theta)/2))
            E_H = (np.pi/2)**2 * np.cos(theta) * np.cos(k*a*np.sin(theta)/2) / ((np.pi/2)**2 - (k*a*np.sin(theta)/2)**2)
            #E_co = (np.sin(phi)*np.cos(phi)*(E_E-E_H))**2 # Rotating the E field to match the co-polarisation measurement
            E_co = (E_E*np.sin(phi))**2 + (E_H*np.cos(phi))**2
            E_co = E_co/np.nanmax(np.abs(E_co))
            probe_pattern.append(E_co)   

        if measurement.fourier_coordinate_system == 'tcostsin':
            tcos,tsin = np.meshgrid(coordinates[0][:,i],coordinates[1][:,i])
            theta = np.radians(np.sqrt(tcos**2+tsin**2))
            phi = np.arctan2(tsin,tcos).T

            E_E = ((1+beta_k*np.cos(theta))*(np.sin(k*b*np.sin(theta)/2)))/((1+beta_k)*(k*b*np.sin(theta)/2))
            E_H = (np.pi/2)**2 * np.cos(theta) * np.cos(k*a*np.sin(theta)/2) / ((np.pi/2)**2 - (k*a*np.sin(theta)/2)**2)
            #E_co = (np.sin(phi)*np.cos(phi)*(E_E-E_H))**2 # Rotating the E field to match the co-polarisation measurement
            E_co = (E_E*np.sin(phi))**2 + (E_H*np.cos(phi))**2
            E_co = E_co/np.nanmax(np.abs(E_co))
            probe_pattern.append(E_co)

        if measurement.fourier_coordinate_system =='txty':
            tx,ty = np.meshgrid(coordinates[0][:,i],coordinates[1][:,i])
            theta = np.arccos(np.sqrt(1-tx**2-ty**2))
            phi = np.arctan2(tx,ty)

            E_E = ((1+beta_k*np.cos(theta))*(np.sin(k*b*np.sin(theta)/2)))/((1+beta_k)*(k*b*np.sin(theta)/2))
            E_H = (np.pi/2)**2 * np.cos(theta) * np.cos(k*a*np.sin(theta)/2) / ((np.pi/2)**2 - (k*a*np.sin(theta)/2)**2)
            #E_co = (np.sin(phi)*np.cos(phi)*(E_E-E_H))**2 # Rotating the E field to match the co-polarisation measurement
            E_co = (E_E*np.sin(phi))**2 + (E_H*np.cos(phi))**2
            E_co = E_co/np.nanmax(np.abs(E_co))
            probe_pattern.append(E_co)

    return np.array(probe_pattern)

def theta_mask(coordinates0,coordinates1,coordinate_system,theta_range):
    """
    Returns a mask to select the data within a given theta range.
    """
    if coordinate_system == 'uv':
        u,v = np.meshgrid(coordinates0,coordinates1)
        theta = np.degrees(np.arccos(np.sqrt(1-u**2-v**2)))
    if coordinate_system == 'txty':
        tx,ty = coordinates0,coordinates1
        u,v = np.meshgrid(np.sin(tx),np.sin(ty))
        theta = np.degrees(np.arccos(np.sqrt(1-u**2-v**2)))
    if coordinate_system == 'tcostsin':
        tcos,tsin = np.meshgrid(coordinates0,coordinates1)
        theta = (np.sqrt(tcos**2+tsin**2))
    mask = (theta > theta_range[0]) & (theta < theta_range[1])
    return mask



def theta_mask_measurement(measurement,theta_range):
    """
    Returns the measurement masked by the theta_range.
    """
    masked_measurement = deepcopy(measurement)
    for i in range (len(masked_measurement.freqlist)):
        mask = theta_mask(masked_measurement.fourier_coordinates[0][:,i],masked_measurement.fourier_coordinates[1][:,i],masked_measurement.fourier_coordinate_system,theta_range)
        masked_measurement.fourier_datacube[i,:,:][~mask] = np.nan
    return masked_measurement
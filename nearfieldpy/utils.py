import numpy as np
import re
from datetime import datetime

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
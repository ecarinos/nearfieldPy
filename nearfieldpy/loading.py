import csv
import pandas as pd 
import numpy as np 
from datetime import datetime
from scipy.interpolate import griddata, CubicSpline
import matplotlib.pyplot as plt
import re


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
    rad_phase=np.radians(phase)
    complex_amp = 10**(amplitude/(2*10))*np.exp(1j*rad_phase)
    return complex_amp

def map_2d_correction(dataref,time,lenx,leny,freq='140.0G'):
    """
    Returns the correction map associated with one frequency measurement.
    It is made by extrapolating the reference measurements across the 
    time-series of the measurement run in order to map out the reference
    complex amplitudes across the 2D x,y plane.
    
    Parameters
    ----------

    dataref : pandas DataFrame object 
        DataFrame containing the loaded reference data of the measurement at all frequencies.
    time : 1D array 
        Times (in second) at which the measurements were taken.
    lenx : int
        Length of the data array in the x direction.
    leny : int
        Length of the data array in the y direction.
    freq : string 
        Name of the frequency band of the measurement you want to select
    
    
    Returns
    -------
    dataref_2d : 2D numpy array
        A 2 dimensional numpy array containing the complex amplitude measured in the x,y plane.

    Examples
    --------     
    
    """
    time_ref = str_time(dataref['Time'])
    amp_ref=complex_amplitude(dataref['Amp.'+freq],dataref['Phase'+freq])
    cs = CubicSpline(time_ref,np.array(amp_ref))
    extrapolated_amp_ref = cs(time)
    dataref_2d=extrapolated_amp_ref.reshape(lenx,leny)
    return dataref_2d

def make_data_2d(data,dataref,freq='140.0'):
    """
    Returns the 2D array associated with one frequency measurement.
    
    Parameters
    ----------
    data : pandas DataFrame object
        DataFrame containing the loaded data of the measurement at all frequencies.
    dataref : pandas DataFrame object (optional)
        DataFrame containing the loaded reference data of the measurement at all frequencies.
    freq : string 
        Name of the frequency band of the measurement you want to select
    
    
    Returns
    -------
    data2d : 2D numpy array
        A 2 dimensional numpy array containing the complex amplitude measured in the x,y plane.

    Examples
    --------     

    """
    lenx = int(np.sqrt(data['x'].size))
    leny = int(np.sqrt(data['y'].size))
    amp=complex_amplitude(data['Amp.'+freq+'G'],data['Phase'+freq+'G'])
    data2d=np.array(amp)
    data2d=data2d.reshape(lenx,leny)
    if dataref is None:
        time = str_time(data['Time'])
        dataref_2d = map_2d_correction(dataref,time,lenx,leny,freq)
        data2d = data2d/dataref_2d
    return data2d

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
    l=[]
    for t in freqs:
        l.append(re.findall(r'\d+\.\d+', t))
    freqlist = np.unique(l)
    return freqlist

    
def compute_2D_fft_datacube(data,datacube,freqlist,fourier_samples=(512,512)):
    """
    From the measurement data, the sorted datacube, list of frequencies and number of samples,
    returns the 2D FFT for each frequency, as well as the associated u,v coordinates.
    """
    fourier_datacube=[]
    for i in range(datacube.shape[0]):
        fourier_datacube.append(np.fft.fftshift(np.fft.fft2(datacube[i],s=fourier_samples)))
    fourier_datacube=np.array(fourier_datacube)

    dx,dy = np.max(np.diff(data['x'])),np.max(np.diff(data['y'])) #find the x and y resolution, the steps used in the measurement
    freqlist_ghz = np.array(freqlist,dtype=float) #get the values of the frequencies in GHz

    k0=2*np.pi*freqlist_ghz/299.792458 #k0 in mm-1 # compute the base wavenumber of the transform

    #compute the u,v Fourier coordinates
    u_all = np.linspace(-np.pi/(dx*k0),np.pi/(dx*k0),fourier_samples[0]) #spatial frequencies go from -pi/k0 to pi/k0
    v_all = np.linspace(-np.pi/(dy*k0),np.pi/(dy*k0),fourier_samples[1])

    return fourier_datacube, u_all, v_all

class Measurement(object):
    """The Measurement class stores the data from one set of near field (NF) measurements, 
    initialized with the load_nf_data method and the paths of the .pkl data files.
    It contains the data DataFrame as well as the sorted NF measurements in a 3D Array, the
    list of frequencies at which the measurements were made, the 2D FFT for each measurements
    with the associated sampling and u,v coordinates.

    How to properly recover the frequency list from the measurement file is still a work in progress.
    For now, use the ***temporary*** functions get_freqlist and get_freq_ghz.

    Attributes
    ----------
    data : pandas DataFrame
        DataFrame from the data file
    dataref : pandas DataFrame
        DataFrame from the reference data file
    datacube : Array
        3D array with the shape (frequencies,nx,ny) containing
        the complex amplitude of the measurement at each sampled frequency
    freqlist : list
        List of strings containing the sampled frequencies
    fourier_datacube : Array
        3D array, with the shape (frequencies,*fourier_samples) containing
        the 2D FFT for each slice of datacube
    fourier_coordinates : list of Array
        u and v coordinates for each sampled frequency, each with
        the shapes (fourier_samples[0],frequencies) and (fourier_samples[1],frequencies)
    fourier_samples : int tuple
        Number of samples to zero-pad the FFT. Defaults to (512,512).
        Should always be a power of 2 greater than the length (in points)
        of each measurement axis.

    """
    def __init__(self,data,dataref,datacube,freqlist,fourier_datacube,fourier_coordinates,fourier_samples=(512,512)):
        self.data = data
        self.dataref = dataref
        self.datacube = datacube
        self.freqlist = freqlist
        self.fourier_datacube = fourier_datacube
        self.fourier_samples = fourier_samples
        self.fourier_coordinates = fourier_coordinates

    @property
    def freqlist_ghz(self):
        return np.array(self.freqlist,dtype=float)
    
    @staticmethod
    def load_from_file(path,pathref,fourier_samples=(512,512)):
        """
        Loads a Measurement from data files with ISAS format.       
        """

        data=pd.read_pickle(path)
        data_sorted=data.sort_values(['y','x'])
        try:
            dataref=pd.read_pickle(pathref)
        except:
            dataref=None

        freqlist = get_freqlist(data)
        
        data2dlist=[]
        for freq in freqlist:
            dat = make_data_2d(data_sorted,dataref,freq)
            data2dlist.append(dat)

        datacube = np.array(data2dlist)
        fourier_datacube, *fourier_coordinates = compute_2D_fft_datacube(data,datacube,freqlist,fourier_samples)

        return Measurement(data,dataref,datacube,freqlist,fourier_datacube,np.array(fourier_coordinates),fourier_samples)
    
    def plot_antenna_pattern(self, frequency_index, ax=None, xlabel=None, ylabel=None, title='',db=True,
             cmap='viridis', style='default', **kwargs):
        """Plots the antenna patern for a given frequency index.

        Parameters
        ----------
        frequency_index : int
            Index of the corresponding frequency of the antenna pattern
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title. Defaults to the frequency of the measurement.
        db : bool
            Whether to plot in dB scale or arbitrary units,
        cmap : str or matplotlib colormap object
            Colormap for the antenna pattern
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.pcolormesh`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots()
            if title == '':
                title = self.freqlist[frequency_index]+'Hz'
            # Plot wavelet power spectrum
            if db:
                pattern = get_db(self.fourier_datacube[frequency_index])
            else:
                pattern = self.fourier_datacube[frequency_index]
            ax.pcolormesh(self.fourier_coordinates[0][:,frequency_index], self.fourier_coordinates[1][:,frequency_index], pattern, 
                          shading='auto', cmap=cmap, **kwargs)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        return ax

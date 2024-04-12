import csv
import pandas as pd 
import numpy as np 
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import re
from nearfieldpy.utils import *

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
    amp_ref = complex_amplitude(dataref['Amp.'+freq],dataref['Phase'+freq])
    cs = CubicSpline(time_ref,np.array(amp_ref))
    extrapolated_amp_ref = cs(time)
    dataref_2d = extrapolated_amp_ref.reshape(lenx,leny)
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
    amp = complex_amplitude(data['Amp.'+freq+'G'],data['Phase'+freq+'G'])
    data2d = np.array(amp)
    data2d = data2d.reshape(lenx,leny)
    if dataref is not None:
        time = str_time(data['Time'])
        dataref_2d = map_2d_correction(dataref,time,lenx,leny,freq)
        data2d = data2d/dataref_2d
    return data2d

    
def compute_2D_fft_datacube(data,datacube,freqlist,fourier_samples=(512,512)):
    """
    From the measurement data, the sorted datacube, list of frequencies and number of samples,
    returns the 2D FFT for each frequency, as well as the associated u,v coordinates.
    """
    fourier_datacube = []
    for i in range(datacube.shape[0]):
        fourier_datacube.append(np.fft.fftshift(np.fft.fft2(datacube[i],s=fourier_samples)))
    fourier_datacube = np.array(fourier_datacube)
    fourier_datacube = fourier_datacube/np.nanmax(np.abs(fourier_datacube)) #normalize the spectra

    dx,dy = np.max(np.diff(data['x'])),np.max(np.diff(data['y'])) #find the x and y resolution, the steps used in the measurement
    freqlist_ghz = np.array(freqlist,dtype=float) #get the values of the frequencies in GHz

    k0 = 2*np.pi*freqlist_ghz/299.792458 #k0 in mm-1 # compute the base wavenumber of the transform

    #compute the u,v Fourier coordinates
    u_all = np.linspace(-np.pi/(dx*k0),np.pi/(dx*k0),fourier_samples[0]) #spatial frequencies go from -pi/k0 to pi/k0
    v_all = np.linspace(-np.pi/(dy*k0),np.pi/(dy*k0),fourier_samples[1])

    return fourier_datacube, u_all, v_all

def probe_beam_farfield_correction(freqlist,fourier_datacube,fourier_coordinates,fourier_coordinate_system):
    """
    Returns the measured farfield antenna pattern corrected for the farfield pattern of the probe.
    Based on A. Yaghjian, “Approximate formulas for the far field and gain of open-ended rectangular waveguide,” IEEE AP, 32, 4, 378, 1984. 
    (doi: 10.1109/TAP.1984.1143332)
    Equations 1, 2 and 5
    """
    a = 1.2954  #mm Dimensions of the probe used
    b = 0.6477 #mm

    coordinates = fourier_coordinates
    frequencies = np.array(freqlist,dtype=float)
    probe_pattern = []

    for i in range(len(frequencies)):
        k = 2*np.pi*frequencies[i] /299.792458
        beta_k = np.sqrt(1 - (np.pi /(k*a))**2)
        if fourier_coordinate_system == "uv":

            u,v = np.meshgrid(coordinates[0][:,i],coordinates[1][:,i])
            theta = np.arccos(np.sqrt(1-u**2-v**2)) 
            phi = np.arctan2(u,v)

            E_E = ((1+beta_k*np.cos(theta))*(np.sin(k*b*np.sin(theta)/2)))/((1+beta_k)*(k*b*np.sin(theta)/2))
            E_H = (np.pi/2)**2 * np.cos(theta) * np.cos(k*a*np.sin(theta)/2) / ((np.pi/2)**2 - (k*a*np.sin(theta)/2)**2)
            #E_co = (np.sin(phi)*np.cos(phi)*(E_E-E_H))**2 # Rotating the E field to match the co-polarisation measurement
            E_co = (E_E*np.sin(phi))**2 + (E_H*np.cos(phi))**2
            E_co = E_co/np.nanmax(np.abs(E_co)) # 

            probe_pattern.append(E_co)   
        if fourier_coordinate_system == 'tcostsin':
            tcos,tsin = np.meshgrid(coordinates[0][:,i],coordinates[1][:,i])
            theta = np.radians(np.sqrt(tcos**2+tsin**2))
            phi = np.arctan2(tsin,tcos).T

            E_E = ((1+beta_k*np.cos(theta))*(np.sin(k*b*np.sin(theta)/2)))/((1+beta_k)*(k*b*np.sin(theta)/2))
            E_H = (np.pi/2)**2 * np.cos(theta) * np.cos(k*a*np.sin(theta)/2) / ((np.pi/2)**2 - (k*a*np.sin(theta)/2)**2)
            #E_co = (np.sin(phi)*np.cos(phi)*(E_E-E_H))**2 # Rotating the E field to match the co-polarisation measurement
            E_co = (E_E*np.sin(phi))**2 + (E_H*np.cos(phi))**2
            E_co = E_co/np.nanmax(np.abs(E_co))
            probe_pattern.append(E_co)

        if fourier_coordinate_system =='txty':
            tx,ty = np.meshgrid(coordinates[0][:,i],coordinates[1][:,i])
            u = np.arcsin(tx)
            v = np.arcsin(ty)
            theta = np.sqrt(1-u**2+v**2)
            phi = np.arctan2(u,v)

            E_E = ((1+beta_k*np.cos(theta))*(np.sin(k*b*np.sin(theta)/2)))/((1+beta_k)*(k*b*np.sin(theta)/2))
            E_H = (np.pi/2)**2 * np.cos(theta) * np.cos(k*a*np.sin(theta)/2) / ((np.pi/2)**2 - (k*a*np.sin(theta)/2)**2)
            #E_co = (np.sin(phi)*np.cos(phi)*(E_E-E_H))**2 # Rotating the E field to match the co-polarisation measurement
            E_co = (E_E*np.sin(phi))**2 + (E_H*np.cos(phi))**2
            E_co = E_co/np.nanmax(np.abs(E_co))
            probe_pattern.append(E_co)
    fourier_datacube = fourier_datacube/np.array(probe_pattern)
    return fourier_datacube

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
        Coordinates for each sampled frequency, each with
        the shapes (fourier_samples[0],frequencies) and (fourier_samples[1],frequencies)
    fourier_coordinate_system : str
        Coordinate system used for the 2D FFT. Defaults to 'uv'.
    fourier_samples : int tuple
        Number of samples to zero-pad the FFT. Defaults to (512,512).
        Should always be a power of 2 greater than the length (in points)
        of each measurement axis.

    """
    def __init__(self,data,dataref,datacube,freqlist,fourier_datacube,fourier_coordinates,fourier_coordinate_system,fourier_samples=(512,512)):
        self.data = data
        self.dataref = dataref
        self.datacube = datacube
        self.freqlist = freqlist
        self.fourier_datacube = fourier_datacube
        self.fourier_samples = fourier_samples
        self.fourier_coordinates = fourier_coordinates
        self.fourier_coordinate_system = fourier_coordinate_system

    @property
    def freqlist_ghz(self):
        return np.array(self.freqlist,dtype=float)
    
    @staticmethod
    def load_from_file(path,pathref,fourier_samples=(512,512)):
        """
        Loads a Measurement from data files with ISAS format.       
        """

        data = pd.read_pickle(path)
        data_sorted = data.sort_values(['y','x'])
        try:
            dataref = pd.read_pickle(pathref)
        except:
            dataref = None

        freqlist = get_freqlist(data)
        
        data2dlist = []
        for freq in freqlist:
            dat = make_data_2d(data_sorted,dataref,freq)
            data2dlist.append(dat)

        datacube = np.array(data2dlist)
        fourier_datacube, *fourier_coordinates = compute_2D_fft_datacube(data,datacube,freqlist,fourier_samples)
        fourier_coordinate_system = 'uv'
        fourier_datacube = probe_beam_farfield_correction(freqlist,fourier_datacube,fourier_coordinates,fourier_coordinate_system) #Apply probe correction

        return Measurement(data,dataref,datacube,freqlist,fourier_datacube,np.array(fourier_coordinates),fourier_coordinate_system,fourier_samples)
    
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
            Whether to plot in dB scale or arbitrary units.
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
            if db:
                pattern = get_db(self.fourier_datacube[frequency_index])
                cbar_label = 'Amplitude (dB)'
            else:
                pattern = np.abs(self.fourier_datacube[frequency_index])
                cbar_label = 'Amplitude (arbitrary units)'
            im = ax.pcolormesh(self.fourier_coordinates[0][:,frequency_index], self.fourier_coordinates[1][:,frequency_index], pattern, 
                          shading='auto', cmap=cmap, **kwargs)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)

            ax.set_aspect('equal')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        return ax
    
    def plot_nearfield(self, frequency_index, ax=None, title='',db=True,
             cmap='viridis', style='default', **kwargs):
        """Plots the nearfield amplitude and phase measurements for a given frequency index, in the xy plane.
        
        Parameters
        ----------
        frequency_index : int
            Index of the corresponding frequency of the antenna pattern
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        title : str
            Plot set_title. Defaults to the frequency of the measurement.
        db : bool
            Whether to plot in dB scale or arbitrary units.
        cmap : str or matplotlib colormap object
            Colormap for the antenna pattern.
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
                fig, ax = plt.subplots(2)
            if title == '':
                title = self.freqlist[frequency_index]+'GHz'
            if db:
                amp = get_db(self.datacube[frequency_index])
            else:
                amp = np.abs(self.datacube[frequency_index])
            phase = np.angle(self.datacube[frequency_index],deg=True)
            im1 = ax[0].pcolormesh(amp, 
                          shading='auto', cmap=cmap, **kwargs)
            im2 = ax[1].pcolormesh(phase,
                             shading='auto', cmap=cmap, **kwargs)

            cbar1 = plt.colorbar(im1, ax=ax[0])
            cbar2 = plt.colorbar(im2, ax=ax[1])
            cbar1.set_label('Amplitude (dB)')
            cbar2.set_label('Phase (degrees)')

            ax[0].set_aspect('equal')
            ax[1].set_aspect('equal')

            ax[0].set_title(title)
        return ax
    
    def plot_measurement(self,frequency_index,ax=None,xlabel=None,ylabel=None,title='',db=True,
                         cmap='viridis', style='default', **kwargs):
        """Plots the antenna pattern and nearfield measurements for a given frequency index,
        both in the xy plane and in the Fourier space.
        """
        if ax is None:
            fig, ax = plt.subplots(3)
        if title == '':
            title = self.freqlist[frequency_index]+'GHz'
        ax[:2] = self.plot_nearfield(frequency_index,ax=ax[:2],title=title,db=db,cmap=cmap,style=style,**kwargs)
        ax[2] = self.plot_antenna_pattern(frequency_index,ax=ax[2],xlabel=xlabel,ylabel=ylabel,title=title,db=db,cmap=cmap,style=style,**kwargs)

        return ax

class Holographic(object):
    """The Holographic class stores the data from one set of holographic measurements,
    initialized with the load_holo_data method and the paths of the .pkl data files.
    It contains Measurement objects for the signal, reference and hologram measurements.    
    """
    def __init__(self,signal,reference,hologram):
        self.signal = signal
        self.reference = reference
        self.hologram = hologram
    
    def load_from_file(path_signal,path_reference,path_hologram,path_signal_ref,path_reference_ref,path_hologram_ref,fourier_samples=(512,512)):
        """
        Loads a Holographic object from data files with ISAS format.       
        """
        signal = Measurement.load_from_file(path_signal,path_signal_ref,fourier_samples)
        reference = Measurement.load_from_file(path_reference,path_reference_ref,fourier_samples)
        hologram = Measurement.load_from_file(path_hologram,path_hologram_ref,fourier_samples)
        return Holographic(signal,reference,hologram)
    
    def plot_hologram(self, frequency_index, ax=None, xlabel=None, ylabel=None, title='',db=True,
                      cmap='viridis', style='default', **kwargs):
        """Plots the 3 holographic measurements for a given frequency index,
        both in the xy plane and in the Fourier space.

        Parameters
        ----------
        frequency_index : int
            Index of the corresponding frequency of the hologram
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
        if ax is None:
            fig, ax = plt.subplots(3,3)
        if title == '':
            title = self.signal.freqlist[frequency_index]+'GHz'
        ax[:,0] = self.signal.plot_measurement(frequency_index,ax=ax[:,0],xlabel=xlabel,ylabel=ylabel,title='Signal',db=db,cmap=cmap,style=style,**kwargs)
        ax[:,1] = self.reference.plot_measurement(frequency_index,ax=ax[:,1],xlabel=xlabel,ylabel=ylabel,title='Reference',db=db,cmap=cmap,style=style,**kwargs)
        ax[:,2] = self.hologram.plot_measurement(frequency_index,ax=ax[:,2],xlabel=xlabel,ylabel=ylabel,title='Hologram',db=db,cmap=cmap,style=style,**kwargs)
        
        return ax
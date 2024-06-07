import ssqueezepy as ssq
import numpy as np 
import scipy
import pandas as pd 
from scipy.interpolate import griddata
import nearfieldpy.loading





def interpolate_two_fourier_arrays(fourier,u1,v1,u2,v2):
    """
    Interpolates a 2D fourier array to the coordinates of another fourier array.
    Parameters
    ----------
    fourier : 2D array
        Fourier transform of the near field measurement, in uv coordinates.
    u1 : 1D array
        u coordinates of the first measurement.
    v1 : 1D array
        v coordinates of the first measurement.
    u2 : 1D array
        u coordinates of the second measurement.
    v2 : 1D array   
        v coordinates of the second measurement.
    Returns
    -------
    fourier_interpolated : 2D array
        Interpolated fourier transform of the second measurement.
    """
    # Interpolate the second measurement to the coordinates of the first measurement
    fourier_interpolated = griddata((u2,v2),fourier.flatten(),(u1,v1),method='linear')
    return fourier_interpolated

def interpolate_two_measurements(measurement1,measurement2):
    """
    Interpolates the second measurement to the coordinates of the first measurement
    in the Fourier space.
    """
    # Interpolate the second measurement to the coordinates of the first measurement
    fourier2 = measurement2.fourier_datacube
    u1,v1 = measurement1.fourier_coordinates
    u2,v2 = measurement2.fourier_coordinates
    nfreq = len(measurement1.freqlist)
    for i in range(nfreq):
        fourier2[i,:,:] = interpolate_two_fourier_arrays(fourier2[i,:,:],
                                                                        u1[:,i],v1[:,i],u2[:,i],v2[:,i])
    measurement2.fourier_datacube = fourier2
    measurement2.fourier_coordinates = u1,v1
    return measurement1,measurement2

def difference_pattern(fourier1,fourier2,u1,v1,u2,v2):
    """
    Returns the difference pattern between two fourier spectra in the uv plane.
    """
    # Interpolate the second measurement to the coordinates of the first measurement
    fourier2_interpolated = interpolate_two_fourier_arrays(fourier2,u1,v1,u2,v2)
    # Calculate the difference pattern
    difference_pattern = fourier1-fourier2_interpolated
    return difference_pattern

def difference_pattern_measurement(measurement1,measurement2):
    """
    Returns the difference pattern between two measurements in uv coordinates.
    """
    # Interpolate the second measurement to the coordinates of the first measurement then differentiate
    fourier1 = measurement1.fourier_datacube
    fourier2 = measurement2.fourier_datacube
    fourier_diff = np.zeros(fourier1.shape)
    u1,v1 = measurement1.fourier_coordinates
    u2,v2 = measurement2.fourier_coordinates
    coord1 = measurement1.fourier_coordinate_system
    coord2 = measurement2.fourier_coordinate_system
    nfreq = len(measurement1.freqlist)
    for i in range(nfreq):
        if coord1 != coord2:
            fourier_diff[i,:,:] = difference_pattern(fourier1[i,:,:],fourier2[i,:,:],
                                                                           u1[:,i],v1[:,i],u2[:,i],v2[:,i])
        else: 
            fourier_diff[i,:,:] = fourier1[i,:,:]-fourier2[i,:,:]
    return fourier_diff
    
def transform_uv_coordinates(fourier_uv,u,v,coordinate_system='txty'):
    """
    For one 2D fourier array in uv coordinates, transform it to another coordinate system,
    mostly for plotting purposes.

    Parameters
    ----------
    fourier_uv : 2D array
        Fourier transform of the antenna response in uv coordinates.
    u : 1D array
        u coordinates.
    v : 1D array
        v coordinates.
    coordinates : str, optional
        Coordinate system to transform to. The default is 'txty'
        which is theta_x = arcsin(u),theta_y=np.arcsin(v).
        Other options are 'tcostsin' which is tcos = theta*cos(phi),
        tsin = theta*sin(phi), and 'uv' which is the original uv coordinates.
    
    Returns
    -------
    
    """
    
    uu, vv = np.meshgrid(u,v)
    circle_mask = uu**2+vv**2 <= 1 #only consider the points where there's propagation
    print(u.shape,v.shape,uu.shape,vv.shape,circle_mask.shape,fourier_uv.shape)
    fourier_uv[~circle_mask] = np.nan
    

    if coordinate_system == 'txty': #theta_x = arcsin(u),theta_y=np.arcsin(v)
        tx = np.arcsin(uu)
        ty = np.arcsin(vv)
        return tx, ty, fourier_uv


        
    
    if coordinate_system == 'tcostsin':

        phi=np.arctan2(vv,uu)
        w = np.sqrt(1-uu**2-vv**2)
        theta=np.degrees(np.arccos(w))
        tcos = theta*np.cos(phi)
        tsin = theta*np.sin(phi)

        return tcos, tsin, fourier_uv
    
    if coordinate_system =='uv':
        return uu,vv,fourier_uv


def interpolate_transformed_antenna_response(fourier_uv,u,v,coordinate_system='txty'):
    """
    For one 2D fourier array in uv coordinates, transform and interpolate it to another coordinate system,
    mostly for plotting purposes. Too slow to really use.

    Parameters
    ----------
    fourier_uv : 2D array
        Fourier transform of the antenna response in uv coordinates.
    u : 1D array
        u coordinates.
    v : 1D array
        v coordinates.
    coordinates : str, optional
        Coordinate system to transform to. The default is 'txty'
        which is theta_x = arcsin(u),theta_y=np.arcsin(v).
        Other options are 'tcostsin' which is tcos = theta*cos(phi),
        tsin = theta*sin(phi), and 'uv' which is the original uv coordinates.
    
    Returns
    -------
    
    """
    nx, ny = fourier_uv.shape
    uu, vv = np.meshgrid(u,v)
    circle_mask = uu**2+vv**2 <= 1 #only consider the points where there's propagation

    
    nans = ~np.isnan(fourier_uv)
    circle_mask = circle_mask*nans
    uu_flat = uu[circle_mask]
    vv_flat = vv[circle_mask]
    fourier_flat = fourier_uv[circle_mask]

    if coordinate_system == 'txty': #theta_x = arcsin(u),theta_y=np.arcsin(v)

        # Define a grid for interpolation
        tx,ty = np.linspace(-90, 90, nx), np.linspace(-90, 90, ny)
        tx_grid, ty_grid = np.meshgrid(tx,ty)

        # Perform interpolation
        fourier_interpolated = griddata((np.degrees(np.arcsin(uu_flat)), 
                                         np.degrees(np.arcsin(vv_flat))), 
                                         np.abs(fourier_flat), 
                                         (tx_grid, ty_grid), method='linear')
        
        return (tx), (ty), fourier_interpolated/np.nanmax(np.abs(fourier_interpolated)) # Don't forget to keep the spectrum normalized
    
    if coordinate_system == 'tcostsin':

        phi=np.arctan2(vv_flat,uu_flat)
        w = np.sqrt(1-uu_flat**2-vv_flat**2)
        theta=np.degrees(np.arccos(w))

        # Define a grid for interpolation (tcos,tsin should go from -90 to 90)
        tcos, tsin = np.linspace(-np.nanmax(theta*np.cos(phi)), np.nanmax(theta*np.cos(phi)), nx),np.linspace(-np.nanmax(theta*np.sin(phi)), np.nanmax(theta*np.sin(phi)), ny)
        tcos_grid, tsin_grid = np.meshgrid(tcos,tsin)


        # Perform interpolation
        fourier_interpolated = griddata((theta*np.cos(phi), 
                                         theta*np.sin(phi)), 
                                         np.abs(fourier_flat), 
                                         (tcos_grid, tsin_grid), method='linear')
        
        return (tcos), (tsin), fourier_interpolated/np.nanmax(np.abs(fourier_interpolated))
    if coordinate_system =='uv':
        u,v = np.linspace(-1, 1, nx),np.linspace(-1, 1, ny)
    
        u_grid, v_grid = np.meshgrid(u,v)
        fourier_interpolated = griddata((uu_flat,
                                         vv_flat), 
                                        np.abs(fourier_flat), 
                                        (u_grid,v_grid),method='linear')
        
        return u,v,fourier_interpolated/np.nanmax(np.abs(fourier_interpolated))

def interpolate_measurement(measurement,coordinate_system='txty'):
    """
    Interpolates the whole fourier datacube to a new coordinate system.
    Changes the fourier_datacube, fourier_coordinates, and fourier_coordinate_system attributes of the measurement object.
    Not recommended for large datacubes, since the computations can take a while.
    """
    fourier_datacube = measurement.fourier_datacube
    u,v = measurement.fourier_coordinates
    nfreq = len(measurement.freqlist)
    for i in range(nfreq):
        u[:,i], v[:,i], fourier_datacube[i,:,:] = interpolate_transformed_antenna_response(fourier_datacube[i,:,:],
                                                                           u[:,i],v[:,i],coordinate_system)
    measurement.fourier_datacube = fourier_datacube
    measurement.fourier_coordinates = u,v
    measurement.fourier_coordinate_system = coordinate_system

    return measurement

def transform_measurement_coordinates(measurement,coordinate_system='tcostsin'):
    """
    Transforms the coordinates of the measurement to a new coordinate system.
    Changes the fourier_coordinates and fourier_coordinate_system attributes of the measurement object.
    """
    fourier_datacube = measurement.fourier_datacube
    u,v = measurement.fourier_coordinates
    nfreq = len(measurement.freqlist)
    uu_all = []
    vv_all = []

    for i in range(nfreq):
        uu, vv, fourier_datacube[i,:,:] = transform_uv_coordinates(fourier_datacube[i,:,:],
                                                                           u[:,i],v[:,i],coordinate_system)
        uu_all.append(uu)
        vv_all.append(vv)
    
    measurement.fourier_datacube = fourier_datacube
    measurement.fourier_coordinates = np.array(uu_all),np.array(vv_all)
    measurement.fourier_coordinate_system = coordinate_system

    return measurement
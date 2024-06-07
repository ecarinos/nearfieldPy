import ssqueezepy as ssq
import numpy as np 
import scipy
import pandas as pd 
from nearfieldpy.transform import difference_pattern_measurement, interpolate_two_measurements
from nearfieldpy.utils import *


def residuals_measurement(measurement1,measurement2,smooth_factor=0.1,theta_range=None):
    """
    Returns the residuals between two measurements.
    """
    # Interpolate the second measurement to the coordinates of the first measurement
    #measurement1,measurement2 = interpolate_two_measurements(measurement1,measurement2)
    if theta_range is not None:
        measurement1 = theta_mask_measurement(measurement1,theta_range)
        measurement2 = theta_mask_measurement(measurement2,theta_range)
    diff = np.abs(difference_pattern_measurement(measurement1,measurement2))
    mean = (measurement1.fourier_datacube + measurement2.fourier_datacube) / 2
    residuals = get_db(diff+mean)-get_db(mean)
    return mean, residuals





import numpy as np 
import math

def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

def cos_sum(a, b):
    """To work with tabulate."""
    return(math.cos(a+b))

def create_time_serie(size, time, t):
    """Generate a time serie of length size and dynamic with respect to time."""
    # Generating time-series
    support = np.arange(0, size)
    serie = np.cos(support + float(time))
    return(t, serie)

def gramian_angular_field(serie):
    """Compute the Gramian Angular Field of an image"""
    # Min-Max scaling
    min_ = np.amin(serie)
    max_ = np.amax(serie)
    scaled_serie = (2*serie - max_ - min_)/(max_ - min_)

    # Floating point inaccuracy!
    scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
    scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

    # Polar encoding
    phi = np.arccos(scaled_serie)
    # Note! The computation of r is not necessary
    r = np.linspace(0, 1, len(scaled_serie))

    # GAF Computation (every term of the matrix)
    gaf = tabulate(phi, phi, cos_sum)

    return(gaf, phi, r, scaled_serie)
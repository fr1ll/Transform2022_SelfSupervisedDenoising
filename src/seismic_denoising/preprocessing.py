"""Preprocessing functions for blind-trace denoising."""

import numpy as np


def multi_active_pixels(patch, 
                        active_number,
                        noise_level):
    """ Function to identify multiple active pixels and replace with values from a random distribution
    
    Parameters
    ----------
    patch : numpy 2D array
        Noisy patch of data to be processed
    active_number : int
        Number of active pixels to be selected within the patch
    noise_level : float
        Random values from a uniform distribution over
        [-noise_level, noise_level] will be used to corrupt the traces belonging to the active pixels 
        to generate the corrupted data
        
    Returns
    -------
        cp_ptch : numpy 2D array
            Processed patch 
        mask : numpy 2D array
            Mask showing location of corrupted traces within the patch 
    """
        
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # STEP ONE: SELECT ACTIVE PIXEL LOCATIONS
    corr=[]                     
    for i in range( active_number*2):
        corr.append(np.random.randint(0,patch.shape[1],1))
    corr=np.array(corr).reshape([active_number,2])    
    
    
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # STEP TWO: REPLACE ACTIVE PIXEL's TRACE VALUES 
    cp_ptch=patch.copy()    
    cp_ptch[:,tuple( corr.T)[1]] = np.random.rand(patch.shape[0],corr.shape[0])*noise_level*2 - noise_level
          
    
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    # STEP THREE: Make mask for calculating loss 
    mask = np.ones_like(patch)    
    mask[:,tuple(corr.T)[1]] = 0
    
       
    return cp_ptch, mask


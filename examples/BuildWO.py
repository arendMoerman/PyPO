import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as pt

#import src.Python.System as System
from src.POPPy.System import System

"""
In this script, we build the warm optics (WO) block as described for DESHIMA.
The script can be loaded into any other script and returns a system object with the WO as two separate elements.
"""

def MakeWO():
    # Global constants
    _A_ELLIPSE  = 3689.3421 / 2             # Semi-major axis in mm
    _C_ELLIPSE  = 1836.4965 / 2             # Focii distance in mm
    _B_ELLIPSE  = 3199.769638 / 2         # Semi-minor axis in mm
    
    _A_HYPERBO  = 1226.5776 / 2             # Vertex distance in mm
    _C_HYPERBO  = 2535.878 / 2               # Focii distance in mm
    _B_HYPERBO  = 2219.500985 / 2            # Semi-minor axis in mm
    
    _X_LIM_ELL  = np.array([1435, 1545]) # In frame where ellipse vertex is at origin
    _Y_LIM_ELL  = np.array([-200, 200])
    
    _X_LIM_HYP  = np.array([160, 480])
    _Y_LIM_HYP  = np.array([-160, 160])
    
    # First, define focii of hyperbola and ellipsoid
    
    h_f0 = np.zeros(3) # Cold focus
    h_f1 = np.array([2298.285978374926, 0.0, 1071.7083523464814])
    
    e_f0 = np.array([486.3974883317985, 0.0, 1371.340617233771]) # Warm focus
    e_f1 = h_f1
    
    e_center = (e_f1 + e_f0) / 2 
    diff = (e_f1 - e_f0) / np.sqrt(np.dot(e_f1 - e_f0, e_f1 - e_f0))
    theta = np.degrees(np.arccos(np.dot(np.array([1,0,0]), diff)))
    print(theta)
    
    # Initialize system
    s = System()
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    h_coeff = [_B_HYPERBO, _B_HYPERBO, _A_HYPERBO]
    h_gridsize = [601, 401]
    
    # Define ellipse coefficients. Note that _X_LIM_ELL is in frame where ellipse vertex is at origin
    # In our definition, ellipse is centered at origin. Semi-major axis is along x-axis
    #_X_LIM_ELL -= _A_ELLIPSE
    
    e_coeff = [_B_ELLIPSE, _B_ELLIPSE, _A_ELLIPSE]
    e_gridsize = [401, 401]
    
    #s.addParabola(name="e1", coef=e_coeff, lims_x=_X_LIM_ELL, lims_y=_Y_LIM_ELL, gridsize=e_gridsize, pmode='man', gmode='xy')
    s.addEllipse(name="e1", coef=e_coeff, lims_x=_X_LIM_ELL, lims_y=_Y_LIM_ELL, gridsize=e_gridsize, pmode='man', gmode='xy', ori='vertical')
    s.system['e1'].rotateGrid(np.array([0, 90, 0]))
    s.system['e1'].translateGrid(e_center)
    s.system['e1'].cRot = e_center
    s.system['e1'].rotateGrid(np.array([0, theta, 0]))
    
    s.addHyperbola(name="h1", coef=h_coeff, lims_x=_X_LIM_HYP, lims_y=_Y_LIM_HYP, gridsize=h_gridsize, pmode='man', gmode='xy', sec='lower')
    s.system['h1'].translateGrid(np.array([0, 0, _C_HYPERBO]))
    s.system['h1'].rotateGrid(np.array([0, 65, 0]))
    
    #s.plotSystem(focus_1=True, focus_2=True)
    
    return s
    
if __name__ == "__main__":
    print("In this script, we build the warm optics (WO) block as described for DESHIMA. The script can be loaded into any other script and returns a system object with the WO as two separate elements.")
    
    
    
    
    
    
    

#####################################################################
# File to apply the Maréchal approximation for the NGS_* mode with
# * = VIS 40x40 or IR 9x9 SH-WFS
#
# The reference file is ./config/GPAO_NGS_*_0.ini
#
# Created: 2023.09 (yyyy.mm)
# Author: Anthony Berdeu (LESIA - Observatoire de Paris)
# License: GPL3 (see LICENSE)
#
# This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 101004719.
#
#####################################################################





#######################
# IMPORTING LIBRARIES #
#######################

# Python
import importlib

import aspro as aspro

# aspro
importlib.reload(aspro)


##############
# PARAMETERS #
##############

##### User parameters #####
# Mode to simulate
flag_mode = 'NGS_VIS' # NGS_VIS/NGS_IR

# Turbulence
config_turbulence = {}
config_turbulence['r_0'] = 0.100 # Fried's parameter @500 nm (m)
config_turbulence['v_0'] = [25.0] # Wind speed of the turbulence layers (m.s-1) (could be a list) (for isoplanetism)
config_turbulence['h_0'] = [10000.0] # Altitude of the turbulent layers (m) (could be a list) (for isoplanetism)
config_turbulence['Cn2'] = [1] # Cn2 weight (could be a list) (for collapsing h_0 and v_0 to an equivalent individual layer)

# NGS
config_NGS = {}
config_NGS['magnitude'] = 8 # Magnitude of the NGS
config_NGS['zenith'] = 0.0  # For the airmass (deg), 0.0 for zenith

# Target
config_target = {}
config_target['wavelength'] = 2.2e-06 # Wavelength of the target (science or fringe tracker) channel (m)
config_target['theta'] = 0 # Angle between the target (science or fringe tracker) and the NGS (arcsecond)

# AO system
config_ao = {}
config_ao['TelescopeDiameter'] = 8.0 # Telescope diameter (m)
config_ao['transmission'] = 0.3 # Global transmission of the WFS channel (to compute the number of photons)
config_ao['sig_RON'] = 0.2 # Readout noise of the camera
config_ao['ExcessNoiseFactor'] = 2 # Excess noise factor
config_ao['g_loop'] = 0.5 # Loop gain
##### User parameters #####


##### Mode-dependent variables #####
# config_NGS['wavelength']  -> Wavelength of the HO NGS channel (m)
# config_NGS['mag2flux']    -> Convertion magnitude to flux / Magnitude 0-point (ph/s/m2 for mag=0)
# config_ao['n_mode']       -> Number of corrected modes corrected models (to compute the equivalent DM number of actuators)
# config_ao['f_loop']       -> Loop frequency (Hz)
# config_ao['SH_diam']      -> SH-WFS diameter (number of lenslets)
# config_ao['pixScale']     -> pixel scale (milliarcsecond / pixel)
# config_ao['n_pix']        -> number of pixels per lenslet
if flag_mode[4:7] == 'VIS':
    config_NGS['wavelength'] = 750e-9
    config_NGS['mag2flux'] = 2.63e10
    config_ao['n_mode'] = 800
    config_ao['f_loop'] = 1000.0
    config_ao['SH_diam'] = 40
    config_ao['pixScale'] = 420
    config_ao['n_pix'] = 6
elif flag_mode[4:7] == 'IR':
    config_NGS['wavelength'] = 2.2e-6
    config_NGS['mag2flux'] = 1.66e9
    config_ao['n_mode'] = 44
    config_ao['f_loop'] = 500.0
    config_ao['SH_diam'] = 9
    config_ao['pixScale'] = 510
    config_ao['n_pix'] = 8
else:
    raise ValueError(flag_mode + \
        ' -> Unknown mode (*_VIS / *_IR)')
##### Mode-dependent variables #####



##### Calibration of the Maréchal approximation #####
# Values obtained with TIPTOP
config_Strehl = {}
if flag_mode[4:7] == 'VIS':
    config_Strehl['geom'] = [0.26705087, 0.98968173]
    config_Strehl['lag'] = [8.48317135, 2.15500641]
    config_Strehl['ph'] = [11.97305155]
    config_Strehl['ron'] = [0.51996901]
    config_Strehl['iso'] = [4.33657467, 1.86425362]
elif flag_mode[4:7] == 'IR':
    config_Strehl['geom'] = [0.24405723, 0.86477159]
    config_Strehl['lag'] = [2.08400088, 2.09918214]
    config_Strehl['ph'] = [15.17856885]
    config_Strehl['ron'] = [1.65331745]
    config_Strehl['iso'] = [1.74957095, 1.97261581]
else:
    raise ValueError(flag_mode + \
        ' -> Unknown mode (*_VIS / *_IR)')
##### Calibration of the Maréchal approximation #####




##################################
# TESTING MARECHAL APPROXIMATION #
##################################

# Running Maréchal approximation
SR_Maréchal = aspro.compute_Marechal_NGS(config_NGS, config_target, config_ao, config_turbulence, config_Strehl)
print('Strehl ratio: ', SR_Maréchal)


#####################################################################
# 'aspro' module defining different functions and methodes
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

# Python
import numpy as np

# IMPORTING LIBRARIES #
#######################





####################
# STRHEL FUNCTIONS #

# Geometric error = fitting + aliasing
# INPUTS
#   - coeff: damping factor
#   - airmass: secant of the zenith angle (1/cos(zenith_angle))
#   - DM_pitch: pitch of the DM on the pupil (m)
#   - r_0: Fried's parameter @500nm (m)
#   - wavelength: wavelength at which the Strehl must be computed (m)
# OUTPUT: the Strehl ratio
def Strehl_geom(coeff, airmass, DM_pitch, r_0, wavelength):
    if len(coeff) == 1:
        return np.exp(-coeff[0] * (DM_pitch / (airmass**(-3.0 / 5.0) * r02rlambda(r_0, wavelength)))**(5.0 / 3.0))
    elif len(coeff) == 2:
        return coeff[1] * np.exp(-coeff[0] * (DM_pitch**(5.0 / 3.0)) / (airmass**(-3.0 / 5.0) * r02rlambda(r_0, wavelength))**(5.0 / 3.0))
    else:
        raise ValueError('Invalid number of coefficients!')

# Servo-lag error
# INPUTS
#   - coeff: damping factor and the power value
#   - airmass: secant of the zenith angle (1/cos(zenith_angle))
#   - v_0: velocity of the turbulent layer (m.s-1)
#   - r_0: Fried's parameter @500nm (m)
#   - wavelength: wavelength at which the Strehl must be computed (m)
#   - f_loop: frequency of the loop (Hz)
#   - g_loop: gain of the loop
# OUTPUT: the Strehl ratio
def Strehl_lag(coeff, airmass, v_0, r_0, wavelength, f_loop, g_loop):
    if len(coeff) == 1:
        return np.exp(-coeff[0] * (v_0 / (airmass**(-3.0 / 5.0) * r02rlambda(r_0, wavelength) * f_loop * g_loop))**(5.0 / 3.0))
    elif len(coeff) == 2:
        return np.exp(-coeff[0] * (v_0 / (airmass**(-3.0 / 5.0) * r02rlambda(r_0, wavelength) * f_loop * g_loop))**(coeff[1]))
    else:
        raise ValueError('Invalid number of coefficients!')


# Photon noise error
# INPUTS
#   - coeff: damping factor
#   - N_ph: number of photons
#   - wavelength: wavelength at which the Strehl must be computed (m)
#   - wavelength_AO: wavelength at which the AO loop is closed (m)
#   - g_loop: gain of the loop (m)
#   - ExcessNoiseFactor: Excess noise factor (2 for EMCCDs)
# OUTPUT: the Strehl ratio
def Strehl_ph(coeff, N_ph, wavelength, wavelength_AO, g_loop, ExcessNoiseFactor):
    if len(coeff) == 1:
        return np.exp(-coeff[0] * (wavelength_AO / wavelength)**2 * ExcessNoiseFactor * g_loop / (2.0 - g_loop) * 1 / N_ph)
    else:
        raise ValueError('Invalid number of coefficients!')


# Readout noise error
# INPUTS
#   - coeff: damping factor
#   - sigRON: single pixel readout noise
#   - N_ph: number of photons
#   - pixScale: pixel scale (arcsecond)
#   - ExcessNoiseFactor: Excess noise factor (2 for EMCCDs)
#   - N_pix: number of pixel (side of the lenslet box)
# OUTPUT: the Strehl ratio
def Strehl_ron(coeff, sigRON, N_ph, pixScale, N_pix, g_loop):
    if len(coeff) == 1:
        return np.exp(-coeff[0] * pixScale**2 * N_pix**4 * sigRON**2 * g_loop / (2.0 - g_loop) * 1 / N_ph**2)
    else:
        raise ValueError('Invalid number of coefficients!')


# Isoplanetic and isokinetic error
# INPUTS
#   - coeff: damping factor and the power value
#   - airmass: secant of the zenith angle (1/cos(zenith_angle))
#   - theta: separation (arcsecond)
#   - h_0: altitude of the turbulent layer (m)
#   - r_0: Fried's parameter @500nm (m)
#   - wavelength: wavelength at which the Strehl must be computed (m)
# OUTPUT: the Strehl ratio
def Strehl_iso(coeff, airmass, theta, h_0, r_0, wavelength):
    if len(coeff) == 1:
        return np.exp(-coeff[0] * (as2rad(theta) * airmass * h_0 / (airmass**(-3.0 / 5.0) * r02rlambda(r_0, wavelength)))**(5.0 / 3.0))
    elif len(coeff) == 2:
        return np.exp(-coeff[0] * (as2rad(theta) * airmass * h_0 / (airmass**(-3.0 / 5.0) * r02rlambda(r_0, wavelength)))**(coeff[1]))
    else:
        raise ValueError('Invalid number of coefficients!')


# Function to compute the Strehl ratio of a NGS configuration file using
# the calibrated Maréchal approximation
# INPUTS
#   - config_NGS: configuration of the NGS
#       ['magnitude'] -> Magnitude of the NGS
#       ['zenith'] -> For the airmass (deg)
#       ['wavelength'] -> Wavelength of the HO NGS channel (m)
#       ['mag2flux'] -> Convertion magnitude to flux / Magnitude 0-point (ph/s/m2 for mag=0)
#   - config_target: configuration of the target
#       ['wavelength'] -> Wavelength of the target (science or fringe tracker) channel (m)
#       ['theta'] -> Angle between the target (science or fringe tracker) and the NGS (arcsecond)
#   - config_ao: configuration of the AO system
#       ['TelescopeDiameter']   -> Telescope diameter (m)
#       ['transmission']        -> Global transmission of the WFS channel (to compute the number of photons)
#       ['sig_RON']             -> Readout noise of the camera
#       ['n_mode']              -> Number of corrected modes corrected models
#                                  (to compute the equivalent DM number of actuators)
#       ['f_loop']              -> Loop frequency (Hz)
#       ['g_loop']              -> Loop gain
#       ['SH_diam']             -> SH-WFS diameter (number of lenslets)
#       ['pixScale']            -> pixel scale (milliarcsecond / pixel)
#       ['n_pix']               -> number of pixels per lenslet
#       [ExcessNoiseFactor]     -> Excess noise factor
#   - config_turbulence: configuration of the turbulence
#       ['r_0'] -> Fried's parameter @500 nm (m)
#       ['v_0'] -> Wind speed of the turbulence layers (m.s-1) (could be a list) (for isoplanetism)
#       ['h_0'] -> Altitude of the turbulent layers (m) (could be a list) (for isoplanetism)
#       ['Cn2'] -> Cn2 weight (could be a list) (for collapsing h_0 and v_0 to an equivalent individual layer)
#   - config_Strehl: configuration of the Strehl parameters
# OUTPUTS
#   - SR: the global Strehl ratio
def compute_Marechal_NGS(config_NGS, config_target, config_ao, config_turbulence, config_Strehl):


    ##### Loading configuration #####

    # Loading atmosphere
    r_0 = config_turbulence['r_0']
    Cn2 = config_turbulence['Cn2']
    h_0 = config_turbulence['h_0']
    h_0 = (np.sum(Cn2 * np.power(h_0, 5.0 / 3.0)) / np.sum(Cn2))**(3.0 / 5.0)
    v_0 = config_turbulence['v_0']
    v_0 = (np.sum(Cn2 * np.power(np.abs(v_0), 5.0 / 3.0)) / np.sum(Cn2))**(3.0 / 5.0)

    # Loading AO system
    ExcessNoiseFactor = config_ao['ExcessNoiseFactor']
    sigRON = config_ao['sig_RON']
    pixScale = config_ao['pixScale'] / 1000.0 # arcsecond
    [eqDM_pitch, eqDMn_act] = modes2eqDM(config_ao)
    f_loop = config_ao['f_loop']
    g_loop = config_ao['g_loop']
    n_pix = config_ao['n_pix']
    D_WFS = config_ao['TelescopeDiameter'] / config_ao['SH_diam']


    # Loading NGS
    wavelength_NGS = config_NGS['wavelength']
    n_ph = config_ao['transmission'] * D_WFS**2 * \
        config_NGS['mag2flux'] * 10.0**(-config_NGS['magnitude'] / 2.5) / f_loop
    zenith_angle = config_NGS['zenith']
    airmass = 1.0 / np.cos(np.radians(zenith_angle))



    # Loading target
    wavelength_target = config_target['wavelength']
    theta = config_target['theta']

    # Loading Strehl damping coefficient
    coeff_geom = config_Strehl['geom']
    coeff_lag = config_Strehl['lag']
    coeff_ph = config_Strehl['ph']
    coeff_ron = config_Strehl['ron']
    coeff_iso = config_Strehl['iso']
    ##### Loading configuration #####



    ##### Computing individual Strehl contributions #####
    SR_geom = Strehl_geom(coeff_geom, airmass, eqDM_pitch, r_0, wavelength_target)
    SR_lag = Strehl_lag(coeff_lag, airmass, v_0, r_0, wavelength_target, f_loop, g_loop)
    SR_ph = Strehl_ph(coeff_ph, n_ph, wavelength_target, wavelength_NGS, g_loop, ExcessNoiseFactor)
    SR_ron = Strehl_ron(coeff_ron, sigRON, n_ph, pixScale, n_pix, g_loop)
    SR_iso = Strehl_iso(coeff_iso, airmass, theta, h_0, r_0, wavelength_target)
    ##### Computing individual Strehl contributions #####



    ##### Output #####
    SR = SR_geom * SR_lag * SR_ph * SR_ron * SR_iso
    return SR
    ##### Output #####


# STRHEL FUNCTIONS #
####################








#################
# MISCELLANEOUS #

# Function to convert the number of modes into an equivalent DM
# INPUTS
#   - config_ao: configuration of the AO system
#       ['TelescopeDiameter']   -> Telescope diameter (m)
#       ['n_mode']              -> Number of corrected modes corrected models
#                                  (to compute the equivalent DM number of actuators)
# OUTPUTS
#   - eqDM_pitch: the pitch of the equivalent DM
#   - eqDMn_act: the DM number of actuator of the equivalent DM
def modes2eqDM(config_ao):
    # Equivalent number of actuators accross the pupil
    eqDMn_act = 2.0 * (config_ao['n_mode'] / np.pi)**0.5

    # Equivalent actuator pitch (-1 actuator)
    eqDM_pitch = config_ao['TelescopeDiameter'] / (eqDMn_act - 1.0)

    # Rounding the number of actuator to get an integer
    eqDMn_act = round(eqDMn_act)

    # Returning results
    return [eqDM_pitch, eqDMn_act]


# Function to convert the Fried parameter 'r_0' (m) according to
# the 'wavelength' (m)
# Note: The reference wavelength of the atmosphere is 500nm (m)
r02rlambda = lambda r_0, wavelength: r_0 * (wavelength / 500e-9)**(6.0 / 5.0)


# Function to convert the Fried parameter r_0 (m) to the
# equivalent seeing (arcsecond)
# Note: The reference wavelength of the atmosphere is 500nm (m)
r02seeing = lambda r_0: 0.98 * 500e-9 / r_0 * 180.0 / np.pi * 3600.0


# Function to convert the seeing (arcsecond) to the
# equivalent Fried parameter r_0 (m)
# Note: The reference wavelength of the atmosphere is 500nm (m)
seeing2r0 = lambda seeing: r02seeing(seeing)

# Function to convert the angle (arcsecond) to rad
as2rad = lambda theta: theta * np.pi / (180.0 * 3600.0)


# MISCELLANEOUS #
#################




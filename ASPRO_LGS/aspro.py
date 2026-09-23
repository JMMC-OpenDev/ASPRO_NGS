# 'aspro' module defining different functions and methodes
#
# Created: 09/21/2023 (mm/dd/yyyy)
# Author: Anthony Berdeu (LESIA - Observatoire de Paris)
#####################################################################

#######################
# IMPORTING LIBRARIES #

# Python
import numpy as np

# IMPORTING LIBRARIES #
#######################


# LBO: AO configuration
def get_mode_config_ao(flag_mode):
    config_ao = {}
    if flag_mode[0:3] == 'NGS':
        config_ao['magnitude_NGS'] = 0.0
        config_ao['n_mode'] = 500.0
        config_ao['f_loop_NGS'] = 1000.0
        config_ao['g_loop_NGS'] = 0.5
        config_ao['f_loop_LGS'] = 1000.0
        config_ao['g_loop_LGS'] = 0.5
    else:
        config_ao['magnitude_NGS'] = 0.0
        config_ao['n_mode'] = 500.0
        config_ao['f_loop_NGS'] = 500.0
        config_ao['g_loop_NGS'] = 0.3
        config_ao['f_loop_LGS'] = 1000.0
        config_ao['g_loop_LGS'] = 0.5

    config_ao['theta_NGS'] = 0.0
    config_ao['theta_LGS'] = 0.0

    return config_ao


##############
# GPAO modes #
# Function to load the GPAO modes configuration based on the TIPTOP fit
#   - flag_mode: 'NGS_VIS' / 'NGS_IR' / 'LGS_VIS' / 'LGS_IR'
# OUTPUTS
#   - config_Strehl: Strehl parameters fitted with TIPTOP
#   - config_WFS_NGS: configuration of the NGS WFS (HO in NGS modes / LO in LGS modes)
#   - config_WFS_LGS: configuration of the LGS WFS (LGS modes)
def get_mode_config(flag_mode):

    ##### General parameters #####
    D_tel = 8.0 # Telescope diameter (m)
    transmission = 0.3 # Global transmission of the WFS channel (to compute the number of photons)
    sig_RON = 0.2 # Readout noise of the camera
    ExcessNoiseFactor = 2.0 # Excess noise factor
    ##### General parameters #####

    ##### Initialization #####
    # Initialization of the outputs
    config_Strehl = {}
    config_WFS_NGS = {}
    config_WFS_LGS = {}

    ##### Mode-dependent variables #####
    # config_WFS_NGS['wavelength']  -> Wavelength of the NGS channel (m)
    # config_WFS_NGS['mag2flux']    -> Convertion magnitude to flux / Magnitude 0-point (ph/s/m2 for mag=0)
    # config_WFS_*['SH_diam']       -> SH-WFS diameter (number of lenslets)
    # config_WFS_*['pixScale']      -> pixel scale (milliarcsecond / pixel)
    # config_WFS_*['n_pix']         -> number of pixels per lenslet

    # config_WFS_NGS
    config_WFS_NGS['D_tel'] = D_tel
    config_WFS_NGS['sig_RON'] = sig_RON
    config_WFS_NGS['ExcessNoiseFactor'] = ExcessNoiseFactor
    config_WFS_NGS['transmission'] = transmission

    if flag_mode == 'NGS_VIS':
        # config_WFS_NGS
        config_WFS_NGS['SH_diam'] = 40.0
        config_WFS_NGS['pixScale'] = 420.0/1000.0 # arcsecond
        config_WFS_NGS['n_pix'] = 6.0

    elif flag_mode == 'LGS_VIS':
        # config_WFS_NGS
        config_WFS_NGS['SH_diam'] = 4.0
        config_WFS_NGS['pixScale'] = 210.0/1000.0 # arcsecond
        config_WFS_NGS['n_pix'] = 12.0

    elif flag_mode[4:6] == 'IR':
        # config_WFS_NGS
        config_WFS_NGS['SH_diam'] = 9.0
        config_WFS_NGS['pixScale'] = 510.0/1000.0 # arcsecond
        config_WFS_NGS['n_pix'] = 8.0

    # Lenslet diameter
    config_WFS_NGS['D_WFS'] = D_tel / config_WFS_NGS['SH_diam']

    # config_WFS_LGS
    config_WFS_LGS['sig_RON'] = sig_RON
    config_WFS_LGS['ExcessNoiseFactor'] = ExcessNoiseFactor

    config_WFS_LGS['SH_diam'] = 9.0
    config_WFS_LGS['pixScale'] = 800.0/1000.0 # arcsecond
    config_WFS_LGS['n_pix'] = 6.0

    config_WFS_LGS['h_LGS'] = 90000.0 # Sodium layer height (m)
    config_WFS_LGS['n_ph'] = 50.0

    # Lenslet diameter
    config_WFS_LGS['D_WFS'] = D_tel / config_WFS_LGS['SH_diam']
    ##### Initialization #####


    if flag_mode == 'NGS_VIS':
       # Strehl
        config_Strehl['geom'] = [0.267, 0.995]
        config_Strehl['lag'] = [8.49 , 2.157]
        config_Strehl['ph'] = [11.973]
        config_Strehl['ron'] = [0.52]
        config_Strehl['iso'] = [4.337, 1.864]

    elif flag_mode == 'NGS_IR':
        # Strehl
        config_Strehl['geom'] = [0.244, 0.869]
        config_Strehl['lag'] = [2.08 , 2.101]
        config_Strehl['ph'] = [15.179]
        config_Strehl['ron'] = [1.653]
        config_Strehl['iso'] = [1.75 , 1.973]

    elif flag_mode == 'LGS_VIS':
        # Strehl
        config_Strehl['geom'] = [0.26 , 1.014]
        config_Strehl['cone'] = [0.722, 1.899]
        config_Strehl['lag'] = [8.46 , 0.413, 2.185]
        config_Strehl['ph_ron_LO'] = [ 5.487e+01, -1.072e-04]
        config_Strehl['ph_ron_LGS'] = [5.874, 0.159]
        config_Strehl['iso'] = [4.326, 0.39 , 1.985]

    elif flag_mode == 'LGS_IR':
        # Strehl
        config_Strehl['geom'] = [0.257, 0.995]
        config_Strehl['cone'] = [0.683, 1.86 ]
        config_Strehl['lag'] = [4.899, 0.857, 1.818]
        config_Strehl['ph_ron_LO'] = [ 7.575e+00, -9.438e-05]

        #config_Strehl['ph_ron_LGS'] = [4.169, 0.35 , 1.958]
        #config_Strehl['iso'] = [5.666, 0.154]
        # mistake ?
        config_Strehl['ph_ron_LGS'] = [5.666, 0.154]
        config_Strehl['iso'] = [4.169, 0.35 , 1.958]

    else:
        raise ValueError(flag_mode + \
            ' -> Unknown mode (NGS_VIS / NGS_IR / LGS_VIS / LGS_IR)')


    if flag_mode[4:7] == 'VIS':
        # config_WFS_NGS
        config_WFS_NGS['wavelength'] = 750e-9
        config_WFS_NGS['mag2flux'] = 2.63e10

    elif flag_mode[4:6] == 'IR':
        # config_WFS_NGS
        config_WFS_NGS['wavelength'] = 2.2e-6
        config_WFS_NGS['mag2flux'] = 1.66e9

    else:
        raise ValueError(flag_mode + \
            ' -> Unknown mode (*_VIS / *_IR)')

    ##### Mode-dependent variables #####

    return [config_Strehl, config_WFS_NGS, config_WFS_LGS]

# GPAO modes #
##############



####################
# STREHL FUNCTIONS #

# Geometric error = fitting + aliasing
# INPUTS
#   - coeff: damping factor
#   - airmass: secant of the zenith angle (1/cos(zenith_angle))
#   - DM_pitch: pitch of the DM on the pupil (m)
#   - r_0: Fried's parameter @500nm (m)
#   - wavelength: wavelength at which the Strehl must be computed (m)
# OUTPUT: the Strehl ratio
def Strehl_geom(coeff, airmass, DM_pitch, r_0, wavelength):
    if len(coeff)==1:
        return np.exp(-coeff[0]*(DM_pitch/(airmass**(-3/5)*r02rlambda(r_0, wavelength)))**(5/3))
    elif len(coeff)==2:
        return coeff[1]*np.exp(-coeff[0]*(DM_pitch**(5/3))/(airmass**(-3/5)*r02rlambda(r_0, wavelength))**(5/3))
        # return np.exp(-coeff[0]*(DM_pitch**(5/3)+coeff[1])/(airmass**(-3/5)*r02rlambda(r_0, wavelength))**(5/3))
        # return np.exp(-coeff[0]*(DM_pitch/(airmass**(-3/5)*r02rlambda(r_0, wavelength)))**(coeff[1]))
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
    if len(coeff)==1:
        return np.exp(-coeff[0]*(v_0/(airmass**(-3/5)*r02rlambda(r_0, wavelength)*f_loop*g_loop))**(5/3))
    elif len(coeff)==2:
        return np.exp(-coeff[0]*(v_0/(airmass**(-3/5)*r02rlambda(r_0, wavelength)*f_loop*g_loop))**(coeff[1]))
    else:
        raise ValueError('Invalid number of coefficients!')


# Servo-lag error for the LGS mode
# INPUTS
#   - coeff: damping factor and the power value
#   - airmass: secant of the zenith angle (1/cos(zenith_angle))
#   - v_0: velocity of the turbulent layer (m.s-1)
#   - r_0: Fried's parameter @500nm (m)
#   - wavelength: wavelength at which the Strehl must be computed (m)
#   - f_loop_LGS: frequency of the loop (Hz) for the LGS part
#   - g_loop_LGS: gain of the loop  for the LGS part
#   - f_loop_LO: frequency of the loop (Hz) for the LO part
#   - g_loop_LO: gain of the loop  for the LO part
# OUTPUT: the Strehl ratio
def Strehl_lag_LGS(coeff, airmass, v_0, r_0, wavelength, f_loop_LGS, g_loop_LGS, f_loop_LO, g_loop_LO):
    if len(coeff)==2:
        coef_LGS = [coeff[0]]
        coef_LO = [coeff[1]]
    elif len(coeff)==3:
        coef_LGS = [coeff[0], coeff[2]]
        coef_LO = [coeff[1], coeff[2]]
    else:
        raise ValueError('Invalid number of coefficients!')

    return \
        Strehl_lag(coef_LGS, airmass, v_0, r_0, wavelength, f_loop_LGS, g_loop_LGS) * \
        Strehl_lag(coef_LO, airmass, v_0, r_0, wavelength, f_loop_LO, g_loop_LO)


# Photon noise error
# INPUTS
#   - coeff: damping factor
#   - N_ph: number of photons
#   - wavelength: wavelength at which the Strehl must be computed (m)
#   - wavelength_eq: equivalent wavelength of the spot (diffraction limited = wavelength_AO / LGS = 1"/D_WFS_LGS) (m)
#   - g_loop: gain of the loop (m)
#   - ExcessNoiseFactor: Excess noise factor (2 for EMCCDs)
# OUTPUT: the Strehl ratio
def Strehl_ph(coeff, N_ph, wavelength, wavelength_eq, g_loop, ExcessNoiseFactor):
    if len(coeff)==1:
        return np.exp(-coeff[0]*(wavelength_eq/wavelength)**2*ExcessNoiseFactor*g_loop/(2-g_loop)*1/N_ph)
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
    if len(coeff)==1:
        return np.exp(-coeff[0]*pixScale**2*N_pix**4*sigRON**2*g_loop/(2-g_loop)*1/N_ph**2)
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
    if len(coeff)==1:
        return np.exp(-coeff[0]*(theta*np.pi/180/3600*airmass*h_0/(airmass**(-3/5)*r02rlambda(r_0, wavelength)))**(5/3))
    elif len(coeff)==2:
        return np.exp(-coeff[0]*(theta*np.pi/180/3600*airmass*h_0/(airmass**(-3/5)*r02rlambda(r_0, wavelength)))**(coeff[1]))
    else:
        raise ValueError('Invalid number of coefficients!')


# Isoplanetic and isokinetic error for the LGS mode
# INPUTS
#   - coeff: damping factor and the power value
#   - airmass: secant of the zenith angle (1/cos(zenith_angle))
#   - theta_LGS: separation (arcsecond) of the LGS source
#   - theta_LO: separation (arcsecond) of the LO source
#   - h_0: altitude of the turbulent layer (m)
#   - r_0: Fried's parameter @500nm (m)
#   - wavelength: wavelength at which the Strehl must be computed (m)
# OUTPUT: the Strehl ratio
def Strehl_iso_LGS(coeff, airmass, theta_LGS, theta_LO, h_0, r_0, wavelength):
    if len(coeff)==2:
        coef_LGS = [coeff[0]]
        coef_LO = [coeff[1]]
    elif len(coeff)==3:
        coef_LGS = [coeff[0], coeff[2]]
        coef_LO = [coeff[1], coeff[2]]
    else:
        raise ValueError('Invalid number of coefficients!')

    return \
        Strehl_iso(coef_LGS, airmass, theta_LGS, h_0, r_0, wavelength) * \
        Strehl_iso(coef_LO, airmass, theta_LO, h_0, r_0, wavelength)


# Cone effect
# INPUTS
#   - coeff: damping factor
#   - airmass: secant of the zenith angle (1/cos(zenith_angle))
#   - h_0: altitude of the turbulent layer (m)
#   - h_lgs: altitude of the sodium layer (m)
#   - D_tel: diameter of the telescope (m)
#   - r_0: Fried's parameter @500nm (m)
#   - wavelength: wavelength at which the Strehl must be computed (m)
# OUTPUT: the Strehl ratio
def Strehl_cone(coeff, airmass, h_0, h_lgs, D_tel, r_0, wavelength):
    if len(coeff)==1:
        beta = (5/3)
    elif len(coeff)==2:
        beta = coeff[1]
    else:
        raise ValueError('Invalid number of coefficients!')
    return np.exp(-coeff[0]*(D_tel/(airmass**(-3/5)*r02rlambda(r_0, wavelength))*h_0/h_lgs)**beta)


# Function to compute the Strehl ratio with the calibrated Maréchal approximation calibrated with TIPTOP
# INPUTS
#   - flag_mode: 'NGS_VIS' / 'NGS_IR' / 'LGS_VIS' / 'LGS_IR'
#   - config_target: configuration of the target
#       ['wavelength'] -> Wavelength of the target (science or fringe tracker) channel (m)
#       ['zenith'] -> For the airmass (deg)
#   - config_turbulence: configuration of the turbulence
#       ['r_0'] -> Fried's parameter @500 nm (m)
#       ['v_0'] -> Wind speed of the turbulence layers (m.s-1) (could be a list) (for isoplanetism)
#       ['h_0'] -> Altitude of the turbulent layers (m) (could be a list) (for isoplanetism)
#       ['Cn2'] -> Cn2 weight (could be a list) (for collapsing h_0 and v_0 to an equivalent individual layer)
#   - config_ao: configuration of the AO system
#       ['magnitude_NGS']       -> Magnitude of the NGS
#       ['f_loop_NGS']          -> Loop frequency of the NGS loop (Hz)
#       ['f_loop_LGS']          -> Loop frequency of the LGS loop (Hz) (for LGS modes only)
#       ['g_loop_NGS']          -> Loop gain of the NGS loop
#       ['g_loop_LGS']          -> Loop gain of the LGS loop (for LGS modes only)
#       ['n_mode']              -> Number of corrected modes (to compute the equivalent DM number of actuators)
#       ['theta_NGS']           -> Angle between the target (science or fringe tracker) and the NGS (arcsecond)
#       ['theta_LGS']           -> Angle between the target (science or fringe tracker) and the LGS (arcsecond) (for LGS modes only)
#   - config_Strehl: Strehl parameters fitted with TIPTOP
#   - config_WFS_NGS: configuration of the NGS WFS (HO in NGS modes / LO in LGS modes)
#       ['D_tel']               -> Telescope diameter (m)
#       ['transmission']        -> Global transmission of the WFS channel (to compute the number of photons)
#       ['sig_RON']             -> Readout noise of the camera
#       ['ExcessNoiseFactor']   -> Excess noise factor
#       ['SH_diam']             -> SH-WFS diameter (lenslet)
#       ['pixScale']            -> Sensor pixel scale (arcsecond)
#       ['n_pix']               -> Number of pixels (per box)
#   - config_WFS_LGS: configuration of the LGS WFS (LGS modes)
#       ['transmission']        -> Global transmission of the WFS channel (to compute the number of photons)
#       ['sig_RON']             -> Readout noise of the camera
#       ['ExcessNoiseFactor']   -> Excess noise factor
#       ['SH_diam']             -> SH-WFS diameter (lenslet)
#       ['pixScale']            -> Sensor pixel scale (arcsecond)
#       ['n_pix']               -> Number of pixels (per box)
#       ['n_ph']                -> Number of photons (per box)
#       ['h_LGS']               -> Sodium layer height (m)
# OUTPUTS
#   - SR: the global Strehl ratio
def compute_Marechal(flag_mode, config_target, config_turbulence, config_ao, config_Strehl, config_WFS_NGS, config_WFS_LGS):
    ##### Loading configuration #####
    # Loading target
    wavelength_target = config_target['wavelength']
    zenith_angle = config_target['zenith']
    airmass = 1/np.cos(np.radians(zenith_angle))


    # Loading atmosphere
    r_0 = config_turbulence['r_0']
    Cn2 = config_turbulence['Cn2']
    h_0 = config_turbulence['h_0']
    h_0 = (np.sum(Cn2*np.power(h_0, 5/3))/np.sum(Cn2))**(3/5)
    v_0 = config_turbulence['v_0']
    v_0 = (np.sum(Cn2*np.power(np.abs(v_0), 5/3))/np.sum(Cn2))**(3/5)

    # Loading AO system
    [eqDM_pitch, eqDMn_act] = modes2eqDM(config_ao['n_mode'], config_WFS_NGS['D_tel'])
    f_loop_NGS = config_ao['f_loop_NGS']
    g_loop_NGS = config_ao['g_loop_NGS']
    n_ph_NGS = mag2nph(config_ao['magnitude_NGS'], config_WFS_NGS['mag2flux'], config_WFS_NGS['transmission'], \
        config_WFS_NGS['D_WFS'], f_loop_NGS)


    if flag_mode[0:3] == 'NGS':
        ##### Loading Strehl damping coefficient #####
        coeff_geom = config_Strehl['geom']
        coeff_lag = config_Strehl['lag']
        coeff_ph = config_Strehl['ph']
        coeff_ron = config_Strehl['ron']
        coeff_iso = config_Strehl['iso']
        ##### Loading Strehl damping coefficient #####


        ##### Computing individual Strehl contributions #####
        SR_geom = Strehl_geom(coeff_geom, airmass, eqDM_pitch, r_0, wavelength_target)
        SR_lag = Strehl_lag(coeff_lag, airmass, v_0, r_0, wavelength_target, f_loop_NGS, g_loop_NGS)
        SR_ph = Strehl_ph(coeff_ph, n_ph_NGS, wavelength_target, config_WFS_NGS['wavelength'], g_loop_NGS, config_WFS_NGS['ExcessNoiseFactor'])
        SR_ron = Strehl_ron(coeff_ron, config_WFS_NGS['sig_RON'], n_ph_NGS, config_WFS_NGS['pixScale'], config_WFS_NGS['n_pix'], g_loop_NGS)
        SR_iso = Strehl_iso(coeff_iso, airmass, np.abs(config_ao['theta_NGS']), h_0, r_0, wavelength_target)
        ##### Computing individual Strehl contributions #####


        ##### Output #####
        SR = SR_geom*SR_lag*SR_ph*SR_ron*SR_iso
        return SR
        ##### Output #####

    elif flag_mode[0:3] == 'LGS':
        ##### Loading Strehl damping coefficient #####
        coeff_geom = config_Strehl['geom']
        coeff_cone = config_Strehl['cone']
        coeff_lag = config_Strehl['lag']
        coeff_ph_ron_LO = config_Strehl['ph_ron_LO']
        coeff_ph_ron_LGS = config_Strehl['ph_ron_LGS']
        coeff_iso = config_Strehl['iso']
        ##### Loading Strehl damping coefficient #####


        ##### Computing individual Strehl contributions #####
        g_loop_LGS = config_ao['g_loop_LGS']
        n_ph_LGS = config_WFS_LGS['n_ph']
        SR_geom = Strehl_geom(coeff_geom, airmass, eqDM_pitch, r_0, wavelength_target)
        SR_cone = Strehl_cone(coeff_cone, airmass, h_0, config_WFS_LGS['h_LGS'], config_WFS_NGS['D_tel'], r_0, wavelength_target)
        SR_lag = Strehl_lag_LGS(coeff_lag, airmass, v_0, r_0, wavelength_target, config_ao['f_loop_LGS'], g_loop_LGS, f_loop_NGS, g_loop_NGS)
        SR_ph = \
            Strehl_ph([coeff_ph_ron_LGS[0]], n_ph_LGS, wavelength_target, arcsecond2rad(1) * config_WFS_LGS['D_WFS'], g_loop_LGS, config_WFS_LGS['ExcessNoiseFactor']) * \
            Strehl_ph([coeff_ph_ron_LO[0]], n_ph_NGS, wavelength_target, config_WFS_NGS['wavelength'], g_loop_NGS, config_WFS_NGS['ExcessNoiseFactor'])
        SR_ron = \
            Strehl_ron([coeff_ph_ron_LGS[1]], config_WFS_LGS['sig_RON'], n_ph_LGS, config_WFS_LGS['pixScale'], config_WFS_LGS['n_pix'], g_loop_LGS) * \
            Strehl_ron([coeff_ph_ron_LO[1]], config_WFS_NGS['sig_RON'], n_ph_NGS, config_WFS_NGS['pixScale'], config_WFS_NGS['n_pix'], g_loop_NGS)
        SR_iso = Strehl_iso_LGS(coeff_iso, airmass, np.abs(config_ao['theta_LGS']), np.abs(config_ao['theta_NGS']), h_0, r_0, wavelength_target)
        ##### Computing individual Strehl contributions #####


        ##### Output #####
        SR = SR_geom*SR_cone*SR_lag*SR_ph*SR_ron*SR_iso
        return SR
        ##### Output #####

    else:
        raise ValueError(flag_mode + \
            ' -> Unknown mode (NGS* / LGS*)')

# STREHL FUNCTIONS #
####################



#################
# MISCELLANEOUS #

# Function to convert arcsecond in radians
arcsecond2rad = lambda arcsec: arcsec/3600*np.pi/180

# Function to convert a magnitude into a number of photons per subaperture
# INPUTS
#   - magnitude: the magnitude
#   - mag2flux: the zero point magnitude -> flux
#   - transmission: the transmission
#   - D_WFS: the diameter of the WFS lenslet
#   - f_loop: the loop frequency
# OUTPUTS
#   - n_ph: the number of photons per subaperture
def mag2nph(magnitude, mag2flux, transmission, D_WFS, f_loop):
    return transmission * D_WFS**2 * mag2flux*10**(-magnitude/2.5) / f_loop

# Function to convert the number of modes into an equivalent DM
# INPUTS
#   - n_mode_AO: number of corrected modes by the AO system
#   - D_tel: telescope diameter
# OUTPUTS
#   - eqDM_pitch: the pitch of the equivalent DM
#   - eqDMn_act: the DM number of actuator of the equivalent DM
def modes2eqDM(n_mode_AO, D_tel):
    # Equivalent number of actuators accross the pupil
    eqDMn_act = 2*(n_mode_AO/np.pi)**0.5

    # Equivalent actuator pitch (-1 actuator)
    eqDM_pitch = D_tel / (eqDMn_act-1)

    # Rounding the number of actuator to get an integer
    eqDMn_act = round(eqDMn_act)

    # Returning results
    return [eqDM_pitch, eqDMn_act]


# Function to convert the Fried parameter 'r_0' (m) according to
# the 'wavelength' (m)
# Note: The reference wavelength of the atmosphere is 500nm (m)
r02rlambda = lambda r_0, wavelength: r_0*(wavelength/500e-9)**(6/5)


# Function to convert the Fried parameter r_0 (m) to the
# equivalent seeing (arcsecond)
# Note: The reference wavelength of the atmosphere is 500nm (m)
r02seeing = lambda r_0: 0.98*500e-9/r_0*180/np.pi*3600

# Function to convert the seeing (arcsecond) to the
# equivalent Fried parameter r_0 (m)
# Note: The reference wavelength of the atmosphere is 500nm (m)
seeing2r0 = lambda seeing: r02seeing(seeing)

# MISCELLANEOUS #
#################


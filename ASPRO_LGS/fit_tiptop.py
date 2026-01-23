# 'aspro' module defining different functions and methodes
# 
# Created: 09/21/2023 (mm/dd/yyyy)
# Author: Anthony Berdeu (LESIA - Observatoire de Paris)
#####################################################################

#######################
# IMPORTING LIBRARIES #

# Python
import os
import matplotlib.pyplot as plt
import configparser
import numpy as np
import scipy.optimize
from astropy.io import fits

# Tiptop
from tiptop import overallSimulation
from p3.aoSystem import fourierModel as FoMod
from p3.aoSystem import FourierUtils as FoUtils

# ASPRO Maréchal model
import aspro as aspro

# IMPORTING LIBRARIES #
#######################




##################################
# DEFINING CALIBRATION PROCEDURE #

# Function to run the TIPTOP simulation while looping on a parameter
# INPUTS
#   - path_results: root folder to save the results. This folder must
#     contain the file config_0.ini which is the reference configuration
#     defining all the needed parameters.
#   - name_error: name of the folder in the root folder to save the
#     simulation results. The log of the simulated PSFs will be stored
#     in a PSF_log subfolder.
#   - list_val: list of the values to loop
#   - func_update_config: function defining how the configuration must
#     be updated when looping on the parameter of interest
# OUTPUTS
#   - list_SR: the list of the Strehl ratios simulated with TIPTOP
def run_simu(path_results, name_error, list_val, \
    func_update_config, x_label):

    ##### Parameters #####
    # File parameter
    path_results_error = path_results + '/' + name_error + '/'
    path_log = path_results_error + 'PSF_log/'

    # Number of values to test
    var_nb = list_val.shape[0]
    ##### Parameters #####


    ##### Loading default configuration #####
    config = configparser.ConfigParser()
    config.optionxform = str # Needed to be case sensitive
    config.read(path_results + 'config_0.ini')
    ##### Loading default configuration #####


    ##### Creating folders #####
    if not os.path.exists(path_results_error):
        os.makedirs(path_results_error)
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    ##### Creating folders #####


    ##### Loop on the parameters #####
    # Saving variables
    hdu = fits.PrimaryHDU(list_val)
    hdu.writeto(path_results_error + 'list_val.fits', overwrite=True)

    list_SR = np.zeros([var_nb, 1])
    for i in range(var_nb):
        
        ###### Updating config ######
        suf_file = '_%i'%(i)
        # config = func_update_config(config, list_val[i])
        config = func_update_config(config, list_val[i,:])

        with open(path_results_error + name_error + suf_file + '.ini', 'w') \
            as configfile: config.write(configfile)
        ###### Updating config ######


        ###### Running simulations ######
        # Running TIPTOP
        [list_SR[i], PSF_AO] = run_TIPTOP(path_results_error, name_error + suf_file)

        # Saving the PSF log
        logFile = path_log + name_error + suf_file
        hdu = fits.PrimaryHDU(np.log10(PSF_AO))
        hdu.writeto(logFile + '.fits', overwrite=True)

        # Saving Strehl
        hdu = fits.PrimaryHDU(list_SR)
        hdu.writeto(path_results_error + 'list_SR.fits', overwrite=True)
        ###### Running simulations ######

    ##### Loop on the parameters #####

        
    ##### Display results #####
    for i in range(list_val.shape[1]):
        plt.figure(i)
        plt.clf()
        plt.plot(list_val[:,i], list_SR, label = 'TIPTOP')
        plt.xlabel(x_label[i])
        plt.ylabel('Strehl')
        plt.grid()
        plt.legend()
    ##### Display results #####

    return list_SR


# Function to fit the TIPTOP simulation
#   - list_val: list of the looped values
#   - list_SR: list of the simulated Strehl ratios
#   - func_strehl: function defining the Strehl ratio in terms of the
#     damping coefficient 'coeff' to be fitted and 'list_var'
#   - coeff_0: the first guess for the coefficients
#   - x_label: x-label for the plot
# OUTPUTS
#   - coeff: the fitted coefficient of the Maréchal approximation
#   - SR_0: the Strehl ratio asymptote for the configuration when
#     the tested value tends towards 0.
def fit_error(list_val, list_SR, func_strehl, coeff_0, x_label):

    ###### Fitting parameter ######
    cost = lambda x: np.sum((list_SR-x[0]*func_strehl(x[1:], list_val))**2)
    xopt = scipy.optimize.fmin(func=cost, x0=np.append(1, coeff_0))
    SR_0 = xopt[0]
    coeff = xopt[1:]
    ###### Fitting parameter ######

    
    ##### Display results #####
    for i in range(list_val.shape[1]):
        plt.figure(i)
        plt.clf()
        plt.plot(list_val[:,i], list_SR, label = 'TIPTOP')
        plt.plot(list_val[:,i], xopt[0]*func_strehl(xopt[1:], list_val), \
            label = 'Best fit: ' + np.array2string(coeff, separator=', '), \
            linestyle = 'dashed')
        plt.xlabel(x_label[i])
        plt.ylabel('Strehl')
        plt.grid()
        plt.legend()
    ##### Display results #####

    return [coeff, SR_0]


# Function to calibrate a specific error term in the Maréchal
# approximation using TIPTOP simulations
# INPUTS
#   - path_results: root folder to save the results. This folder must
#     contain the file config_0.ini which is the reference configuration
#     defining all the needed parameters.
#   - name_error: name of the folder in the root folder to save the
#     simulation results. The log of the simulated PSFs will be stored
#     in a PSF_log subfolder.
#   - list_val: list of the values to loop
#   - func_update_config: function defining how the configuration must
#     be updated when looping on the parameter of interest
#   - func_strehl: function defining the Strehl ratio in terms of the
#     damping coefficient 'coeff' to be fitted and 'list_var'
#   - coeff_0: the first guess for the coefficients
#   - x_label: x-label for the plot
# OUTPUTS
#   - list_SR: the list of the Strehl ratios simulated with TIPTOP
#   - coeff: the fitted coefficient of the Maréchal approximation
#   - SR_0: the Strehl ratio asymptote for the configuration when
#     the tested value tends towards 0.
def calib_error(path_results, name_error, list_val, \
    func_update_config, func_strehl, coeff_0, x_label):

    ###### Running simulation ######
    list_SR = run_simu(path_results, name_error, list_val, \
        func_update_config, x_label)
    ###### Running simulation ######
    
    ###### Fitting parameter ######
    [coeff, SR_0] = fit_error(list_val, list_SR, func_strehl, \
        coeff_0, x_label)
    ###### Fitting parameter ######

    return [list_SR, coeff, SR_0]
# DEFINING CALIBRATION PROCEDURE #
##################################




####################
# STREHL FUNCTIONS #

# Function to compute the Strehl ratio of a NGS configuration file using
# the calibrated Maréchal approximation
# INPUTS
#   - path_config: path of the configuration file
#   - name_config: name of the configuration file
# OUTPUTS
#   - SR: the global Strehl ratio
def compute_Maréchal_NGS(path_config, name_config):


    ##### Loading default configuration #####
    # Opening configuration
    config = configparser.ConfigParser()
    config.optionxform = str # Needed to be case sensitive
    config.read(path_config + name_config + '.ini')

    # Loading atmosphere
    seeing = eval(config['atmosphere']['Seeing'])
    r_0 = aspro.seeing2r0(seeing)
    Cn2Weights = eval(config['atmosphere']['Cn2Weights'])
    h_0 = eval(config['atmosphere']['Cn2Heights'])
    h_0 = (np.sum(Cn2Weights*np.power(h_0, 5/3))/np.sum(Cn2Weights))**(3/5)
    v_0 = eval(config['atmosphere']['WindSpeed'])
    v_0 = (np.sum(Cn2Weights*np.power(np.abs(v_0), 5/3))/np.sum(Cn2Weights))**(3/5)

    # Loading AO system
    wavelength_AO = eval(config['sources_HO']['Wavelength'])[0]
    ExcessNoiseFactor = eval(config['sensor_HO']['ExcessNoiseFactor'])
    sigRON = eval(config['sensor_HO']['SigmaRON'])
    pixScale = eval(config['sensor_HO']['PixelScale'])/1000 # arcsecond
    DM_pitch = eval(config['DM']['DmPitchs'])[0]
    f_loop = eval(config['RTC']['SensorFrameRate_HO'])
    g_loop = eval(config['RTC']['LoopGain_HO'])
    N_ph = eval(config['sensor_HO']['NumberPhotons'])[0]
    N_pix = eval(config['sensor_HO']['FieldOfView'])
    

    # Loading science
    wavelength_sci = eval(config['sources_science']['Wavelength'])[0]
    zenith_angle = eval(config['telescope']['ZenithAngle'])
    theta = np.abs(eval(config['sources_HO']['Zenith'])[0] - eval(config['sources_science']['Zenith'])[0])
    airmass = 1/np.cos(np.radians(zenith_angle))

    # Loading Strehl damping coefficient
    coeff_geom = eval(config['Strehl']['geom'])
    coeff_lag = eval(config['Strehl']['lag'])
    coeff_ph = eval(config['Strehl']['ph'])
    coeff_ron = eval(config['Strehl']['ron'])
    coeff_iso = eval(config['Strehl']['iso'])
    ##### Loading default configuration #####



    ##### Computing individual Strehl contributions #####
    SR_geom = aspro.Strehl_geom(coeff_geom, airmass, DM_pitch, r_0, wavelength_sci)
    SR_lag = aspro.Strehl_lag(coeff_lag, airmass, v_0, r_0, wavelength_sci, f_loop, g_loop)
    SR_ph = aspro.Strehl_ph(coeff_ph, N_ph, wavelength_sci, wavelength_AO, g_loop, ExcessNoiseFactor)
    SR_ron = aspro.Strehl_ron(coeff_ron, sigRON, N_ph, pixScale, N_pix, g_loop)
    SR_iso = aspro.Strehl_iso(coeff_iso, airmass, theta, h_0, r_0, wavelength_sci)
    ##### Computing individual Strehl contributions #####



    ##### Output #####
    SR = SR_geom*SR_lag*SR_ph*SR_ron*SR_iso
    return SR
    ##### Output #####




# Function to compute the Strehl ratio of a LGS configuration file using
# the calibrated Maréchal approximation
# INPUTS
#   - path_config: path of the configuration file
#   - name_config: name of the configuration file
# OUTPUTS
#   - SR: the global Strehl ratio
def compute_Maréchal_LGS(path_config, name_config):


    ##### Loading default configuration #####
    # Opening configuration
    config = configparser.ConfigParser()
    config.optionxform = str # Needed to be case sensitive
    config.read(path_config + name_config + '.ini')

    # Loading atmosphere
    seeing = eval(config['atmosphere']['Seeing'])
    r_0 = aspro.seeing2r0(seeing)
    Cn2Weights = eval(config['atmosphere']['Cn2Weights'])
    h_0 = eval(config['atmosphere']['Cn2Heights'])
    h_0 = (np.sum(Cn2Weights*np.power(h_0, 5/3))/np.sum(Cn2Weights))**(3/5)
    v_0 = eval(config['atmosphere']['WindSpeed'])
    v_0 = (np.sum(Cn2Weights*np.power(np.abs(v_0), 5/3))/np.sum(Cn2Weights))**(3/5)

    # Loading AO system [LGS]
    DM_pitch = eval(config['DM']['DmPitchs'])[0]
    D_tel = eval(config['telescope']['TelescopeDiameter'])
    # LGS
    wavelength_LGS = eval(config['sources_HO']['Wavelength'])[0]
    ExcessNoiseFactor_LGS = eval(config['sensor_HO']['ExcessNoiseFactor'])
    sigRON_LGS = eval(config['sensor_HO']['SigmaRON'])
    pixScale_LGS = eval(config['sensor_HO']['PixelScale'])/1000 # arcsecond
    f_loop_LGS = eval(config['RTC']['SensorFrameRate_HO'])
    g_loop_LGS = eval(config['RTC']['LoopGain_HO'])
    N_ph_LGS = eval(config['sensor_HO']['NumberPhotons'])[0]
    N_pix_LGS = eval(config['sensor_HO']['FieldOfView'])
    h_lgs = eval(config['sources_HO']['Height'])
    D_WFS_LGS = \
        eval(config['telescope']['TelescopeDiameter']) / \
        eval(config['sensor_HO']['NumberLenslets'])[0]
    # LO
    wavelength_LO = eval(config['sources_LO']['Wavelength'])[0]
    ExcessNoiseFactor_LO = eval(config['sensor_LO']['ExcessNoiseFactor'])
    sigRON_LO = eval(config['sensor_LO']['SigmaRON'])
    pixScale_LO = eval(config['sensor_LO']['PixelScale'])/1000 # arcsecond
    f_loop_LO = eval(config['RTC']['SensorFrameRate_LO'])
    g_loop_LO = eval(config['RTC']['LoopGain_LO'])
    N_ph_LO = eval(config['sensor_LO']['NumberPhotons'])[0]
    N_pix_LO = eval(config['sensor_LO']['FieldOfView'])
    D_WFS_LO = \
        eval(config['telescope']['TelescopeDiameter']) / \
        eval(config['sensor_LO']['NumberLenslets'])[0]
    

    # Loading science
    wavelength_sci = eval(config['sources_science']['Wavelength'])[0]
    zenith_angle = eval(config['telescope']['ZenithAngle'])
    theta_LGS = np.abs(eval(config['sources_HO']['Zenith'])[0] - eval(config['sources_science']['Zenith'])[0])
    theta_LO = np.abs(eval(config['sources_LO']['Zenith'])[0] - eval(config['sources_science']['Zenith'])[0])
    airmass = 1/np.cos(np.radians(zenith_angle))

    # Loading Strehl damping coefficient
    coeff_geom = eval(config['Strehl']['geom'])
    coeff_cone = eval(config['Strehl']['cone'])
    coeff_lag = eval(config['Strehl']['lag'])
    coeff_ph_ron_LGS = eval(config['Strehl']['ph_ron_LGS'])
    coeff_ph_ron_LO = eval(config['Strehl']['ph_ron_LO'])
    coeff_iso = eval(config['Strehl']['iso'])
    ##### Loading default configuration #####



    ##### Computing individual Strehl contributions #####
    SR_geom = aspro.Strehl_geom(coeff_geom, airmass, DM_pitch, r_0, wavelength_sci)
    SR_cone = aspro.Strehl_cone(coeff_cone, airmass, h_0, h_lgs, D_tel, r_0, wavelength_sci)
    SR_lag = aspro.Strehl_lag_LGS(coeff_lag, airmass, v_0, r_0, wavelength_sci, f_loop_LGS, g_loop_LGS, f_loop_LO, g_loop_LO)
    SR_ph = \
        aspro.Strehl_ph([coeff_ph_ron_LGS[0]], N_ph_LGS, wavelength_sci, aspro.arcsecond2rad(1) * D_WFS_LGS, g_loop_LGS, ExcessNoiseFactor_LGS) * \
        aspro.Strehl_ph([coeff_ph_ron_LO[0]], N_ph_LO, wavelength_sci, wavelength_LO, g_loop_LO, ExcessNoiseFactor_LO)
    SR_ron = \
        aspro.Strehl_ron([coeff_ph_ron_LGS[1]], sigRON_LGS, N_ph_LGS, pixScale_LGS, N_pix_LGS, g_loop_LGS) * \
        aspro.Strehl_ron([coeff_ph_ron_LO[1]], sigRON_LO, N_ph_LO, pixScale_LO, N_pix_LO, g_loop_LO)
    SR_iso = aspro.Strehl_iso_LGS(coeff_iso, airmass, theta_LGS, theta_LO, h_0, r_0, wavelength_sci)
    ##### Computing individual Strehl contributions #####



    ##### Output #####
    SR = SR_geom*SR_cone*SR_lag*SR_ph*SR_ron*SR_iso
    return SR
    ##### Output #####

# STREHL FUNCTIONS #
####################



#################
# MISCELLANEOUS #

# Function to run TIPTOP overralsimulation
# INPUTS
#   - path_config: path of the configuration file
#   - name_config: name of the configuration file
# OUTPUTS
#   - SR: Strehl ratio
#   - PSF_AO: Simulated AO PSF
def run_TIPTOP(path_config, name_config):

    # Running TIPTOP
    overallSimulation(path_config, name_config, \
        path_config, name_config, \
        doPlot=False, doConvolve=True, verbose=True, \
        returnRes=False)

    # Getting the Strehl and extracting the PSF
    inputFile = path_config + name_config
    PSF_AO = fits.getdata(inputFile + '.fits', 1)[0]
    fao = FoMod.fourierModel(inputFile + '.ini', calcPSF=False, \
        display=False, computeFocalAnisoCov=False)
    SR = FoUtils.getStrehl(PSF_AO, fao.ao.tel.pupil, fao.freq.sampRef)

    # Output
    return [SR, PSF_AO]

# MISCELLANEOUS #
#################




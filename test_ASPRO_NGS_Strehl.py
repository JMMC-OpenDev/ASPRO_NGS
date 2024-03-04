#####################################################################
# 'test_ASPRO_NGS_Strehl.py'
#
# Created: 2024.02 (yyyy.mm)
# Author: Laurent  Bourgès (JMMC - OSUG, CNRS)
# License: GPL3 (see LICENSE)
#
#####################################################################

import aspro as aspro
import numpy as np
import matplotlib.pyplot as plt


trace = True

config_turbulence = {}


def setConfigTurbulence(seeing, tau0, h0):

    config_turbulence['seeing'] = seeing
    config_turbulence['tau0'] = tau0  # (ms)
    config_turbulence['h_0'] = h0  # Altitude of the turbulent layers (m) (could be a list) (for isoplanetism)
    config_turbulence['Cn2'] = 1.0  # Cn2 weight (could be a list) (for collapsing h_0 and v_0 to an equivalent individual layer)

    # seeing as gives r0:
    # Fried's parameter @500 nm (m):
    config_turbulence['r_0'] = (1.028993 * (0.5e-6 / config_turbulence['seeing']) / np.pi * (180.0 * 3600.0))  # m

    # tau0 (+r0) gives v0:
    # # Wind speed of the turbulence layers (m.s-1) (could be a list) (for isoplanetism)
    # unused by strehl_iso:
    config_turbulence['v_0'] = (1000.0 * config_turbulence['r_0'] / config_turbulence['tau0'])  # (m.s-1)

    # Derive seeing & tau0:
    config_turbulence['seeing'] = (1.028993 * (0.5e-6 / config_turbulence['r_0']) / np.pi * (180.0 * 3600.0))  # as
    config_turbulence['tau0'] = (1000.0 * config_turbulence['r_0'] / config_turbulence['v_0'])  # (ms)

    if trace:
        print("setConfigTurbulence:")
        print(f"- seeing: {config_turbulence['seeing']:.2f} as")
        print(f"- h0:     {config_turbulence['h_0']:.3f} m")
        print(f"- r0:     {100.0 * config_turbulence['r_0']:.2f} cm")
        print(f"- tau0:   {config_turbulence['tau0']:.2f} ms")
        print(f"- v0:     {config_turbulence['v_0']:.3f} m.s-1")


def computeStrehl_UT_NGS(flag_mode, target_ao_mag, distance_ao_as, iso=False):
    ##### User parameters #####
    # Mode to simulate
    # flag_mode = 'NGS_IR' or ''NGS_VIS'
    if flag_mode[4:7] != "VIS" and flag_mode[4:7] != "IR":
        raise ValueError(flag_mode + " -> Unknown mode (*_VIS / *_IR)")

    # NGS
    config_NGS = {}
    config_NGS['magnitude'] = target_ao_mag  # Magnitude of the NGS
    config_NGS['zenith'] = 0.0  # For the airmass (deg), 0.0 for zenith

    # Target
    config_target = {}
    config_target['wavelength'] = 2.2e-06 # Wavelength of the target (science or fringe tracker) channel (m)
    config_target['theta'] = distance_ao_as # Angle between the target (science or fringe tracker) and the NGS (arcsecond)

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
    if flag_mode[4:7] == "VIS":
        config_NGS["wavelength"] = 750e-9
        config_NGS["mag2flux"] = 2.63e10
        config_ao["n_mode"] = 800
        config_ao["f_loop"] = 1000.0
        config_ao["SH_diam"] = 40
        config_ao["pixScale"] = 420
        config_ao["n_pix"] = 6
    elif flag_mode[4:7] == "IR":
        config_NGS["wavelength"] = 2.2e-6
        config_NGS["mag2flux"] = 1.66e9
        config_ao["n_mode"] = 44
        config_ao["f_loop"] = 500.0
        config_ao["SH_diam"] = 9
        config_ao["pixScale"] = 510
        config_ao["n_pix"] = 8
    ##### Mode-dependent variables #####

    ##### Calibration of the Maréchal approximation #####
    # Values obtained with TIPTOP
    config_Strehl = {}
    if flag_mode[4:7] == "VIS":
        config_Strehl["geom"] = [0.26705087, 0.98968173]
        config_Strehl["lag"] = [8.48317135, 2.15500641]
        config_Strehl["ph"] = [11.97305155]
        config_Strehl["ron"] = [0.51996901]
        config_Strehl["iso"] = [4.33657467, 1.86425362]
    elif flag_mode[4:7] == "IR":
        config_Strehl["geom"] = [0.24405723, 0.86477159]
        config_Strehl["lag"] = [2.08400088, 2.09918214]
        config_Strehl["ph"] = [15.17856885]
        config_Strehl["ron"] = [1.65331745]
        config_Strehl["iso"] = [1.74957095, 1.97261581]
    ##### Calibration of the Maréchal approximation #####


    # Running Maréchal approximation (Anthony Berdeu, LESIA, OBSPM)
    # return aspro.compute_Marechal_NGS(config_NGS, config_target, config_ao, config_turbulence, config_Strehl)

    ##### Computing individual Strehl contributions #####

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
    [eqDM_pitch, eqDMn_act] = aspro.modes2eqDM(config_ao)
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



    if iso:
        # print(f"Strehl_iso: coeff_iso={coeff_iso} airmass={airmass} theta={theta} h_0={h_0} r_0={r_0} wavelength={wavelength_target}")
        return aspro.Strehl_iso(coeff_iso, airmass, theta, h_0, r_0, wavelength_target)

    ##### Computing individual Strehl contributions #####
    SR_geom = aspro.Strehl_geom(coeff_geom, airmass, eqDM_pitch, r_0, wavelength_target)
    SR_lag = aspro.Strehl_lag(coeff_lag, airmass, v_0, r_0, wavelength_target, f_loop, g_loop)
    SR_ph = aspro.Strehl_ph(coeff_ph, n_ph, wavelength_target, wavelength_NGS, g_loop, ExcessNoiseFactor)
    SR_ron = aspro.Strehl_ron(coeff_ron, sigRON, n_ph, pixScale, n_pix, g_loop)
    SR_iso = aspro.Strehl_iso(coeff_iso, airmass, theta, h_0, r_0, wavelength_target)
    ##### Computing individual Strehl contributions #####

    # print(f"SR_geom: {SR_geom}")
    # print(f"SR_lag:  {SR_lag}")
    # print(f"SR_ph:   {SR_ph}")
    # print(f"SR_ron:  {SR_ron}")
    # print(f"SR_iso:  {SR_iso}")

    ##### Output #####
    SR = SR_geom * SR_lag * SR_ph * SR_ron * SR_iso
    return SR
    ##### Output #####


def plotStrehlIso(flag_mode):
    ao_Rmag = 5.0

    plt.figure(figsize=(20, 10))

    dists_AO = np.arange(0.0, 30.0, 0.2, dtype=float)

    for i in range(len(seeing_values)):
        seeing = seeing_values[i]
        setConfigTurbulence(seeing, tau0_values[i], ho_values[i])

        sr_iso = np.zeros_like(dists_AO)

        print("distance_ao_as\tstrehl_ratio")

        for i in range(len(dists_AO)):
            distance_ao_as = dists_AO[i]  # as

            sr_iso[i] = computeStrehl_UT_NGS(flag_mode, ao_Rmag, distance_ao_as, True)
            print(f"{distance_ao_as:.2f}\t{sr_iso[i]:.4e}")

        plt.plot(dists_AO, sr_iso, marker='o', label=f"seeing: {seeing:.2f}")

    plt.xlabel('dist (as)')
    plt.ylabel('SR_iso')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.grid(True)
    plt.legend()
    plt.show()


def plotStrehlMag(flag_mode):
    plt.figure(figsize=(20, 10))

    distance_ao_as = 0.0
    mags_AO = np.arange(0.0, 20.0, 0.25, dtype=float)

    for i in range(len(seeing_values)):
        seeing = seeing_values[i]
        setConfigTurbulence(seeing, tau0_values[i], ho_values[i])

        sr = np.zeros_like(mags_AO)

        print("ao_mag\tstrehl_ratio")

        for i in range(len(mags_AO)):
            ao_mag = mags_AO[i]  # as

            sr[i] = computeStrehl_UT_NGS(flag_mode, ao_mag, distance_ao_as, False)
            print(f"{ao_mag:.2f}\t{sr[i]:.4e}")

        plt.plot(mags_AO, sr, marker='o', label=f"seeing: {seeing:.2f}")

    plt.xlabel('AO mag')
    plt.ylabel('SR_GPAO')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.grid(True)
    plt.legend()
    plt.show()

# --- main ---
if __name__ == "__main__":
    flag_mode = "NGS_VIS"

    # Using ESO turbulence categories:
    #     - GRAVITY: https://www.eso.org/sci/observing/phase2/ObsConditions.GRAVITY.html
    #         More specifically, the categories are:
    #                 T < 10%, corresponding to seeing ≤ 0.60“ and τ0 > 5.2ms
    #                 T < 20%, corresponding to seeing ≤ 0.70“ and τ0 > 4.4ms
    #                 T < 30%, corresponding to seeing ≤ 0.80“ and τ0 > 4.1ms
    #                 T < 50%, corresponding to seeing ≤ 1.00“ and τ0 > 3.2ms
    #                 T < 70%, corresponding to seeing ≤ 1.15“ and τ0 > 2.2ms
    #                 T < 85%, corresponding to seeing ≤ 1.40“ and τ0 > 1.6ms
    #         For conditions worse than T = 85%, no GRAVITY operations are possible

    seeing_values = np.array([0.60, 0.70, 0.80, 1.00, 1.15, 1.40])
    tau0_values = np.array([5.2, 4.4, 4.1, 3.2, 2.2, 1.6])

    # from http://archive.eso.org/wdb/wdb/asm/mass_paranal/form:
    # Median $8 * $10 = median (MASS Turb Altitude [m] * MASS-DIMM Cn2 fraction at ground)
    ho_values = np.array([5850.0, 5250.0, 4650.0, 3700.0, 3200.0, 2700.0])

    plotStrehlIso(flag_mode)
    plotStrehlMag(flag_mode)
    plotStrehlMag("NGS_IR")

    print("That's All, folks !'")

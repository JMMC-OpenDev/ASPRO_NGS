# File to apply the Maréchal approximation for the different GPAOs modes
#   - NGS_VIS   -> 40x40 SH-WFS
#   - NGS_IR    -> 9x9 SH-WFS
#   - LGS_VIS   -> 30x30 LGS + 4x4 SH-WFS
#   - LGS_IR    -> 30x30 LGS + 9x9 SH-WFS
#
# Created: 06/09/2023 (mm/dd/yyyy)
# Author: Anthony Berdeu (LIRA - Observatoire de Paris)
########################################################################

#######################
# IMPORTING LIBRARIES #
#######################

# Python
import os
import numpy as np
import importlib

import matplotlib.pyplot as plt

import aspro as aspro

# aspro
importlib.reload(aspro)



def setConfigTurbulence(config_turbulence, seeing, tau0, h0):

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

    if 1:
        print("setConfigTurbulence:")
        print(f"- seeing: {config_turbulence['seeing']:.2f} as")
        print(f"- h0:     {config_turbulence['h_0']:.3f} m")
        print(f"- r0:     {100.0 * config_turbulence['r_0']:.2f} cm")
        print(f"- tau0:   {config_turbulence['tau0']:.2f} ms")
        print(f"- v0:     {config_turbulence['v_0']:.3f} m.s-1")


###################
# Comparing modes #
###################


# Loop on magnitude
mag_min = 0
mag_max = 22
mag_delta = 0.25
mag_nb = round((mag_max-mag_min)/mag_delta+1)
list_mag = np.zeros([mag_nb, 1])
list_mag[:,0] = np.linspace(mag_min, mag_max, mag_nb)


# Target
config_target = {}
config_target['wavelength'] = 2.2e-06 # Wavelength of the target (science or fringe tracker) channel (m)
config_target['zenith'] = 0.0 # Pointing angle to zenith for the airmass (deg)


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


# --- main ---
if __name__ == "__main__":

    name_error = 'NGS_vs_LGS'
    xlabel = 'Magnitude (NGS)'

    # Defining cases
    nb_test = 2
    list_flag_mode = ['NGS_VIS', 'LGS_VIS']

    # Turbulence
    config_turbulence = {}
    config_turbulence['r_0'] = 0.100 # Fried's parameter @500 nm (m)
    config_turbulence['v_0'] = [25.0] # Wind speed of the turbulence layers (m.s-1) (could be a list) (for isoplanetism)
    config_turbulence['h_0'] = [1500.0] # Altitude of the turbulent layers (m) (could be a list) (for isoplanetism)
    config_turbulence['Cn2'] = [1.0] # Cn2 weight (could be a list) (for collapsing h_0 and v_0 to an equivalent individual layer)

    # Generating cases
    list_label = [None] * nb_test
    list_config_target = [None] * nb_test
    list_config_turbulence = [None] * nb_test
    list_config_ao = [None] * nb_test
    list_config_Strehl = [None] * nb_test
    list_config_WFS_NGS = [None] * nb_test
    list_config_WFS_LGS = [None] * nb_test
    for i in range(nb_test):
        list_label[i] = list_flag_mode[i]

        list_config_target[i] = config_target
        list_config_turbulence[i] = config_turbulence

        # AO configuration
        if list_flag_mode[i][0:3] == 'NGS':
            config_ao = {}
            config_ao['magnitude_NGS'] = 0
            config_ao['n_mode'] = 500
            config_ao['f_loop_NGS'] = 1000
            config_ao['g_loop_NGS'] = 0.5
            config_ao['f_loop_LGS'] = 1000
            config_ao['g_loop_LGS'] = 0.5
        else:
            config_ao = {}
            config_ao['magnitude_NGS'] = 0
            config_ao['n_mode'] = 500
            config_ao['f_loop_NGS'] = 500
            config_ao['g_loop_NGS'] = 0.3
            config_ao['f_loop_LGS'] = 1000
            config_ao['g_loop_LGS'] = 0.5

        # no distance:
        config_ao['theta_NGS'] = 0
        config_ao['theta_LGS'] = 0

        list_config_ao[i] = config_ao
        [list_config_Strehl[i], list_config_WFS_NGS[i], list_config_WFS_LGS[i]] = aspro.get_mode_config(list_flag_mode[i])

    ##### Parameters #####


    plt.figure(figsize=(20, 10))

    ##### Maréchal #####

    for i in range(nb_test):
        print(f"\nflag_mode = '{list_flag_mode[i]}'")

        plt.gca().set_prop_cycle(None)

        for s in range(len(seeing_values)):
            seeing = seeing_values[s]
            setConfigTurbulence(list_config_turbulence[i], seeing, tau0_values[s], ho_values[s])

            list_SR_Maréchal = np.zeros([mag_nb])

            for mag in range(mag_nb):
                # Running Maréchal approximation
                list_config_ao[i]['magnitude_NGS'] = list_mag[mag, 0] # TODO: fix dim

                list_SR_Maréchal[mag] = aspro.compute_Maréchal(list_flag_mode[i], list_config_target[i], list_config_turbulence[i], list_config_ao[i], list_config_Strehl[i], list_config_WFS_NGS[i], list_config_WFS_LGS[i])

                print(f"magnitude_NGS = {list_mag[mag,0]:.2f} - SR = {list_SR_Maréchal[mag]:.3f} ({list_flag_mode[i]})")

            plt.plot(list_mag, list_SR_Maréchal, label = f"{list_label[i]} seeing: {seeing:.2f}")

    ##### Maréchal #####

    plt.xlabel(xlabel)
    plt.ylabel('Strehl')
    plt.title(f"{name_error} - {str(config_target)}")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.show()

    ##### Display results #####

#####################################################################
# 'test_ASPRO_NGS_LGS_vs_mag.py'
#
# File to apply the Maréchal approximation for the different GPAOs modes
#   - NGS_VIS   -> 40x40 SH-WFS
#   - NGS_IR    -> 9x9 SH-WFS
#   - LGS_VIS   -> 30x30 LGS + 4x4 SH-WFS
#   - LGS_IR    -> 30x30 LGS + 9x9 SH-WFS
#
# Created: 2026.09 (yyyy.mm)
# Author: Anthony Berdeu (LIRA - Observatoire de Paris)
# License: GPL3 (see LICENSE)
#
# This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 101004719.
#
#####################################################################





#######################
# IMPORTING LIBRARIES #
#######################

# Python
import aspro as aspro
import numpy as np
import matplotlib.pyplot as plt


# --- aspro wrapper ---
trace = True

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


# Turbulence  derived from (seeing, tau0 and h0 values)
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

# --- aspro wrapper ---


###################
# Comparing modes #
###################

# Loop on magnitude
mag_min = 0
mag_max = 22
mag_delta = 0.25
mag_nb = round((mag_max-mag_min)/mag_delta+1)
list_mag = np.zeros([mag_nb])
list_mag[:] = np.linspace(mag_min, mag_max, mag_nb)


# Defining cases
nb_test = 2
list_flag_mode = ['NGS_VIS', 'LGS_VIS']

# Target
config_target = {}
config_target['wavelength'] = 2.2e-06 # Wavelength of the target (science or fringe tracker) channel (m)
config_target['zenith'] = 0.0 # Pointing angle to zenith for the airmass (deg)

# Generating cases
list_label = [None] * nb_test
list_config_ao = [None] * nb_test
list_config_Strehl = [None] * nb_test
list_config_WFS_NGS = [None] * nb_test
list_config_WFS_LGS = [None] * nb_test

for i in range(nb_test):
    flag_mode = list_flag_mode[i]
    list_label[i] = flag_mode

    # AO configuration
    list_config_ao[i] = aspro.get_mode_config_ao(flag_mode)

    # Loading TIPTOP fit and configuration
    [list_config_Strehl[i], list_config_WFS_NGS[i], list_config_WFS_LGS[i]] = aspro.get_mode_config(flag_mode)

##### Parameters #####


plt.figure(figsize=(20, 10))

##### Maréchal #####
linestyle='solid'

for i in range(nb_test):
    print(f"\nflag_mode = '{list_flag_mode[i]}'")

    plt.gca().set_prop_cycle(None)

    if (i != 0):
        linestyle = 'dashed'

    for s in range(len(seeing_values)):
        seeing = seeing_values[s]
        setConfigTurbulence(seeing, tau0_values[s], ho_values[s])

        list_SR_Marechal = np.zeros([mag_nb])

        for mag in range(mag_nb):
            list_config_ao[i]['magnitude_NGS'] = list_mag[mag]

            # Running Maréchal approximation
            list_SR_Marechal[mag] = aspro.compute_Marechal(list_flag_mode[i], config_target, config_turbulence, list_config_ao[i], list_config_Strehl[i], list_config_WFS_NGS[i], list_config_WFS_LGS[i])

            print(f"magnitude_NGS = {list_mag[mag]:.2f} - SR = {list_SR_Marechal[mag]:.3f} ({list_flag_mode[i]})")

        plt.plot(list_mag, list_SR_Marechal, label = f"{list_label[i]} seeing: {seeing:.2f}", linestyle=linestyle)

##### Maréchal #####

plt.xlabel('Magnitude (NGS)')
plt.ylabel('Strehl (%)')
plt.title(f"NGS vs LGS - {str(config_target)}")
plt.xlim(mag_min, mag_max)
plt.locator_params(axis='x', nbins=mag_max)
plt.locator_params(axis='y', nbins=10)
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout(pad=0.05)
plt.show()

##### Display results #####

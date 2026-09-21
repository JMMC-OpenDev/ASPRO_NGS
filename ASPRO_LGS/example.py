#####################################################################
# File to apply the Maréchal approximation for the different GPAOs modes
#   - NGS_VIS   -> 40x40 SH-WFS
#   - NGS_IR    -> 9x9 SH-WFS
#   - LGS_VIS   -> 30x30 LGS + 4x4 SH-WFS
#   - LGS_IR    -> 30x30 LGS + 9x9 SH-WFS
#
# Created: 06/09/2023 (mm/dd/yyyy)
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
import os
import numpy as np
import importlib

import aspro as aspro

# aspro
importlib.reload(aspro)


##############################
# Example on a single Strehl #
##############################

flag_mode = 'LGS_VIS'

# Target
config_target = {}
config_target['wavelength'] = 2.2e-06 # Wavelength of the target (science or fringe tracker) channel (m)
config_target['zenith'] = 0.0 # Pointing angle to zenith for the airmass (deg)

# Turbulence
config_turbulence = {}
config_turbulence['r_0'] = 0.100 # Fried's parameter @500 nm (m)
config_turbulence['v_0'] = [15.0] # Wind speed of the turbulence layers (m.s-1) (could be a list) (for isoplanetism)
config_turbulence['h_0'] = [1500.0] # Altitude of the turbulent layers (m) (could be a list) (for isoplanetism)
config_turbulence['Cn2'] = [1] # Cn2 weight (could be a list) (for collapsing h_0 and v_0 to an equivalent individual layer)
#config_turbulence['v_0'] = [12.3, 8.3, 30.4, 56, 32] # Wind speed of the turbulence layers (m.s-1) (could be a list) (for isoplanetism)
#config_turbulence['h_0'] = [30, 562, 4500, 7750, 14000] # Altitude of the turbulent layers (m) (could be a list) (for isoplanetism)
# config_turbulence['Cn2'] = [12.3, 8.3, 30.4, 56, 32] # Cn2 weight (could be a list) (for collapsing h_0 and v_0 to an equivalent individual layer)

# AO configuration
config_ao = aspro.get_mode_config_ao(flag_mode)

# Loading TIPTOP fit and configuration
[config_Strehl, config_WFS_NGS, config_WFS_LGS] = aspro.get_mode_config(flag_mode)


##################################
# TESTING MARECHAL APPROXIMATION #
##################################

# Running Maréchal approximation
SR_Marechal = aspro.compute_Marechal(flag_mode, config_target, config_turbulence, config_ao, config_Strehl, config_WFS_NGS, config_WFS_LGS)
print('Strehl ratio: ', SR_Marechal)

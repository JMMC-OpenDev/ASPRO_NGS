package fr.jmmc.jmal.gpao;

/*
 * 'aspro' module defining different functions and methods
 * (Java translation of ASPRO_LGS/aspro.py)
 *
 * Created: 09/21/2023 (mm/dd/yyyy)
 * Author: Anthony Berdeu (LESIA - Observatoire de Paris)
 *
 * This code is part of a project that has received funding from the European Union's
 * Horizon 2020 research and innovation programme under grant agreement No 101004719.
 * License: GPL v3
 */

import java.util.Arrays;

/**
 * GPAO Strehl ratio estimation (NGS / LGS modes, VIS / IR) using the Maréchal approximation
 * calibrated with TIPTOP.
 * <p>
 * Modes: {@code NGS_VIS}, {@code NGS_IR}, {@code LGS_VIS}, {@code LGS_IR}.
 * <p>
 * All the code lives in this single class; the configuration containers are nested static classes
 * mirroring the Python dictionaries of the original module.
 */
public final class AsproLGS {

    /** Reference wavelength of the atmosphere (m) */
    public static final double WAVELENGTH_REF = 500e-9;

    private AsproLGS() {
        // utility class
    }

    // ------------------------------------------------------------------
    // Configuration containers (mirroring the Python dictionaries)
    // ------------------------------------------------------------------
    /**
     * Strehl damping parameters fitted with TIPTOP (config_Strehl).
     * Unused coefficient sets for a given mode are left {@code null}.
     */
    public static final class StrehlConfig {

        /** geometric error (fitting + aliasing) */
        public double[] geom;
        /** cone effect (LGS modes only) */
        public double[] cone;
        /** servo-lag error */
        public double[] lag;
        /** photon noise error (NGS modes only) */
        public double[] ph;
        /** readout noise error (NGS modes only) */
        public double[] ron;
        /** photon (index 0) and readout (index 1) noise of the LO channel (LGS modes only) */
        public double[] ph_ron_LO;
        /** photon (index 0) and readout (index 1) noise of the LGS channel (LGS modes only) */
        public double[] ph_ron_LGS;
        /** isoplanetic and isokinetic error */
        public double[] iso;

        @Override
        public String toString() {
            return "StrehlConfig{geom=" + Arrays.toString(geom)
                    + ", cone=" + Arrays.toString(cone)
                    + ", lag=" + Arrays.toString(lag)
                    + ", ph=" + Arrays.toString(ph)
                    + ", ron=" + Arrays.toString(ron)
                    + ", ph_ron_LO=" + Arrays.toString(ph_ron_LO)
                    + ", ph_ron_LGS=" + Arrays.toString(ph_ron_LGS)
                    + ", iso=" + Arrays.toString(iso) + '}';
        }
    }

    /**
     * Configuration of a SH wave-front sensor (config_WFS_NGS / config_WFS_LGS).
     * Fields not relevant to a given WFS are left as {@code NaN}.
     */
    public static final class WfsConfig {

        /** Telescope diameter (m) */
        public double D_tel = Double.NaN;
        /** Global transmission of the WFS channel (to compute the number of photons) */
        public double transmission = Double.NaN;
        /** Readout noise of the camera */
        public double sig_RON = Double.NaN;
        /** Excess noise factor (2 for EMCCDs) */
        public double ExcessNoiseFactor = Double.NaN;
        /** SH-WFS diameter (number of lenslets) */
        public double SH_diam = Double.NaN;
        /** Sensor pixel scale (arcsecond / pixel) */
        public double pixScale = Double.NaN;
        /** Number of pixels per lenslet (side of the lenslet box) */
        public double n_pix = Double.NaN;
        /** Lenslet diameter (m) */
        public double D_WFS = Double.NaN;
        /** Wavelength of the NGS channel (m) */
        public double wavelength = Double.NaN;
        /** Conversion magnitude to flux / Magnitude 0-point (ph/s/m2 for mag=0) */
        public double mag2flux = Double.NaN;
        /** Sodium layer height (m) (LGS WFS only) */
        public double h_LGS = Double.NaN;
        /** Number of photons per box (LGS WFS only) */
        public double n_ph = Double.NaN;

        @Override
        public String toString() {
            return "WfsConfig{D_tel=" + D_tel + ", transmission=" + transmission
                    + ", sig_RON=" + sig_RON + ", ExcessNoiseFactor=" + ExcessNoiseFactor
                    + ", SH_diam=" + SH_diam + ", pixScale=" + pixScale + ", n_pix=" + n_pix
                    + ", D_WFS=" + D_WFS + ", wavelength=" + wavelength + ", mag2flux=" + mag2flux
                    + ", h_LGS=" + h_LGS + ", n_ph=" + n_ph + '}';
        }
    }

    /** Result of {@link #getModeConfig(String)}: the three configurations of a GPAO mode. */
    public static final class ModeConfig {

        /** Strehl parameters fitted with TIPTOP */
        public final StrehlConfig configStrehl;
        /** configuration of the NGS WFS (HO in NGS modes / LO in LGS modes) */
        public final WfsConfig configWFS_NGS;
        /** configuration of the LGS WFS (LGS modes) */
        public final WfsConfig configWFS_LGS;

        ModeConfig(final StrehlConfig configStrehl, final WfsConfig configWFS_NGS, final WfsConfig configWFS_LGS) {
            this.configStrehl = configStrehl;
            this.configWFS_NGS = configWFS_NGS;
            this.configWFS_LGS = configWFS_LGS;
        }
    }

    /** Configuration of the target (config_target). */
    public static final class TargetConfig {

        /** Wavelength of the target (science or fringe tracker) channel (m) */
        public double wavelength;
        /** Pointing angle to zenith for the airmass (deg) */
        public double zenith;

        public TargetConfig() {
        }

        public TargetConfig(final double wavelength, final double zenith) {
            this.wavelength = wavelength;
            this.zenith = zenith;
        }
    }

    /** Configuration of the turbulence (config_turbulence). */
    public static final class TurbulenceConfig {

        /** Fried's parameter @500 nm (m) */
        public double r_0;
        /** Wind speed of the turbulence layers (m.s-1) (one value per layer) (for isoplanetism) */
        public double v_0;
        /** Altitude of the turbulent layers (m) (one value per layer) (for isoplanetism) */
        public double h_0;
        /** Cn2 weight (one value per layer) (for collapsing h_0 and v_0 to an equivalent individual layer) */
        public double Cn2;

        public TurbulenceConfig() {
        }

        public TurbulenceConfig(final double r_0, final double v_0, final double h_0, final double Cn2) {
            this.r_0 = r_0;
            this.v_0 = v_0;
            this.h_0 = h_0;
            this.Cn2 = Cn2;
        }

        /* derived */
        public double seeing;
        public double tau0;

        public void setTurbulenceConfig(final double seeing, final double tau0, final double h0) {
            this.seeing = seeing; // as
            this.tau0 = tau0; // ms

            this.h_0 = h0;
            this.Cn2 = 1.0;

            // seeing (as) gives r0:
            this.r_0 = (1.028993 * (0.5e-6 / this.seeing) / Math.PI * (180.0 * 3600.0)); // m

            // tau0 (+r0) gives v0:
            this.v_0 = (1000.0 * this.r_0 / this.tau0); //  # (m.s-1)

            // Derive seeing & tau0:
            this.seeing = (1.028993 * (0.5e-6 / this.r_0) / Math.PI * (180.0 * 3600.0)); // as
            this.tau0 = (1000.0 * this.r_0 / this.v_0); // (ms)
        }
    }

    /** Configuration of the AO system (config_ao). LGS-only fields may be left unset in NGS modes. */
    public static final class AoConfig {

        /** Magnitude of the NGS */
        public double magnitude_NGS;
        /** Loop frequency of the NGS loop (Hz) */
        public double f_loop_NGS;
        /** Loop frequency of the LGS loop (Hz) (for LGS modes only) */
        public double f_loop_LGS = Double.NaN;
        /** Loop gain of the NGS loop */
        public double g_loop_NGS;
        /** Loop gain of the LGS loop (for LGS modes only) */
        public double g_loop_LGS = Double.NaN;
        /** Number of corrected modes (to compute the equivalent DM number of actuators) */
        public double n_mode;
        /** Angle between the target (science or fringe tracker) and the NGS (arcsecond) */
        public double theta_NGS;
        /** Angle between the target (science or fringe tracker) and the LGS (arcsecond) (for LGS modes only) */
        public double theta_LGS = Double.NaN;
    }

    /** Result of {@link #modes2eqDM(double, double)}. */
    public static final class EqDM {

        /** the pitch of the equivalent DM (m) */
        public final double eqDM_pitch;
        /** the DM number of actuators of the equivalent DM (across the pupil) */
        public final long eqDMn_act;

        EqDM(final double eqDM_pitch, final long eqDMn_act) {
            this.eqDM_pitch = eqDM_pitch;
            this.eqDMn_act = eqDMn_act;
        }
    }

    // ------------------------------------------------------------------
    // LBO: AO configuration
    // ------------------------------------------------------------------
    /**
     * Return the default AOConfig    
     * @param flagMode 'NGS_VIS' / 'NGS_IR' / 'LGS_VIS' / 'LGS_IR'
     * @return default AOConfig    
     */
    public static AoConfig getAoConfig(final String flagMode) {
        final AoConfig configAo = new AoConfig();
        configAo.magnitude_NGS = 0.0;

        if (isNGS(flagMode)) {
            configAo.n_mode = 500.0;
            configAo.f_loop_NGS = 1000.0;
            configAo.g_loop_NGS = 0.5;
            configAo.f_loop_LGS = 1000.0;
            configAo.g_loop_LGS = 0.5;
        } else {
            configAo.magnitude_NGS = 0.0;
            configAo.n_mode = 500.0;
            configAo.f_loop_NGS = 500.0;
            configAo.g_loop_NGS = 0.3;
            configAo.f_loop_LGS = 1000.0;
            configAo.g_loop_LGS = 0.5;
        }
        configAo.theta_NGS = 0.0;
        configAo.theta_LGS = 0.0;

        return configAo;
    }

    // ------------------------------------------------------------------
    // GPAO modes
    // ------------------------------------------------------------------
    /**
     * Load the GPAO modes configuration based on the TIPTOP fit.
     *
     * @param flagMode 'NGS_VIS' / 'NGS_IR' / 'LGS_VIS' / 'LGS_IR'
     * @return the Strehl parameters fitted with TIPTOP, the configuration of the NGS WFS
     *         (HO in NGS modes / LO in LGS modes) and the configuration of the LGS WFS (LGS modes)
     * @throws IllegalArgumentException if the mode is unknown
     */
    public static ModeConfig getModeConfig(final String flagMode) {

        // ##### General parameters #####
        final double D_tel = 8.0; // Telescope diameter (m)
        final double transmission = 0.3; // Global transmission of the WFS channel (to compute the number of photons)
        final double sig_RON = 0.2; // Readout noise of the camera
        final double ExcessNoiseFactor = 2; // Excess noise factor

        // ##### Initialization #####
        final StrehlConfig configStrehl = new StrehlConfig();
        final WfsConfig configWFS_NGS = new WfsConfig();
        final WfsConfig configWFS_LGS = new WfsConfig();

        final boolean isVIS = isVIS(flagMode);
        final boolean isIR = isIR(flagMode);

        // config_WFS_NGS
        configWFS_NGS.D_tel = D_tel;
        configWFS_NGS.sig_RON = sig_RON;
        configWFS_NGS.ExcessNoiseFactor = ExcessNoiseFactor;
        configWFS_NGS.transmission = transmission;

        if ("NGS_VIS".equals(flagMode)) {
            configWFS_NGS.SH_diam = 40;
            configWFS_NGS.pixScale = 420.0 / 1000.0; // arcsecond
            configWFS_NGS.n_pix = 6;

        } else if ("LGS_VIS".equals(flagMode)) {
            configWFS_NGS.SH_diam = 4;
            configWFS_NGS.pixScale = 210.0 / 1000.0; // arcsecond
            configWFS_NGS.n_pix = 12;

        } else if (isIR) {
            configWFS_NGS.SH_diam = 9;
            configWFS_NGS.pixScale = 510.0 / 1000.0; // arcsecond
            configWFS_NGS.n_pix = 8;
        }

        // Lenslet diameter
        configWFS_NGS.D_WFS = D_tel / configWFS_NGS.SH_diam;

        // config_WFS_LGS
        configWFS_LGS.sig_RON = sig_RON;
        configWFS_LGS.ExcessNoiseFactor = ExcessNoiseFactor;

        configWFS_LGS.SH_diam = 9;
        configWFS_LGS.pixScale = 800.0 / 1000.0; // arcsecond
        configWFS_LGS.n_pix = 6;

        configWFS_LGS.h_LGS = 90000; // Sodium layer height (m)
        configWFS_LGS.n_ph = 50;

        // Lenslet diameter
        configWFS_LGS.D_WFS = D_tel / configWFS_LGS.SH_diam;

        // ##### Mode-dependent variables #####
        switch (flagMode) {
            case "NGS_VIS":
                configStrehl.geom = new double[]{0.267, 0.995};
                configStrehl.lag = new double[]{8.49, 2.157};
                configStrehl.ph = new double[]{11.973};
                configStrehl.ron = new double[]{0.52};
                configStrehl.iso = new double[]{4.337, 1.864};
                break;

            case "NGS_IR":
                configStrehl.geom = new double[]{0.244, 0.869};
                configStrehl.lag = new double[]{2.08, 2.101};
                configStrehl.ph = new double[]{15.179};
                configStrehl.ron = new double[]{1.653};
                configStrehl.iso = new double[]{1.75, 1.973};
                break;

            case "LGS_VIS":
                configStrehl.geom = new double[]{0.26, 1.014};
                configStrehl.cone = new double[]{0.722, 1.899};
                configStrehl.lag = new double[]{8.46, 0.413, 2.185};
                configStrehl.ph_ron_LO = new double[]{5.487e+01, -1.072e-04};
                configStrehl.ph_ron_LGS = new double[]{5.874, 0.159};
                configStrehl.iso = new double[]{4.326, 0.39, 1.985};
                break;

            case "LGS_IR":
                configStrehl.geom = new double[]{0.257, 0.995};
                configStrehl.cone = new double[]{0.683, 1.86};
                configStrehl.lag = new double[]{4.899, 0.857, 1.818};
                configStrehl.ph_ron_LO = new double[]{7.575e+00, -9.438e-05};
                configStrehl.ph_ron_LGS = new double[]{4.169, 0.35, 1.958};
                configStrehl.iso = new double[]{5.666, 0.154};
                break;

            default:
                throw new IllegalArgumentException(flagMode
                        + " -> Unknown mode (NGS_VIS / NGS_IR / LGS_VIS / LGS_IR)");
        }

        if (isVIS) {
            configWFS_NGS.wavelength = 750e-9;
            configWFS_NGS.mag2flux = 2.63e10;

        } else if (isIR) {
            configWFS_NGS.wavelength = 2.2e-6;
            configWFS_NGS.mag2flux = 1.66e9;

        } else {
            throw new IllegalArgumentException(flagMode + " -> Unknown mode (*_VIS / *_IR)");
        }

        return new ModeConfig(configStrehl, configWFS_NGS, configWFS_LGS);
    }

    /** @return true if flag_mode[0:3] == 'NGS' */
    public static boolean isNGS(final String flagMode) {
        return flagMode != null && flagMode.startsWith("NGS");
    }

    /** @return true if flag_mode[0:3] == 'LGS' */
    public static boolean isLGS(final String flagMode) {
        return flagMode != null && flagMode.startsWith("LGS");
    }

    /** @return true if flag_mode[4:7] == 'VIS' */
    public static boolean isVIS(final String flagMode) {
        return flagMode != null && flagMode.length() >= 7 && flagMode.startsWith("VIS", 4);
    }

    /** @return true if flag_mode[4:6] == 'IR' */
    public static boolean isIR(final String flagMode) {
        return flagMode != null && flagMode.length() >= 6 && flagMode.startsWith("IR", 4);
    }

    // ------------------------------------------------------------------
    // Strehl functions
    // ------------------------------------------------------------------
    /**
     * Geometric error = fitting + aliasing.
     *
     * @param coeff      damping factor (1 or 2 coefficients)
     * @param airmass    secant of the zenith angle (1/cos(zenith_angle))
     * @param DM_pitch   pitch of the DM on the pupil (m)
     * @param r_0        Fried's parameter @500nm (m)
     * @param wavelength wavelength at which the Strehl must be computed (m)
     * @return the Strehl ratio
     */
    public static double strehlGeom(final double[] coeff, final double airmass, final double DM_pitch,
                                    final double r_0, final double wavelength) {
        final double rl = Math.pow(airmass, -3.0 / 5.0) * r02rlambda(r_0, wavelength);
        switch (coeff.length) {
            case 1:
                return Math.exp(-coeff[0] * Math.pow(DM_pitch / rl, 5.0 / 3.0));
            case 2:
                return coeff[1] * Math.exp(-coeff[0] * Math.pow(DM_pitch, 5.0 / 3.0) / Math.pow(rl, 5.0 / 3.0));
            default:
                throw new IllegalArgumentException("Invalid number of coefficients!");
        }
    }

    /**
     * Servo-lag error.
     *
     * @param coeff      damping factor and the power value (1 or 2 coefficients)
     * @param airmass    secant of the zenith angle (1/cos(zenith_angle))
     * @param v_0        velocity of the turbulent layer (m.s-1)
     * @param r_0        Fried's parameter @500nm (m)
     * @param wavelength wavelength at which the Strehl must be computed (m)
     * @param f_loop     frequency of the loop (Hz)
     * @param g_loop     gain of the loop
     * @return the Strehl ratio
     */
    public static double strehlLag(final double[] coeff, final double airmass, final double v_0, final double r_0,
                                   final double wavelength, final double f_loop, final double g_loop) {
        final double x = v_0 / (Math.pow(airmass, -3.0 / 5.0) * r02rlambda(r_0, wavelength) * f_loop * g_loop);
        switch (coeff.length) {
            case 1:
                return Math.exp(-coeff[0] * Math.pow(x, 5.0 / 3.0));
            case 2:
                return Math.exp(-coeff[0] * Math.pow(x, coeff[1]));
            default:
                throw new IllegalArgumentException("Invalid number of coefficients!");
        }
    }

    /**
     * Servo-lag error for the LGS mode.
     *
     * @param coeff      damping factors (LGS, LO) and optionally the shared power value (2 or 3 coefficients)
     * @param airmass    secant of the zenith angle (1/cos(zenith_angle))
     * @param v_0        velocity of the turbulent layer (m.s-1)
     * @param r_0        Fried's parameter @500nm (m)
     * @param wavelength wavelength at which the Strehl must be computed (m)
     * @param f_loop_LGS frequency of the loop (Hz) for the LGS part
     * @param g_loop_LGS gain of the loop for the LGS part
     * @param f_loop_LO  frequency of the loop (Hz) for the LO part
     * @param g_loop_LO  gain of the loop for the LO part
     * @return the Strehl ratio
     */
    public static double strehlLagLGS(final double[] coeff, final double airmass, final double v_0, final double r_0,
                                      final double wavelength, final double f_loop_LGS, final double g_loop_LGS,
                                      final double f_loop_LO, final double g_loop_LO) {
        final double[] coefLGS;
        final double[] coefLO;
        switch (coeff.length) {
            case 2:
                coefLGS = new double[]{coeff[0]};
                coefLO = new double[]{coeff[1]};
                break;
            case 3:
                coefLGS = new double[]{coeff[0], coeff[2]};
                coefLO = new double[]{coeff[1], coeff[2]};
                break;
            default:
                throw new IllegalArgumentException("Invalid number of coefficients!");
        }
        return strehlLag(coefLGS, airmass, v_0, r_0, wavelength, f_loop_LGS, g_loop_LGS)
                * strehlLag(coefLO, airmass, v_0, r_0, wavelength, f_loop_LO, g_loop_LO);
    }

    /**
     * Photon noise error.
     *
     * @param coeff             damping factor (1 coefficient)
     * @param N_ph              number of photons
     * @param wavelength        wavelength at which the Strehl must be computed (m)
     * @param wavelength_eq     equivalent wavelength of the spot (diffraction limited = wavelength_AO / LGS = 1"/D_WFS_LGS) (m)
     * @param g_loop            gain of the loop
     * @param ExcessNoiseFactor Excess noise factor (2 for EMCCDs)
     * @return the Strehl ratio
     */
    public static double strehlPh(final double[] coeff, final double N_ph, final double wavelength,
                                  final double wavelength_eq, final double g_loop, final double ExcessNoiseFactor) {
        if (coeff.length == 1) {
            final double ratio = wavelength_eq / wavelength;
            return Math.exp(-coeff[0] * ratio * ratio * ExcessNoiseFactor * g_loop / (2.0 - g_loop) / N_ph);
        }
        throw new IllegalArgumentException("Invalid number of coefficients!");
    }

    /**
     * Readout noise error.
     *
     * @param coeff    damping factor (1 coefficient)
     * @param sigRON   single pixel readout noise
     * @param N_ph     number of photons
     * @param pixScale pixel scale (arcsecond)
     * @param N_pix    number of pixel (side of the lenslet box)
     * @param g_loop   gain of the loop
     * @return the Strehl ratio
     */
    public static double strehlRon(final double[] coeff, final double sigRON, final double N_ph,
                                   final double pixScale, final double N_pix, final double g_loop) {
        if (coeff.length == 1) {
            return Math.exp(-coeff[0] * pixScale * pixScale * Math.pow(N_pix, 4) * sigRON * sigRON
                    * g_loop / (2.0 - g_loop) / (N_ph * N_ph));
        }
        throw new IllegalArgumentException("Invalid number of coefficients!");
    }

    /**
     * Isoplanetic and isokinetic error.
     *
     * @param coeff      damping factor and the power value (1 or 2 coefficients)
     * @param airmass    secant of the zenith angle (1/cos(zenith_angle))
     * @param theta      separation (arcsecond)
     * @param h_0        altitude of the turbulent layer (m)
     * @param r_0        Fried's parameter @500nm (m)
     * @param wavelength wavelength at which the Strehl must be computed (m)
     * @return the Strehl ratio
     */
    public static double strehlIso(final double[] coeff, final double airmass, final double theta, final double h_0,
                                   final double r_0, final double wavelength) {
        final double x = theta * Math.PI / 180.0 / 3600.0 * airmass * h_0
                / (Math.pow(airmass, -3.0 / 5.0) * r02rlambda(r_0, wavelength));
        switch (coeff.length) {
            case 1:
                return Math.exp(-coeff[0] * Math.pow(x, 5.0 / 3.0));
            case 2:
                return Math.exp(-coeff[0] * Math.pow(x, coeff[1]));
            default:
                throw new IllegalArgumentException("Invalid number of coefficients!");
        }
    }

    /**
     * Isoplanetic and isokinetic error for the LGS mode.
     *
     * @param coeff      damping factors (LGS, LO) and optionally the shared power value (2 or 3 coefficients)
     * @param airmass    secant of the zenith angle (1/cos(zenith_angle))
     * @param theta_LGS  separation (arcsecond) of the LGS source
     * @param theta_LO   separation (arcsecond) of the LO source
     * @param h_0        altitude of the turbulent layer (m)
     * @param r_0        Fried's parameter @500nm (m)
     * @param wavelength wavelength at which the Strehl must be computed (m)
     * @return the Strehl ratio
     */
    public static double strehlIsoLGS(final double[] coeff, final double airmass, final double theta_LGS,
                                      final double theta_LO, final double h_0, final double r_0,
                                      final double wavelength) {
        final double[] coefLGS;
        final double[] coefLO;
        switch (coeff.length) {
            case 2:
                coefLGS = new double[]{coeff[0]};
                coefLO = new double[]{coeff[1]};
                break;
            case 3:
                coefLGS = new double[]{coeff[0], coeff[2]};
                coefLO = new double[]{coeff[1], coeff[2]};
                break;
            default:
                throw new IllegalArgumentException("Invalid number of coefficients!");
        }
        return strehlIso(coefLGS, airmass, theta_LGS, h_0, r_0, wavelength)
                * strehlIso(coefLO, airmass, theta_LO, h_0, r_0, wavelength);
    }

    /**
     * Cone effect.
     *
     * @param coeff      damping factor and optionally the power value (1 or 2 coefficients)
     * @param airmass    secant of the zenith angle (1/cos(zenith_angle))
     * @param h_0        altitude of the turbulent layer (m)
     * @param h_lgs      altitude of the sodium layer (m)
     * @param D_tel      diameter of the telescope (m)
     * @param r_0        Fried's parameter @500nm (m)
     * @param wavelength wavelength at which the Strehl must be computed (m)
     * @return the Strehl ratio
     */
    public static double strehlCone(final double[] coeff, final double airmass, final double h_0, final double h_lgs,
                                    final double D_tel, final double r_0, final double wavelength) {
        final double beta;
        switch (coeff.length) {
            case 1:
                beta = 5.0 / 3.0;
                break;
            case 2:
                beta = coeff[1];
                break;
            default:
                throw new IllegalArgumentException("Invalid number of coefficients!");
        }
        final double x = D_tel / (Math.pow(airmass, -3.0 / 5.0) * r02rlambda(r_0, wavelength)) * h_0 / h_lgs;
        return Math.exp(-coeff[0] * Math.pow(x, beta));
    }

    /**
     * Compute the Strehl ratio with the calibrated Maréchal approximation calibrated with TIPTOP.
     *
     * @param flagMode         'NGS_VIS' / 'NGS_IR' / 'LGS_VIS' / 'LGS_IR'
     * @param configTarget     configuration of the target
     * @param configTurbulence configuration of the turbulence
     * @param configAo         configuration of the AO system
     * @param modeConfig       Strehl parameters fitted with TIPTOP and WFS configurations (see {@link #getModeConfig(String)})
     * @return the global Strehl ratio
     */
    public static double computeMarechal(final String flagMode, final TargetConfig configTarget,
                                         final TurbulenceConfig configTurbulence, final AoConfig configAo,
                                         final ModeConfig modeConfig) {
        return computeMarechal(flagMode, configTarget, configTurbulence, configAo,
                modeConfig.configStrehl, modeConfig.configWFS_NGS, modeConfig.configWFS_LGS);
    }

    /**
     * Compute the Strehl ratio with the calibrated Maréchal approximation calibrated with TIPTOP.
     *
     * @param flagMode         'NGS_VIS' / 'NGS_IR' / 'LGS_VIS' / 'LGS_IR'
     * @param configTarget     configuration of the target
     * @param configTurbulence configuration of the turbulence
     * @param configAo         configuration of the AO system
     * @param configStrehl     Strehl parameters fitted with TIPTOP
     * @param configWFS_NGS    configuration of the NGS WFS (HO in NGS modes / LO in LGS modes)
     * @param configWFS_LGS    configuration of the LGS WFS (LGS modes) (may be null in NGS modes)
     * @return the global Strehl ratio
     * @throws IllegalArgumentException if the mode is unknown
     */
    public static double computeMarechal(final String flagMode, final TargetConfig configTarget,
                                         final TurbulenceConfig configTurbulence, final AoConfig configAo,
                                         final StrehlConfig configStrehl, final WfsConfig configWFS_NGS,
                                         final WfsConfig configWFS_LGS) {
        // ##### Loading configuration #####
        // Loading target
        final double wavelength_target = configTarget.wavelength;
        final double zenith_angle = configTarget.zenith;
        final double airmass = 1.0 / Math.cos(Math.toRadians(zenith_angle));

        // Loading atmosphere
        final double r_0 = configTurbulence.r_0;
        final double Cn2 = configTurbulence.Cn2;
        final double h_0 = cn2WeightedLayer(Cn2, configTurbulence.h_0, false);
        final double v_0 = cn2WeightedLayer(Cn2, configTurbulence.v_0, true);

        // Loading AO system
        final EqDM eqDM = modes2eqDM(configAo.n_mode, configWFS_NGS.D_tel);
        final double eqDM_pitch = eqDM.eqDM_pitch;
        final double f_loop_NGS = configAo.f_loop_NGS;
        final double g_loop_NGS = configAo.g_loop_NGS;
        final double n_ph_NGS = mag2nph(configAo.magnitude_NGS, configWFS_NGS.mag2flux, configWFS_NGS.transmission,
                configWFS_NGS.D_WFS, f_loop_NGS);

        if (isNGS(flagMode)) {
            // ##### Loading Strehl damping coefficient #####
            final double[] coeff_geom = configStrehl.geom;
            final double[] coeff_lag = configStrehl.lag;
            final double[] coeff_ph = configStrehl.ph;
            final double[] coeff_ron = configStrehl.ron;
            final double[] coeff_iso = configStrehl.iso;

            // ##### Computing individual Strehl contributions #####
            final double SR_geom = strehlGeom(coeff_geom, airmass, eqDM_pitch, r_0, wavelength_target);
            final double SR_lag = strehlLag(coeff_lag, airmass, v_0, r_0, wavelength_target, f_loop_NGS, g_loop_NGS);
            final double SR_ph = strehlPh(coeff_ph, n_ph_NGS, wavelength_target, configWFS_NGS.wavelength,
                    g_loop_NGS, configWFS_NGS.ExcessNoiseFactor);
            final double SR_ron = strehlRon(coeff_ron, configWFS_NGS.sig_RON, n_ph_NGS, configWFS_NGS.pixScale,
                    configWFS_NGS.n_pix, g_loop_NGS);
            final double SR_iso = strehlIso(coeff_iso, airmass, Math.abs(configAo.theta_NGS), h_0, r_0,
                    wavelength_target);

            // ##### Output #####
            return SR_geom * SR_lag * SR_ph * SR_ron * SR_iso;

        } else if (isLGS(flagMode)) {
            // ##### Loading Strehl damping coefficient #####
            final double[] coeff_geom = configStrehl.geom;
            final double[] coeff_cone = configStrehl.cone;
            final double[] coeff_lag = configStrehl.lag;
            final double[] coeff_ph_ron_LO = configStrehl.ph_ron_LO;
            final double[] coeff_ph_ron_LGS = configStrehl.ph_ron_LGS;
            final double[] coeff_iso = configStrehl.iso;

            // ##### Computing individual Strehl contributions #####
            final double g_loop_LGS = configAo.g_loop_LGS;
            final double n_ph_LGS = configWFS_LGS.n_ph;
            final double SR_geom = strehlGeom(coeff_geom, airmass, eqDM_pitch, r_0, wavelength_target);
            final double SR_cone = strehlCone(coeff_cone, airmass, h_0, configWFS_LGS.h_LGS, configWFS_NGS.D_tel,
                    r_0, wavelength_target);
            final double SR_lag = strehlLagLGS(coeff_lag, airmass, v_0, r_0, wavelength_target,
                    configAo.f_loop_LGS, g_loop_LGS, f_loop_NGS, g_loop_NGS);
            final double SR_ph
                         = strehlPh(new double[]{coeff_ph_ron_LGS[0]}, n_ph_LGS, wavelength_target,
                    arcsecond2rad(1.0) * configWFS_LGS.D_WFS, g_loop_LGS, configWFS_LGS.ExcessNoiseFactor)
                    * strehlPh(new double[]{coeff_ph_ron_LO[0]}, n_ph_NGS, wavelength_target,
                    configWFS_NGS.wavelength, g_loop_NGS, configWFS_NGS.ExcessNoiseFactor);
            final double SR_ron
                         = strehlRon(new double[]{coeff_ph_ron_LGS[1]}, configWFS_LGS.sig_RON, n_ph_LGS,
                    configWFS_LGS.pixScale, configWFS_LGS.n_pix, g_loop_LGS)
                    * strehlRon(new double[]{coeff_ph_ron_LO[1]}, configWFS_NGS.sig_RON, n_ph_NGS,
                    configWFS_NGS.pixScale, configWFS_NGS.n_pix, g_loop_NGS);
            final double SR_iso = strehlIsoLGS(coeff_iso, airmass, Math.abs(configAo.theta_LGS),
                    Math.abs(configAo.theta_NGS), h_0, r_0, wavelength_target);

            // ##### Output #####
            return SR_geom * SR_cone * SR_lag * SR_ph * SR_ron * SR_iso;

        } else {
            throw new IllegalArgumentException(flagMode + " -> Unknown mode (NGS* / LGS*)");
        }
    }

    /**
     * Collapse per-layer values into an equivalent single layer using the Cn2 weights:
     * {@code (sum(Cn2 * |x|^(5/3)) / sum(Cn2))^(3/5)}.
     *
     * @param Cn2    Cn2 weights (one per layer)
     * @param values per-layer values (altitudes or wind speeds)
     * @param useAbs if true, the absolute value of each layer value is used
     * @return the equivalent single-layer value
     */
    private static double cn2WeightedLayer(final double Cn2, final double values, final boolean useAbs) {
        double num = 0.0;
        double den = 0.0;
//        for (int i = 0; i < Cn2.length; i++) {
        final double v = useAbs ? Math.abs(values/*[i]*/) : values/*[i]*/;
        num += Cn2/*[i]*/ * Math.pow(v, 5.0 / 3.0);
        den += Cn2/*[i]*/;
        //      }
        return Math.pow(num / den, 3.0 / 5.0);
    }

    // ------------------------------------------------------------------
    // Miscellaneous
    // ------------------------------------------------------------------
    /**
     * Convert arcsecond in radians.
     *
     * @param arcsec angle (arcsecond)
     * @return angle (radian)
     */
    public static double arcsecond2rad(final double arcsec) {
        return arcsec / 3600.0 * Math.PI / 180.0;
    }

    /**
     * Convert a magnitude into a number of photons per subaperture.
     *
     * @param magnitude    the magnitude
     * @param mag2flux     the zero point magnitude -> flux (ph/s/m2 for mag=0)
     * @param transmission the transmission
     * @param D_WFS        the diameter of the WFS lenslet (m)
     * @param f_loop       the loop frequency (Hz)
     * @return the number of photons per subaperture
     */
    public static double mag2nph(final double magnitude, final double mag2flux, final double transmission,
                                 final double D_WFS, final double f_loop) {
        return transmission * D_WFS * D_WFS * mag2flux * Math.pow(10.0, -magnitude / 2.5) / f_loop;
    }

    /**
     * Convert the number of modes into an equivalent DM.
     *
     * @param n_mode_AO number of corrected modes by the AO system
     * @param D_tel     telescope diameter (m)
     * @return the pitch of the equivalent DM and the number of actuators of the equivalent DM
     */
    public static EqDM modes2eqDM(final double n_mode_AO, final double D_tel) {
        // Equivalent number of actuators across the pupil
        final double eqDMn_act = 2.0 * Math.sqrt(n_mode_AO / Math.PI);

        // Equivalent actuator pitch (-1 actuator)
        final double eqDM_pitch = D_tel / (eqDMn_act - 1.0);

        // Rounding the number of actuator to get an integer (Math.rint = round half to even, as Python's round)
        return new EqDM(eqDM_pitch, (long) Math.rint(eqDMn_act));
    }

    /**
     * Convert the Fried parameter 'r_0' (m) according to the 'wavelength' (m).
     * Note: The reference wavelength of the atmosphere is 500nm.
     *
     * @param r_0        Fried's parameter @500nm (m)
     * @param wavelength wavelength (m)
     * @return Fried's parameter at the given wavelength (m)
     */
    public static double r02rlambda(final double r_0, final double wavelength) {
        return r_0 * Math.pow(wavelength / WAVELENGTH_REF, 6.0 / 5.0);
    }

    /**
     * Convert the Fried parameter r_0 (m) to the equivalent seeing (arcsecond).
     * Note: The reference wavelength of the atmosphere is 500nm.
     *
     * @param r_0 Fried's parameter @500nm (m)
     * @return seeing (arcsecond)
     */
    public static double r02seeing(final double r_0) {
        return 0.98 * WAVELENGTH_REF / r_0 * 180.0 / Math.PI * 3600.0;
    }

    /**
     * Convert the seeing (arcsecond) to the equivalent Fried parameter r_0 (m).
     * Note: The reference wavelength of the atmosphere is 500nm.
     *
     * @param seeing seeing (arcsecond)
     * @return Fried's parameter @500nm (m)
     */
    public static double seeing2r0(final double seeing) {
        return r02seeing(seeing);
    }

    // ------------------------------------------------------------------
    // Example (single Strehl, see examples_ASPRO.ipynb)
    // ------------------------------------------------------------------
    /**
     * Example on a single Strehl: reproduces the first example of examples_ASPRO.ipynb.
     *
     * @param args unused
     */
    public static void main(final String[] args) {
        // Target
        final String flagMode = "LGS_VIS";
        final TargetConfig configTarget = new TargetConfig(2.2e-06, 0.0);

        // Turbulence
        final TurbulenceConfig configTurbulence = new TurbulenceConfig(0.100, 15.0, 1500.0, 1.0);

        // Loading TIPTOP fit and configuration
        final ModeConfig modeConfig = getModeConfig(flagMode);

        // AO configuration
        final AoConfig configAo = getAoConfig(flagMode);
        configAo.magnitude_NGS = 8;

        // Running Maréchal approximation
        final double SR = computeMarechal(flagMode, configTarget, configTurbulence, configAo, modeConfig);
        System.out.println("Strehl ratio: " + SR);
    }
}

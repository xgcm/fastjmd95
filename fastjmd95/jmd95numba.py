#
# created by mlosch on 2002-08-09
# converted to python by jahn on 2010-04-29

import sys
import numpy as np

from numba import vectorize, float64, float32

# coefficients nonlinear equation of state in pressure coordinates for
# 1. density of fresh water at p = 0
# pop: unt0-unt5
eosJMDCFw = np.array(
    [999.842594, 6.793952e-02, -9.095290e-03, 1.001685e-04, -1.120083e-06, 6.536332e-09]
)
# 2. density of sea water at p = 0
# pop: uns1t0-uns1t4, unsqt0-unsqt2, uns2t0
eosJMDCSw = np.array(
    [
        8.244930e-01,
        -4.089900e-03,
        7.643800e-05,
        -8.246700e-07,
        5.387500e-09,
        -5.724660e-03,
        1.022700e-04,
        -1.654600e-06,
        4.831400e-04,
    ]
)
# coefficients in pressure coordinates for
# 3. secant bulk modulus K of fresh water at p = 0
# pop: bup0s0t0-bup0s0t4
eosJMDCKFw = np.array(
    [1.965933e04, 1.444304e02, -1.706103e00, 9.648704e-03, -4.190253e-05]
)
# 4. secant bulk modulus K of sea water at p = 0
# pop: bup0s1t0-bup0s1t3, bup0sqt0-bup0sqt2
eosJMDCKSw = np.array(
    [
        5.284855e01,
        -3.101089e-01,
        6.283263e-03,
        -5.084188e-05,
        3.886640e-01,
        9.085835e-03,
        -4.619924e-04,
    ]
)
# 5. secant bulk modulus K of sea water at p
# pop: bup1s0t0-bup1s0t3, bup1s1t0-bup1s1t2, bup1sqt0, bup2s0t0-bup2s0t2, bup2s1t0-bup2s1t2
eosJMDCKP = np.array(
    [
        3.186519e00,
        2.212276e-02,
        -2.984642e-04,
        1.956415e-06,
        6.704388e-03,
        -1.847318e-04,
        2.059331e-07,
        1.480266e-04,
        2.102898e-04,
        -1.202016e-05,
        1.394680e-07,
        -2.040237e-06,
        6.128773e-08,
        6.207323e-10,
    ]
)


@vectorize(
    [float64(float64, float64, float64), float32(float32, float32, float32)],
    nopython=True,
)
def _bulkmodjmd95(s, t, p):
    """ Compute bulk modulus
    """

    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t

    s3o2 = s * np.sqrt(s)

    # p = pressure(i,j,k,bi,bj)*SItoBar
    p2 = p * p
    # secant bulk modulus of fresh water at the surface
    bulkmod = (
        eosJMDCKFw[0]
        + eosJMDCKFw[1] * t
        + eosJMDCKFw[2] * t2
        + eosJMDCKFw[3] * t3
        + eosJMDCKFw[4] * t4
    )
    # secant bulk modulus of sea water at the surface
    bulkmod = (
        bulkmod
        + s
        * (eosJMDCKSw[0] + eosJMDCKSw[1] * t + eosJMDCKSw[2] * t2 + eosJMDCKSw[3] * t3)
        + s3o2 * (eosJMDCKSw[4] + eosJMDCKSw[5] * t + eosJMDCKSw[6] * t2)
    )
    # secant bulk modulus of sea water at pressure p
    bulkmod = (
        bulkmod
        + p * (eosJMDCKP[0] + eosJMDCKP[1] * t + eosJMDCKP[2] * t2 + eosJMDCKP[3] * t3)
        + p * s * (eosJMDCKP[4] + eosJMDCKP[5] * t + eosJMDCKP[6] * t2)
        + p * s3o2 * eosJMDCKP[7]
        + p2 * (eosJMDCKP[8] + eosJMDCKP[9] * t + eosJMDCKP[10] * t2)
        + p2 * s * (eosJMDCKP[11] + eosJMDCKP[12] * t + eosJMDCKP[13] * t2)
    )

    return bulkmod


@vectorize([float64(float64, float64), float32(float32, float32)], nopython=True)
def _rho_s(s, t):

    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    s3o2 = s * np.sqrt(s)

    # density of freshwater at the surface
    rho_fw = (
        eosJMDCFw[0]
        + eosJMDCFw[1] * t
        + eosJMDCFw[2] * t2
        + eosJMDCFw[3] * t3
        + eosJMDCFw[4] * t4
        + eosJMDCFw[5] * t4 * t
    )
    # density of sea water at the surface
    rho_s = (
        rho_fw
        + s
        * (
            eosJMDCSw[0]
            + eosJMDCSw[1] * t
            + eosJMDCSw[2] * t2
            + eosJMDCSw[3] * t3
            + eosJMDCSw[4] * t4
        )
        + s3o2 * (eosJMDCSw[5] + eosJMDCSw[6] * t + eosJMDCSw[7] * t2)
        + eosJMDCSw[8] * s * s
    )
    return rho_s


@vectorize(
    [float64(float64, float64, float64), float32(float32, float32, float32)],
    nopython=True,
)
def rho(s, t, p):
    """
    Computes in-situ density of sea water using Jackett and McDougall 1995
    polynomial [1]_.

    Parameters
    ----------
    s : array_like
        practical salinity [psu (PSS-78)]
    t : array_like
        potential temperature [degree C (IPTS-68)];
        same shape as s
    p : array_like
        pressure [dbar]; broadcastable to shape of s

    Returns
    -------
    dens : array
        density [kg/m^3]

    Example
    -------
    >>> rho(35.5, 3., 3000.)
    1041.83267

    Notes
    -----
    Adopted from `MITgcm python utils <https://github.com/MITgcm/MITgcm/blob/master/utils/python/MITgcmutils/MITgcmutils/jmd95.py>`_.

    .. [1] Jackett, D.R. and T.J. Mcdougall, 1995: Minimal Adjustment of
    Hydrographic Profiles to Achieve Static Stability. J. Atmos. Oceanic
    Technol., 12, 381–389, https://doi.org/10.1175/1520-0426(1995)012<0381:MAOHPT>2.0.CO;2
    """

    # convert pressure to bar
    p = 0.1 * p

    # density of freshwater at the surface
    rho_s = _rho_s(s, t)

    bulk_mod = _bulkmodjmd95(s, t, p)
    # the pop formumlation
    # rho = rho_s * bulk_mod * denomk
    # original formulation
    rho = rho_s / (1.0 - p / bulk_mod)
    return rho


@vectorize(
    [float64(float64, float64, float64), float32(float32, float32, float32)],
    nopython=True,
)
def drhodt(s, t, p):
    """
    Computes partial derivative of density with respect to potential temperature
    using Jackett and McDougall 1995 polynomial [1]_.

    Parameters
    ----------
    s : array_like
        practical salinity [psu (PSS-78)]
    t : array_like
        potential temperature [degree C (IPTS-68)];
        same shape as s
    p : array_like
        pressure [dbar]; broadcastable to shape of s

    Returns
    -------
    drhods : array
        partial derivative of density with respect to potential temperature
        [kg/m^3/deg C]

    Example
    -------
    >>> drhodt(35.5, 3., 3000.)
    -0.17244

    Notes
    -----
    Adopted from `MITgcm python utils <https://github.com/MITgcm/MITgcm/blob/master/utils/python/MITgcmutils/MITgcmutils/jmd95.py>`_.

    .. [1] Jackett, D.R. and T.J. Mcdougall, 1995: Minimal Adjustment of
    Hydrographic Profiles to Achieve Static Stability. J. Atmos. Oceanic
    Technol., 12, 381–389, https://doi.org/10.1175/1520-0426(1995)012<0381:MAOHPT>2.0.CO;2
    """
    p = 0.1 * p
    p2 = p * p
    t2 = t * t

    # thermal expansion
    sqr = np.sqrt(s)

    DRDT0 = (
        eosJMDCFw[1]
        + 2 * eosJMDCFw[2] * t
        + (3 * eosJMDCFw[3] + 4 * eosJMDCFw[4] * t + 5 * eosJMDCFw[5] * t2) * t2
        + (
            eosJMDCSw[1]
            + 2 * eosJMDCSw[2] * t
            + (3 * eosJMDCSw[3] + 4 * eosJMDCSw[4] * t) * t2
            + (eosJMDCSw[6] + 2 * eosJMDCSw[7] * t) * sqr
        )
        * s
    )
    DKDT = (
        eosJMDCKFw[1]
        + 2 * eosJMDCKFw[2] * t
        + (3 * eosJMDCKFw[3] + 4 * eosJMDCKFw[4] * t) * t2
        + p * (eosJMDCKP[1] + 2 * eosJMDCKP[2] * t + 3 * eosJMDCKP[3] * t)
        + p2 * (eosJMDCKP[9] + 2 * eosJMDCKP[10] * t)
        + s
        * (
            eosJMDCKSw[1]
            + 2 * eosJMDCKSw[2] * t
            + 3 * eosJMDCKSw[3] * t2
            + p * (eosJMDCKP[5] + 2 * eosJMDCKP[6] * t)
            + p2 * (eosJMDCKP[12] + 2 * eosJMDCKP[13] * t)
            + sqr * (eosJMDCKSw[5] + 2 * eosJMDCKSw[6] * t)
        )
    )

    rho_s = _rho_s(s, t)
    bulk_mod = _bulkmodjmd95(s, t, p)
    denomk = 1.0 / (bulk_mod - p)
    DRHODT = denomk * (DRDT0 * bulk_mod - p * rho_s * DKDT * denomk)
    return DRHODT
    # return rho, DRHODT


@vectorize(
    [float64(float64, float64, float64), float32(float32, float32, float32)],
    nopython=True,
)
def drhods(s, t, p):
    """
    Computes partial derivative of density with respect to practical salinity
    using Jackett and McDougall 1995 polynomial [1]_.

    Parameters
    ----------
    s : array_like
        practical salinity [psu (PSS-78)]
    theta : array_like
        potential temperature [degree C (IPTS-68)];
        same shape as s
    p : array_like
        pressure [dbar]; broadcastable to shape of s

    Returns
    -------
    drhods : array
        partial derivative of density with respect to practical salinity
        [kg/m^3/psu]

    Example
    -------
    >>> drhods(35.5, 3., 3000.)
    0.77481

    Notes
    -----
    Adopted from `MITgcm python utils <https://github.com/MITgcm/MITgcm/blob/master/utils/python/MITgcmutils/MITgcmutils/jmd95.py>`_.

    .. [1] Jackett, D.R. and T.J. Mcdougall, 1995: Minimal Adjustment of
    Hydrographic Profiles to Achieve Static Stability. J. Atmos. Oceanic
    Technol., 12, 381–389, https://doi.org/10.1175/1520-0426(1995)012<0381:MAOHPT>2.0.CO;2
    """

    p = 0.1 * p
    p2 = p * p
    t2 = t * t
    t3 = t2 * t

    # thermal expansion
    sqr = np.sqrt(s)

    work1 = (
        eosJMDCSw[0]
        + eosJMDCSw[1] * t
        + (eosJMDCSw[2] + eosJMDCSw[3] * t + eosJMDCSw[4] * t2) * t2
    )
    work2 = sqr * (eosJMDCSw[5] + eosJMDCSw[6] * t + eosJMDCSw[7] * t2)
    work3 = (
        eosJMDCKSw[0]
        + eosJMDCKSw[1] * t
        + (eosJMDCKSw[2] + eosJMDCKSw[3] * t) * t2
        + p * (eosJMDCKP[4] + eosJMDCKP[5] * t + eosJMDCKP[6] * t2)
        + p2 * (eosJMDCKP[11] + eosJMDCKP[12] * t + eosJMDCKP[13] * t3)
    )
    work4 = sqr * (
        eosJMDCKSw[4] + eosJMDCKSw[5] * t + eosJMDCKSw[6] * t2 + eosJMDCKP[7] * p
    )

    # didn't work for some reason
    #     bulk_mod = (
    #         eosJMDCKFw[9]
    #         + eosJMDCKFw[1] * t
    #         + (eosJMDCKFw[2] + eosJMDCKFw[3] * t + eosJMDCKFw[4] * t2) * t2
    #         + p * (eosJMDCKP[0] + eosJMDCKP[1] * t + (eosJMDCKP[2] + eosJMDCKP[3] * t) * t2)
    #         + p2 * (eosJMDCKP[8] + eosJMDCKP[9] * t + eosJMDCKP[10] * t2)
    #         + s * (work3 + work4)
    #     )

    bulk_mod = _bulkmodjmd95(s, t, p)
    denomk = 1.0 / (bulk_mod - p)

    drds0 = 2 * eosJMDCSw[8] * s + work1 + 1.5 * work2
    dkds = work3 + 1.5 * work4
    rho_s = _rho_s(s, t)
    drhods = denomk * (drds0 * bulk_mod - p * rho_s * dkds * denomk)
    return drhods
    # return bulk_mod

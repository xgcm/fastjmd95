import numpy as np
from itertools import product
import pytest

from fastjmd95 import rho, drhodt, drhods
from .reference_values import rho_expected, drhodt_expected, drhods_expected


@pytest.fixture
def s_t_p():
    s0 = np.arange(30, 41, 2.0)
    t0 = np.arange(-2, 35, 2.0)
    p0 = np.arange(0, 5000.0, 1000.0)
    p, t, s = np.array(list(product(p0, t0, s0))).transpose()
    return s, t, p


def test_rho(s_t_p):
    s, t, p = s_t_p
    rho_actual = rho(s, t, p)
    np.testing.assert_allclose(rho_actual, rho_expected)


def test_drhot(s_t_p):
    s, t, p = s_t_p
    drhodt_actual = drhodt(s, t, p)
    np.testing.assert_allclose(drhodt_actual, drhodt_expected, rtol=1e-2)


def test_drhos(s_t_p):
    s, t, p = s_t_p
    drhods_actual = drhods(s, t, p)
    np.testing.assert_allclose(drhods_actual, drhods_expected, rtol=1e-2)

import numpy as np
from itertools import product
import pytest

from fastjmd95 import rho, drhodt, drhods
from .reference_values import rho_expected, drhodt_expected, drhods_expected

import dask
import dask.array
import xarray as xr

@pytest.fixture
def s_t_p():
    s0 = np.arange(30, 41, 2.0)
    t0 = np.arange(-2, 35, 2.0)
    p0 = np.arange(0, 5000.0, 1000.0)
    p, t, s = np.array(list(product(p0, t0, s0))).transpose()
    return s, t, p

def _chunk(*args):
    return [dask.array.from_array(a, chunks=(100,)) for a in args]

def _make_xarray(*args,withdask=False):
    if withdask==True:
        args = _chunk(*args)
    return [xr.DataArray(a, coords=[np.arange(0,len(a))], dims=["i"]) for a in args]

@pytest.fixture
def no_client():
    return None


@pytest.fixture
def threaded_client():
    with dask.config.set(scheduler='threads'):
        yield


@pytest.fixture
def processes_client():
    with dask.config.set(scheduler='processes'):
        yield


@pytest.fixture(scope='module')
def distributed_client():
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(threads_per_worker=1,
                           n_workers=2,
                           processes=True)
    client = Client(cluster)
    yield
    client.close()
    del client
    cluster.close()
    del cluster


all_clients = ['no_client', 'threaded_client', 'processes_client', 'distributed_client']
all_arrays = ['dask_arrays', 'xarrays']
# https://stackoverflow.com/questions/45225950/passing-yield-fixtures-as-test-parameters-with-a-temp-directory
@pytest.mark.parametrize('client', all_clients)
@pytest.mark.parametrize('array_type', all_arrays)
@pytest.mark.parametrize('function,expected',
                         [(rho, rho_expected),
                         (drhodt, drhodt_expected),
                         (drhods, drhods_expected)])
def test_functions(request, client, array_type, s_t_p, function, expected):
    s, t, p = s_t_p
    if array_type == 'dask_arrays':
        if client != 'no_client':
            s, t, p = _chunk(s, t, p)
    elif array_type == 'xarrays':
        if client != 'no_client':
            s, t, p = _make_xarray(s, t, p, withdask=True)
        else:
            s, t, p = _make_xarray(s, t, p, withdask=False)
    client = request.getfixturevalue(client)
    actual = function(s, t, p)
    np.testing.assert_allclose(actual, expected, rtol=1e-2)

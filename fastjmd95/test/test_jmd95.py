import numpy as np
from itertools import product
import pytest

from fastjmd95 import rho, drhodt, drhods
from .reference_values import rho_expected, drhodt_expected, drhods_expected

import dask
import dask.array

@pytest.fixture
def s_t_p():
    s0 = np.arange(30, 41, 2.0)
    t0 = np.arange(-2, 35, 2.0)
    p0 = np.arange(0, 5000.0, 1000.0)
    p, t, s = np.array(list(product(p0, t0, s0))).transpose()
    return s, t, p

def _chunk(*args):
    return [dask.array.from_array(a, chunks=(5,)) for a in args]


def threaded_client():
    with dask.config.set(scheduler='threads'):
        print("yeild from threaded_client")
        yield
    print("back to threaded_client")


def processes_client():
    with dask.config.set(scheduler='processes'):
        print("yeild from processes_client")
        yield
    print("back to processes_client")


def distributed_client():
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=2)
    client = Client(cluster)
    print('yielding from distributed_client')
    yield
    print('back to distributed_client')
    client.close()
    del client
    cluster.close()
    del cluster
    print('Shut down cluster')


@pytest.fixture(scope="function",
                params=[None, "threaded", "processes", "distributed"])
    def client(request):
        if request.param is None:
            yield
        elif request.param == "threaded"
            with threaded_client():
                yield
        elif request.param == "processes"
            with processes_client():
                yield
        elif request.param == "distributed"
            with distributed_client():
                yield

#all_clients = [None, threaded_client, processes_client, distributed_client]
#@pytest.mark.parametrize('client', all_clients)
def test_rho(client, s_t_p):
    s, t, p = s_t_p
    if client:
        print(client)
        s, t, p = _chunk(s, t, p)
        print("chunking")
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

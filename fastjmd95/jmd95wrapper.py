
import numpy as np
import dask.array as dsa
import dask
import xarray as xr

import fastjmd95.jmd95numba as jmd95numba


def rho_x(s, t, p):
    if any([isinstance(s,dask.array.core.Array),
           isinstance(t,dask.array.core.Array),
           isinstance(p,dask.array.core.Array)]):
        rho = dsa.map_blocks(jmd95numba.rho,s,t,p)
    elif any([[isinstance(s,xr.DataArray),
           isinstance(t,xr.DataArray),
           isinstance(p,xr.DataArray)]]):
        rho = xr.apply_ufunc(jmd95numba.rho,s,t,p,dask='allowed')
    else:
        rho = jmd95numba.rho(s,t,p)
    return rho

def drhodt_x(s, t, p):
    if any([isinstance(s,dask.array.core.Array),
           isinstance(t,dask.array.core.Array),
           isinstance(p,dask.array.core.Array)]):
        drhodt = dsa.map_blocks(jmd95numba.drhodt,s,t,p)
    elif any([[isinstance(s,xr.DataArray),
           isinstance(t,xr.DataArray),
           isinstance(p,xr.DataArray)]]):
        drhodt = xr.apply_ufunc(jmd95numba.drhodt,s,t,p,dask='allowed')
    else:
        drhodt = jmd95numba.rho(s,t,p)
    return drhodt

def drhods_x(s, t, p):
    if any([isinstance(s,dask.array.core.Array),
           isinstance(t,dask.array.core.Array),
           isinstance(p,dask.array.core.Array)]):
        drhods = dsa.map_blocks(jmd95numba.drhods,s,t,p)
    elif any([[isinstance(s,xr.DataArray),
           isinstance(t,xr.DataArray),
           isinstance(p,xr.DataArray)]]):
        drhods = xr.apply_ufunc(jmd95numba.drhods,s,t,p,dask='allowed')
    else:
        drhods = jmd95numba.rho(s,t,p)
    return drhods
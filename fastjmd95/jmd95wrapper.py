
import numpy as np
import dask.array as dsa
import dask
import xarray as xr

import fastjmd95.jmd95numba as jmd95numba

def _any_dask_array(*args):
    return any([isinstance(a, dask.array.core.Array) for a in args])

def _any_xarray(*args):
    return any([isinstance(a, xr.DataArray) for a in args])

def maybe_wrap_arrays(func):
    def wrapper(*args):
        if _any_dask_array(*args):
            rho = dsa.map_blocks(func,*args)
        elif _any_xarray(*args):
            rho = xr.apply_ufunc(func,*args,output_dtypes=[float],dask='parallelized')
        else:
            rho = func(*args)
        return rho
    return wrapper

@maybe_wrap_arrays
def rho(s,t,p):
    return jmd95numba.rho(s,t,p)
    
@maybe_wrap_arrays
def drhodt(s,t,p):
    return jmd95numba.drhodt(s,t,p)

@maybe_wrap_arrays
def drhods(s,t,p):
    return jmd95numba.drhods(s,t,p)

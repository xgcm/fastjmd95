
import numpy as np
import dask.array as dsa
import dask
import xarray as xr

import fastjmd95.jmd95numba as jmd95numba

def _any_dask_array(a,b,c):
    any_dask = any([isinstance(a,dask.array.core.Array),
           isinstance(b,dask.array.core.Array),
           isinstance(c,dask.array.core.Array)])
    return any_dask

def _any_xarray(a,b,c):
    any_xarray = any([[isinstance(a,xr.DataArray),
           isinstance(b,xr.DataArray),
           isinstance(c,xr.DataArray)]])
    return any_xarray

def my_decorator(func):
    def wrapper(*args):
        if _any_dask_array(*args):
            rho = dsa.map_blocks(func,*args)
        elif _any_xarray(*args):
            rho = xr.apply_ufunc(func,*args,dask='allowed')
        else:
            rho = func(*args)
        return rho
    return wrapper

@my_decorator
def rho(s,t,p):
    return jmd95numba.rho(s,t,p)
    
@my_decorator
def drhodt(s,t,p):
    return jmd95numba.drhodt(s,t,p)

@my_decorator
def drhods(s,t,p):
    return jmd95numba.drhods(s,t,p)

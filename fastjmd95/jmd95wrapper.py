
import fastjmd95.jmd95numba as jmd95numba

try:
    import dask.array as dsa
except ImportError:
    dsa = None

try:
    import xarray as xr
except ImportError:
    xr = None

def _any_dask_array(*args):
    if dsa:
        return any([isinstance(a, dsa.core.Array) for a in args])
    else:
        return False

def _any_xarray(*args):
    if xr:
        return any([isinstance(a, xr.DataArray) for a in args])
    else:
        return False

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

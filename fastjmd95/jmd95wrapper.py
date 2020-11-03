
import fastjmd95.jmd95numba as jmd95numba

def _any_dask_array(*args):
    try:
        import dask.array as dsa
        return any([isinstance(a, dsa.core.Array) for a in args])
    except ImportError:
        print("Can't parse dask arrays if dask isn't installed")
        return False

def _any_xarray(*args):
    try:
        import xarray as xr
        return any([isinstance(a, xr.DataArray) for a in args])
    except ImportError:
        print("Can't parse xarrays if xarray isn't installed")
        return False

def maybe_wrap_arrays(func):
    def wrapper(*args):
        if _any_dask_array(*args):
            import dask.array as dsa
            rho = dsa.map_blocks(func,*args)
        elif _any_xarray(*args):
            import xarray as xr
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

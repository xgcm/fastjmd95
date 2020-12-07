from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .jmd95wrapper import rho, drhodt, drhods

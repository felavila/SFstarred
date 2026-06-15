
import inspect
import os
import sys

__all__ = []

# Needed only if there is no path pointing to the root directory. Mostly for testing purposes.
path_ = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
sys.path.append(path_)
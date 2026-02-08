import scanpy as sc
from packaging.version import Version
from collections.abc import Iterable
import joblib
import contextlib

if Version(sc.__version__) >= Version("1.11.0"):
    from scanpy._utils import SeedLike as RandomState
else:
    from scanpy._utils import AnyRandom as RandomState

def isiterable(x) -> bool:
    return not isinstance(x, str) and isinstance(x, Iterable)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

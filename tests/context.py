import sys
import os
import multiprocessing
from pathlib import Path


os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

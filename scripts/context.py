"""
This file is used to add current project path into the python path.
"""

import sys
import os
import multiprocessing
from pathlib import Path

# 设置LOKY_MAX_CPU_COUNT环境变量以解决joblib警告
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

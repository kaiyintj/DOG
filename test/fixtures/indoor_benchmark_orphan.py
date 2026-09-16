"""Simulator-launch fixture that exits while its child is still running."""

import os
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    child = subprocess.Popen([
        sys.executable, str(Path(__file__).with_name('indoor_benchmark_simulator.py'))])
    Path(os.environ['BENCHMARK_TEST_ORPHAN_PID_FILE']).write_text(str(child.pid))

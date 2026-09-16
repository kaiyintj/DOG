"""Run the ROS integration test in its own process, isolated from pytest forks."""

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import psutil

from semantic_mapping.gazebo.indoor_benchmark import main


def run():
    """Replace only external model-cache and simulator launch dependencies."""
    real_popen = subprocess.Popen
    launched = []
    unrelated = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'])

    def simulator(command, **kwargs):
        assert command[:3] == ['ros2', 'launch', 'semantic_mapping']
        assert 'seed:=7' in command
        assert 'rviz:=false' in command
        fixture = Path(__file__).with_name('indoor_benchmark_simulator.py')
        if os.environ.get('BENCHMARK_TEST_PARENT_EXIT'):
            fixture = fixture.with_name('indoor_benchmark_orphan.py')
        process = real_popen([sys.executable, str(fixture)], **kwargs)
        launched.append(process)
        return process

    try:
        with patch('subprocess.Popen', simulator), patch(
            'huggingface_hub.snapshot_download',
            return_value=os.environ['BENCHMARK_TEST_CHECKPOINT'],
        ):
            result = main(sys.argv[1:])
        assert len(launched) == 1 and launched[0].poll() is not None
        assert unrelated.poll() is None, 'runner stopped an unrelated process'
        if os.environ.get('BENCHMARK_TEST_PARENT_EXIT'):
            pid = int(Path(os.environ['BENCHMARK_TEST_ORPHAN_PID_FILE']).read_text())
            assert (
                not psutil.pid_exists(pid)
                or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            )
        print('runner fixture: cleanup verified', flush=True)
        return result
    finally:
        if os.environ.get('BENCHMARK_TEST_PARENT_EXIT'):
            pid_path = Path(os.environ['BENCHMARK_TEST_ORPHAN_PID_FILE'])
            if pid_path.exists():
                try:
                    orphan = psutil.Process(int(pid_path.read_text()))
                    orphan.terminate()
                    orphan.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    pass
        unrelated.terminate()
        unrelated.wait(timeout=3)


if __name__ == '__main__':
    raise SystemExit(run())

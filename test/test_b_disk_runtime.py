"""Interface tests for the read-only B-disk runtime check."""

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_b_disk_runtime.py"
CLIP_DISTRIBUTIONS = {
    "numpy": "1.26.4",
    "scipy": "1.11.4",
    "Pillow": "12.0.0",
    "open_clip_torch": "3.3.0",
    "setuptools": "79.0.1",
}
CLIP_MODULES = (
    "numpy",
    "scipy",
    "rclpy",
    "rosbag2_py",
    "cv_bridge",
    "torch",
    "PIL",
    "open_clip",
)
SEGFORMER_DISTRIBUTIONS = {
    "transformers": "4.46.3",
    "huggingface-hub": "0.36.0",
    "tokenizers": "0.20.3",
}
SEGFORMER_MODULES = ("transformers", "huggingface_hub", "tokenizers")


def _write_distribution(site, name, version):
    normalized_name = name.replace("-", "_")
    metadata = (
        site
        / "{}-{}.dist-info".format(normalized_name, version)
        / "METADATA"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "Metadata-Version: 2.1\nName: {}\nVersion: {}\n".format(
            name, version),
        encoding="utf-8",
    )


def _write_module(site, module_name):
    parts = module_name.split(".")
    if len(parts) == 1:
        (site / "{}.py".format(parts[0])).write_text(
            "VALUE = 1\n", encoding="utf-8")
        return
    package = site
    for part in parts:
        package = package / part
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text(
            "VALUE = 1\n", encoding="utf-8")


def _fake_environment(tmp_path, numpy_version="1.26.4"):
    site = tmp_path / "site"
    binaries = tmp_path / "bin"
    site.mkdir()
    binaries.mkdir()
    versions = dict(CLIP_DISTRIBUTIONS)
    versions["numpy"] = numpy_version
    for name, version in versions.items():
        _write_distribution(site, name, version)
    for module_name in CLIP_MODULES:
        _write_module(site, module_name)
    _write_module(site, "livox_ros_driver2.msg")
    (site / "numpy.py").write_text(
        "uint8 = object()\n"
        "class Array:\n"
        "    pass\n"
        "def zeros(unused_shape, dtype=None):\n"
        "    return Array()\n",
        encoding="utf-8",
    )
    (site / "cv_bridge.py").write_text(
        "class CvBridge:\n"
        "    def cv2_to_imgmsg(self, array, encoding):\n"
        "        return array, encoding\n"
        "    def imgmsg_to_cv2(self, message, desired_encoding):\n"
        "        return message[0]\n",
        encoding="utf-8",
    )

    ros2 = binaries / "ros2"
    ros2.write_text(
        "#!/bin/sh\n"
        "if [ \"$2\" = prefix ]; then\n"
        "  echo \"/fake/$3\"\n"
        "elif [ \"$2\" = executables ]; then\n"
        "  if [ \"$3\" = fast_lio ]; then\n"
        "    echo 'fast_lio fastlio_mapping'\n"
        "  elif [ \"$3\" = semantic_mapping ]; then\n"
        "    echo 'semantic_mapping clip_node'\n"
        "    echo 'semantic_mapping ga_bsvm_node'\n"
        "  fi\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    ros2.chmod(0o755)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(site)
    environment["PATH"] = "{}:{}".format(
        binaries, environment.get("PATH", ""))
    return environment, ros2


def _run_check(environment, backend="clip"):
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--backend", backend],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


def test_cli_accepts_one_complete_runtime(tmp_path):
    """Report ready through the command interface when every probe passes."""
    environment, unused_ros2 = _fake_environment(tmp_path)

    completed = _run_check(environment)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DIST_NUMPY=PASS" in completed.stdout
    assert "IMPORT_LIVOX_ROS_DRIVER2_MSG=PASS" in completed.stdout
    assert "CV_BRIDGE_RGB_ROUNDTRIP=PASS" in completed.stdout
    assert "ROS_PACKAGE_SEMANTIC_MAPPING=PASS" in completed.stdout
    assert "ROS_EXECUTABLE_FAST_LIO_FASTLIO_MAPPING=PASS" in completed.stdout
    assert "ROS_EXECUTABLE_SEMANTIC_MAPPING_CLIP_NODE=PASS" in completed.stdout
    assert "OVERALL=B_DISK_RUNTIME_READY" in completed.stdout


def test_cli_accepts_segformer_runtime(tmp_path):
    """Use the same interface with the SegFormer dependency adapter."""
    environment, unused_ros2 = _fake_environment(tmp_path)
    site = Path(environment["PYTHONPATH"])
    for name, version in SEGFORMER_DISTRIBUTIONS.items():
        _write_distribution(site, name, version)
    for module_name in SEGFORMER_MODULES:
        _write_module(site, module_name)

    completed = _run_check(environment, backend="segformer")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "BACKEND=segformer" in completed.stdout
    assert "DIST_TRANSFORMERS=PASS" in completed.stdout
    assert "IMPORT_TRANSFORMERS=PASS" in completed.stdout
    assert "OVERALL=B_DISK_RUNTIME_READY" in completed.stdout


def test_cli_explains_version_and_overlay_failures(tmp_path):
    """Expose actionable version and ROS-overlay failures at the interface."""
    environment, ros2 = _fake_environment(
        tmp_path, numpy_version="2.2.6")
    site = Path(environment["PYTHONPATH"])
    (site / "cv_bridge.py").write_text(
        "import sys\n"
        "print('AttributeError: _ARRAY_API not found', file=sys.stderr)\n",
        encoding="utf-8",
    )
    ros2.write_text(
        "#!/bin/sh\n"
        "if [ \"$2\" = prefix ] && [ \"$3\" = fast_lio ]; then\n"
        "  echo 'package not found' >&2\n"
        "  exit 1\n"
        "elif [ \"$2\" = prefix ]; then\n"
        "  echo \"/fake/$3\"\n"
        "elif [ \"$2\" = executables ] && "
        "[ \"$3\" = semantic_mapping ]; then\n"
        "  echo 'semantic_mapping clip_node'\n"
        "  echo 'semantic_mapping ga_bsvm_node'\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )

    completed = _run_check(environment)

    assert completed.returncode == 1
    assert "DIST_NUMPY=FAIL" in completed.stdout
    assert "actual=2.2.6 required[==1.26.4]" in completed.stdout
    assert "IMPORT_CV_BRIDGE=FAIL" in completed.stdout
    assert "ROS_PACKAGE_FAST_LIO=FAIL" in completed.stdout
    assert "OVERALL=B_DISK_RUNTIME_NOT_READY" in completed.stdout


def test_cli_rejects_broken_cv_bridge_rgb_conversion(tmp_path):
    """Catch ABI mismatches that import-only checks cannot expose."""
    environment, unused_ros2 = _fake_environment(tmp_path)
    site = Path(environment["PYTHONPATH"])
    (site / "cv_bridge.py").write_text(
        "class CvBridge:\n"
        "    def cv2_to_imgmsg(self, unused_array, encoding):\n"
        "        raise KeyError(16)\n",
        encoding="utf-8",
    )

    completed = _run_check(environment)

    assert completed.returncode == 1
    assert "IMPORT_CV_BRIDGE=PASS" in completed.stdout
    assert "CV_BRIDGE_RGB_ROUNDTRIP=FAIL" in completed.stdout
    assert "OVERALL=B_DISK_RUNTIME_NOT_READY" in completed.stdout

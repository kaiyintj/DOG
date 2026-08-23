#!/usr/bin/env python3
"""Check whether the migrated B-disk ROS/Python runtime is ready."""

import argparse
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import re
import shutil
import subprocess
import sys

BACKEND_REQUIREMENTS = {
    "clip": "requirements-clip.txt",
    "segformer": "requirements-segformer.txt",
}
BASE_IMPORTS = (
    "numpy",
    "scipy",
    "rclpy",
    "rosbag2_py",
    "cv_bridge",
    "livox_ros_driver2.msg",
)
BACKEND_IMPORTS = {
    "clip": ("torch", "PIL", "open_clip"),
    "segformer": (
        "torch",
        "PIL",
        "transformers",
        "huggingface_hub",
        "tokenizers",
    ),
}
ROS_PACKAGES = ("livox_ros_driver2", "fast_lio", "semantic_mapping")
ROS_EXECUTABLES = {
    "fast_lio": ("fastlio_mapping",),
    "semantic_mapping": ("clip_node", "ga_bsvm_node"),
}
IMPORT_FAILURE_MARKERS = (
    "_ARRAY_API not found",
    "compiled using NumPy 1.x",
    "numpy.core.multiarray failed to import",
)


@dataclass(frozen=True)
class Check:
    """Describe one observable runtime readiness result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RequirementSpec:
    """Store one distribution name and its numeric version constraints."""

    name: str
    constraints: tuple


class RuntimeCheckError(RuntimeError):
    """Report an invalid or unreadable readiness specification."""


def _status_name(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def _parse_requirement(line, path, line_number):
    match = re.fullmatch(
        r"([A-Za-z0-9_.-]+)\s*((?:(?:==|>=|<=|>|<)[^,]+)"
        r"(?:,(?:==|>=|<=|>|<)[^,]+)*)?",
        line,
    )
    if match is None:
        raise RuntimeCheckError(
            "{}:{} has unsupported requirement {!r}".format(
                path, line_number, line))
    constraints = []
    for clause in filter(None, match.group(2).split(",")):
        constraint = re.fullmatch(r"(==|>=|<=|>|<)\s*(\d+(?:\.\d+)*)", clause)
        if constraint is None:
            raise RuntimeCheckError(
                "{}:{} has unsupported constraint {!r}".format(
                    path, line_number, clause))
        constraints.append((constraint.group(1), constraint.group(2)))
    return RequirementSpec(match.group(1), tuple(constraints))


def _read_requirements(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        return []
    seen.add(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeCheckError(
            "cannot read requirements {}: {}".format(path, error)) from error

    requirements = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            requirements.extend(
                _read_requirements(path.parent / line[3:].strip(), seen))
            continue
        requirements.append(_parse_requirement(line, path, line_number))
    return requirements


def _numeric_version(value):
    match = re.fullmatch(r"\d+(?:\.\d+)*", value)
    if match is None:
        raise ValueError(value)
    return tuple(int(part) for part in value.split("."))


def _compare_versions(left, right):
    width = max(len(left), len(right))
    return (
        left + (0,) * (width - len(left)),
        right + (0,) * (width - len(right)),
    )


def _satisfies(actual, constraints):
    actual_version = _numeric_version(actual)
    comparisons = {
        "==": lambda left, right: left == right,
        ">=": lambda left, right: left >= right,
        "<=": lambda left, right: left <= right,
        ">": lambda left, right: left > right,
        "<": lambda left, right: left < right,
    }
    for operator, expected in constraints:
        left, right = _compare_versions(
            actual_version, _numeric_version(expected))
        if not comparisons[operator](left, right):
            return False
    return True


def _constraint_text(requirement):
    if not requirement.constraints:
        return "any"
    return ",".join(
        "{}{}".format(operator, version)
        for operator, version in requirement.constraints
    )


def _distribution_checks(project_root, backend):
    requirements = _read_requirements(
        project_root / BACKEND_REQUIREMENTS[backend])
    unique = {requirement.name.lower(): requirement
              for requirement in requirements}
    checks = []
    for key in sorted(unique):
        requirement = unique[key]
        name = "DIST_{}".format(_status_name(requirement.name))
        try:
            actual = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            checks.append(Check(name, False, "not installed"))
            continue
        try:
            allowed = _satisfies(actual, requirement.constraints)
        except ValueError:
            checks.append(Check(
                name, False, "invalid installed version {}".format(actual)))
            continue
        detail = "actual={} required[{}]".format(
            actual, _constraint_text(requirement))
        checks.append(Check(name, allowed, detail))
    return checks


def _failure_summary(completed):
    lines = [
        line.strip()
        for line in completed.stderr.splitlines()
        if line.strip()
    ]
    if not lines:
        return "import exited {}".format(completed.returncode)
    return lines[-1][-240:]


def _import_check(module_name):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib,sys; importlib.import_module(sys.argv[1])",
            module_name,
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    diagnostic_failure = any(
        marker in completed.stderr
        for marker in IMPORT_FAILURE_MARKERS
    )
    passed = completed.returncode == 0 and not diagnostic_failure
    return Check(
        "IMPORT_{}".format(_status_name(module_name)),
        passed,
        (
            "importable"
            if passed
            else _failure_summary(completed)
        ),
    )


def _ros_package_checks():
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return [Check("ROS2_COMMAND", False, "ros2 is not on PATH")]

    checks = [Check("ROS2_COMMAND", True, ros2)]
    for package in ROS_PACKAGES:
        completed = subprocess.run(
            [ros2, "pkg", "prefix", package],
            capture_output=True,
            check=False,
            text=True,
        )
        detail = (
            completed.stdout.strip()
            if completed.returncode == 0
            else _failure_summary(completed)
        )
        checks.append(Check(
            "ROS_PACKAGE_{}".format(_status_name(package)),
            completed.returncode == 0,
            detail,
        ))
        if completed.returncode != 0 or package not in ROS_EXECUTABLES:
            continue
        executables = subprocess.run(
            [ros2, "pkg", "executables", package],
            capture_output=True,
            check=False,
            text=True,
        )
        installed = {
            fields[1]
            for line in executables.stdout.splitlines()
            if len(fields := line.split(maxsplit=1)) == 2
        }
        for executable in ROS_EXECUTABLES[package]:
            checks.append(Check(
                "ROS_EXECUTABLE_{}_{}".format(
                    _status_name(package), _status_name(executable)),
                executables.returncode == 0 and executable in installed,
                (
                    "installed"
                    if executable in installed
                    else "not installed"
                ),
            ))
    return checks


def check_runtime(project_root, backend):
    """Return all checks exposed by the runtime readiness interface."""
    project_root = Path(project_root).resolve()
    if backend not in BACKEND_REQUIREMENTS:
        raise RuntimeCheckError("unsupported backend: {}".format(backend))
    checks = _distribution_checks(project_root, backend)
    checks.extend(
        _import_check(module_name)
        for module_name in BASE_IMPORTS + BACKEND_IMPORTS[backend]
    )
    checks.extend(_ros_package_checks())
    return checks


def parse_args(argv=None):
    """Parse the single varying runtime choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_REQUIREMENTS),
        default="clip",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Print stable readiness lines and return failure until all pass."""
    arguments = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        checks = check_runtime(project_root, arguments.backend)
    except RuntimeCheckError as error:
        print("ERROR={}".format(error), file=sys.stderr)
        return 2

    print("BACKEND={}".format(arguments.backend))
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print("{}={}".format(check.name, status))
        if not check.passed:
            print("{}_DETAIL={}".format(check.name, check.detail))
    ready = all(check.passed for check in checks)
    print("OVERALL={}".format(
        "B_DISK_RUNTIME_READY"
        if ready
        else "B_DISK_RUNTIME_NOT_READY"
    ))
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())

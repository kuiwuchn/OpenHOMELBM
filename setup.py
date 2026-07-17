"""Setuptools configuration for OpenHOMELBM."""

from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements() -> list[str]:
    """Return non-empty, non-comment dependency lines."""
    return [
        line
        for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]


setup(
    name="OpenHOMELBM",
    version="0.1.0",
    description=(
        "GPU-oriented HOME-LBM environments with MuJoCo-Warp coupling "
        "and reinforcement learning"
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/kuiwuchn/OpenHOMELBM",
    project_urls={
        "Documentation": "https://kuiwuchn.github.io/OpenHOMELBM/",
        "Paper": "https://kuiwuchn.github.io/homelbm.html",
        "Source": "https://github.com/kuiwuchn/OpenHOMELBM",
    },
    license="GPL-3.0-or-later",
    license_files=("LICENSE",),
    python_requires=">=3.11",
    packages=find_namespace_packages(include=["envs", "envs.*"]),
    package_data={
        "envs.lbm.eel": ["*.xml"],
        "envs.lbm.karman": ["*.xml"],
        "envs.lbm3d.eel": ["*.xml"],
        "envs.lbm3d.karman": ["*.xml"],
    },
    include_package_data=True,
    install_requires=read_requirements(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)

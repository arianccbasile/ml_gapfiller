import re

from setuptools import find_packages, setup

PACKAGE_NAME = "ml_gapfill"
SOURCE_DIRECTORY = "src"
SOURCE_PACKAGE_REGEX = re.compile(rf"^{SOURCE_DIRECTORY}")

source_packages = find_packages(include=[SOURCE_DIRECTORY, f"{SOURCE_DIRECTORY}.*"])
proj_packages = [
    SOURCE_PACKAGE_REGEX.sub(PACKAGE_NAME, name) for name in source_packages
]

setup(
    name=PACKAGE_NAME,
    packages=proj_packages,
    package_dir={PACKAGE_NAME: SOURCE_DIRECTORY},
)

#!/usr/bin/env python
from setuptools import Extension, setup, find_packages
import flair

PACKAGES = find_packages()


def setup_package():
    setup(
        name='flair-project',
        version=flair.__version__,
        packages=PACKAGES,
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'flair-cli=flair_project.__main__:main',
            ],
        },
    )


if __name__ == "__main__":
    setup_package()

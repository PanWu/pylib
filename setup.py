#!/usr/bin/env python

from setuptools import setup, find_packages

# for tests
import os
os.environ['TESTING'] = '1'

install_requires = []
setup_requires=[
    "numpy>=1.12.1",
    "scipy>=0.19.0",
    "pandas>=0.19.2",
    "scikit-learn>=0.18.1",
]
tests_require=[]

setup(
    name="pylib",
    version="0.1.0",
    description='Python utility functions library',
    author='Pan Wu',
    author_email='ustcwupan@gmail.com',
    url='https://github.com/PanWu/pylib',

    packages=find_packages(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,

    zip_safe=False
)

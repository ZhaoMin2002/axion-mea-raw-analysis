"""ad_organoids setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('ad_organoids', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.rst') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name = 'ad_organoids',
    version = __version__,
    description = 'Alzheimer disease organoid analyses.',
    long_description = long_description,
    python_requires = '>=3.6',
    author = 'Voytek and Lipton Labs',
    author_email = 'rhammonds@ucsd.edu',
    maintainer = 'Ryan Hammonds',
    maintainer_email = 'rhammonds@ucsd.edu',
    url = 'https://github.com/voyteklab/ad_organoids',
    packages = find_packages(),
    license = 'Apache License, 2.0',
    keywords = ['neuroscience', 'organoids', 'neural oscillations', 'time series analysis'
                , 'local field potentials', 'spectral analysis', 'electrophysiology'],
    install_requires = install_requires,
    #tests_require = ['pytest'],
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
        ],
    platforms = 'any'
)

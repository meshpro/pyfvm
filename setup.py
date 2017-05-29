# -*- coding: utf-8 -*-
#
from setuptools import setup, find_packages
import os
import codecs

# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, 'pyfvm', '__about__.py')) as f:
    exec(f.read(), about)


def read(fname):
    try:
        content = codecs.open(
            os.path.join(os.path.dirname(__file__), fname),
            encoding='utf-8'
            ).read()
    except Exception:
        content = ''
    return content


setup(
    name='PyFVM',
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    packages=find_packages(),
    description='Finite Volume Discretizations for Python',
    long_description=read('README.rst'),
    url='https://github.com/nschloe/pyfvm',
    download_url='https://github.com/nschloe/pyfvm/releases',
    license=about['__license__'],
    platforms='any',
    install_requires=[
        'sphinxcontrib-bibtex',
        'code_extract',
        'krypy',
        'meshzoo',
        'numpy',
        'pyamg',
        'pygmsh',
        'scipy',
        'sympy',
        'voropy',
        ],
    classifiers=[
        about['__license__'],
        about['__status__'],
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics'
        ]
    )

# -*- coding: utf-8 -*-
#
import os
from setuptools import setup
import codecs

from pyfvm import __version__, __author__, __author_email__


def read(fname):
    try:
        content = codecs.open(
            os.path.join(os.path.dirname(__file__), fname),
            encoding='utf-8'
            ).read()
    except Exception:
        content = ''
    return content

setup(name='pyfvm',
      version=__version__,
      author=__author__,
      author_email=__author_email__,
      packages=['pyfvm'],
      description='Finite Volume Method for Python',
      long_description=read('README.rst'),
      url='https://github.com/nschloe/pyfvm',
      download_url='https://github.com/nschloe/pyfvm/releases',
      license='License :: OSI Approved :: BSD License',
      platforms='any',
      requires=['numpy', 'scipy', 'vtk'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Mathematics'
          ],
      scripts=[
          'nfc/form-compiler',
          ],
      )
# Don't install test files.
#        data_files=[('tests', ['tests/cubesmall.e',
#                               'tests/pacman.e',
#                               'tests/rectanglesmall.e',
#                               'tests/test.e',
#                               'tests/tetrahedron.e'])]

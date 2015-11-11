# VoroPy

[![Build Status](https://travis-ci.org/nschloe/voropy.svg?branch=master)](https://travis-ci.org/nschloe/voropy)
[![Code Health](https://landscape.io/github/nschloe/voropy/master/landscape.png)](https://landscape.io/github/nschloe/voropy/master)
[![Coverage Status](https://img.shields.io/coveralls/nschloe/voropy.svg)](https://coveralls.io/r/nschloe/voropy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/voropy/badge/?version=latest)](https://readthedocs.org/projects/voropy/?badge=latest)
[![PyPi Version](https://img.shields.io/pypi/v/voropy.svg)](https://pypi.python.org/pypi/voropy)
[![PyPi Downloads](https://img.shields.io/pypi/dm/voropy.svg)](https://pypi.python.org/pypi/voropy)

A package for handling simplex-meshes in two and three dimensions with support for Voronoi regions.

![](https://nschloe.github.io/voropy/moebius2.png)

*A twice-folded Möbius strip, created with VoroPy's `moebius_tri -i 2 out.e`. Visualization with [ParaView](http://www.paraview.org/).*

This is of specific interest to everyone who wants to use the finite volume method, but many of the provided methods can come in handy for finite element problems, too.

For example, this package is for you if you want to:

* have an easy interface for I/O from/to VTK and Exodus files;
* create relations between nodes, edges, faces, and cells of a mesh;
* compute the control volumes of a given input mesh;
* display meshes with corresponding Voronoi meshes in matplotlib.

Some of the methods in this packages are merely convenience wrappers around the VTK Python interface which needs to be installed in your system. Further dependencies include Numpy and SciPy.

### License

VoroPy is released under the 3-clause BSD license.

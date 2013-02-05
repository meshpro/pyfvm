A package for handling simplex-meshes in two and three dimensions with support for Voronoi regions.

This is of specific interest to everyone who wants to use the finite volume method, but many of the provided methods can come in handy for finite element problems, too.

For example, this package is for you if you want to:

* have an easy interface for I/O from/to VTK and Exodus files;
* create relations between nodes, edges, faces, and cells of a mesh;
* compute the control volumes of a given input mesh;
* display meshes with corresponding Voronoi meshes in matplotlib.

Some of the methods in this packages are merely convenience wrappers around the VTK Python interface which needs to be installed in your system. Further dependencies include Numpy and SciPy.

[![Build Status](https://travis-ci.org/nschloe/VoroPy.png?branch=master)](https://travis-ci.org/nschloe/VoroPy)

# -*- coding: utf-8 -*-
#
from .code_generator_eigen import *
# from .code_generator_tpetra import *
from .dirichlet import *
from .expression import *
from .form_language import *
from .fvm_matrix import *
from .fvm_operator import *
from .helpers import *
from .linear_fvm_problem import *
from .integral_boundary import *
from .integral_edge import *
from .integral_vertex import *
from .operator import *
from .subdomain import *

import inspect
from string import Template
import os

# <http://stackoverflow.com/a/67692/353337>
# import importlib.machinery
import imp

import tokenize
import autopep8

# <http://stackoverflow.com/a/7472878/353337>
# Python 2:
import StringIO
# Python 3:
# import io

# Python 3 compat
TokenInfo = getattr(tokenize, 'TokenInfo', lambda *a: a)


def _semicolon_to_newline(tokens):
    '''Replaces semicolons in Python code with newlines and the appropriate
    indentation.
    '''
    # From <http://stackoverflow.com/a/37788839/353337>.

    line_offset = 0
    last_indent = None
    col_offset = None  # None or an integer
    for ttype, tstr, (slno, scol), (elno, ecol), line in tokens:
        slno, elno = slno + line_offset, elno + line_offset
        if ttype in (tokenize.INDENT, tokenize.DEDENT):
            last_indent = ecol  # block is indented to this column
        elif ttype == tokenize.OP and tstr == ';':
            # swap out semicolon with a newline
            ttype = tokenize.NEWLINE
            tstr = '\n'
            line_offset += 1
            if col_offset is not None:
                scol, ecol = scol - col_offset, ecol - col_offset
            col_offset = 0  # next tokens should start at the current indent
        elif col_offset is not None:
            if not col_offset:
                # adjust column by starting column of next token
                col_offset = scol - last_indent
            scol, ecol = scol - col_offset, ecol - col_offset
            if ttype == tokenize.NEWLINE:
                col_offset = None
        yield TokenInfo(ttype, tstr, (slno, scol), (elno, ecol), line)


def _semicolons_to_newlines(source):
    generator = tokenize.generate_tokens(StringIO.StringIO(source).readline)
    return tokenize.untokenize(_semicolon_to_newline(generator))


def compile(infile, outfile, backend=None):
    inmod_name = 'inmod'
    inmod = imp.load_source(inmod_name, infile)
    # inmod = importlib.machinery.SourceFileLoader(
    #     inmod_name,
    #     infile
    #     ).load_module()

    namespace = sanitize_identifier_cxx(
            os.path.splitext(os.path.basename(infile))[0]
            )

    # Collect relevant classes
    classes = []
    for _, obj in inmod.__dict__.items():
        # Only inspect classes from inmod
        if not inspect.isclass(obj) or obj.__module__ != inmod_name:
            continue
        classes.append(obj)

    return compile_classes(namespace, classes, outfile, backend)


def compile_classes(classes, namespace, outfile=None, backend='scipy'):
    if outfile is None and backend is None:
        raise RuntimeError('One of outfile and backend must be specified.')
    elif outfile is None:
        if backend == 'scipy':
            outfile = namespace + '.py'
        elif backend == 'nosh':
            outfile = namespace + '.hpp'
        else:
            raise ValueError('Illegal backend \'%s\'.' % backend)
    elif backend is None:
        if os.path.splitext(os.path.basename(outfile))[1] == '.py':
            backend = 'scipy'
        elif os.path.splitext(os.path.basename(outfile))[1] == '.hpp':
            backend = 'nosh'
        else:
            raise ValueError(
                'Not backend specified and coudn\'t determine '
                'from filename extension'
                )

    # Loop over all locally defined entities and collect everything we can
    # convert.
    # Between the entities, there are some dependencies between the entities
    # which are not necessarily reflected in the order they appear here. For
    # example, in the output HPP, the boundary conditions classes must
    # typically be defined _before_ the fvm_matrix classes, since they are used
    # there.
    # Build directed dependency graph as a dictionary, see
    # <https://www.python.org/doc/essays/graphs/>.
    def get_generator(cls):
        if issubclass(cls, FvmMatrix):
            return FvmMatrixCode(namespace, cls)
        elif issubclass(cls, LinearFvmProblem):
            return LinearFvmProblemCode(namespace, cls)
        elif issubclass(cls, FvmOperator):
            return FvmOperatorCode(namespace, cls)
        elif issubclass(cls, Subdomain):
            return SubdomainCode(cls)
        # elif issubclass(var, EdgeCore):
        #     instance = var()
        #     return get_code_matrix_core_edge(namespace, name, instance)
        elif issubclass(cls, Expression):
            return ExpressionCode(cls)
        else:
            raise RuntimeError('Unknown class \'%s\'.' % cls.__name__)

    deps = {}
    generators = {}

    # Recursively go through all generators, get the dependencies, and build
    # the dependency tree.
    def insert_dependencies(generator):
        if generator.class_name_cxx in deps:
            return
        generators[generator.class_name_cxx] = generator
        deps[generator.class_name_cxx] = [
            dep.class_name_cxx for dep in generator.get_dependencies()
            ]
        for dep in generator.get_dependencies():
            insert_dependencies(dep)

    # Loop over all inmod classes to create the dependency tree
    for obj in classes:
        generator = get_generator(obj)
        insert_dependencies(generator)

    # Now that we have all generators and dependencies in place,
    # go through dependency graph and collect code.
    visited = set([])
    missing_objects = set([])

    def collect_class_objects(name):
        if name not in deps:
            missing_objects.add(name)
            return []
        # take care of the dependencies first
        class_objects = []
        for dep in deps[name]:
            class_objects.extend(collect_class_objects(dep))
        if name not in visited:
            if backend == 'scipy':
                class_object = \
                    generators[name].get_python_class_object(class_objects)
            elif backend == 'nosh':
                class_object = \
                    generators[name].get_cxx_class_object(class_objects)
            else:
                raise ValueError('Illegal backend \'%s\'.' % backend)
            visited.add(name)
            class_objects.append(class_object)
            return class_objects

        return []

    class_objects = []
    for name in deps:
        class_objects.extend(collect_class_objects(name))

    if missing_objects:
        print()
        print('You will need to manually define the objects')
        print('     %s' % ', '.join(missing_objects))
        print('in the namespace `%s` before #including the generated header.' %
              namespace)
        print()

    # Plug it all together in main
    if backend == 'scipy':
        # Python classes want two newlines between them
        code = '\n\n'.join(
                class_object['code'] for class_object in class_objects
                )

        main_template_python = '''
import pyfvm
from numpy import *

${content}'''
        main_src = Template(main_template_python)
        main_content = main_src.substitute({'content': code})
        main_content = _semicolons_to_newlines(main_content)
        main_content = autopep8.fix_code(
                main_content,
                options={'aggressive': 2}
                )
    elif backend == 'nosh':
        code = '\n'.join(
                class_object['code'] for class_object in class_objects
                )
        main_template_cxx = '''
#ifndef ${namespace_uppercase}_HPP
#define ${namespace_uppercase}_HPP

#include <nosh.hpp>

namespace ${namespace}
{
${content}
} // namespace ${namespace}'''
        main_src = Template(main_template_cxx)
        main_content = main_src.substitute({
            'namespace': namespace,
            'namespace_uppercase': namespace.upper(),
            'content': code
            })
    else:
        raise ValueError('Illegal backend \'%s\'.' % backend)

    # write it
    # outfile = os.path.splitext(infile)[0] + '.hpp'
    with open(outfile, 'w') as f:
        f.write(main_content)

    if backend == 'nosh':
        # Make sure it's formatted nicely.
        # TODO check out uncrustify
        run('astyle --style=ansi -s2 %s' % outfile)

    return

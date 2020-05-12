#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/12/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

try:
    from jacinle.jit.cext import auto_travis
    auto_travis(__file__)
except:
    pass

try:
    from .pygco import cut_general_graph, cut_grid_graph, cut_grid_graph_simple
except ImportError as e:
    raise ImportError('Auto travis for pygco failed. Run ./travis.sh manually.') from e


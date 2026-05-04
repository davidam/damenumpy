#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (C) 2026  David Arroyo Menéndez

# Author: David Arroyo Menéndez <davidam@gmail.com>
# Maintainer: David Arroyo Menéndez <davidam@gmail.com>

# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.

# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Damenumpy; see the file LICENSE.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301 USA,

from unittest import TestCase
import numpy as np
import collections
collections.Callable = collections.abc.Callable


class TestOperations(TestCase):
    def test_iteranting_1d(self):
        x = np.array([1, 2])
        y = str(type(x))
        self.assertEqual("<class 'numpy.ndarray'>", y)

    def test_iterating_1d(self):
        arr = np.array([1,2,3])
        string1 = ""
        for x in arr:
            string1 = string1 + str(x)
        self.assertEqual("123", string1)

    def test_iterating_2d(self):    
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        string1 = ""
        for x in arr:
            string1 = string1 + str(x)
        res1 = "[1 2 3][4 5 6]"
        self.assertEqual(string1, res1)
        string2 = ""
        for x in arr:
            for y in x:
                string2 = string2 + str(y)
        res2 = "123456"
        self.assertEqual(string2, res2)

    def test_iterating_3d(self):    
        arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        string1 = ""
        for x in arr:
            for y in x:
                for z in y:
                    string1 = string1 + str(z) + " "
        res1 = "1 2 3 4 5 6 7 8 9 10 11 12 "
        self.assertEqual(string1, res1)
        string2 = ""
        for x in np.nditer(arr):
            string2 = string2 + str(x) + " "
        self.assertEqual(string2, res1)

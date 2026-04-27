#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (C) 2019  David Arroyo Menéndez

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

# This file is about simple exercises with numpy
# You can read external documentation
# + https://numpy.org
# + https://www.w3schools.com/python/numpy/numpy_creating_arrays.asp
# + https://www.geeksforgeeks.org/python/numpy-tutorial/

from unittest import TestCase
import numpy as np
import collections
collections.Callable = collections.abc.Callable


class TestBasics(TestCase):

    def test_indexing(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])   # Create a rank 2 array
        self.assertEqual(arr1[0, 0], 1)
        self.assertEqual(arr1[0, 1], 2)
        self.assertEqual(arr1[1, 0], 4)

    def test_indexing_2d(self):
        arr1 = np.array([[10, 20, 30], [40, 5, 66], [70, 88, 94]])  # Create an array
        arr2 = arr1[[0,2]]
        res = np.array([[10, 20, 30], [70, 88, 94]])  # Create an array
        self.assertTrue(np.array_equal(arr2, res))
        
        arr3 = np.array([[101, 20, 3, 10], [40, 5, 66, 7], [70, 88, 9, 141]])
        arr4 = arr3[1]
        res = np.array([40, 5, 66, 7])
        self.assertTrue(np.array_equal(arr4, res))

        arr5 = np.array([[12, 15, 18], 
                         [25, 30, 35], 
                         [40, 45, 50]])
        arr6 = arr5[:2, :2] # first 2 rows, first 2 columns
        res = np.array([[12, 15],
                        [25, 30]])
        self.assertTrue(np.array_equal(arr6, res))

    def test_indexing_3d(self):
        arr1 = np.array([[[10, 25, 70],
                          [30, 45, 55],
                          [20, 45, 7]],
                         [[50, 65, 8],
                          [70, 85, 10],
                          [11, 22, 33]]])
        arr2 = arr1[:,[1]] # Accessing the Middle rows of 3D NumPy array
        res = np.array([[[30, 45, 55]],
                        [[70, 85, 10]]])
        self.assertTrue(np.array_equal(arr2, res))
        
        arr3 = np.array([[[10, 25, 70], [30, 45, 55], [20, 45, 7]], 
                         [[50, 65, 8], [70, 85, 10], [11, 22, 33]],
                         [[19, 69, 36], [1, 5, 24], [4, 20, 96]]])
        arr4 = arr3[:,[0, 2]] # Accessing the First and Last rows of 3D NumPy array
        res =  np.array([[[10, 25, 70], [20, 45, 7]],
                         [[50, 65, 8], [11, 22, 33]],
                         [[19, 69, 36], [4, 20, 96]]])
        self.assertTrue(np.array_equal(arr4, res))

        
        
    def test_arange(self):
        # first make an array from zero to nine
        a = np.arange(10)
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertTrue(np.array_equal(a, x))
        # second make an array form two to nine using floats
        b = np.arange(2, 10, dtype=float)
        y = np.array([2., 3., 4., 5., 6., 7., 8., 9.])
        self.assertTrue(np.array_equal(b, y))
        
    def test_any_all(self):
        # Create a NumPy array 'arr' containing Boolean values
        arr = np.array([True, False, True])
        # Create another NumPy array 'arr2' containing all False values
        arr2 = np.array([False, False, False])
        # At least one element in the first array is True
        self.assertTrue(arr.any())
        # There not element in the second array valued with True
        self.assertFalse(arr2.any())
        # Not all elements in the first array is True
        self.assertFalse(arr.all())
        # All elements in the second array is True
        self.assertFalse(arr.all())

    def test_logical_values(self):
        # Create two larger NumPy arrays with Boolean values
        array1 = np.array([True, False, True, False, True])
        array2 = np.array([False, True, True, False, False])

        # Calculate the element-wise logical AND between array1 and array2
        result_and = np.logical_and(array1, array2)  
        self.assertTrue(np.array_equal(result_and, np.array([False, False, True, False, False])))
        
        # Calculate the element-wise logical OR between array1 and array2
        result_or = np.logical_or(array1, array2)  
        self.assertTrue(np.array_equal(result_or, np.array([True, True, True, False, True])))

        
    def test_shape(self):
        b = np.array([[1, 2, 3], [4, 5, 6]])   # Create a rank 2 array
        self.assertEqual((2, 3), b.shape)

    def test_sum(self):
        x = np.array([[1, 2], [3, 4]])
        x2 = np.array([4, 6])
        self.assertEqual(np.sum(x), 10)
        self.assertEqual(np.sum(x2), 10)

    def test_sum_zero(self):
        x = np.zeros((2, 2))
        res = np.sum(x)
        self.assertEqual(res, 0)

    def test_equal(self):
        self.assertTrue(np.array_equal([1, 2], [1, 2]))
        self.assertTrue(np.array_equal(np.array([1, 2]), np.array([1, 2])))

    def test_zeros(self):
        x = np.zeros((3, 3))
        arr1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertTrue(np.array_equal(x, arr1))

    def test_ones(self):
        x = np.ones((2, 2))
        self.assertTrue(np.array_equal(x, np.array([[1, 1], [1, 1]])))

    def test_full(self):
        x = np.full((2, 2), 7)
        self.assertTrue(np.array_equal(x, np.array([[7, 7], [7, 7]])))

    def test_eye(self):
        x = np.eye(3)
        arr1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.array_equal(x, arr1))

    def test_diag(self):
        x = np.diag([1,2,3])
        arr1 = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.assertTrue(np.array_equal(x, arr1))

    def test_linspace(self):
        x = np.linspace(1., 4., 6)
        arr1 = np.array([1. ,  1.6,  2.2,  2.8,  3.4,  4. ])
        self.assertTrue(np.array_equal(x, arr1))
        
    def test_bool_array(self):
        a = np.array([[1, 2], [3, 4], [5, 6]])
        bool_idx = (a > 2)
        # Find the elements of a that are bigger than 2;
        # this returns a numpy array of Booleans of the same
        # shape as a, where each slot of bool_idx tells
        # whether that element of a is > 2.
        self.assertEqual(bool_idx[0, 0], False)
        self.assertEqual(bool_idx[0, 1], False)
        self.assertEqual(bool_idx[1, 0], True)
        self.assertEqual(bool_idx[1, 1], True)

    def test_tile(self):
        v = np.array([1, 0, 1])
        vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
        arr1 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]])
        self.assertTrue(np.array_equal(vv, arr1))

    def test_empty_like(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        v = np.array([1, 0, 1])
        # Create an empty matrix with the same shape as x
        y = np.empty_like(x)
        # Add the vector v to each row of the matrix x with an explicit loop
        for i in range(4):
            y[i, :] = x[i, :] + v
        arr1 = np.array([[2, 2, 4], [5, 5, 7], [8, 8, 10], [11, 11, 13]])
        self.assertTrue(np.array_equal(arr1, y))

    def test_transpose(self):
        a = np.array([[1, 2], [3, 4]])
        arr1 = np.array([[1, 3], [2, 4]])
        self.assertTrue(np.array_equal(arr1, a.transpose()))

    def test_array_split(self):
        arr = np.array([1, 2, 3, 4, 5, 6])
        res1 = np.array_split(arr, 3)
        res2 = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.array_equal(res1, res2))

    def test_concatenate(self):
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        arr = np.concatenate((arr1, arr2), axis=1)
        res = np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
        self.assertTrue(np.array_equal(arr, res))

    def test_where(self):
        arr = np.array([1, 2, 3, 4, 5, 4, 4])
        x = np.where(arr == 4)
        res = np.array([[3, 5, 6],])
        self.assertTrue(np.array_equal(res, x))

        arr = np.array([10, 14, 93, 41, 8, 7])
        x = np.where(arr%2 == 0)
        res = np.array([[0, 1, 4],])
        self.assertTrue(np.array_equal(res, x))     

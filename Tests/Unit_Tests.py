'''
>>> import math
>>> import numpy as np
>>> import scipy

math.pi == pi

>>> math.pi == np.pi == scipy.pi
'''

def calculate_area_of_circle(radius):
 # Use value of pi
  pi = 3.14
  area = pi * radius ** 2
  return area

def calculate_area_of_circle(radius):
 # Use value of pi
  pi = 3.141592653589793
  area = pi * radius ** 2
  return area



def test_calculate_area_of_circle():
    # Various test cases
    assert calculate_area_of_circle(2) == 12.56
    assert calculate_area_of_circle(4) == 50.24
    assert calculate_area_of_circle(0) == 0.0



import numpy as np
from numba import njit
from numba.experimental import jitclass
import numba.types as nb_types

@jitclass([('val', nb_types.int32)])
class MyReg:
    def __init__(self, val):
        self.val = val
    def do_stuff(self):
        return self.val * 2

reg = MyReg(10)

def make_func(r):
    @njit(cache=False)
    def f():
        return r.do_stuff()
    return f

fn = make_func(reg)
try:
    print(fn())
except Exception as e:
    print('ERROR:', e)

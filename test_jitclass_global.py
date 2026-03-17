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

global_ns = {'reg_obj': reg, 'njit': njit}
code = """
@njit
def f():
    return reg_obj.do_stuff()
"""
exec(code, global_ns)
fn = global_ns['f']
print("fn() returned:", fn())

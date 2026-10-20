"""NEW exact static-origin defect bound, never a candidate acceptance tolerance."""
from fractions import Fraction as Q
import numpy as np

def origin_l1_defect(decimal_xyz,compiled_xyz):
 exact=[Q(v)for v in decimal_xyz.split()];native=[Q.from_float(float(v))for v in compiled_xyz]
 return sum((abs(a-b)for a,b in zip(exact,native,strict=True)),Q(0))

def path_defect_budget(parent_budget,decimal_xyz,compiled_xyz):
 return parent_budget+origin_l1_defect(decimal_xyz,compiled_xyz)

def frame_residual_is_bounded(component_difference,exact_l1_path_budget):
 # A NEW arithmetic evaluation guard of1e-12m bounds floating matrix operations
 # ONLY. It is NOT a change to source, URDF angle or native penetration gates.
 return float(component_difference)<=float(exact_l1_path_budget)+1e-12

"""NEW lossless exact-rational proof table packets, not numeric repair."""
import numpy as np
from fractions import Fraction as Q

def rational_packet(rows):
 numer=bytearray();denom=bytearray();npointer=[0];dpointer=[0]
 for row in rows:
  for x in row:
   n,d=x.numerator,x.denominator
   numer.extend(n.to_bytes(max(1,(abs(n).bit_length()+8)//8),'little',signed=True))
   denom.extend(d.to_bytes(max(1,(d.bit_length()+7)//8),'little',signed=False))
   npointer.append(len(numer));dpointer.append(len(denom))
 return dict(shape=(len(rows),4),numerator_bytes=np.frombuffer(bytes(numer),dtype=np.uint8),denominator_bytes=np.frombuffer(bytes(denom),dtype=np.uint8),
  numerator_ptr=np.asarray(npointer,dtype=np.int64),denominator_ptr=np.asarray(dpointer,dtype=np.int64),
  columns=('ideal_mean','ALL_tie_bound','RN_budget','total_bound'),integer_order='little',numerator_signed=True,denominator_positive=True)

def restore_NEW_packet_control(p):
 n=p['numerator_bytes'].tobytes();d=p['denominator_bytes'].tobytes();a=p['numerator_ptr'];b=p['denominator_ptr'];v=[]
 for i in range(len(a)-1):v.append(Q(int.from_bytes(n[a[i]:a[i+1]],'little',signed=True),int.from_bytes(d[b[i]:b[i+1]],'little')))
 return [v[i:i+4] for i in range(0,len(v),4)]

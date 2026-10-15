import os,inspect,numpy as np
os.environ['PYOPENGL_PLATFORM']='egl'
for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('str',str),('unicode',str)]:
 if name not in np.__dict__:setattr(np,name,value)
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
import torch
from hamer.models import load_hamer
print('IMPORTS_OK',torch.__version__,flush=True)
model,cfg=load_hamer('/workspace/hand-continuity-repair/hamer-transfer/runtime/weights')
model=model.cuda().eval()
print('MODEL_LOADED',cfg.MODEL.IMAGE_SIZE,cfg.EXTRA.FOCAL_LENGTH,flush=True)

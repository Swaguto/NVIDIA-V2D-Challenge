"""NEW lossless return writer for future NEW joint-method inputs.

Motivated by the retained tuple-key metadata failure. It does NOT reconstruct or
rerun that loader or its missing state. Arrays/sparse pointers get full NPZ
payloads; mapping keys retain types. No floating repair/normalization occurs.
"""
from joint_constraint_interface_v1 import DurableRecorder
from pathlib import Path
import json,hashlib,os
import numpy as np
from scipy import sparse


def seal_return(directory,name,value):
    directory=Path(directory);arrays={}
    def pack(v):
        if sparse.issparse(v):
            if v.format not in ['csc','csr']:raise TypeError('explicit CSC/CSR return required')
            return dict(type='sparse',format=v.format,shape=list(v.shape),
                        data=pack(v.data),indices=pack(v.indices),indptr=pack(v.indptr))
        if isinstance(v,np.ndarray):
            if v.dtype.hasobject:raise TypeError('object arrays not lossless numeric returns')
            key=f'array_{len(arrays):06d}';arrays[key]=v
            return dict(type='array',key=key,dtype=v.dtype.str,shape=list(v.shape),
                        bytes_sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest())
        if isinstance(v,np.generic):
            return dict(type='numpy_scalar',dtype=v.dtype.str,bytes=v.tobytes().hex())
        if isinstance(v,dict):
            return dict(type='mapping',items=[[pack(k),pack(x)] for k,x in v.items()])
        if isinstance(v,tuple):return dict(type='tuple',items=[pack(x) for x in v])
        if isinstance(v,list):return dict(type='list',items=[pack(x) for x in v])
        if isinstance(v,Path):return dict(type='path',value=str(v))
        if isinstance(v,bool):return dict(type='bool',value=v)
        if isinstance(v,int):return dict(type='int',value=str(v))
        if isinstance(v,float):return dict(type='float',hex=v.hex())
        if v is None:return dict(type='none')
        if isinstance(v,str):return dict(type='str',value=v)
        raise TypeError('unsupported raw return '+type(v).__name__)
    tree=pack(value);array_path=directory/(name+'.npz');tree_path=directory/(name+'.json')
    with array_path.open('xb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    with tree_path.open('x') as f:
        json.dump(dict(schema='v2d.NEW-lossless-joint-return.v1',tree=tree,array_file=array_path.name),f,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    return tree_path


def restore_NEW_synthetic(path):
    """Used only by NEW controls; not a lost episode/state reconstruction tool."""
    path=Path(path);doc=json.loads(path.read_text())
    with np.load(path.parent/doc['array_file'],allow_pickle=False) as z:
        def unpack(t):
            kind=t['type']
            if kind=='array':
                a=z[t['key']]
                if a.dtype.str!=t['dtype'] or list(a.shape)!=t['shape'] or hashlib.sha256(a.tobytes()).hexdigest()!=t['bytes_sha256']:raise ValueError('array integrity')
                return a
            if kind=='numpy_scalar':return np.frombuffer(bytes.fromhex(t['bytes']),dtype=np.dtype(t['dtype']))[0]
            if kind=='mapping':return {unpack(k):unpack(v) for k,v in t['items']}
            if kind=='tuple':return tuple(unpack(v) for v in t['items'])
            if kind=='list':return [unpack(v) for v in t['items']]
            if kind=='path':return Path(t['value'])
            if kind=='float':return float.fromhex(t['hex'])
            if kind=='int':return int(t['value'])
            if kind=='bool':return t['value']
            if kind=='none':return None
            if kind=='str':return t['value']
            if kind=='sparse':
                ctor=sparse.csc_matrix if t['format']=='csc' else sparse.csr_matrix
                return ctor((unpack(t['data']),unpack(t['indices']),unpack(t['indptr'])),shape=t['shape'])
            raise ValueError(kind)
        return unpack(doc['tree'])


class LosslessRecorder(DurableRecorder):
    def __call__(self,name,value):
        path=seal_return(self.directory,f'{self.count:06d}-{name}',value)
        self.count+=1
        return path

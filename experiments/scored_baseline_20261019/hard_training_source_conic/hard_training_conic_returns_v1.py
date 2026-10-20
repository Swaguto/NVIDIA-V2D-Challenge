"""Immediate snapshots of actual conic objects; x/z/s are saved separately."""
def snapshot(obj,exclude=()):
    fields={}
    for key in dir(obj):
        if key.startswith('_')or key in exclude:continue
        value=getattr(obj,key)
        if callable(value):continue
        fields[key]=value if isinstance(value,(type(None),bool,int,float,str))else repr(value)
    return dict(actual_type=type(obj).__name__,opaque_repr=repr(obj)if not exclude else None,fields=fields)

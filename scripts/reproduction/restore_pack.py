"""Verify a hash-pinned episode archive and restore into a new directory."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path,PurePosixPath

def restore(archive,expected,out):
    if out.exists():raise FileExistsError(out)
    with archive.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=expected:
            raise ValueError('Archive SHA256 mismatch')
    with tarfile.open(archive) as tar:
        members=tar.getmembers();names=set()
        for member in members:
            path=PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                or '\\' in member.name or ':' in member.name or member.name in names):
                raise ValueError(f'Unsafe or duplicate archive member: {member.name}')
            names.add(member.name)
        manifest=json.load(tar.extractfile('artifact-manifest.json'))
        if set(manifest['files'])|{'artifact-manifest.json'}!=names:
            raise ValueError('Archive member catalog differs')
        for member in members:
            if member.name=='artifact-manifest.json':continue
            expected_file=manifest['files'][member.name]
            with tar.extractfile(member) as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            if member.size!=expected_file['bytes'] or digest!=expected_file['sha256']:
                raise ValueError(f'Invalid member: {member.name}')
        out.mkdir(parents=True)
        # Members have been checked; avoid permissions/ownership effects of extraction.
        for member in members:
            dest=out/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,dest.open('wb') as target:
                for chunk in iter(lambda:source.read(1<<20),b''):target.write(chunk)
    print(json.dumps({'status':'verified_restore','files':len(names),'archive_sha256':expected}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();restore(args.archive,args.sha256,args.out)

"""NEW conditional ALL-pair geometric predicates, not native numerical contacts.

Static filtering follows the pinned public MuJoCo 3.8.1 collision driver.
Public reference and license are retained separately. No native callback or
installed-binary equivalence is asserted. Every pair remains represented.
"""
from joint_collision_maps_v1 import *


def builtin_pair_reasons(raw, a, b):
    ba, bb = raw['geom_bodyid'][a], raw['geom_bodyid'][b]
    wa, wb = raw['body_weldid'][ba], raw['body_weldid'][bb]
    # Native weldparent is the WELDED representative of the weld body's parent.
    pa = raw['body_weldid'][raw['body_parentid'][wa]]
    pb = raw['body_weldid'][raw['body_parentid'][wb]]
    dsbl = bool(raw['disableflags'] & raw['FILTERPARENT_bit'])
    reasons = []
    if wa == wb:
        reasons.append('same_weld')
    if not dsbl and wa != 0 and wb != 0 and (wa == pb or wb == pa):
        reasons.append('welded_parent')
    if 65536 * min(ba, bb) + max(ba, bb) in raw['exclude_signature']:
        reasons.append('explicit_body_exclusion')
    if not ((raw['geom_contype'][a] & raw['geom_conaffinity'][b]) or
            (raw['geom_contype'][b] & raw['geom_conaffinity'][a])):
        reasons.append('geom_bitmask')
    return reasons

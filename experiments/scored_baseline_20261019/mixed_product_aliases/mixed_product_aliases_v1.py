"""New mixed t_a*Y_ij identities; existing tensor powers/scopes only."""
from collections import Counter

def request(variable,pair):
 r=Counter(pair);r[variable]+=1;return dict(sorted(r.items()))
def key(req):return tuple(sorted(req.items()))
def full_mixed_cell(variable_index,pair_index):return (1+variable_index,11+pair_index)
def domains_equal(canonical,copies,names):return all(canonical.get(v)==copies.get(v)and v in canonical and v in copies for v in names)
def mixed_zero_variance_case(mass,mean_t,second_t,mean_Y,mixed):
 if mass<=0:return dict(classification='UNRESOLVED_zero_signed_mass_NO_normalization')
 variance_numerator=mass*second_t-mean_t*mean_t
 if variance_numerator!=0:return dict(classification='UNRESOLVED_nonzero_or_negative_variance_NOT_zero_pivot')
 return dict(classification='NECESSARY_zero_variance_mixed_alias_ONLY',mixed_residual=mass*mixed-mean_t*mean_Y)

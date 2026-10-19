import itertools

import numpy as np
import pytest

from v2d_dex3_flashchord.temporal_pose_graph import finite_temporal_reachability


def solve(banks, **overrides):
    banks = [np.asarray(b, dtype=float).reshape(-1, 1) for b in banks]
    args = dict(original_reference=np.zeros((len(banks), 1)),
                timestamps_s=np.arange(len(banks), dtype=float), lower_limits=[-1.],
                upper_limits=[1.], velocity_limits=[1.], component_cap=.25,
                adjacent_added_cap=.1, second_added_cap=.05)
    args.update(overrides)
    return finite_temporal_reachability(banks, **args)


def test_pair_state_avoids_greedy_dead_end_and_trims_backward():
    r = solve([[0], [0, .04], [.1], [.2]])
    assert r.path_indices == (0, 1, 0, 0)
    assert r.viable_node_indices == ((0,), (1,), (0,), (0,))
    assert r.forward_pair_counts[0] == 2
    assert r.backward_pair_counts[0] == 1


def test_actual_velocity_uses_timestamp_intervals():
    assert solve([[0], [.11], [.11]], timestamps_s=[0, .1, .2],
                 adjacent_added_cap=1., second_added_cap=1.).path_indices is None


def test_correction_bounds_do_not_bound_raw_source_motion():
    r = solve([[0], [.3], [.6]], original_reference=np.array([[0], [.3], [.6]]),
              component_cap=0., adjacent_added_cap=0., second_added_cap=0.)
    assert r.path_indices == (0, 0, 0)


def test_second_difference_rejects_individually_valid_edges():
    r = solve([[0], [.04], [0]], adjacent_added_cap=.05)
    assert r.forward_pair_counts == (1, 0)
    assert r.path_indices is None


def test_empty_bank_retains_missing_index_and_no_complete_viability():
    r = solve([[0], [], [0]])
    assert r.empty_bank_indices == (1,)
    assert r.path_indices is None
    assert r.viable_node_indices == ((), (), ())


def test_costs_select_complete_path_not_cheapest_last_node_alone():
    r = solve([[0, .02], [.02, .04], [.04, .06]],
              node_costs=[[10, 0], [0, 10], [0, 10]])
    assert r.path_indices == (1, 0, 0)
    assert r.path_cost == pytest.approx(0.)


@pytest.mark.parametrize('overrides', [
    dict(timestamps_s=[0, 0, 1]), dict(timestamps_s=[0, np.nan, 2]),
    dict(velocity_limits=[0.]), dict(lower_limits=[1.]),
    dict(component_cap=-1.), dict(second_added_cap=np.inf),
    dict(node_costs=[[0], [0], [-1]]), dict(node_costs=[[0]]),
])
def test_invalid_contracts_fail(overrides):
    with pytest.raises(ValueError):
        solve([[0], [0], [0]], **overrides)


def test_out_of_original_component_bank_is_not_silently_filtered():
    with pytest.raises(ValueError):
        solve([[0], [.3], [0]])


def test_exhaustive_small_graphs_match_path_and_all_viable_nodes():
    rng = np.random.default_rng(52)
    for _ in range(40):
        banks = [np.round(rng.uniform(-.12, .12, 3), 3) for _ in range(4)]
        costs = [rng.uniform(0, 1, 3) for _ in banks]
        valid = []
        for path in itertools.product(range(3), repeat=4):
            values = np.array([b[i] for b, i in zip(banks, path, strict=True)])
            if (abs(np.diff(values)) <= .1 + 1e-9).all() and (abs(np.diff(values, n=2)) <= .05 + 1e-9).all():
                valid.append((sum(c[i] for c, i in zip(costs, path, strict=True)), path))
        r = solve(banks, node_costs=costs)
        if not valid:
            assert r.path_indices is None
            assert all(not v for v in r.viable_node_indices)
        else:
            best = min(valid)
            assert r.path_cost == pytest.approx(best[0])
            for frame in range(4):
                assert r.viable_node_indices[frame] == tuple(sorted({p[frame] for _, p in valid}))

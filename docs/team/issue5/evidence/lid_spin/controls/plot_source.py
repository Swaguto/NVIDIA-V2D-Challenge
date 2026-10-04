"""Plot frozen spin diagnostics and separately evaluated development errors."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent
pred = json.loads((base/'portable-lid-spin/probe_results.json').read_text())
before = json.loads((base/'portable-pose-candidate-support-relative.json').read_text())['episodes'][0]
after = json.loads((base/'portable-lid-spin-relative.json').read_text())['episodes'][0]
out = base/'lid-spin-figures-v2'
out.mkdir(exist_ok=False)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False})
fig, axes = plt.subplots(4, 2, figsize=(11, 12), layout='constrained')
for row, ax in zip(pred['frames'], axes.flat):
    d = row['lid_spin_diagnostic']
    curves = d['coarse_curve']
    theta = [p['angle_deg'] for p in curves]
    ax.plot(theta, [p['cost'] for p in curves], color='#175b9d', label='Gradient appearance cost')
    ax.axhline(.75, color='#777777', lw=.8, ls=':', label='Appearance limit')
    ax.set(xlim=(0, 360), ylim=(.35, 1.25), xticks=[0, 90, 180, 270, 360],
           xlabel='Spin about measured CAD axis (degrees)', ylabel='1 − correlation')
    geo = ax.twinx()
    geo.plot(theta, [p['geometric_residual_m']*1000 for p in curves], color='#ba671b', alpha=.8,
             label='Symmetric trimmed CAD distance')
    geo.set(ylim=(5, 11), ylabel='Geometric residual (mm)')
    state = 'UPDATE APPLIED' if d['update_applied'] else 'AMBIGUOUS: BASELINE RETAINED'
    color = '#20814a' if d['update_applied'] else '#666666'
    ax.axvline(d['selected']['angle_deg'] % 360, color=color, ls='--', lw=1.2)
    ax.set_title(f"Frame {row['frame_index']} · {state}\npeak gap {d['separated_peak_cost_gap']:.3f}", color=color)
    ax.grid(alpha=.2)
handles, labels = axes.flat[0].get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color='#ba671b'))
labels.append('Geometric residual (right axis)')
fig.legend(handles, labels, loc='outside lower center', ncol=3)
fig.suptitle('Episode 2: fixed translation and axis tilt, full 360° spin sweep\n'
             'Eight development samples; no GT read during inference. Geometry varies by about 1 mm.', fontsize=12)
fig.savefig(out/'spin_curves.png', dpi=170)
fig.savefig(out/'spin_curves.pdf')
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
frames = [p['frame_index'] for p in pred['frames']]
old = np.array([p['pair_errors'][0]['canonical_rotation_error_deg'] for p in before['frames']])
new = np.array([p['pair_errors'][0]['canonical_rotation_error_deg'] for p in after['frames']])
for i, row in enumerate(pred['frames']):
    color = '#20814a' if row['lid_spin_diagnostic']['update_applied'] else '#999999'
    ax.plot([i-.12, i+.12], [old[i], new[i]], color=color, lw=2)
ax.scatter(np.arange(8)-.12, old, color='#175b9d', label='Published pot-support baseline', zorder=3)
ax.scatter(np.arange(8)+.12, new, color='#20814a', marker='s', label='Guarded lid spin', zorder=3)
ax.axhline(5, color='#ba671b', ls='--', label='5° criterion (not met)')
ax.set(xticks=np.arange(8), xticklabels=frames, ylabel='Canonical relative rotation error (degrees)',
       xlabel='Source frame index', ylim=(0, 160),
       title=f'Episode 2 median: {np.median(old):.2f}° → {np.median(new):.2f}°\n'
             'Four lid updates; occluded-pot errors persist. Ordered positional RPE remains 7.09 cm.')
ax.grid(axis='y', alpha=.2)
ax.legend(loc='upper left')
fig.savefig(out/'canonical_rotation_errors.png', dpi=170)
fig.savefig(out/'canonical_rotation_errors.pdf')
print(out)

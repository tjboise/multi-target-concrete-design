import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import glob, os

def load_pareto(pattern):
    fronts = []
    for p in sorted(glob.glob(pattern)):
        df = pd.read_csv(p)
        fronts.append(df[['GWP','28day']].values)
    return fronts

hyb_fronts  = load_pareto(r'results\grid_hyb_f20_n10_rep0*\pareto_front.csv')
base_fronts = load_pareto(r'results\grid_base_f20_n10_rep0*\pareto_front.csv')
ref_df      = pd.read_csv(r'results\nsga2_reference.csv')
print(f"Loaded {len(hyb_fronts)} hybrid fronts, {len(base_fronts)} baseline fronts, {len(ref_df)} ref pts")

def avg_front(fronts, x_min=112, x_max=270, n=120, gwp_cap=300):
    xs = np.linspace(x_min, x_max, n)
    ys_all = []
    for front in fronts:
        front = front[front[:, 0] <= gwp_cap]
        idx   = np.argsort(front[:, 0])
        gx, gy = front[idx, 0], front[idx, 1]
        yi = np.interp(xs, gx, gy, left=gy[0], right=gy[-1])
        ys_all.append(yi)
    return xs, np.mean(ys_all, axis=0)

xs_hyb,  ys_hyb  = avg_front(hyb_fronts)
xs_base, ys_base = avg_front(base_fronts)
ref_sorted = ref_df[ref_df['gwp'] <= 300].sort_values('gwp')

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(ref_sorted['gwp'], ref_sorted['pred_28day'],
        color='#888', lw=1.8, ls=(0,(6,3)), label='Reference (200 gen × 100 pop)')
ax.plot(xs_base, ys_base,
        color='#2a78d6', lw=2.2, ls=(0,(5,3)), label='NSGA-II baseline (avg of 5 runs)')
ax.plot(xs_hyb, ys_hyb,
        color='#eb6834', lw=2.4, label='LLM-hybrid F=20, N=10 (avg of 5 runs)')

mid_x = 195
mid_y = (float(np.interp(mid_x, xs_hyb, ys_hyb)) + float(np.interp(mid_x, xs_base, ys_base))) / 2
ax.annotate('+6.07% HV advantage', xy=(mid_x, mid_y),
            xytext=(212, 56.5), fontsize=9, color='#c04a10',
            arrowprops=dict(arrowstyle='->', color='#c04a10', lw=1.2))

ax.set_xlabel('GWP (kg CO₂-eq / m³)', fontsize=11)
ax.set_ylabel('28-day compressive strength (MPa)', fontsize=11)
ax.set_xlim(108, 278)
ax.set_ylim(38, 80)
ax.grid(True, color='#ddd', lw=0.6)
ax.legend(fontsize=9, framealpha=0.9)
ax.tick_params(labelsize=9)
fig.suptitle('Pareto Front: LLM-Hybrid vs NSGA-II Baseline\n'
             'pop=50, gen=100, no knowledge table', fontsize=11, y=1.0)
fig.tight_layout()

os.makedirs(os.path.join('results', 'figures'), exist_ok=True)
out = os.path.join('results', 'figures', 'pareto_front_hybrid_vs_nsga2.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")

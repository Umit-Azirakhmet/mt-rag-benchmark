#!/usr/bin/env python3
"""
Generate visualizations for the retrieval experiment report.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load experiment data
backup_dir = Path("backup-retrieval-1126/retrieval")
with open(backup_dir / "experiment_summary.json", 'r') as f:
    data = json.load(f)

# Extract experiment results
experiments = []
for phase in ['phase1', 'phase2', 'phase3']:
    if phase in data['results']:
        for exp in data['results'][phase]:
            exp_name = exp['experiment']
            results = exp['results']
            experiments.append({
                'name': exp_name,
                'phase': phase,
                'r5': results['averages']['Recall@5'],
                'r10': results['averages']['Recall@10'],
                'ndcg5': results['averages']['nDCG@5'],
                'ndcg10': results['averages']['nDCG@10'],
                'domains': results['domain_results']
            })

# Baseline values
baseline_r10 = 0.38
baseline_ndcg10 = 0.30

# Create output directory
output_dir = Path("report_figures")
output_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# 1. Overall Performance Comparison (Recall@10 and nDCG@10)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

exp_names = [e['name'] for e in experiments]
r10_values = [e['r10'] for e in experiments]
ndcg10_values = [e['ndcg10'] for e in experiments]

x_pos = np.arange(len(exp_names))
width = 0.6

ax1.barh(x_pos, r10_values, width, label='Our Results', color='steelblue')
ax1.axvline(baseline_r10, color='red', linestyle='--', linewidth=2, label='Baseline (0.38)')
ax1.set_yticks(x_pos)
ax1.set_yticklabels(exp_names, fontsize=9)
ax1.set_xlabel('Recall@10', fontsize=12, fontweight='bold')
ax1.set_title('Recall@10 Comparison Across Experiments', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

ax2.barh(x_pos, ndcg10_values, width, label='Our Results', color='darkgreen')
ax2.axvline(baseline_ndcg10, color='red', linestyle='--', linewidth=2, label='Baseline (0.30)')
ax2.set_yticks(x_pos)
ax2.set_yticklabels(exp_names, fontsize=9)
ax2.set_xlabel('nDCG@10', fontsize=12, fontweight='bold')
ax2.set_title('nDCG@10 Comparison Across Experiments', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'overall_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Improvement over Baseline
fig, ax = plt.subplots(figsize=(12, 6))

improvements_r10 = [(e['r10'] - baseline_r10) * 100 for e in experiments]
improvements_ndcg10 = [(e['ndcg10'] - baseline_ndcg10) * 100 for e in experiments]

x = np.arange(len(exp_names))
width = 0.35

bars1 = ax.bar(x - width/2, improvements_r10, width, label='Recall@10 Improvement (%)', color='steelblue')
bars2 = ax.bar(x + width/2, improvements_ndcg10, width, label='nDCG@10 Improvement (%)', color='darkgreen')

ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax.set_ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Improvement over Baseline', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'improvement_over_baseline.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'improvement_over_baseline.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Phase-wise Progression
fig, ax = plt.subplots(figsize=(12, 6))

phase1_exps = [e for e in experiments if e['phase'] == 'phase1']
phase2_exps = [e for e in experiments if e['phase'] == 'phase2']
phase3_exps = [e for e in experiments if e['phase'] == 'phase3']

phase1_r10 = [e['r10'] for e in phase1_exps]
phase2_r10 = [e['r10'] for e in phase2_exps]
phase3_r10 = [e['r10'] for e in phase3_exps] if phase3_exps else []

x1 = np.arange(len(phase1_exps))
x2 = np.arange(len(phase2_exps))
x3 = np.arange(len(phase3_exps)) if phase3_exps else []

ax.plot(x1, phase1_r10, 'o-', label='Phase 1', linewidth=2, markersize=8, color='blue')
ax.plot(x2, phase2_r10, 's-', label='Phase 2', linewidth=2, markersize=8, color='green')
if phase3_exps:
    ax.plot(x3, phase3_r10, '^-', label='Phase 3', linewidth=2, markersize=8, color='orange')
ax.axhline(baseline_r10, color='red', linestyle='--', linewidth=2, label='Baseline')

ax.set_xlabel('Experiment Index in Phase', fontsize=12, fontweight='bold')
ax.set_ylabel('Recall@10', fontsize=12, fontweight='bold')
ax.set_title('Performance Progression Across Phases', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'phase_progression.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'phase_progression.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Domain-specific Performance (Best Model: phase2_augmentation)
best_exp = next(e for e in experiments if e['name'] == 'phase2_augmentation')
domains = ['clapnq', 'fiqa', 'govt', 'cloud']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

domain_r10 = [best_exp['domains'][d]['Recall@10'] for d in domains]
domain_ndcg10 = [best_exp['domains'][d]['nDCG@10'] for d in domains]

x = np.arange(len(domains))
width = 0.6

bars1 = ax1.bar(x, domain_r10, width, color='steelblue', alpha=0.8)
ax1.set_xlabel('Domain', fontsize=12, fontweight='bold')
ax1.set_ylabel('Recall@10', fontsize=12, fontweight='bold')
ax1.set_title('Domain-specific Recall@10 (Best Model: phase2_augmentation)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(domains, fontsize=10)
ax1.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars1, domain_r10)):
    ax1.text(bar.get_x() + bar.get_width()/2., val,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)

bars2 = ax2.bar(x, domain_ndcg10, width, color='darkgreen', alpha=0.8)
ax2.set_xlabel('Domain', fontsize=12, fontweight='bold')
ax2.set_ylabel('nDCG@10', fontsize=12, fontweight='bold')
ax2.set_title('Domain-specific nDCG@10 (Best Model: phase2_augmentation)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(domains, fontsize=10)
ax2.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars2, domain_ndcg10)):
    ax2.text(bar.get_x() + bar.get_width()/2., val,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'domain_performance.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'domain_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Heatmap: All Metrics for All Experiments
fig, ax = plt.subplots(figsize=(14, 8))

metrics = ['R@5', 'R@10', 'nDCG@5', 'nDCG@10']
metric_data = {
    'R@5': [e['r5'] for e in experiments],
    'R@10': [e['r10'] for e in experiments],
    'nDCG@5': [e['ndcg5'] for e in experiments],
    'nDCG@10': [e['ndcg10'] for e in experiments]
}

data_matrix = np.array([metric_data[m] for m in metrics])

im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
ax.set_xticks(np.arange(len(exp_names)))
ax.set_yticks(np.arange(len(metrics)))
ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(metrics, fontsize=10)

# Add text annotations
for i in range(len(metrics)):
    for j in range(len(exp_names)):
        text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=7)

ax.set_title('Performance Heatmap: All Metrics Across Experiments', fontsize=14, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax, label='Score')
plt.tight_layout()
plt.savefig(output_dir / 'performance_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ All visualizations saved to {output_dir}/")


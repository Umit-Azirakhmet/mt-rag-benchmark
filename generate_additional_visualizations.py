#!/usr/bin/env python3
"""
Generate additional detailed visualizations for the retrieval experiment report.
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
            config = results.get('config', {})
            experiments.append({
                'name': exp_name,
                'phase': phase,
                'r5': results['averages']['Recall@5'],
                'r10': results['averages']['Recall@10'],
                'ndcg5': results['averages']['nDCG@5'],
                'ndcg10': results['averages']['nDCG@10'],
                'domains': results['domain_results'],
                'epochs': config.get('epochs', 'N/A'),
                'batch_size': config.get('batch_size', 'N/A'),
                'lr': config.get('learning_rate', 'N/A'),
                'augmentation': config.get('use_augmentation', False),
                'validation': config.get('use_validation', False)
            })

# Baseline values
baseline_r10 = 0.38
baseline_ndcg10 = 0.30
baseline_r5 = 0.30
baseline_ndcg5 = 0.27

# Create output directory
output_dir = Path("report_figures")
output_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# 1. Detailed Configuration Comparison Table Visualization
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

table_data = []
headers = ['Experiment', 'Phase', 'Epochs', 'Batch Size', 'LR', 'Augment', 'Valid', 'R@5', 'R@10', 'nDCG@5', 'nDCG@10']

for exp in experiments:
    lr_str = f"{exp['lr']:.0e}" if isinstance(exp['lr'], float) else str(exp['lr'])
    table_data.append([
        exp['name'],
        exp['phase'],
        str(exp['epochs']),
        str(exp['batch_size']),
        lr_str,
        'Yes' if exp['augmentation'] else 'No',
        'Yes' if exp['validation'] else 'No',
        f"{exp['r5']:.4f}",
        f"{exp['r10']:.4f}",
        f"{exp['ndcg5']:.4f}",
        f"{exp['ndcg10']:.4f}"
    ])

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

# Color code by phase
for i in range(len(experiments)):
    phase = experiments[i]['phase']
    if phase == 'phase1':
        color = '#E8F4F8'
    elif phase == 'phase2':
        color = '#E8F8E8'
    else:
        color = '#FFF8E8'
    for j in range(len(headers)):
        table[(i+1, j)].set_facecolor(color)

# Highlight header
for j in range(len(headers)):
    table[(0, j)].set_facecolor('#4A90E2')
    table[(0, j)].set_text_props(weight='bold', color='white')

plt.title('Detailed Experiment Configuration and Results', fontsize=16, fontweight='bold', pad=20)
plt.savefig(output_dir / 'detailed_config_table.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'detailed_config_table.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. All Metrics Comparison (R@5, R@10, nDCG@5, nDCG@10)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

exp_names = [e['name'] for e in experiments]
metrics_data = {
    'Recall@5': [e['r5'] for e in experiments],
    'Recall@10': [e['r10'] for e in experiments],
    'nDCG@5': [e['ndcg5'] for e in experiments],
    'nDCG@10': [e['ndcg10'] for e in experiments]
}

baselines = {
    'Recall@5': baseline_r5,
    'Recall@10': baseline_r10,
    'nDCG@5': baseline_ndcg5,
    'nDCG@10': baseline_ndcg10
}

axes_flat = axes.flatten()
for idx, (metric, values) in enumerate(metrics_data.items()):
    ax = axes_flat[idx]
    x_pos = np.arange(len(exp_names))
    bars = ax.barh(x_pos, values, 0.6, color='steelblue', alpha=0.8)
    ax.axvline(baselines[metric], color='red', linestyle='--', linewidth=2, label=f'Baseline ({baselines[metric]:.3f})')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(exp_names, fontsize=8)
    ax.set_xlabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val, i, f' {val:.3f}', va='center', fontsize=7)

plt.suptitle('Comprehensive Metrics Comparison Across All Experiments', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(output_dir / 'all_metrics_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'all_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Domain-wise Performance for All Experiments
domains = ['clapnq', 'fiqa', 'govt', 'cloud']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, domain in enumerate(domains):
    ax = axes[idx // 2, idx % 2]
    
    domain_r10 = [e['domains'].get(domain, {}).get('Recall@10', 0) for e in experiments]
    domain_ndcg10 = [e['domains'].get(domain, {}).get('nDCG@10', 0) for e in experiments]
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, domain_r10, width, label='Recall@10', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, domain_ndcg10, width, label='nDCG@10', color='darkgreen', alpha=0.8)
    
    ax.set_xlabel('Experiment', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax.set_title(f'{domain.upper()} Domain Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(max(domain_r10), max(domain_ndcg10)) * 1.15])

plt.suptitle('Domain-specific Performance Across All Experiments', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(output_dir / 'domain_wise_all_experiments.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'domain_wise_all_experiments.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Training Configuration Impact Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Epochs impact
phase1_exps = [e for e in experiments if e['phase'] == 'phase1']
epochs_data = {'epochs': [], 'r10': [], 'ndcg10': []}
for exp in phase1_exps:
    if isinstance(exp['epochs'], (int, float)):
        epochs_data['epochs'].append(exp['epochs'])
        epochs_data['r10'].append(exp['r10'])
        epochs_data['ndcg10'].append(exp['ndcg10'])

if epochs_data['epochs']:
    ax = axes[0, 0]
    ax.plot(epochs_data['epochs'], epochs_data['r10'], 'o-', linewidth=2, markersize=10, label='Recall@10', color='blue')
    ax.plot(epochs_data['epochs'], epochs_data['ndcg10'], 's-', linewidth=2, markersize=10, label='nDCG@10', color='green')
    ax.axhline(baseline_r10, color='red', linestyle='--', linewidth=1.5, label='Baseline R@10')
    ax.axhline(baseline_ndcg10, color='orange', linestyle='--', linewidth=1.5, label='Baseline nDCG@10')
    ax.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Training Epochs (Phase 1)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

# Learning rate impact
phase2_lr_exps = [e for e in experiments if e['phase'] == 'phase2' and isinstance(e['lr'], float)]
if phase2_lr_exps:
    ax = axes[0, 1]
    lr_values = [e['lr'] for e in phase2_lr_exps]
    r10_values = [e['r10'] for e in phase2_lr_exps]
    ndcg10_values = [e['ndcg10'] for e in phase2_lr_exps]
    
    ax.plot(lr_values, r10_values, 'o-', linewidth=2, markersize=10, label='Recall@10', color='blue')
    ax.plot(lr_values, ndcg10_values, 's-', linewidth=2, markersize=10, label='nDCG@10', color='green')
    ax.axhline(baseline_r10, color='red', linestyle='--', linewidth=1.5, label='Baseline R@10')
    ax.axhline(baseline_ndcg10, color='orange', linestyle='--', linewidth=1.5, label='Baseline nDCG@10')
    ax.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Learning Rate (Phase 2)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

# Augmentation impact
aug_exp = next((e for e in experiments if e['name'] == 'phase2_augmentation'), None)
no_aug_exp = next((e for e in experiments if e['phase'] == 'phase2' and not e['augmentation'] and e['name'] != 'phase2_cosine_loss'), None)
if aug_exp and no_aug_exp:
    ax = axes[1, 0]
    categories = ['Without\nAugmentation', 'With\nAugmentation']
    r10_vals = [no_aug_exp['r10'], aug_exp['r10']]
    ndcg10_vals = [no_aug_exp['ndcg10'], aug_exp['ndcg10']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, r10_vals, width, label='Recall@10', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, ndcg10_vals, width, label='nDCG@10', color='darkgreen', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Data Augmentation', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Phase comparison
ax = axes[1, 1]
phase_avg_r10 = {}
phase_avg_ndcg10 = {}
for phase in ['phase1', 'phase2', 'phase3']:
    phase_exps = [e for e in experiments if e['phase'] == phase]
    if phase_exps:
        phase_avg_r10[phase] = np.mean([e['r10'] for e in phase_exps])
        phase_avg_ndcg10[phase] = np.mean([e['ndcg10'] for e in phase_exps])

if phase_avg_r10:
    phases = list(phase_avg_r10.keys())
    r10_vals = [phase_avg_r10[p] for p in phases]
    ndcg10_vals = [phase_avg_ndcg10[p] for p in phases]
    
    x = np.arange(len(phases))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, r10_vals, width, label='Recall@10', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, ndcg10_vals, width, label='nDCG@10', color='darkgreen', alpha=0.8)
    
    ax.set_xlabel('Phase', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=11, fontweight='bold')
    ax.set_title('Average Performance by Phase', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Phase 1', 'Phase 2', 'Phase 3'], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(baseline_r10, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(baseline_ndcg10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

plt.suptitle('Training Configuration Impact Analysis', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(output_dir / 'configuration_impact.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'configuration_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Improvement Percentage Visualization
fig, ax = plt.subplots(figsize=(14, 8))

exp_names_short = [e['name'].replace('phase', 'P') for e in experiments]
improvements_r10 = [(e['r10'] - baseline_r10) / baseline_r10 * 100 for e in experiments]
improvements_ndcg10 = [(e['ndcg10'] - baseline_ndcg10) / baseline_ndcg10 * 100 for e in experiments]

x = np.arange(len(exp_names_short))
width = 0.35

bars1 = ax.bar(x - width/2, improvements_r10, width, label='Recall@10 Improvement (%)', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, improvements_ndcg10, width, label='nDCG@10 Improvement (%)', color='darkgreen', alpha=0.8)

ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax.set_ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
ax.set_title('Percentage Improvement over Baseline', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names_short, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 1:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'improvement_percentage.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'improvement_percentage.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Best Model Domain Breakdown (Detailed)
best_exp = next((e for e in experiments if e['name'] == 'phase2_augmentation'), None)
if best_exp:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    domains = ['clapnq', 'fiqa', 'govt', 'cloud']
    domain_r5 = [best_exp['domains'][d]['Recall@5'] for d in domains]
    domain_r10 = [best_exp['domains'][d]['Recall@10'] for d in domains]
    domain_ndcg5 = [best_exp['domains'][d]['nDCG@5'] for d in domains]
    domain_ndcg10 = [best_exp['domains'][d]['nDCG@10'] for d in domains]
    
    x = np.arange(len(domains))
    width = 0.2
    
    ax = axes[0]
    ax.bar(x - 1.5*width, domain_r5, width, label='Recall@5', color='#3498db', alpha=0.8)
    ax.bar(x - 0.5*width, domain_r10, width, label='Recall@10', color='#2980b9', alpha=0.8)
    ax.bar(x + 0.5*width, domain_ndcg5, width, label='nDCG@5', color='#27ae60', alpha=0.8)
    ax.bar(x + 1.5*width, domain_ndcg10, width, label='nDCG@10', color='#229954', alpha=0.8)
    
    ax.set_xlabel('Domain', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Best Model (phase2_augmentation): All Metrics by Domain', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in domains], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 0.65])
    
    # Add value labels
    for i, domain in enumerate(domains):
        ax.text(i, domain_r10[i] + 0.02, f'{domain_r10[i]:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Radar chart style comparison
    ax = axes[1]
    metrics = ['R@5', 'R@10', 'nDCG@5', 'nDCG@10']
    domain_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    x_metrics = np.arange(len(metrics))
    for i, domain in enumerate(domains):
        values = [
            best_exp['domains'][domain]['Recall@5'],
            best_exp['domains'][domain]['Recall@10'],
            best_exp['domains'][domain]['nDCG@5'],
            best_exp['domains'][domain]['nDCG@10']
        ]
        ax.plot(x_metrics, values, 'o-', linewidth=2, markersize=8, 
               label=domain.upper(), color=domain_colors[i], alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Best Model: Domain Comparison Across Metrics', fontsize=13, fontweight='bold')
    ax.set_xticks(x_metrics)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 0.65])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_model_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'best_model_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"✅ All additional visualizations saved to {output_dir}/")


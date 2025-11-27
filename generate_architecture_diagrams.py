#!/usr/bin/env python3
"""
Generate architecture and methodology diagrams for the conference paper.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Set style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'

output_dir = "report_figures"
import pathlib
pathlib.Path(output_dir).mkdir(exist_ok=True)

# 1. System Architecture Diagram
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(5, 7.5, 'Multi-Turn RAG Retrieval System Architecture', 
        ha='center', fontsize=16, fontweight='bold')

# Input Layer
input_box = FancyBboxPatch((1, 6), 8, 0.8, boxstyle="round,pad=0.1", 
                           facecolor='#E8F4F8', edgecolor='#3498db', linewidth=2)
ax.add_patch(input_box)
ax.text(5, 6.4, 'Multi-Turn Conversations', ha='center', fontsize=12, fontweight='bold')

# Query Processing
query_box = FancyBboxPatch((1, 4.5), 3.5, 1, boxstyle="round,pad=0.1", 
                           facecolor='#FFF8E8', edgecolor='#f39c12', linewidth=2)
ax.add_patch(query_box)
ax.text(2.75, 5.2, 'Query\nProcessing', ha='center', fontsize=11, fontweight='bold', va='center')
ax.text(2.75, 4.8, '• Query Extraction\n• Context Handling', ha='center', fontsize=9, va='center')

# Fine-tuned BGE Model
model_box = FancyBboxPatch((5.5, 4.5), 3.5, 1, boxstyle="round,pad=0.1", 
                           facecolor='#E8F8E8', edgecolor='#27ae60', linewidth=2)
ax.add_patch(model_box)
ax.text(7.25, 5.2, 'Fine-tuned BGE\nEmbedding Model', ha='center', fontsize=11, fontweight='bold', va='center')
ax.text(7.25, 4.8, 'BAAI/bge-base-en-v1.5\n+ Domain Fine-tuning', ha='center', fontsize=9, va='center')

# Arrows
arrow1 = FancyArrowPatch((5, 6.2), (2.75, 5.5), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='#34495e')
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((4.5, 5), (5.5, 5), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='#34495e')
ax.add_patch(arrow2)

# Retrieval Layer
retrieval_box = FancyBboxPatch((1, 2.5), 8, 1, boxstyle="round,pad=0.1", 
                               facecolor='#F8E8E8', edgecolor='#e74c3c', linewidth=2)
ax.add_patch(retrieval_box)
ax.text(5, 3.2, 'Dense Retrieval', ha='center', fontsize=12, fontweight='bold')
ax.text(2.5, 2.8, '• Corpus Embedding', ha='left', fontsize=10)
ax.text(5, 2.8, '• Similarity Search', ha='center', fontsize=10)
ax.text(7.5, 2.8, '• Top-K Retrieval', ha='right', fontsize=10)

# Arrow from model to retrieval
arrow3 = FancyArrowPatch((5, 4.5), (5, 3.5), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='#34495e')
ax.add_patch(arrow3)

# Output
output_box = FancyBboxPatch((1, 0.5), 8, 1, boxstyle="round,pad=0.1", 
                            facecolor='#F0F0F0', edgecolor='#7f8c8d', linewidth=2)
ax.add_patch(output_box)
ax.text(5, 1.2, 'Retrieved Documents', ha='center', fontsize=12, fontweight='bold')
ax.text(5, 0.8, 'Top-K Relevant Passages (K=5, 10)', ha='center', fontsize=10)

# Arrow from retrieval to output
arrow4 = FancyArrowPatch((5, 2.5), (5, 1.5), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='#34495e')
ax.add_patch(arrow4)

plt.tight_layout()
plt.savefig(f'{output_dir}/system_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/system_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Fine-tuning Pipeline Diagram
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

ax.text(7, 5.5, 'Fine-tuning Pipeline', ha='center', fontsize=16, fontweight='bold')

# Base Model
base_box = FancyBboxPatch((0.5, 3.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                         facecolor='#E8F4F8', edgecolor='#3498db', linewidth=2)
ax.add_patch(base_box)
ax.text(1.75, 4.5, 'Base Model', ha='center', fontsize=11, fontweight='bold')
ax.text(1.75, 4.1, 'BAAI/bge-base-en-v1.5', ha='center', fontsize=9)
ax.text(1.75, 3.8, 'Pre-trained', ha='center', fontsize=9, style='italic')

# Training Data
data_box = FancyBboxPatch((0.5, 1), 2.5, 1.5, boxstyle="round,pad=0.1", 
                         facecolor='#FFF8E8', edgecolor='#f39c12', linewidth=2)
ax.add_patch(data_box)
ax.text(1.75, 2, 'Training Data', ha='center', fontsize=11, fontweight='bold')
ax.text(1.75, 1.6, 'MT-RAG Domains', ha='center', fontsize=9)
ax.text(1.75, 1.3, 'Query-Passage Pairs', ha='center', fontsize=9)

# Arrow
arrow1 = FancyArrowPatch((3, 2.75), (4, 2.75), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='#34495e')
ax.add_patch(arrow1)

# Fine-tuning Process
train_box = FancyBboxPatch((4, 1.5), 3, 3, boxstyle="round,pad=0.1", 
                          facecolor='#E8F8E8', edgecolor='#27ae60', linewidth=2)
ax.add_patch(train_box)
ax.text(5.5, 4.2, 'Fine-tuning', ha='center', fontsize=12, fontweight='bold')
ax.text(5.5, 3.8, '• Epochs: 3-5', ha='center', fontsize=9)
ax.text(5.5, 3.5, '• Batch Size: 32', ha='center', fontsize=9)
ax.text(5.5, 3.2, '• LR: 2e-5', ha='center', fontsize=9)
ax.text(5.5, 2.9, '• Loss: MNR', ha='center', fontsize=9)
ax.text(5.5, 2.6, '• Augmentation: Yes', ha='center', fontsize=9)
ax.text(5.5, 2.3, '• Validation: Yes', ha='center', fontsize=9)

# Arrow
arrow2 = FancyArrowPatch((7, 3), (8, 3), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='#34495e')
ax.add_patch(arrow2)

# Fine-tuned Model
ft_box = FancyBboxPatch((8, 1.5), 2.5, 3, boxstyle="round,pad=0.1", 
                        facecolor='#F8E8E8', edgecolor='#e74c3c', linewidth=2)
ax.add_patch(ft_box)
ax.text(9.25, 4.2, 'Fine-tuned', ha='center', fontsize=11, fontweight='bold')
ax.text(9.25, 3.9, 'BGE Model', ha='center', fontsize=11, fontweight='bold')
ax.text(9.25, 3.5, 'Domain-adapted', ha='center', fontsize=9, style='italic')
ax.text(9.25, 3.2, 'Embeddings', ha='center', fontsize=9)

# Evaluation
eval_box = FancyBboxPatch((11, 1.5), 2.5, 3, boxstyle="round,pad=0.1", 
                          facecolor='#F0F0F0', edgecolor='#7f8c8d', linewidth=2)
ax.add_patch(eval_box)
ax.text(12.25, 4.2, 'Evaluation', ha='center', fontsize=11, fontweight='bold')
ax.text(12.25, 3.8, '• Recall@K', ha='center', fontsize=9)
ax.text(12.25, 3.5, '• nDCG@K', ha='center', fontsize=9)
ax.text(12.25, 3.2, '• 4 Domains', ha='center', fontsize=9)
ax.text(12.25, 2.9, '• Test Set', ha='center', fontsize=9)

# Arrow
arrow3 = FancyArrowPatch((10.5, 3), (11, 3), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='#34495e')
ax.add_patch(arrow3)

plt.tight_layout()
plt.savefig(f'{output_dir}/fine_tuning_pipeline.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/fine_tuning_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Experimental Setup Diagram
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

ax.text(5, 7.5, 'Experimental Setup: Three-Phase Approach', 
        ha='center', fontsize=16, fontweight='bold')

# Phase 1
phase1_box = FancyBboxPatch((0.5, 5), 2.8, 2, boxstyle="round,pad=0.1", 
                            facecolor='#E8F4F8', edgecolor='#3498db', linewidth=2)
ax.add_patch(phase1_box)
ax.text(1.9, 6.5, 'Phase 1', ha='center', fontsize=12, fontweight='bold')
ax.text(1.9, 6.2, 'Quick Wins', ha='center', fontsize=11, fontweight='bold')
ax.text(1.9, 5.8, '• Epochs: 1→3→5', ha='center', fontsize=9)
ax.text(1.9, 5.5, '• Batch: 16→32', ha='center', fontsize=9)
ax.text(1.9, 5.2, '• Validation: Yes', ha='center', fontsize=9)

# Phase 2
phase2_box = FancyBboxPatch((3.6, 5), 2.8, 2, boxstyle="round,pad=0.1", 
                            facecolor='#E8F8E8', edgecolor='#27ae60', linewidth=2)
ax.add_patch(phase2_box)
ax.text(5, 6.5, 'Phase 2', ha='center', fontsize=12, fontweight='bold')
ax.text(5, 6.2, 'Training Improvements', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 5.8, '• LR: 1e-5, 5e-5', ha='center', fontsize=9)
ax.text(5, 5.5, '• Data Augmentation', ha='center', fontsize=9)
ax.text(5, 5.2, '• Best Model', ha='center', fontsize=9, fontweight='bold')

# Phase 3
phase3_box = FancyBboxPatch((6.7, 5), 2.8, 2, boxstyle="round,pad=0.1", 
                            facecolor='#FFF8E8', edgecolor='#f39c12', linewidth=2)
ax.add_patch(phase3_box)
ax.text(8.1, 6.5, 'Phase 3', ha='center', fontsize=12, fontweight='bold')
ax.text(8.1, 6.2, 'Retrieval Strategy', ha='center', fontsize=11, fontweight='bold')
ax.text(8.1, 5.8, '• Hybrid (Dense+BM25)', ha='center', fontsize=9)
ax.text(8.1, 5.5, '• Reranking', ha='center', fontsize=9)
ax.text(8.1, 5.2, '• Future Work', ha='center', fontsize=9, style='italic')

# Arrows
arrow1 = FancyArrowPatch((3.3, 6), (3.6, 6), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='#34495e')
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((6.4, 6), (6.7, 6), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='#34495e')
ax.add_patch(arrow2)

# Domains
domain_box = FancyBboxPatch((1, 2), 8, 1.5, boxstyle="round,pad=0.1", 
                           facecolor='#F0F0F0', edgecolor='#7f8c8d', linewidth=2)
ax.add_patch(domain_box)
ax.text(5, 3.2, 'Evaluation Domains', ha='center', fontsize=12, fontweight='bold')
ax.text(2.5, 2.6, 'CLAPNQ', ha='center', fontsize=10, fontweight='bold')
ax.text(4, 2.6, 'FIQA', ha='center', fontsize=10, fontweight='bold')
ax.text(5.5, 2.6, 'GOVT', ha='center', fontsize=10, fontweight='bold')
ax.text(7, 2.6, 'CLOUD', ha='center', fontsize=10, fontweight='bold')
ax.text(2.5, 2.3, 'Legal/Patents', ha='center', fontsize=8)
ax.text(4, 2.3, 'Financial', ha='center', fontsize=8)
ax.text(5.5, 2.3, 'Government', ha='center', fontsize=8)
ax.text(7, 2.3, 'Cloud Tech', ha='center', fontsize=8)

# Metrics
metrics_box = FancyBboxPatch((1, 0.2), 8, 1, boxstyle="round,pad=0.1", 
                             facecolor='#F8E8E8', edgecolor='#e74c3c', linewidth=2)
ax.add_patch(metrics_box)
ax.text(5, 0.9, 'Evaluation Metrics', ha='center', fontsize=12, fontweight='bold')
ax.text(2.5, 0.5, 'Recall@5', ha='center', fontsize=10)
ax.text(4, 0.5, 'Recall@10', ha='center', fontsize=10)
ax.text(5.5, 0.5, 'nDCG@5', ha='center', fontsize=10)
ax.text(7, 0.5, 'nDCG@10', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/experimental_setup.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/experimental_setup.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Data Flow Diagram
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')

ax.text(6, 3.5, 'Data Flow in Retrieval System', ha='center', fontsize=16, fontweight='bold')

# Components
components = [
    ('Query', 1, 2, '#E8F4F8', '#3498db'),
    ('Embedding', 3.5, 2, '#E8F8E8', '#27ae60'),
    ('Corpus', 6, 2, '#FFF8E8', '#f39c12'),
    ('Similarity', 8.5, 2, '#F8E8E8', '#e74c3c'),
    ('Top-K', 10.5, 2, '#F0F0F0', '#7f8c8d')
]

for i, (label, x, y, face, edge) in enumerate(components):
    box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.05", 
                         facecolor=face, edgecolor=edge, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', fontsize=10, fontweight='bold')
    
    if i < len(components) - 1:
        arrow = FancyArrowPatch((x+0.4, y), (x+0.9, y), 
                               arrowstyle='->', mutation_scale=15, linewidth=1.5, color='#34495e')
        ax.add_patch(arrow)

# Labels below
ax.text(1, 1, 'User Query', ha='center', fontsize=9)
ax.text(3.5, 1, 'Query\nEmbedding', ha='center', fontsize=9)
ax.text(6, 1, 'Document\nEmbeddings', ha='center', fontsize=9)
ax.text(8.5, 1, 'Cosine\nSimilarity', ha='center', fontsize=9)
ax.text(10.5, 1, 'Ranked\nResults', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/data_flow.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/data_flow.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ All architecture diagrams saved to {output_dir}/")



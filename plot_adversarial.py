import matplotlib.pyplot as plt
import seaborn as sns

# Metrics extracted from Phase 3 (Baseline) and Phase 4 (Adversarial)
rounds = [1, 2, 3, 4, 5]

# Loss Comparison
loss_baseline = [102.71, 63.29, 42.37, 33.53, 29.38]
loss_adversarial = [177.36, 199.98, 202.03, 189.13, 197.21]

# Accuracy Tracking (Averaged from terminal logs)
acc_baseline_avg = [0.9388, 0.9767, 0.9871, 0.9892, 0.9908]
acc_adv_good_nodes = [0.9055, 0.9108, 0.9461, 0.9467, 0.9465]
acc_adv_malicious = [0.9339, 0.9273, 0.8864, 0.8596, 0.8502]

sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: The Impact of Poisoning on Global Loss
ax1.plot(rounds, loss_baseline, marker='o', color='#5cb85c', linewidth=2.5, label='Healthy Network (Phase 3)')
ax1.plot(rounds, loss_adversarial, marker='X', color='#d9534f', linewidth=2.5, linestyle='--', label='Poisoned Network (Phase 4)')
ax1.set_title('Impact of Malicious Node on Global Loss', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Communication Round', fontsize=12)
ax1.set_ylabel('Aggregated Training Loss', fontsize=12)
ax1.set_xticks(rounds)
ax1.legend(fontsize=11)

# Plot 2: Accuracy Divergence (Good vs Malicious)
ax2.plot(rounds, acc_baseline_avg, marker='o', color='#5cb85c', linewidth=2.5, label='Baseline Global Accuracy')
ax2.plot(rounds, acc_adv_good_nodes, marker='s', color='#f0ad4e', linewidth=2.5, label='Good Nodes (Under Attack)')
ax2.plot(rounds, acc_adv_malicious, marker='X', color='#d9534f', linewidth=2.5, linestyle='--', label='Malicious Node Accuracy')
ax2.set_title('Client Accuracy Divergence Under Attack', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Communication Round', fontsize=12)
ax2.set_ylabel('Client Detection Accuracy', fontsize=12)
ax2.set_xticks(rounds)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('phase4_adversarial_impact.png', dpi=300, bbox_inches='tight')
print("Graph successfully saved as 'phase4_adversarial_impact.png'")
plt.show()
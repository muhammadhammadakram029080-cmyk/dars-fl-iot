import matplotlib.pyplot as plt
import seaborn as sns

# Data extracted directly from your Phase 3 simulation terminal output
rounds = [1, 2, 3, 4, 5]
loss = [102.71, 63.29, 42.37, 33.53, 29.38]

# Calculated average accuracy across all 4 clients per round
accuracy = [0.9388, 0.9767, 0.9871, 0.9892, 0.9908]

# Set Seaborn styling for professional formatting
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Federated Loss Reduction
ax1.plot(rounds, loss, marker='o', color='#d9534f', linewidth=2.5, markersize=8)
ax1.set_title('Phase 3: Distributed Loss Reduction', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Communication Round', fontsize=12)
ax1.set_ylabel('Aggregated Training Loss', fontsize=12)
ax1.set_xticks(rounds)

# Plot 2: Federated Accuracy Convergence
ax2.plot(rounds, accuracy, marker='s', color='#5cb85c', linewidth=2.5, markersize=8)
ax2.set_title('Phase 3: Threat Detection Accuracy Convergence', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Communication Round', fontsize=12)
ax2.set_ylabel('Average Client Accuracy', fontsize=12)
ax2.set_xticks(rounds)
ax2.set_ylim(0.92, 1.00)

plt.tight_layout()

# Save the high-resolution image for your research paper
plt.savefig('phase3_convergence_metrics.png', dpi=300, bbox_inches='tight')
print("Graph successfully saved as 'phase3_convergence_metrics.png'")
plt.show()
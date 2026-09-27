import matplotlib.pyplot as plt
import seaborn as sns

# --- UPDATE THESE 2 VARIABLES WITH YOUR EXACT TERMINAL OUTPUT ---
centralized_time = 45.50  # Replace with the seconds from your centralized run
centralized_acc = 0.9950  # Replace with the accuracy from your centralized run

# Data from your Phase 3 healthy federated simulation
federated_time = 113.43   
federated_acc = 0.9908    

labels = ['Centralized Baseline', 'Federated (Phase 3)']
times = [centralized_time, federated_time]
accuracies = [centralized_acc, federated_acc]

sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Training Time Comparison
sns.barplot(x=labels, y=times, ax=ax1, palette=['#0275d8', '#5cb85c'])
ax1.set_title('Phase 5: Total Training Time Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Time (Seconds)', fontsize=12)
for i, v in enumerate(times):
    ax1.text(i, v + (max(times)*0.02), f"{v}s", ha='center', fontsize=11, fontweight='bold')

# Plot 2: Accuracy Comparison
sns.barplot(x=labels, y=accuracies, ax=ax2, palette=['#0275d8', '#5cb85c'])
ax2.set_title('Phase 5: Global Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Overall Accuracy', fontsize=12)
ax2.set_ylim(0.90, 1.01) # Set floor to 0.90 to make the difference visually clear
for i, v in enumerate(accuracies):
    ax2.text(i, v + 0.002, f"{v:.4f}", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('phase5_centralized_vs_federated.png', dpi=300, bbox_inches='tight')
print("Graph successfully saved as 'phase5_centralized_vs_federated.png'")
plt.show()
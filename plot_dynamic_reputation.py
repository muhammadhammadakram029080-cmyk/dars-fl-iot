import matplotlib.pyplot as plt
import seaborn as sns

rounds = [1, 2, 3, 4, 5]
# Illustrative data showing the dynamic threshold successfully penalizing the bad node
trust_good = [1.0, 1.0, 1.0, 1.0, 1.0]
trust_malicious = [1.0, 0.7, 0.4, 0.1, 0.0]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
plt.plot(rounds, trust_good, marker='o', color='#5cb85c', linewidth=2.5, label='Healthy Nodes (Trust Score)')
plt.plot(rounds, trust_malicious, marker='X', color='#d9534f', linewidth=2.5, linestyle='--', label='Compromised Node (Trust Score)')
plt.title('Dynamic Reputation System: Malicious Node Isolation', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Adaptive Trust Score', fontsize=12)
plt.xticks(rounds)
plt.ylim(-0.1, 1.1)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('phase4_dynamic_trust_isolation.png', dpi=300, bbox_inches='tight')
print("Graph successfully saved as 'phase4_dynamic_trust_isolation.png'")
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data from your Top-5 tests (for Diff = 1)
data = {
    'Method': [
        'Raw VLAD', 'Super VLAD', 'Raw Fisher', 'Super Fisher',
        'Raw VLAD', 'Super VLAD', 'Raw Fisher', 'Super Fisher',
        'Raw VLAD', 'Super VLAD', 'Raw Fisher', 'Super Fisher'
    ],
    'MPP': [
        '0.5', '0.5', '0.5', '0.5',
        '1.0', '1.0', '1.0', '1.0',
        '2.0', '2.0', '2.0', '2.0'
    ],
    'Success_Rate': [
        49.2, 78.1, 42.5, 43.2,  # 0.5 MPP
        55.8, 72.9, 42.5, 42.5,  # 1.0 MPP
        53.6, 66.4, 42.5, 42.7   # 2.0 MPP
    ]
}

df = pd.DataFrame(data)

# Vytvoření grafu
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Použití palety 'viridis' jako v předchozím případě
ax = sns.barplot(x='Method', y='Success_Rate', hue='MPP', data=df, palette='viridis')

plt.title('UNI-2h consecutive slides (Top-5 Match for Diff=1)', fontsize=15)
plt.ylabel('success rate (%)', fontsize=12)
plt.xlabel('method', fontsize=12)
# Změna limitu na 100, protože u Top-5 jsme už mnohem výše
plt.ylim(0, 100)

# Přidání popisků přímo nad sloupce
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(format(p.get_height(), '.1f') + '%', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=10, fontweight='bold')

plt.legend(title='Resolution (MPP)')
plt.tight_layout()
plt.savefig('srovnani_metod_mpp_top5.png', dpi=300)
plt.show()
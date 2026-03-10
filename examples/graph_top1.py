import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data z tvých testů (pro Diff = 1) - ZŮSTÁVAJÍ STEJNÁ
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
        14.9, 45.4, 8.5, 7.6,  # 0.5 MPP
        17.5, 33.6, 8.5, 8.5,  # 1.0 MPP
        14.8, 25.8, 8.5, 8.3   # 2.0 MPP
    ]
}

df = pd.DataFrame(data)

plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Použití palety viridis jako v tvém originálu
ax = sns.barplot(x='Method', y='Success_Rate', hue='MPP', data=df, palette='viridis')

plt.title('UNI-2h consecutive slides (Top-1 Match for Diff=1)', fontsize=15)
plt.ylabel('success rate (%)', fontsize=12)
plt.xlabel('method', fontsize=12)
plt.ylim(0, 55)

# --- OPRAVENÁ ČÁST PRO POPISKY ---
for p in ax.patches:
    height = p.get_height()
    # Tato podmínka odfiltruje ty "neviditelné" sloupce s nulovou výškou
    if height > 0: 
        ax.annotate(format(height, '.1f') + '%', 
                       (p.get_x() + p.get_width() / 2., height), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=10, fontweight='bold')

plt.legend(title='Resolution (MPP)')
plt.tight_layout()
plt.savefig('srovnani_metod_mpp_top1_FIXED.png', dpi=300)
plt.show()
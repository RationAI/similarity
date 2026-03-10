import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_linking_attack_with_centering(df, column_name):
    # Očištění ID (prefixy a koncovky T1)
    df['sample_id'] = (df['slide_id']
                       .str.replace(r'^(FLASH|MIDI)\d{4}_', '', regex=True)
                       .str.replace(r'T\d+$', 'T', regex=True))
    
    flash_df = df[df['slide_id'].str.contains('FLASH')].copy()
    midi_df = df[df['slide_id'].str.contains('MIDI')].copy()

    # Převedení na matice
    flash_matrix = np.stack(flash_df[column_name].values)
    midi_matrix = np.stack(midi_df[column_name].values)

    # --- CENTERING (VYMAZÁNÍ SCANNER BIASU) ---
    # Tento krok posune těžiště obou skupin do nuly
    flash_matrix -= flash_matrix.mean(axis=0)
    midi_matrix -= midi_matrix.mean(axis=0)

    # Spočítáme podobnost CROSS-SCANNER (FLASH vs MIDI)
    sim_matrix = cosine_similarity(flash_matrix, midi_matrix)
    
    rs_values = []
    for i in range(len(flash_df)):
        sample_id = flash_df['sample_id'].iloc[i]
        
        # Najdeme index dvojčete v midi_df
        matching_row = midi_df[midi_df['sample_id'] == sample_id]
        if matching_row.empty: continue
        
        midi_pos = midi_df.index.get_loc(matching_row.index[0])
        true_score = sim_matrix[i, midi_pos]
        
        # Rs: Kolik jiných MIDI slidů je podobnější než to pravé dvojče?
        rs = np.sum(sim_matrix[i] > true_score)
        rs_values.append(rs)
        
    return rs_values

def calculate_global_linking_with_centering(df, column_name):
    # 1. Očištění ID
    df['sample_id'] = (df['slide_id']
                       .str.replace(r'^(FLASH|MIDI)\d{4}_', '', regex=True)
                       .str.replace(r'T\d+$', 'T', regex=True))
    
    # 2. Rozdělení na skupiny pro Centering
    flash_mask = df['slide_id'].str.contains('FLASH')
    midi_mask = df['slide_id'].str.contains('MIDI')
    
    # Převedení na matice
    # (Předpokládáme, že každý slide patří buď do FLASH nebo MIDI)
    flash_matrix = np.stack(df.loc[flash_mask, column_name].values)
    midi_matrix = np.stack(df.loc[midi_mask, column_name].values)

    # --- CENTERING (VYMAZÁNÍ SCANNER BIASU) ---
    # Centrujeme každou skupinu nezávisle na jejím vlastním průměru
    flash_matrix -= flash_matrix.mean(axis=0)
    midi_matrix -= midi_matrix.mean(axis=0)

    # 3. Spojení vycentrovaných dat zpět do jedné matice v původním pořadí
    # Vytvoříme prázdnou matici stejného tvaru
    combined_matrix = np.zeros((len(df), flash_matrix.shape[1]))
    combined_matrix[flash_mask] = flash_matrix
    combined_matrix[midi_mask] = midi_matrix

    # 4. Spočítáme podobnost "každý s každým" (Globální matice)
    # Výsledek bude čtvercová matice (N x N)
    sim_matrix = cosine_similarity(combined_matrix)
    
    rs_values = []
    for i in range(len(df)):
        current_sample_id = df['sample_id'].iloc[i]
        current_slide_id = df['slide_id'].iloc[i]
        
        # Najdeme skóre pro "pravé dvojče"
        # Dvojče je slide se stejným sample_id, ale jiným slide_id
        # (Abychom nenašli sami sebe)
        match_mask = (df['sample_id'] == current_sample_id) & (df['slide_id'] != current_slide_id)
        matching_rows = df[match_mask]
        
        if matching_rows.empty:
            continue
            
        # Pokud je víc dvojčat (např. T, T1, T2), vezmeme to s nejvyšší podobností
        match_indices = [df.index.get_loc(idx) for idx in matching_rows.index]
        true_score = np.max(sim_matrix[i, match_indices])
        
        # Rs: Kolik JINÝCH slidů (s jiným sample_id) je podobnější než naše dvojče?
        # Ignorujeme diagonálu (sebe sama) a ostatní slidy se stejným sample_id
        other_samples_mask = (df['sample_id'] != current_sample_id)
        other_scores = sim_matrix[i, other_samples_mask]
        
        rs = np.sum(other_scores > true_score)
        rs_values.append(rs)
        
    return rs_values

def plot_success_count(results_dict):
    # Převedeme data do formátu pro Seaborn
    methods = list(results_dict.keys())
    # Spočítáme, kolikrát je v seznamu Rs hodnota 0 (perfektní hit)
    success_counts = [rs_list.count(0) for rs_list in results_dict.values()]
    
    df_plot = pd.DataFrame({
        'Method': methods,
        'Number of success (Top-1)': success_counts
    })

    # Nastavení vzhledu
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Vykreslení barů
    ax = sns.barplot(x='Method', y='Number of success (Top-1)', data=df_plot, 
                     palette=['#e91e63', '#2196f3', '#00bcd4', '#4caf50'],
                     edgecolor='black')

    # Nastavení os a nadpisu
    plt.title('Cross-scanner pairing ($R_s = 0$)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Number of patients ($|H|$)', fontsize=13)
    plt.xlabel('Method', fontsize=13)
    
    # Pevná osa Y podle tvého vzorku (10 pacientů)
    plt.ylim(0, 12) 
    plt.yticks(range(0, 12, 2))

    # Přidání čísel přímo nad sloupce
    for i, v in enumerate(success_counts):
        ax.text(i, v + 0.2, f"{v}/10", ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig("cross.png", dpi=300)
    plt.show()

# --- VÝPOČET A SROVNÁNÍ ---
df = pd.read_pickle("descriptors_comparison_all.pkl")
methods = ['vlad_raw', 'vlad_super', 'fisher_raw', 'fisher_super']

print(f"{'Metoda':<15} | {'Avg Rs':<8} | {'Max Rs':<8} | {'Top-1 (Rs=0)':<12}")
print("-" * 55)

for m in methods:
    # Voláme verzi s CENTROVÁNÍM
    rs_results = calculate_linking_attack_with_centering(df, m)
    
    avg_rs = np.mean(rs_results)
    max_rs = np.max(rs_results)
    hits = rs_results.count(0)
    
    print(f"{m:<15} | {avg_rs:<8.2f} | {max_rs:<8} | {hits}/{len(rs_results)}")

# --- POUŽITÍ ---
# Do tohoto diktátu si ulož výsledky z tvého cyklu
all_results = {}
for m in ['vlad_raw', 'vlad_super', 'fisher_raw', 'fisher_super']:
    all_results[m] = calculate_linking_attack_with_centering(df, m)

plot_success_count(all_results)
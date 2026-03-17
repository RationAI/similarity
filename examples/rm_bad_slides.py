import os
import shutil
import pandas as pd
import urllib.parse
from pathlib import Path

# --- NASTAVENÍ ---
# Cesta k CSV s chybami, které jsi vygeneroval při checku
ERRORS_CSV = "integrity_errors.csv" 
# Základní cesta, kde máš uložené ty "slide_id=%2F..." složky
TARGET_DIR = "/data/fs201053/jb88526/privagams_enc2_mpp05_enhanced"

def cleanup_failed_slides():
    if not os.path.exists(ERRORS_CSV):
        print(f"❌ Soubor {ERRORS_CSV} nenalezen. Není co mazat.")
        return

    errors_df = pd.read_csv(ERRORS_CSV)
    # Pokud tvůj check ukládá cestu v 'slide', použijeme tu
    # Předpokládám, že tam jsou ty problematické TIFFy
    slides_to_delete = errors_df['slide'].tolist()

    print(f"Bude smazáno {len(slides_to_delete)} složek.")
    deleted_count = 0

    for slide_path in slides_to_delete:
        if slide_path.endswith(".tiff"):
            continue

        full_path = os.path.join(TARGET_DIR, slide_path)

        if os.path.exists(full_path):
            try:
                # Smažeme celou složku i s chybnými parquety
                shutil.rmtree(full_path)
                print(f"✅ Smazáno: {full_path}")
                deleted_count += 1
            except Exception as e:
                print(f"⚠️ Chyba při mazání {full_path}: {e}")
        else:
            print(f"ℹ️ Složka nenalezena (již smazána nebo jiná cesta): {full_path}")

    print(f"\n--- HOTOVO ---")
    print(f"Celkem úspěšně smazáno: {deleted_count} složek.")

if __name__ == "__main__":
    # DOPORUČENÍ: Nejdřív si to pusť nanečisto (nahraď shutil.rmtree za print("Chci smazat"))
    confirm = input("Opravdu smazat složky podle seznamu chyb? (ano/ne): ")
    if confirm.lower() == 'ano':
        cleanup_failed_slides()
    else:
        print("Akce zrušena.")
#!/bin/bash

# Cesta k hlavní složce, kde máš složky 1, 2, 3 atd.
PARENT_DIR="/data/fs201053/jb88526/privagams"

# Omezení threadů pro stabilitu na login uzlu
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Ověření, zda složka existuje
if [ ! -d "$PARENT_DIR" ]; then
    echo "Chyba: Složka $PARENT_DIR neexistuje."
    exit 1
fi

echo "Spouštím zpracování složek v: $PARENT_DIR"

# Projde všechny podložky v PARENT_DIR
for dir in "$PARENT_DIR"/*/; do
    # Odstraní koncové lomítko z názvu pro hezčí výpis
    dir_name=$(basename "$dir")
    
    echo "------------------------------------------------"
    echo "Zpracovávám složku: $dir_name"
    echo "Cesta: $dir"
    
    # Spuštění tvého makeru
    # Používáme python -m, aby se zachovaly cesty k modulům
    python -m examples.compute_similarity --slide-path="$dir"
    
    if [ $? -eq 0 ]; then
        echo "Složka $dir_name dokončena úspěšně."
    else
        echo "Chyba při zpracování složky $dir_name!"
    fi
done

echo "Všechny složky byly zpracovány."

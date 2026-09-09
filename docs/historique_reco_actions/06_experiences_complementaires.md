# Phase 2d — Expériences complémentaires

## (a) Fenêtre active « fin d'activité » — RÉSULTAT NÉGATIF
Hypothèse : les erreurs Sit/Fall→Walk viennent de la fenêtre densité-max qui cadre la
MARCHE d'approche (dense) au lieu du geste. Correction testée : ancrer la fin de fenêtre
sur la fin d'activité (viser le geste). Une seule variable change.
Résultat : val 0,969 → ~0,82 ; ET train 0,95 → 0,86.
Lecture : le TRAIN baisse aussi → pas un souci de généralisation mais des images MOINS
INFORMATIVES. Pour Sit/Fall, la « fin d'activité » = personne immobile → peu d'événements ;
la fenêtre « late » cadre l'immobilité. La densité-max captait le segment le plus riche.
→ HYPOTHÈSE INVALIDÉE ; la règle initiale était mieux adaptée qu'il n'y paraissait.

## (b) Chaînes plus longues L=4 (81 classes) — L'ORDRE SCALE
- 40 epochs : val ada 0,854 (mais courbe montait encore → SOUS-ENTRAÎNÉ).
- Repris 60 epochs : val ada ~0,969 = MÊME PLATEAU que L=3. Test ≈ 0,97.
Lecture : 27→81 classes ne dégrade PAS ; il faut juste plus d'epochs (3x plus de données).
Précaution méthodo : une mesure prise avant convergence peut être lue à tort comme un plafond.

## (c) Étude neurone QIF / ALIF (demande du tuteur) — EN COURS
- spikingjelly : QIFNode présent ; ALIFNode ABSENT (le plus proche = ParametricLIFNode/PLIF).
- Neurone paramétré via env NEURON dans block_tdbn.py (défaut lif).
- QIF (mêmes tdBN/L=4/T=40) : bs=4 OOM (~44 Go, +6,5 Go vs LIF — dynamique quadratique) →
  lancé en bs=3 (~33 Go). Constat : QIF nettement plus gourmand en mémoire.
- Résultats QIF/PLIF : à compléter.

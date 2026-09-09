# Phase 0 — DVS-Gesture : le benchmark insuffisant

## Contexte
DVS-Gesture (Amir 2017) : 11 gestes bras/mains, caméra événementielle 128x128,
plusieurs dizaines de sujets. Benchmark le plus utilisé pour l'action recognition
événementielle.

## Constat (article Vicente-Sola 2025)
- ANN (BN) : 97,35 % · ANN (BNTT) : 96,95 %
- SNN (BNTT) : 89,82 % · SNN* (pré-entraîné) : 94,84 %
- L'ANN produit une prédiction PAR IMAGE puis somme sur la séquence : aucune mémoire,
  aucune perception d'ordre. Pourtant il fait aussi bien / mieux.

## Hypothèse posée
Si la tâche se résout sans mémoire temporelle, les performances rapportées NE MESURENT
PAS l'avantage supposé des SNN. → il faut une tâche qui EXIGE le temporel.

## Suite
Réponse = protocole DVS-GC (chaînage), où l'ordre devient la classe. → phase 1.

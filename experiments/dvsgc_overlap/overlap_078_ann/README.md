# ANN-BN sur DVS-GC overlap 0/7/8 (temoin SNN vs ANN)

But : montrer que l'ANN non-temporel echoue a percevoir l'ordre, la ou le SNN reussit.
- Meme archi que le SResNet-38, mais LIF -> ReLU et BNTT -> BatchNorm classique.
- Sortie = somme des predictions sur les pas de temps (pas de dynamique temporelle).
- Tache : overlap 0/7/8, seq3, 27 classes, T=40 (meme que le SNN Run A, val ~0.82).
- Resultat attendu : accuracy proche du plafond "sans ordre" (bien < SNN).
Lancer : python train_ann.py   (BS, EPOCHS en variables env)

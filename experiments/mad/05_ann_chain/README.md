# ANN-BN sur MAD-Chain (temoin SNN vs ANN)

Meme archi que le SResNet-38 mais LIF -> ReLU et BNTT -> BatchNorm classique
(aucune dynamique temporelle ; sortie = somme sur les pas de temps).
Tache : MAD-Chain Walk/Sit/Fall, seq3, 27 classes, T=40, split 70/15/15 (par participant).
Attendu : accuracy proche du plafond "sans ordre" (bien < SNN ~0.73-0.78).
Lancer : python train_ann_chain.py   (BS, EPOCHS en env)

# MAD-Chain — chaines d'activites (SResNet / SNN)

But : tester la perception d'ordre des SNN sur des chaines d'activites REELLES.

## Config actuelle
- Primitives Walk(1) / Sit Down(3) / Fall(9), seq_len=3 -> 27 classes.
- 100 participants, split PAR PARTICIPANT : 70 train / 15 val / 15 test.
- T=40, crop+downsample 128, fenetre active glissante, batch 6.
- Eval : Val(std) + Val(ada) (BN recalculee sur la val, test-time BN adaptation).

## Scripts
- train_chain.py        : entrainement (variables env BS, EPOCHS, RESUME).
- train_chain_swa.py    : baseline + SWA (moyennage de poids).
- train_chain_coslr.py  : baseline + cosine LR (levier en cours de test).
- test_eval.py          : accuracy sur le test set (best_chain.pth).
- error_analysis.py     : confusion primitives + accuracy par position + R-error.
- analyze_train.py, plot_curves.py : courbes et statistiques de fluctuation.

## Journal
Voir EXPERIMENTS.md pour le recap de tous les runs et les enseignements.

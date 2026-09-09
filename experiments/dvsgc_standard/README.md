# DVS-GC standard — primitives {1, 3, 8}, L=4, 81 classes

Reproduction du protocole de référence **DVS-Gesture-Chain** (Vicente-Sola et al.,
2025) pour valider la ré-implémentation de S-ResNet-38 avant de modifier la
difficulté ou le modèle.

## Configuration

- Primitives : 1 (Right Hand Wave), 3 (Right Arm Clockwise), 8 (Air Drums)
- Longueur de chaîne : `seq_len = 4` → `3^4 = 81` classes
- Nombre de trames : `T = 60`
- Architecture : S-ResNet-38 (LIF, reset soustractif), BNTT
- `batch_size = 4`, `accum_steps = 2` (batch effectif 8), `lr = 1e-4`, 100 epochs
- Données : `/users/abdekess61/datasets/SResNet/DVSGC_frames_number_60_split_by_number/`

## Résultat

S-ResNet-38 atteint **97,48 % au test** (vs 95,83 ± 0,62 % dans l'article), ce qui
valide la chaîne de calcul.

## Scripts

- `train.py` : entraînement complet (reproduit le checkpoint
  `best_model_sresnet38_bs4_accum2_lr1e-4_val.pth`).
- `val.py` / `test.py` : évaluation sur validation / test.
- `test_memory_bs8.py` : test d'empreinte mémoire (batch 8).
- `train_before_bs4_accum_*.py` : versions antérieures de la boucle d'entraînement.

## Checkpoints

- `best_model_sresnet38_bs4_accum2_lr1e-4_val.pth` : modèle principal (97,48 % test).
- `best_model_constant_lr1e-4_val.pth` : essai à pas d'apprentissage constant.
- `best_model.pth` / `best_model_val.pth` : versions antérieures.
- `best_model_first_run.pth` : première implémentation (analyse `analysis/dvsgc_138_run1/`).

## Notes

Les analyses d'erreurs de cette campagne se trouvent dans `analysis/dvsgc_138_run1/`
(première version) et `analysis/dvsgc_138_run2/` (S-ResNet-38 fidèle à l'article).

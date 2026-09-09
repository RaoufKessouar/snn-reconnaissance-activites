# Reproductibilité

## Environnement de référence

Les métadonnées W&B du 22 juin 2026 indiquent :

- Python 3.12.3 ;
- PyTorch 2.6.0+cu124 ;
- torchvision 0.21.0+cu124 ;
- SpikingJelly 0.0.0.0.12 ;
- CUDA 12.4 ;
- GPU Quadro RTX 8000, 48 318 382 080 octets annoncés par carte.

`requirements-cuda124.txt` reproduit les versions capturées. Les paquets
PyTorch CUDA proviennent de l'index officiel PyTorch. Sur une machine sans CUDA,
installer d'abord une version CPU compatible de PyTorch/torchvision, puis
`requirements.txt`.

## Contrôle du dépôt

```bash
python scripts/check_repository.py
```

Ce contrôle :

- analyse la syntaxe de tous les fichiers Python ;
- vérifie l'intégrité SHA-256 du rapport et des cinq checkpoints ;
- refuse les anciens chemins absolus dans les parcours maintenus ;
- vérifie qu'aucune recalibration n'utilise un chargeur de validation ou de
  test.

Il ne remplace pas un test numérique GPU.

## Configuration des chemins

```bash
cp configs/cetautomatix.env.example configs/cetautomatix.env
source configs/cetautomatix.env
```

Le fichier sans suffixe `.example` est ignoré par Git. Pour DVS-GC overlap,
remplacer temporairement `DVSGC_DATA_ROOT` par
`/data/abdekess61/SResNet`.

## Évaluations

MAD-Chain L=3, 27 classes :

```bash
python experiments/mad/04_train_chain/eval_test_tdbn.py
```

MAD-Chain L=4, 81 classes :

```bash
python experiments/mad/04_train_chain/eval_test_tdbn_L4.py
python experiments/mad/04_train_chain/eval_L4_positions.py
```

Détection de chute :

```bash
SPLIT=val  python experiments/mad/06_fall_detect/eval_fall_split.py
SPLIT=test python experiments/mad/06_fall_detect/eval_fall_split.py
```

DVS-GC standard :

```bash
export DVSGC_DATA_ROOT=/users/abdekess61/datasets/SResNet
python experiments/dvsgc_standard/test.py
```

Le checkpoint DVS principal n'a pas encore de nouvelle trace d'évaluation
associée dans le dépôt ; conserver la sortie de cette commande dans un artefact
de release, avec la révision Git et le SHA-256 du checkpoint.

## Entraînements principaux

Pour éviter une connexion W&B lors d'un test, utiliser
`WANDB_MODE=offline`. Les valeurs historiques ont souvent été passées par
variables d'environnement ; elles sont rendues explicites ci-dessous.

```bash
# MAD-Chain L=3
SEED=123 DETERMINISTIC=0 BS=4 EPOCHS=60 WANDB_MODE=offline \
python experiments/mad/04_train_chain/train_chain_tdbn.py

# MAD-Chain L=4
SEED=123 DETERMINISTIC=0 BS=4 EPOCHS=60 WANDB_MODE=offline \
python experiments/mad/04_train_chain/train_chain_tdbn_L4.py

# Détection de chute
SEED=123 BS=3 EPOCHS=60 T=50 FALL_BOOST=2.0 SEL_K=3 WANDB_MODE=offline \
python experiments/mad/06_fall_detect/train_fall_kcurve.py

# DVS-GC standard
SEED=123 WANDB_MODE=offline python experiments/dvsgc_standard/train.py
```

`DETERMINISTIC=1` demande des algorithmes PyTorch déterministes et peut réduire
les performances ou échouer si une opération ne possède pas d'implémentation
déterministe. Pour une campagne scientifique, exécuter au moins trois graines
(`SEED`) et publier moyenne, écart-type et liste des checkpoints.

## Protocole de normalisation

La fonction de recalibration doit recevoir un chargeur d'entraînement. Le
checkpoint est ensuite évalué en mode `eval` sur validation ou test. Les
étiquettes ne sont pas utilisées pendant la recalibration, mais utiliser les
entrées du test constituerait tout de même une adaptation transductive.

Les checkpoints historiques tdBN ont été sauvegardés après une mise à jour des
statistiques sur la validation. Les résultats d'origine doivent donc être
distingués d'une réplication stricte obtenue avec les scripts corrigés.

## Conservation des preuves

Pour chaque nouvelle exécution, archiver :

- révision Git et état du dépôt ;
- commande complète et variables d'environnement non secrètes ;
- versions Python, CUDA et paquets ;
- graine ;
- effectifs exacts de chaque split ;
- meilleur critère de validation et époque ;
- métriques de test finales, une seule fois après gel du choix ;
- SHA-256 du checkpoint.

Ne pas commiter les métadonnées W&B brutes. Exporter plutôt un CSV de métriques
nettoyé et un fichier YAML de configuration sans adresse électronique, UUID
matériel ni chemin personnel.

# Dynamique temporelle des SNN pour la reconnaissance d'activités

Code, protocoles, checkpoints sélectionnés et analyses du stage de fin de M1
d'Abderraouf Tarek Kessouar, réalisé au laboratoire ETIS de mai à août 2026
sous l'encadrement de Mathias Quoy et Camille Simon-Chane.

Le dépôt étudie une question précise : un réseau de neurones impulsionnels
exploite-t-il réellement son état temporel pour reconnaître l'ordre
d'activités issues de caméras événementielles ? La contribution principale est
un protocole de contrôle qui sépare l'effet de la mémoire de celui de la
normalisation.

Le [rapport](docs/report/KESSOUAR_rapport.pdf) donne le contexte
scientifique complet. L'annexe D sur le flot optique est volontairement hors du
périmètre de ce dépôt.

> **Pour reprendre le projet de bout en bout :** commencer par le
> [guide de reprise](docs/GUIDE_REPRISE.md). Il relie l'accès aux données ETIS,
> l'installation, le prétraitement, les évaluations, les entraînements et les
> règles de conservation des nouvelles expériences.

## Résultats essentiels

| Expérience | Modèle | Résultat | Statut de l'audit |
|---|---|---:|---|
| DVS-GC 0/7/8, L=3, T=40 | ANN-BN sans mémoire | 34,64 % validation | trace retrouvée |
| DVS-GC 0/7/8, L=3, T=40 | SNN IF, reset zéro | 72,50 % validation | trace retrouvée |
| DVS-GC 0/7/8, L=3, T=40 | SNN LIF, reset soustractif | 81,96 % validation | trace retrouvée |
| DVS-GC 0/7/8, L=3, T=60 | ANN-BNTT sans mémoire | 98,57 % validation | trace retrouvée ; fuite de position démontrée |
| MAD-Chain, L=3, T=40 | SNN-tdBN | 97,12 % test | trace retrouvée, évaluation standard |
| MAD-Chain, L=4, T=40 | SNN-tdBN | 96,93 % test | trace retrouvée, évaluation standard |
| Détection de chute, T=50 | SNN-tdBN, K=3 | F1 0,945 au test | valeur du rapport ; script et checkpoint présents |

Le détail, les dénominateurs et les écarts entre le rapport et les traces sont
consignés dans [docs/RESULTS.md](docs/RESULTS.md) et
[docs/AUDIT.md](docs/AUDIT.md). Ces réserves sont importantes : le score de
81,96 % provient d'une exécution T=40, bien que le rapport l'associe à T=60, et
les anciens scripts de tdBN recalibraient parfois les statistiques sur la
validation ou le test. Les scripts recommandés dans cette version recalibrent
uniquement à partir de l'entraînement et n'adaptent jamais le modèle au test.

## Démarrage rapide sur le serveur ETIS

Le code cible Python 3.12, PyTorch 2.6.0 et CUDA 12.4.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda124.txt
cp configs/cetautomatix.env.example configs/cetautomatix.env
source configs/cetautomatix.env
python scripts/check_repository.py
```

Le fichier de configuration copié est local et ignoré par Git. Vérifier les
chemins avant de lancer une expérience.

Évaluer le checkpoint MAD-Chain principal :

```bash
python experiments/mad/04_train_chain/eval_test_tdbn.py
```

Réévaluer après une recalibration stricte sur l'entraînement :

```bash
RECALIBRATE_FROM_TRAIN=1 \
python experiments/mad/04_train_chain/eval_test_tdbn.py
```

Autres points d'entrée :

```bash
# MAD-Chain L=4
python experiments/mad/04_train_chain/eval_test_tdbn_L4.py

# Détection de chute : SPLIT vaut val ou test
SPLIT=test python experiments/mad/06_fall_detect/eval_fall_split.py

# DVS-GC standard ; DVSGC_DATA_ROOT doit viser le cache standard
python experiments/dvsgc_standard/test.py
```

Les entraînements complets durent plusieurs dizaines d'heures sur une Quadro
RTX 8000. Voir [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) avant de les
relancer.

## Organisation

```text
.
├── src/                         modèles S-ResNet, BNTT, tdBN et DVS-GC
├── experiments/
│   ├── dvsgc_standard/          reproduction DVS-GC 1/3/8
│   ├── dvsgc_overlap/           étude 0/7/8 et contrôles temporels
│   └── mad/                     MAD-Chain et détection de chute
├── analysis/                    scripts et tables d'analyse DVS-GC
├── figures/                     figures agrégées et scripts de génération
├── docs/
│   ├── report/                  rapport final
│   └── historique_reco_actions/ journal scientifique du stage
├── configs/                     exemples de chemins, sans données privées
├── scripts/                     contrôles du dépôt
└── artifacts/SHA256SUMS         intégrité des poids et du rapport
```

Les scripts exploratoires historiques sont conservés pour la provenance. Ils ne
constituent pas tous des points d'entrée maintenus ; les commandes ci-dessus
sont les parcours de référence.

## Données et checkpoints

Les données MAD ne sont pas redistribuées. DVS-Gesture doit également être
installé ou mis en cache séparément. Les chemins, formats, droits d'accès,
splits par participant et variables d'environnement sont décrits dans
[docs/DATA.md](docs/DATA.md). Les données ETIS et ce dépôt privé sont les deux
parties complémentaires de la transmission.

Cinq checkpoints utiles à la reprise sont inclus. Leurs empreintes SHA-256 sont
dans [artifacts/SHA256SUMS](artifacts/SHA256SUMS). Les checkpoints
intermédiaires, journaux W&B et sorties détaillées restent sur
`cetautomatix` et sont exclus du dépôt Git, notamment parce que les
métadonnées W&B contiennent des chemins personnels, une adresse électronique et
des identifiants matériels.

## Pour poursuivre les recherches

Après le parcours opérationnel du
[guide de reprise](docs/GUIDE_REPRISE.md), consulter
[docs/CONTINUATION.md](docs/CONTINUATION.md). Les priorités sont :
reproduire les résultats avec la recalibration strictement issue de
l'entraînement, évaluer les modèles sur les mêmes chaînes, répéter plusieurs
graines et passer de chaînes concaténées à des flux continus.

## Citation et licence

Les métadonnées de citation sont fournies dans [CITATION.cff](CITATION.cff).
Le dépôt réutilise et adapte des idées et du code issus notamment de
S-ResNet, DVS-Gesture-Chain, SpikingJelly et tdBN ; voir
[THIRD_PARTY.md](THIRD_PARTY.md).

Aucune licence de redistribution n'a été choisie dans les sources remises. Elle
doit être validée avec les encadrants et les auteurs des composants adaptés
avant de rendre le dépôt public. En l'absence de licence, les droits restent
réservés.

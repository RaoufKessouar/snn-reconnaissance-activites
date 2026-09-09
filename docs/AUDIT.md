# Audit du dépôt remis

Audit réalisé le 9 septembre 2026 à partir du rapport final, du dossier
`/users/abdekess61/raouf/SResNet_organise` et des traces restées sur
`cetautomatix`.

## Conclusion

Le travail scientifique essentiel est présent : modèles BNTT et tdBN,
prétraitement MAD, chaînage, contrôles ANN/GRU, détection de chute, checkpoints
principaux, figures et journal d'expériences. Le premier rangement était
toutefois insuffisant pour une remise professionnelle : il déplaçait les
fichiers sans réparer leurs imports, publiait des métadonnées personnelles et
présentait plusieurs résultats comme entièrement reproduits alors que les
traces sont ambiguës.

## Défauts constatés et corrections

### 1. Sources déplacées au lieu d'être copiées

Après le rangement initial, le dossier `SResNet` ne contenait plus que le
checkpoint `last_model_sresnet38_bs4_accum2_lr1e-4.pth`, un lien de données et
des caches Python. Les sources ont vraisemblablement été déplacées dans
`SResNet_organise`. Rien n'a été supprimé du serveur pendant cet audit.

Action recommandée : conserver `SResNet_organise` comme source de vérité et
faire une sauvegarde en lecture seule avant toute nouvelle réorganisation.

### 2. Imports et chemins cassés

Le déplacement de `model.py`, `block.py` et `dvsgc.py` vers `src/` n'avait
pas été répercuté dans les scripts. Plusieurs calculs de `parents[n]` visaient
également `experiments/` au lieu de la racine. Les principaux scripts MAD et
DVS-GC ont été réparés, les modèles utilisent des imports de paquet et les
chemins de données passent par des variables d'environnement.

### 3. Utilisation incorrecte de la recalibration

`update_bntt_running_stats` ne réinitialisait que les modules
`torch.nn.BatchNorm2d`, pas le tdBN personnalisé. Appelé avec un chargeur de
validation ou de test, il mettait le modèle en mode entraînement et modifiait
néanmoins les statistiques tdBN avec ces entrées.

La nouvelle fonction `recalibrate_running_stats` :

- reconnaît les modules BN et tdBN possédant des statistiques courantes ;
- réinitialise réellement moyenne et variance ;
- calcule les statistiques cumulées ;
- restaure le mode du modèle et les momentums ;
- est appelée avec le chargeur d'entraînement dans les scripts corrigés.

Les scripts d'évaluation n'adaptent plus le modèle au test. Une recalibration
optionnelle doit explicitement utiliser l'entraînement.

### 4. Résultats insuffisamment traçables

- 81,96 % est prouvé pour T=40, pas T=60 comme indiqué dans le rapport.
- 97,48 % sur DVS-GC standard n'a pas de trace de test associée au checkpoint
  principal ; la trace de 86,88 % charge un autre checkpoint.
- le journal GRU indique 100,00 %, tandis que le rapport indique 99,93 % ;
- les métriques finales de détection de chute sont dans le rapport, mais pas
  dans un journal de test conservé.

Ces écarts ne justifient pas de remplacer arbitrairement les résultats. Ils sont
marqués selon leur niveau de preuve dans `docs/RESULTS.md`.

### 5. Données personnelles et volume

Les répertoires W&B contenaient notamment une adresse électronique, le nom
d'hôte, des chemins personnels et les UUID des GPU. Ils sont désormais exclus,
ainsi que les journaux bruts et les sorties individuelles. Seules les figures
agrégées et cinq checkpoints de reprise sont destinés à Git.

### 6. Reproductibilité incomplète

Le dépôt initial ne fournissait pas de configuration portable, mélangeait des
versions exactes avec un paquet PyTorch CUDA non installable depuis l'index
standard et ne distinguait pas scripts historiques et parcours maintenus.

Cette version ajoute :

- les dépendances CUDA 12.4 séparées ;
- les variables de chemins documentées ;
- des empreintes SHA-256 ;
- un contrôle automatique de syntaxe, d'intégrité et de protocole ;
- une feuille de route pour les futurs travaux.

## Limites restantes

L'audit n'a pas relancé les entraînements de 30 à 80 heures. Les données MAD
sont privées et ne peuvent pas être embarquées dans le dépôt. Les cinq
checkpoints utilisent le format PyTorch et doivent être considérés comme des
artefacts de confiance limitée : ne jamais charger un checkpoint provenant
d'une source non vérifiée.

Les scripts historiques fixaient les tirages du dataset mais ne fixaient pas
systématiquement Python, NumPy, PyTorch et CUDA avant l'initialisation du modèle.
La graine qualifiée d'« unique » dans le rapport n'est donc pas une preuve de
reproductibilité bit à bit. Les points d'entrée principaux appellent désormais
`seed_from_environment` ; les résultats historiques doivent néanmoins être
interprétés comme des exécutions uniques.

Avant publication publique, les encadrants doivent également choisir une
licence et vérifier les droits de redistribution des adaptations de code tiers.

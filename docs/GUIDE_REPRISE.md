# Guide de reprise du projet

Ce document est le point d'entrée destiné aux encadrants et aux personnes qui
reprendront le travail. Il décrit le parcours complet, depuis l'obtention des
accès jusqu'à la reproduction des expériences et au lancement de nouveaux
travaux.

Le projet repose sur deux éléments complémentaires :

1. ce dépôt Git privé, qui contient le code, les protocoles, le rapport, les
   figures et cinq checkpoints sélectionnés ;
2. les espaces ETIS, qui contiennent les données brutes, les caches volumineux
   et l'archive du dossier de travail historique.

Le dépôt seul permet d'étudier le code, mais pas de relancer les expériences
sur les données réelles. L'accès aux deux éléments est nécessaire pour une
reprise scientifique complète.

## 1. Périmètre scientifique

Le travail conservé ici couvre :

- DVS-Gesture-Chain (DVS-GC), d'abord avec les primitives 1/3/8, puis avec les
  primitives spatialement proches 0/7/8 ;
- MAD-Chain, qui compose des séquences d'activités 1/3/9 du jeu MAD ;
- la détection de chute dans des séquences MAD ;
- les contrôles ANN, ANN-BNTT, GRU et les variantes de neurones SNN ;
- les analyses de l'ordre temporel, de la normalisation et des erreurs.

L'exploration du flot optique présentée dans l'annexe D du rapport est hors du
périmètre de cette transmission. Les données DSEC et MVSEC ne sont donc pas
nécessaires.

## 2. Accès à obtenir avant de commencer

La personne qui reprend le projet doit disposer :

- d'un compte ETIS et d'un accès SSH au serveur de calcul ;
- d'un accès en lecture au dépôt GitHub privé
  `RaoufKessouar/snn-reconnaissance-activites` ;
- d'un accès durable aux trois emplacements suivants :

| Élément | Emplacement ETIS | Utilité |
|---|---|---|
| archive du travail organisé sur le serveur | `/users/abdekess61/raouf/SResNet_organise` | provenance, sorties et fichiers historiques |
| cache DVS-GC standard T=60 | `/users/abdekess61/datasets/SResNet` | reproduction rapide de DVS-GC 1/3/8 |
| données et caches partagés | `/data/abdekess61` | DVS brut, DVS-GC overlap et MAD |

État constaté le 10 septembre 2026 : le dossier parent
`/users/abdekess61` est en mode `700`. Même si certains sous-dossiers sont en
`755`, un autre utilisateur ne peut pas les traverser. L'administrateur doit
donc transférer les dossiers concernés ou accorder explicitement les droits aux
encadrants. `/data/abdekess61` était lisible par les autres utilisateurs ETIS,
mais sa conservation après la fermeture du compte et les droits d'écriture
doivent également être confirmés.

Ne pas déplacer, supprimer ou republier les données MAD sans autorisation du
laboratoire.

## 3. Première installation

Toutes les commandes suivantes sont à exécuter depuis la racine du dépôt.

```bash
git clone https://github.com/RaoufKessouar/snn-reconnaissance-activites.git
cd snn-reconnaissance-activites

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda124.txt
```

L'environnement historique de référence utilisait Python 3.12.3, PyTorch
2.6.0+cu124, torchvision 0.21.0+cu124, SpikingJelly 0.0.0.0.12 et CUDA 12.4.
Sur une machine sans CUDA, installer une version CPU compatible de PyTorch et
torchvision, puis `requirements.txt`. Les entraînements principaux restent
destinés à un GPU.

## 4. Configuration des données

Créer une configuration locale non suivie par Git :

```bash
cp configs/cetautomatix.env.example configs/cetautomatix.env
source configs/cetautomatix.env
```

Vérifier ensuite que les emplacements sont accessibles :

```bash
test -d "$DVSGC_DATA_ROOT"
test -d "$MAD_EXTRACT_DIR"
test -d "$MAD_CACHE_DIR"
find "$MAD_EXTRACT_DIR" -name '*.csv' | head
```

La configuration fournie sélectionne par défaut le cache DVS-GC standard :

```bash
export DVSGC_DATA_ROOT=/users/abdekess61/datasets/SResNet
```

Pour les expériences DVS-GC 0/7/8, utiliser plutôt :

```bash
export DVSGC_DATA_ROOT=/data/abdekess61/SResNet
```

Si le cache standard du dossier personnel n'est pas transféré, il peut être
reconstruit dans un emplacement où le nouvel utilisateur possède les droits
d'écriture. Le constructeur `DVSGestureChain` crée automatiquement les
répertoires `events_np` et `DVSGC_frames_number_*`. Il ne télécharge toutefois
pas DVS-Gesture automatiquement : `DvsGesture.tar.gz`,
`gesture_mapping.csv`, `LICENSE.txt` et `README.txt` doivent être placés dans
`<racine>/download`, conformément aux conditions du jeu de données. Une autre
possibilité consiste à réutiliser en lecture les événements déjà convertis dans
`/data/abdekess61/SResNet/events_np` et à écrire les nouvelles trames dans une
racine de travail distincte.

MAD est lu depuis les CSV déjà extraits. Les fichiers de cache `.npz` sont
créés à la demande par les classes de jeu de données. `MAD_CACHE_DIR` doit donc
désigner un volume volumineux sur lequel l'utilisateur possède les droits
d'écriture.

L'inventaire détaillé, les formats et les effectifs attendus se trouvent dans
[DATA.md](DATA.md).

## 5. Vérification initiale

Avant toute expérience :

```bash
python scripts/check_repository.py
python experiments/mad/00_extract/inventory.py
```

La première commande fonctionne sans jeu de données et vérifie la syntaxe, les
empreintes des checkpoints, la portabilité des chemins et le protocole de
normalisation. La seconde vérifie que les CSV MAD sont trouvés et correctement
interprétés.

Contrôler également l'identité du code et des checkpoints :

```bash
git rev-parse HEAD
sha256sum -c artifacts/SHA256SUMS
```

## 6. Reproduire les évaluations conservées

Les évaluations suivantes constituent l'ordre recommandé. Elles utilisent les
checkpoints suivis dans le dépôt.

### 6.1 DVS-GC standard, primitives 1/3/8

```bash
export DVSGC_DATA_ROOT=/users/abdekess61/datasets/SResNet
python experiments/dvsgc_standard/test.py
```

Le rapport annonce 97,48 % au test pour le checkpoint principal, mais la trace
exacte correspondante n'a pas été retrouvée. Conserver la sortie de cette
réévaluation avec la révision Git et le SHA-256 du checkpoint.

### 6.2 MAD-Chain L=3

Évaluation historique sans nouvelle recalibration :

```bash
python experiments/mad/04_train_chain/eval_test_tdbn.py
```

Évaluation recommandée après recalibration des statistiques uniquement à
partir de l'entraînement :

```bash
RECALIBRATE_FROM_TRAIN=1 \
python experiments/mad/04_train_chain/eval_test_tdbn.py
```

Le premier parcours permet de comparer avec la trace historique à 97,12 %. Le
second applique le protocole corrigé ; son score doit être présenté comme une
nouvelle réplication et non comme la preuve du score historique.

### 6.3 MAD-Chain L=4

```bash
python experiments/mad/04_train_chain/eval_test_tdbn_L4.py
RECALIBRATE_FROM_TRAIN=1 \
python experiments/mad/04_train_chain/eval_test_tdbn_L4.py
python experiments/mad/04_train_chain/eval_L4_positions.py
```

La trace historique indique 96,93 % au test en mode standard.

### 6.4 Détection de chute

```bash
SPLIT=val  python experiments/mad/06_fall_detect/eval_fall_split.py
SPLIT=test python experiments/mad/06_fall_detect/eval_fall_split.py
```

Le seuil K=3 a été choisi sur la validation. Le rapport annonce au test une
sensibilité de 0,991, une précision de 0,902, un F1 de 0,945 et un taux de
fausses alarmes de 0,115. La trace exacte du test doit être régénérée.

## 7. Relancer les entraînements principaux

Les valeurs ci-dessous rendent explicites les paramètres historiques. Les
entraînements complets peuvent durer plusieurs dizaines d'heures.

```bash
# MAD-Chain, L=3
SEED=123 DETERMINISTIC=0 BS=4 EPOCHS=60 WANDB_MODE=offline \
python experiments/mad/04_train_chain/train_chain_tdbn.py

# MAD-Chain, L=4
SEED=123 DETERMINISTIC=0 BS=4 EPOCHS=60 WANDB_MODE=offline \
python experiments/mad/04_train_chain/train_chain_tdbn_L4.py

# Détection de chute
SEED=123 BS=3 EPOCHS=60 T=50 FALL_BOOST=2.0 SEL_K=3 WANDB_MODE=offline \
python experiments/mad/06_fall_detect/train_fall_kcurve.py

# DVS-GC standard
SEED=123 WANDB_MODE=offline \
python experiments/dvsgc_standard/train.py

# DVS-GC overlap 0/7/8, comparaison du neurone à T=40
DVSGC_DATA_ROOT=/data/abdekess61/SResNet NEURON_MODE=lif_sub \
BATCH_SIZE=8 EPOCHS=80 WANDB_MODE=offline \
python experiments/dvsgc_overlap/overlap_078_seq3_T40_neuron_study/train_overlap.py
```

Pour une nouvelle étude, exécuter au moins trois graines et publier la moyenne,
l'écart-type et la liste des checkpoints. `DETERMINISTIC=1` demande les
algorithmes PyTorch déterministes, au prix possible d'une baisse de performance
ou d'une erreur si une opération ne possède pas d'implémentation déterministe.

## 8. Comprendre le prétraitement

### DVS-GC

`src/dvsgc.py` convertit DVS-Gesture en événements `.npz`, construit des
combinaisons de gestes et les intègre en T trames 128 x 128 à deux polarités.
Les expériences standard utilisent les primitives 1/3/8 et des chaînes de
longueur 4. Les expériences overlap utilisent 0/7/8 et des chaînes de longueur
3.

### MAD

Le pipeline MAD :

1. lit les événements `(x, y, polarité, temps)` des CSV 640 x 480 ;
2. sélectionne la fenêtre de trois secondes de densité maximale ;
3. recadre `x=[120,520]` et `y=[80,440]` ;
4. redimensionne en 128 x 128 ;
5. répartit les événements en T trames et deux polarités ;
6. assemble les chaînes de manière déterministe à partir de la graine et met
   les trames en cache.

Les participants 1-70, 71-85 et 86-100 forment respectivement les ensembles
d'entraînement, de validation et de test. Aucun participant ne doit apparaître
dans deux ensembles.

## 9. Conserver la preuve d'une nouvelle exécution

Pour chaque expérience, enregistrer dans un dossier de résultats séparé :

- la révision Git et l'état du dépôt ;
- la commande et les variables d'environnement non secrètes ;
- les versions Python, CUDA et des paquets ;
- la graine et les effectifs exacts des trois ensembles ;
- le critère de sélection et l'époque du meilleur checkpoint ;
- les métriques finales et le SHA-256 du checkpoint.

Ne pas commiter les sorties W&B brutes : elles peuvent contenir une adresse
électronique, des identifiants matériels et des chemins personnels. Exporter
des métriques nettoyées et une configuration sans données personnelles.

## 10. Ordre conseillé pour poursuivre la recherche

1. régénérer les évaluations manquantes et attacher leurs traces aux
   checkpoints existants ;
2. réentraîner MAD-Chain avec recalibration strictement issue de
   l'entraînement ;
3. comparer SNN, ANN-tdBN et GRU sur exactement les mêmes chaînes ;
4. répéter les expériences sur plusieurs graines ;
5. remplacer les chaînes artificiellement concaténées par des flux continus ;
6. mesurer aussi la latence, la mémoire, le coût de calcul et le délai de
   détection.

Les réserves scientifiques détaillées sont dans [RESULTS.md](RESULTS.md) et
[AUDIT.md](AUDIT.md). Les pistes de recherche sont développées dans
[CONTINUATION.md](CONTINUATION.md). Le contexte complet se trouve dans le
[rapport final](report/KESSOUAR_rapport.pdf).

## 11. Critère d'une transmission complète

La reprise est considérée comme opérationnelle lorsque la nouvelle personne
peut :

- cloner le dépôt privé ;
- lire les trois emplacements ETIS et écrire dans un espace de cache ;
- exécuter `scripts/check_repository.py` sans erreur ;
- charger chaque checkpoint et lancer les quatre évaluations de la section 6 ;
- reconstruire un cache dans un espace neuf ;
- conserver une nouvelle exécution avec sa configuration et ses preuves.

Tant que les droits sur les dossiers ETIS n'ont pas été transférés ou rendus
durables, la transmission technique n'est pas complète, même si le dépôt Git
est prêt.

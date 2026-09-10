# Données

Les données ne sont pas redistribuées dans ce dépôt. MAD est un jeu interne à
ETIS et DVS-Gesture doit être obtenu selon les conditions de sa source
officielle. Ne jamais publier les CSV MAD, les archives brutes ni les caches
`.npz` sans autorisation explicite du laboratoire.

## Variables d'environnement

| Variable | Contenu |
|---|---|
| `SRESNET_DATA_ROOT` | racine générique des caches et sorties de données |
| `MAD_DATA_ROOT` | dossier `lab_gesture_dataset` |
| `MAD_ZIP_DIR` | archives MAD ; défaut : `$MAD_DATA_ROOT/zips` |
| `MAD_EXTRACT_DIR` | CSV MAD extraits ; défaut : `$MAD_DATA_ROOT/extracted` |
| `MAD_CACHE_DIR` | cache des trames MAD ; défaut : `$SRESNET_DATA_ROOT/MAD_cache` |
| `DVSGC_DATA_ROOT` | racine DVS-Gesture/DVS-GC utilisée par l'expérience courante |

Sans variable, le code utilise le dossier `data/` du dépôt. Ce comportement
convient à une installation locale mais pas au serveur ETIS.

## Emplacements retrouvés sur cetautomatix

| Contenu | Chemin |
|---|---|
| DVS-Gesture brut | `/data/abdekess61/datasets/DVS_Gesture/` |
| DVS-GC standard T=60 | `/users/abdekess61/datasets/SResNet/` |
| DVS-GC overlap et événements `events_np` | `/data/abdekess61/SResNet/` |
| MAD | `/data/abdekess61/datasets/lab_gesture_dataset/` |
| cache MAD | `/data/abdekess61/SResNet/MAD_cache/` |

Le cache DVS standard et le cache overlap n'ont pas la même racine. Modifier
`DVSGC_DATA_ROOT` avant de passer d'une campagne à l'autre.

Le chemin `/data/raouf`, créé au début du stage, était vide lors de
l'inventaire du 10 septembre 2026. Il ne doit pas être utilisé comme racine de
référence : les données utiles ont été retrouvées sous `/data/abdekess61`.

## Volumes et inventaire de transmission

Les ordres de grandeur constatés le 10 septembre 2026 sont :

| Contenu | Volume approximatif | Éléments observés |
|---|---:|---:|
| DVS-GC standard T=60 | 6,5 Go | cache de trames |
| DVS-Gesture brut | 2,8 Go | archive source |
| MAD brut et extrait | 200 Go | 3 archives et 8 700 CSV extraits |
| DVS-GC overlap et événements | 16 Go | 1 464 événements `.npz` et caches T=40/60/80 |
| cache MAD | inclus dans la racine précédente | 45 312 fichiers `.npz` |

Ces nombres servent à détecter une copie manifestement incomplète ; ils ne
constituent pas un manifeste d'intégrité fichier par fichier.

## Droits d'accès et conservation

L'inventaire a montré que `/users/abdekess61` est en mode `700`. Par
conséquent, les encadrants ne peuvent pas atteindre le dossier de travail ni le
cache DVS-GC standard tant que l'administrateur n'a pas transféré ces dossiers
ou accordé des droits explicites.

`/data/abdekess61` et ses sous-dossiers utiles étaient en mode `755` : ils
étaient lisibles par d'autres comptes ETIS, mais modifiables uniquement par leur
propriétaire. Il faut demander à l'administrateur de confirmer leur conservation
après la fin du compte et, si nécessaire, d'accorder un emplacement de cache en
écriture aux personnes qui poursuivent le projet.

Les trois emplacements à inclure dans la transmission administrative sont :

- `/users/abdekess61/raouf/SResNet_organise` ;
- `/users/abdekess61/datasets/SResNet` ;
- `/data/abdekess61`.

## Prétraitement MAD

Chaque CSV contient les événements `(x, y, polarité, temps)` d'un capteur
640 × 480. Le pipeline :

1. cherche la fenêtre de trois secondes de densité maximale ;
2. recadre `x=[120,520]`, `y=[80,440]` ;
3. redimensionne en 128 × 128 ;
4. accumule les événements en T trames et deux polarités ;
5. indexe le cache par fenêtre, recadrage, résolution, classe et fichiers source.

MAD-Chain utilise les activités 1, 3 et 9 et les participants 1-70, 71-85,
86-100 pour entraînement, validation et test. Le cache doit être placé sur un
volume disposant de suffisamment d'espace ; il ne doit pas être commité.

## Vérifications avant exécution

```bash
test -d "$MAD_EXTRACT_DIR"
test -d "$MAD_CACHE_DIR"
test -d "$DVSGC_DATA_ROOT"
find "$MAD_EXTRACT_DIR" -name '*.csv' | head
```

Une absence de fichiers pour un participant ou une activité réduit
silencieusement le nombre de chaînes générées. Toujours comparer les effectifs
affichés au démarrage avec ceux du protocole :

- MAD-Chain L=3 : 1 890 entraînement, 1 215 validation, 1 215 test ;
- contrôle à un échantillon par classe et participant : 405 ;
- détection de chute : vérifier le nombre de positives et négatives imprimé par
  le script d'évaluation.

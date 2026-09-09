# Feuille de route pour la suite

## Avant toute nouvelle conclusion

1. Réévaluer le checkpoint DVS-GC standard principal et conserver la trace
   associée à son SHA-256.
2. Réévaluer ANN-tdBN et la détection de chute pour matérialiser les résultats
   actuellement présents uniquement dans le rapport.
3. Réentraîner MAD-Chain avec recalibration uniquement sur l'entraînement.
4. Comparer SNN, ANN-tdBN et GRU sur exactement les mêmes 1 215 chaînes.
5. Répéter chaque configuration sur plusieurs graines.

Ces étapes séparent la consolidation du travail existant des nouvelles
expériences.

## Questions scientifiques prioritaires

### Flux continus

Remplacer la concaténation de clips par des flux continus avec transitions
naturelles, nombre variable d'activités et événements inconnus. Évaluer
classification, localisation temporelle, délai de détection et robustesse
ouverte.

### Contrôles équitables

Construire une tête récurrente de capacité comparable à S-ResNet-38 pour la
détection de chute. Comparer nombre de paramètres, opérations actives, mémoire,
latence et énergie, pas seulement l'exactitude.

### Normalisation

Étudier séparément :

- BNTT avec statistiques recalibrées sur l'entraînement ;
- tdBN sans adaptation de validation/test ;
- GroupNorm ou LayerNorm, indépendantes du lot ;
- sensibilité à T et à la taille réelle du lot.

La permutation des trames doit être un test systématique : un contrôle déclaré
sans mémoire doit produire la même sortie, à la tolérance numérique près.

### Robustesse

Publier les intervalles de confiance par graine et par participant. Examiner
les erreurs selon activité, position dans la chaîne, morphologie, densité
d'événements et durée du segment. Pour la chute, rapporter également courbe
précision-rappel, délai de détection et nombre de fausses alertes par heure.

## Améliorations logicielles

- Transformer les scripts historiques en commandes paramétrées par fichiers
  YAML.
- Ajouter des tests unitaires sur la construction des chaînes, les splits et
  l'invariance par permutation de l'ANN.
- Versionner un petit jeu synthétique sans données personnelles pour les tests.
- Produire un manifeste machine-readable par exécution.
- Publier les checkpoints volumineux dans une release ou avec Git LFS si leur
  nombre augmente.

## Règles de non-régression

- aucune personne ne doit apparaître dans deux splits MAD ;
- le test n'est exécuté qu'après gel du checkpoint et des seuils ;
- toute recalibration utilise l'entraînement ;
- tout score mentionne T, L, primitives, split, effectif et graine ;
- aucune donnée MAD ou métadonnée personnelle ne doit être publiée.

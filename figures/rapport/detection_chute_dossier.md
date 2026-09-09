# Détection de chute — dossier complet (démarche + résultats)

## 1. Reformulation
- On quitte la classification de chaîne (une séquence = une classe) pour de la DÉTECTION
  + LOCALISATION d'un événement critique (la chute) dans un flux d'activités.
- Motivation applicative : surveiller un flux continu, pas classer une séquence dont on
  sait qu'elle contient N activités.

## 2. Protocole
- Vocabulaire = 9 activités de MAD. Séquences de longueur 5.
- Chute présente OU absente (~50% des séquences), à une position aléatoire, au plus une.
- Lecture PAR IMAGE : sortie [B, T, 9] (une distribution d'activité par pas de temps).
- Loss : entropie croisée PONDÉRÉE (chute rehaussée, env FALL_BOOST) car chute rare/image.
- Base : normalisation tdBN. Split PAR PARTICIPANT (train 1-70 / val 71-85 / test 86-100).
- Règle de décision : chute déclarée si >= K images-chute dans la séquence. K = point de
  fonctionnement (compromis sensibilité / fausses alarmes).
- Métriques : sensibilité (rappel), taux de fausses alarmes, F1, localisation (frames de
  chute correctement repérées), exactitude par image.

## 3. Versions successives (une variable changée à la fois)
| Version | Changement | Résultat | Enseignement |
|---|---|---|---|
| v1 | T=40, FALL_BOOST=4, 40 ep | rappel ~1,0 mais fausses alarmes ~1,0 ; F1<=0,81 | boost trop agressif -> le modèle crie au loup |
| v2 | T=50, bs=3, FALL_BOOST=2, 60 ep | best F1 0,876 @K=3 (ep24) | boost réduit -> fausses alarmes divisées ; opérating-point réglé |
| v3 | idem v2 + log des métriques à K=1..5 par epoch + sélection best sur F1@K3 | best ep43 | permet de tracer les courbes par K |

Constat transversal : baisser FALL_BOOST de 4 à 2 fait chuter les fausses alarmes sans
perdre le rappel. Constat mémoire/temps : ~76 min/epoch (T=50, bs=3).

## 4. Résultats finaux (v3, VALIDATION, meilleur modèle = epoch 43)
- Exactitude par image (frame accuracy) ~ 0,74 (monte proprement 0,19 -> 0,74).
- Table de fonctionnement par seuil K :

| K | sensibilité | précision | F1 | fausses alarmes |
|---|---|---|---|---|
| 1 | 1,000 | 0,695 | 0,820 | 0,468 |
| 2 | 0,996 | 0,816 | 0,897 | 0,239 |
| 3 | 0,987 | 0,895 | 0,939 | 0,124 |
| 4 | 0,935 | 0,963 | 0,937 | 0,064 |
| 5 | 0,858 | 0,961 | 0,907 | 0,037 |

- Point de fonctionnement retenu : K=3 (rappel 0,99 · F1 0,94 · fausses alarmes 0,12).
  K=4 réduit les fausses alarmes à 0,06 en gardant 0,94 de rappel.

## 5. Hypothèses et analyses
- Les métriques de DÉTECTION oscillent fort d'un epoch à l'autre : la règle "au moins K
  images" est sensible (quelques images-chute isolées font basculer une séquence). La
  frame accuracy sous-jacente, elle, est LISSE -> l'apprentissage est sain.
- L'ordre n'est PAS la clé ici : ce qui distingue une chute, c'est sa SIGNATURE TEMPORELLE
  (effondrement rapide/non contrôlé vs assise/ramassage lents et contrôlés), pas l'ordre
  des activités. C'est là que la dynamique interne du neurone joue.
- Comparateur pertinent : pas l'ANN feed-forward (borne basse), mais un modèle temporel
  comparable (GRU/LSTM) ; l'atout du SNN est l'EFFICIENCE (surveillance continue basse conso).

## 6. Figures disponibles (dossier figs_rapport/)
- fig_chute_operating.pdf/.png : sensibilité vs fausses alarmes selon K (K=3 entouré).
- fig_chute_validation.pdf/.png : exactitude par image + F1@K3 par epoch.
- (schéma) detection_chute.pdf/.png : principe de la règle de décision >= K images.

## 7. En attente
- Éval TEST (sujets 86-100) du checkpoint best_T50_fb2.0_kcurve.pth -> chiffre final
  (les résultats ci-dessus sont en VALIDATION).
- Optionnel : comparaison tête temporelle SNN vs GRU/LSTM.

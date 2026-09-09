# analysis_output_3 — Rendu concret du modele MAD-Chain

Modele : 04_train_chain/best_chain.pth (SResNet38, chaines Walk/Sit Down/Fall, 27 classes)
Test accuracy : 0.7844 (15 sujets inedits, 1215 chaines)

## Contenu
- train_sequences.png : exemples VARIES de sequences d'entree (chaines d'entrainement).
- correct_cases.png   : chaines TEST bien classees.
- wrong_cases.png     : chaines TEST mal classees (V = vrai, P = predit, [o/x] par position).
- cases_manifest.csv  : liste des exemples affiches (index, vrai, predit).

## Rappels analyse d'erreurs
- Walk tres bien reconnu (0.97) ; Sit Down / Fall parfois pris pour Walk.
- Pas d'erreur de repetition (R-error 0.09 << 0.33) -> l'ORDRE est bien percu.
- Position 1 la plus difficile (membrane du SNN pas encore "chauffee").
- Hypothese : la fenetre active capture parfois la phase de MARCHE d'approche
  des clips Sit Down / Fall -> a verifier sur wrong_cases.png.

Genere par build_analysis.py

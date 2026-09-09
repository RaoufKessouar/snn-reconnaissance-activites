# Experience : neurone / reset sur overlap 0,7,8 (T=40)

But : isoler l'effet du couple neurone+reset (papier Table 4) sur la tache overlap,
a config par ailleurs identique et stabilisee.

Ameliorations verrouillees (justifiees par le diagnostic) :
- T=40 (frames plus denses -> stats BNTT plus stables)
- batch REEL 8 sans accumulation (l'accumulation n'ameliore PAS BNTT)
- precise-BN avant chaque validation (val propre + selection checkpoint fiable)
- 80 epochs

Variable sous test (NEURON_MODE) :
- A = lif_sub : LIF + reset par soustraction (baseline, val propre ~0.7679)
- B = if_zero : IF + reset a zero (hypothese)

Reserve : Table 4 est en BN classique, ici BNTT -> resultat a verifier.

Lancer :
  NEURON_MODE=lif_sub python train_overlap.py   # run A
  NEURON_MODE=if_zero python train_overlap.py   # run B
Si OOM : BATCH_SIZE=6 (ou 4) devant la commande.

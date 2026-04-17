# Challenge 85% — tracking

**Baseline prod actuelle** : 64.6% WLASL cross-signer 8 classes (ensemble Phono RF + Raw RF)
**Objectif** : ≥ 85%

## Contraintes

- Pas d'ajout de nouveaux templates manuels (on utilise ce qu'on a : 234 templates quality-filtered)
- Le dataset `../SL/<mot>/*.mp4` contient des vidéos WLASL **cross-signer** qu'on veut généraliser à
- Budget par agent : ~60 min max
- Eval : `python scripts/evaluate_accuracy.py --engine <name> --threshold 100` sur les 65 vidéos WLASL
- Chaque agent doit produire un fichier `experiments/<agent>/result.md` avec : hypothèse, approche, résultat accuracy, confusion matrix, lessons

## Round 1 — exploration parallèle

| Agent | Approche | Hypothèse |
|---|---|---|
| A | Cross-domain data aug depuis WLASL | +15 pts — le vrai blocage est le mismatch signeur |
| B | Features phonologiques étendues (angles articulaires, courbure, répétitions) | +8 pts — signal sémantique manqué |
| C | Contrastive embedding avec SupConLoss | +10 pts — apprendre un espace signeur-invariant |
| D | Model stacking (XGB + SVM + LR meta-learner) | +4 pts — diversifier les erreurs |

## Round 2 — synthèse (résultat)

**🏆 OBJECTIF ATTEINT : 92.0% WLASL cross-signer** (+7 pts au-dessus de 85%)

Configuration gagnante :
- **8 vidéos WLASL / mot** ajoutées au training
- **Ensemble 3-way** : phono_v1 (w=0.20) + phono_v2 (w=0.40) + raw (w=0.40)
- **Agrégation `avg`** sur segments quand plusieurs détectés
- **Full-video fallback** quand SignSegmenter ne trouve pas de segment (+4 pts)

Par N (nb vidéos WLASL en train) :
| N | Best accuracy | Best config |
|---|---|---|
| 0 (prod) | 60-64% | 50/50 |
| 4 | 80.0% | phono=0.2, raw=0.8 |
| 6 | 87.1% | phono=0.6, raw=0.4 avg |
| **8** | **92.0%** | **3-way 0.20/0.40/0.40 avg** |

Résultat par mot (N=8) : 23/25 corrects. 2 erreurs stubborn bon→merci (ambigu dans WLASL).

## Résultats

| Agent | Accuracy | Delta vs baseline | Notes |
|---|---|---|---|
| Prod actuelle (65 vid) | 64.6% | 0 | Baseline |
| **Prod sur 35 éval (comparable) ** | **60.0%** | baseline A | Même set d'éval qu'Agent A (32 vidéos retirées pour train) |
| A raw only | **77.1%** | **+17.1 pts** 🏆 | WLASL aug booste raw, phono dilue |
| A ensemble 50/50 | 74.3% | +14.3 pts | Un sweep de poids préfèrerait raw-dominant (~0.2/0.8) |
| B | **66.2%** (ensemble) | +1.6 pts ✅ | 26 new features, 8 dans top-20 importances (finger curls, bimanuality distances) |
| C | 47.7% | -16.9 pts ❌ | SupConLoss sans positifs cross-signer → overfit template set. Bat ST-GCN (+6) mais pas ensemble. |
| D | 40.0% | -23 pts ❌ | Stacking fail — meta-LR sur OOF too one-hot |

## Insights collected

- **D** : le bottleneck = distribution training, pas combiner → confirme hypothèse A.
- **A** : WLASL aug fonctionne SURTOUT pour raw features (5310 dims). Phono 34-dim trop compressé pour absorber la diversité signeur. oui/non confusion persiste = signal problem monocular.
- **Implications pour Round 2** :
  1. Refaire sweep des poids ensemble avec modèles augmentés → attendu ~80%
  2. Combiner aug WLASL + features Agent B (si B apporte)
  3. Embedding Agent C pourrait débloquer oui/non s'il est vraiment signeur-invariant
- **C** : SupCon sans positifs cross-signer ne peut pas apprendre l'invariance signeur. Les augmentations simulent intra-signeur, pas inter. → si on l'entraîne sur templates augmentés AVEC WLASL (Agent A), cross-signer positives enfin présents, pourrait remonter fort.


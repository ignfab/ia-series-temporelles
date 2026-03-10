# Benchmark — Datation d'apparition de bâtiments dans des séries temporelles d'images aériennes

Ce projet fournit une base de code pour **évaluer et comparer des méthodes de détection de l'apparition de bâtiments** dans des séries temporelles d'orthophotographies aériennes françaises (IGN).

La tâche consiste à identifier, pour un bâtiment donné, le **premier instant** dans une série d'images où ce bâtiment devient visible — c'est-à-dire à prédire un indice temporel `frame_id ∈ [0, T-1]`.

---

## Structure du projet

```
.
├── config.py          # Hyperparamètres globaux (batch size, workers…)
├── data.py            # Dataset PyTorch (BuildingTimeSeriesDataset)
├── main.py            # Point d'entrée : chargement, modèle, entraînement, visu
├── utils.py           # Fonctions de visualisation et métriques
├── requirements.txt   # Dépendances Python
│
├── all_file_paths.json    # Chemins des .tif par département et bâtiment (à fournir)
└── date_apparition.json   # Année d'apparition par bâtiment (à fournir)
```

---

## Données

### Organisation sur disque

```
<root_path>/
  <dep_id>/
    orthoimage/
      <building_id>/
        <YYYY_...>.tif    # Une image par date disponible
        emprise.tif       # Masque binaire de l'emprise du bâtiment
```

### Splits

| Split  | Départements                              | Nb deps |
|--------|-------------------------------------------|---------|
| Train  | 21 → 95                                   | 75      |
| Val    | 01–07, 09–11                              | 10      |
| Test   | 12–19, 2A, 2B                             | 10      |

### Format des images

Chaque image est un GeoTIFF de taille `(C, H, W)` avec `H = W = 512` pixels.  
Les canaux sont harmonisés automatiquement à **4 canaux (R, G, B, IR)** :

| Canaux source | Canaux produits        | `n_channels`   |
|---------------|------------------------|----------------|
| RGB (3)       | R, G, B + IR=0         | [1, 1, 1, 0]   |
| IR seul (1)   | R=0, G=0, B=0, IR      | [0, 0, 0, 1]   |
| RGBIR (4)     | R, G, B, IR            | [1, 1, 1, 1]   |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation rapide

```python
from data import BuildingTimeSeriesDataset

dataset = BuildingTimeSeriesDataset(root_path='/path/to/data', split='train')
sample  = dataset[0]

# sample['images']     → Tensor (T, 4, H, W)
# sample['emprise']    → Tensor (H, W)
# sample['frame_id']   → int  (label de la tâche)
# sample['n_channels'] → Tensor (T, 4)
```

### Visualisation

```python
from utils import visualize_time_series, visualize_prediction

# Série temporelle brute
visualize_time_series(dataset, idx=0)

# Comparaison ground truth / prédiction
visualize_prediction(dataset, idx=0, pred_frame_id=3)
```

---

## Exemple de modèle : AnySat

```python
import torch

AnySat = torch.hub.load('gastruc/anysat', 'anysat', pretrained=True, flash_attn=False)

images = dataset[0]['images']  # (T, 4, H, W)
output = AnySat(
    {"aerial": images},
    patch_size=20,
    output='dense',
    output_modality='aerial'
)
print(output.shape)  # (T, D, H', W')
```

---

## Métriques

Deux métriques sont disponibles dans `utils.py` :

- **Accuracy** (`compute_accuracy`) : pourcentage de prédictions exactes (frame exact).
- **MAE** (`compute_mae`) : erreur absolue moyenne en nombre de frames.

---

## Étendre le benchmark

Pour ajouter une nouvelle méthode :

1. Implémenter le modèle (dans un nouveau fichier, ex. `models/my_model.py`).
2. Décommenter et compléter la boucle d'entraînement dans `main.py`.
3. Évaluer sur le split `test` en reportant Accuracy et MAE.

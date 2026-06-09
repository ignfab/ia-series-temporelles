# Benchmark — Datation d'apparition de bâtiments dans des séries temporelles d'images aériennes

Ce projet fournit une base de code pour **évaluer et comparer des méthodes de détection de l'apparition de bâtiments** dans des séries temporelles d'orthophotographies aériennes françaises (IGN).

La tâche consiste à identifier, pour un bâtiment donné, le **premier instant** dans une série d'images où ce bâtiment devient visible — c'est-à-dire à prédire un indice temporel `frame_id ∈ [0, T-1]`.

---

## Structure du projet

```
.
├── configs/                # Fichiers de configuration (YAML)
├── data/                   # Code de chargement des données
├── models/                 # Implémentations de modèles (AnySat, FLAIR-HUB, etc.)
├── utils/                  # Fonctions utilitaires (visualisation, métriques)
├── train.py                # Script d'entraînement et d'évaluation
|── threshold.py            # Script d'évaluation par seuils (pour config_seuil.yaml)
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

| Split  | Départements                                                     | Nb deps |
|--------|------------------------------------------------------------------|---------|
| Test   | 12, 15, 22, 26, 36, 61, 64, 68, 69, 71, 73, 75, 76, 83, 84, 85   | 16      |
| Val    | 04, 14, 29, 31, 58, 66, 67, 77                                   | 8       |
| Train  | Les départements (Métropole + Corse) restants excepté le 08      | 71      |

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
python -m venv datation-env
source datation-env/bin/activate
pip install -r requirements.txt
```

---

```bash
mkdir results
mkdir ckpt
```

Télécharger les poids de modèles pré-entraînés (ex. FLAIR-HUB) et les placer dans `ckpt/`.
- FLAIR-HUB IR : [lien de téléchargement](https://huggingface.co/IGNF/FLAIR-HUB_LC-A_IR_swinbase-upernet/resolve/main/FLAIR-HUB_LC-A_IR_swinbase-upernet.safetensors?download=true)
- FLAIR-HUB RGB : [lien de téléchargement](https://huggingface.co/IGNF/FLAIR-HUB_LC-A_RGB_swinbase-upernet/resolve/main/FLAIR-HUB_LC-A_RGB_swinbase-upernet.safetensors?download=true)
- FLAIR-INC RGB-IR : [lien de téléchargement](https://huggingface.co/IGNF/FLAIR-INC_rgbi_15cl_resnet34-unet/resolve/main/FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth?download=true)
- FLAIR-INC RGB : [lien de téléchargement](https://huggingface.co/IGNF/FLAIR-INC_rgb_15cl_resnet34-unet/resolve/main/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth?download=true)
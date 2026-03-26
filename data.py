# ==============================================================================
# data.py — Dataset PyTorch pour les séries temporelles d'images aériennes
# ==============================================================================
#
# Structure attendue des données sur disque :
#   <root_path>/
#     <dep_id>/
#       orthoimage/
#         <building_id>/
#           <YYYY_...>.tif     # Images aériennes (une par date)
#           emprise.tif        # Masque de l'emprise du bâtiment
#
# Fichiers JSON requis (à la racine du projet) :
#   - all_file_paths.json  : dict { dep_id -> { building_id -> [liste de chemins .tif] } }
#   - date_apparition.json : dict { building_id -> année d'apparition du bâtiment }
# ==============================================================================

import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import json


# Répartition des départements par split
IDS_PER_SPLIT = {
    "train": [str(k) for k in range(21, 22)],  # 96
    # "val": ["01", "02", "03", "04", "05", "06", "07", "09", "10", "11"],
    "val": ["02"],
    "test": ["12", "13", "14", "15", "2B", "16", "17", "18", "19", "2A"],
}

MEAN = [105.66, 111.35, 102.18, 106.59]
STD = [52.23, 45.62, 44.30, 39.78]


class BuildingTimeSeriesDataset(Dataset):
    """
    Dataset PyTorch pour la tâche de datation d'apparition de bâtiments.

    Pour chaque bâtiment, charge une série temporelle d'images aériennes
    (une image par date disponible) et fournit :
      - la pile d'images (T, 4, H, W) — 4 canaux : R, G, B, IR
      - le masque d'emprise du bâtiment (H, W)
      - l'indice temporel du premier frame où le bâtiment est apparu

    Args:
        root_path (str): Chemin vers le répertoire racine des données.
        split (str): 'train', 'val' ou 'test'.
    """

    def __init__(self, root_path, split="train", norm=False):
        self.root_path = root_path
        self.split = split
        self.dep_ids = IDS_PER_SPLIT[self.split]
        self.norm = norm

        # Charge la liste des échantillons et les métadonnées associées
        self.dep_ids, self.building_ids, self.samples = self._find_samples()

        # Année d'apparition de chaque bâtiment (référence pour le label)
        self.date_apparition = json.load(open("date_apparition.json", "r"))

        print(f"Split '{split}' chargé — {len(self.samples)} bâtiments.")

    def _find_samples(self):
        """
        Lit all_file_paths.json et construit les listes parallèles
        (dep_ids, building_ids, samples) pour tous les départements du split.

        Returns:
            dep_ids      (list[str]): Identifiant de département pour chaque sample.
            building_ids (list[str]): Identifiant de bâtiment pour chaque sample.
            samples      (list[list[str]]): Liste de chemins .tif pour chaque bâtiment.
        """
        all_file_paths = json.load(open("all_file_paths.json", "r"))
        samples, building_ids, dep_ids = [], [], []

        for dep in self.dep_ids:
            dep_data = all_file_paths[dep]
            samples.extend(dep_data.values())
            building_ids.extend(dep_data.keys())
            dep_ids.extend([dep] * len(dep_data))

        return dep_ids, building_ids, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Charge et retourne un échantillon.

        Returns:
            dict avec les clés :
              'images'     : Tensor (T, 4, H, W) — série temporelle d'images
              'emprise'    : Tensor (H, W)        — masque de l'emprise du bâtiment
              'frame_id'   : int                  — indice du premier frame post-apparition
              'n_channels' : Tensor (T, 4)        — masque de validité des canaux par frame
        """
        tif_files = self.samples[idx]
        building_id = self.building_ids[idx]
        dep_id = self.dep_ids[idx]

        year_apparition = int(self.date_apparition[building_id])

        images, years, n_channels = [], [], []

        for path in tif_files:
            # Extraction de l'année depuis le nom de fichier (format : YYYY_...)
            years.append(int(path[:4]))

            with rasterio.open(
                os.path.join(self.root_path, dep_id, "orthoimage", building_id, path)
            ) as src:
                image = src.read().astype(np.float32)  # (C, H, W)

                # Harmonisation à 4 canaux (R, G, B, IR)
                if image.shape[0] == 3:
                    # Image RGB uniquement : ajout d'un canal IR nul
                    image = np.concatenate(
                        [image, np.zeros_like(image[0])[None]], axis=0
                    )
                    n_channels.append([1, 1, 1, 0])

                elif image.shape[0] == 1:
                    # Image IR uniquement : ajout de 3 canaux RGB nuls
                    zeros = np.zeros_like(image[0])[None]
                    image = np.concatenate([zeros, zeros, zeros, image], axis=0)
                    n_channels.append([0, 0, 0, 1])

                else:
                    # Image RGBIR complète
                    n_channels.append([1, 1, 1, 1])

            images.append(image)

        # --- Calcul du label : premier frame où le bâtiment est visible ---
        # On cherche le premier indice t tel que years[t] >= year_apparition
        first_frame_id = 0
        for frame_id, year in enumerate(years):
            if year >= year_apparition:
                break
            first_frame_id += 1

        # Chargement du masque d'emprise du bâtiment
        with rasterio.open(
            os.path.join(
                self.root_path, dep_id, "orthoimage", building_id, "emprise.tif"
            )
        ) as src:
            emprise = src.read()[0].astype(np.float32)  # (H, W)

        if self.norm:
            # Normalisation par canal (en utilisant les stats globales)
            images = [
                (img - np.array(MEAN)[:, None, None]) / np.array(STD)[:, None, None]
                for img in images
            ]

        return {
            "images": torch.from_numpy(np.stack(images, axis=0)),  # (T, 4, H, W)
            "emprise": torch.from_numpy(emprise),  # (H, W)
            "frame_id": first_frame_id,  # scalaire
            "n_channels": torch.tensor(n_channels),  # (T, 4)
        }

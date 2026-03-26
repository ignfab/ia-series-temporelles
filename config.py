# ==============================================================================
# config.py — Hyperparamètres et paramètres globaux du benchmark
# ==============================================================================

# --- DataLoader ---
BATCH_SIZE = 1  # Taille du batch pour l'entraînement et l'évaluation
NUM_WORKERS = 0  # Nombre de workers pour le chargement des données (0 = chargement dans le processus principal)
SEUIL_BATIMENT = 50  # Seuil en % de pixels prédits comme bâtiment pour considérer une frame comme "post-apparition"

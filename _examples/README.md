# 📖 Exemples Excalibur

Exemples d'utilisation de la bibliothèque Excalibur.

## 🎯 Exemples de base

### `example_imports.py`
Comment importer et utiliser les modules de base.

```python
python example_imports.py
```

**Démontre :**
- Imports corrects
- Vérification des modules
- Structure du package

---

### `example_usage.py`
Utilisation basique de bout en bout.

```python
python example_usage.py
```

**Démontre :**
- Création d'un grid
- Ajout d'un champ de potentiel
- Création d'une métrique
- Génération d'un photon
- Intégration simple

---

## 📊 Génération de photons

### `example_multi_photons.py`
Génération de multiples photons avec différentes configurations.

```python
python example_multi_photons.py
```

**Démontre :**
- `generate_grid()` - Photons sur grille régulière
- `generate_cone_random()` - Photons dans un cône
- Sauvegarde dans HDF5

**Output:**
- `multi_photons_grid.h5`
- `multi_photons_random.h5`

---

## 🔬 Visualisation

### `example_field_visualizing.py`
Visualisation des champs de potentiel gravitationnel.

```python
python example_field_visualizing.py
```

**Génère :**
- Profils 1D : Φ(r)
- Carte 2D : slices du champ
- Visualisation 3D : isosurfaces

**Output:**
- `visualizations/field_profiles_1d.png`
- `visualizations/field_visualization_2d.png`
- `visualizations/field_visualization_3d.png`

---

### `visualize_trajectories.py` ⭐
Visualisation complète des trajectoires de photons.

```python
python visualize_trajectories.py ../data/backward_raytracing_trajectories.h5
```

**Génère :**
- Trajectoires 2D (projections)
- Trajectoires 3D (interactive)
- Animation temporelle
- Statistiques (distances, vitesses)
- Évolution temporelle

**Output:**
- `visualizations/trajectories_2d.png`
- `visualizations/trajectories_3d.png`
- `visualizations/trajectories_animation.gif`
- `visualizations/trajectories_stats.png`
- `visualizations/trajectories_time.png`

**Options:**
```bash
# Spécifier fichier
python visualize_trajectories.py path/to/file.h5

# Sous-échantillonner photons
python visualize_trajectories.py file.h5 --max-photons 10

# Sauvegarder figures
python visualize_trajectories.py file.h5 --save
```

---

## 💾 I/O de données

### `example_opening_trajectory.py`
Lire et analyser les fichiers HDF5 de trajectoires.

```python
python example_opening_trajectory.py
```

**Démontre :**
- Ouverture fichier HDF5
- Lecture des datasets
- Extraction positions/vitesses
- Analyse des données
- Calcul de statistiques

**Exemple de code :**
```python
import h5py

with h5py.File('../data/trajectories.h5', 'r') as f:
    # Liste des photons
    photon_names = list(f.keys())
    
    # Lire un photon
    photon_0 = f['photon_0']
    states = photon_0['states'][:]
    
    # Extraire composantes
    eta = states[:, 0]          # Temps conformal
    positions = states[:, 1:4]  # x, y, z
    velocities = states[:, 4:8] # u^0, u^1, u^2, u^3
```

---

## 🎓 Exemples avancés

### Personnaliser la métrique

```python
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast

# Créer métrique personnalisée
metric = PerturbedFLRWMetricFast(
    a_of_eta=my_scale_factor,
    grid=my_grid,
    interpolator=my_interpolator
)

# Calculer Christoffel en un point
x = [eta, x, y, z]
christoffel = metric.christoffel(x)
```

### Intégration avec callback

```python
def callback(photon, step):
    """Appelé à chaque step d'intégration."""
    if step % 100 == 0:
        print(f"Step {step}: position = {photon.x[1:4]}")

integrator = Integrator(metric, dt=dt)
integrator.integrate(photon, n_steps, callback=callback)
```

### Analyse de convergence

```python
# Tester différents time steps
dt_values = [1e14, 5e14, 1e15, 5e15]
results = []

for dt in dt_values:
    integrator = Integrator(metric, dt=-dt)
    photon_copy = copy.deepcopy(photon)
    integrator.integrate(photon_copy, n_steps)
    results.append(photon_copy.x)

# Comparer résultats
for i, (dt, final_pos) in enumerate(zip(dt_values, results)):
    print(f"dt={dt:.2e}: final position = {final_pos[1:4]}")
```

---

## 📝 Workflow typique

### 1. Exploration rapide
```bash
python example_usage.py                    # Comprendre la structure
python example_field_visualizing.py        # Voir le potentiel
```

### 2. Génération de données
```bash
cd ../scientific_runs
python integrate_photons_on_perturbed_flrw_OPTIMIZED.py
```

### 3. Visualisation et analyse
```bash
cd ../examples
python visualize_trajectories.py ../data/backward_raytracing_trajectories_OPTIMIZED_*.h5
python example_opening_trajectory.py       # Analyse détaillée
```

---

## 🔧 Personnalisation

### Modifier un exemple

1. Copier l'exemple : `cp example_usage.py my_test.py`
2. Modifier les paramètres :
   ```python
   # Grid size
   N = 128  # Réduire pour tests rapides
   
   # Mass
   M = 5e20 * one_Msun  # Augmenter la masse
   
   # Integration
   n_steps = 500  # Moins de steps pour tests
   ```
3. Exécuter : `python my_test.py`

### Créer un nouvel exemple

Template :

```python
#!/usr/bin/env python3
"""
Description de l'exemple.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/magri/excalibur_project')

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
# ... autres imports

def main():
    """Fonction principale."""
    print("Mon exemple personnalisé\n")
    
    # Setup
    # ...
    
    # Calculs
    # ...
    
    # Visualisation/Sauvegarde
    # ...
    
    print("✓ Exemple terminé")

if __name__ == '__main__':
    main()
```

---

## 📊 Exemples de visualisations

Les exemples génèrent diverses visualisations dans `../visualizations/` :

### Champs de potentiel
- Profils radiaux Φ(r)
- Cartes 2D avec contours
- Surfaces 3D isopotentielles

### Trajectoires
- Projections 2D (xy, xz, yz)
- Visualisations 3D interactives
- Animations temporelles
- Statistiques (histogrammes, distributions)

### Métriques
- Évolution du temps conformal η(λ)
- Distances parcourues
- Vitesses et accélérations

---

## 🎯 Exemples recommandés pour débuter

1. **`example_imports.py`** - Vérifier l'installation
2. **`example_usage.py`** - Comprendre le workflow
3. **`example_field_visualizing.py`** - Voir les champs
4. **`example_opening_trajectory.py`** - Lire les données
5. **`visualize_trajectories.py`** - Visualisation complète

---

**Retour:** [README principal](../README.md) | [Organisation](../PROJECT_ORGANIZATION.md)

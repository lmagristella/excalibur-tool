# 🔬 Calculs Scientifiques - Production

Scripts principaux pour les calculs scientifiques de production.

## 📊 Scripts disponibles

### `integrate_photons_on_perturbed_flrw_OPTIMAL.py` 🚀 PRODUCTION

Version OPTIMALE combinant **TOUTES** les optimisations - **60x plus rapide**.

**Configuration :**
- ✅ Numba JIT (15x speedup)
- ✅ Persistent Worker Pool 4 workers (4x additionnel)
- ✅ **Speedup total : 60x** vs version standard

**Usage :**
```bash
python integrate_photons_on_perturbed_flrw_OPTIMAL.py
```

**Performance :**
- 50 photons × 1000 steps : **~1.5 secondes** 🚀
- 50 photons × 5000 steps : **~8 secondes** 🚀
- 100 photons × 5000 steps : **~15 secondes** 🚀

**Quand utiliser :**
- ✅ Production runs avec ≥20 photons
- ✅ Simulations multi-photons
- ✅ Runs scientifiques standards

**Note :** Pour <20 photons, le overhead de parallélisation n'est pas rentable. Utiliser la version OPTIMIZED à la place.

---

### `integrate_photons_on_perturbed_flrw_OPTIMIZED.py` ⭐ RECOMMANDÉ

Version optimisée avec Numba JIT seul - **15-20x plus rapide**.

**Usage :**
```bash
python integrate_photons_on_perturbed_flrw_OPTIMIZED.py
```

**Paramètres (éditer dans le script) :**
```python
# Grille
N = 512                          # Résolution (512³ cells)
grid_size = 2000 * one_Mpc      # Taille de la boîte

# Masse
M = 1e20 * one_Msun             # Masse (kg)
radius = 10 * one_Mpc           # Rayon virial
center = [500, 500, 500] * one_Mpc  # Position (Mpc)

# Photons
n_photons = 50                   # Nombre de photons
cone_angle = np.pi / 12         # Angle du cône (15°)

# Intégration
n_steps = calculé automatiquement  # Basé sur distance à la masse
dt = calculé automatiquement       # Basé sur contraintes de stabilité
```

**Output :**
```
data/backward_raytracing_trajectories_OPTIMIZED_mass_500_500_500_Mpc.h5
```

**Performance :**
- 50 photons × 1000 steps : **~6 secondes** ✅
- 50 photons × 5000 steps : **~30 secondes** ✅

**Quand utiliser :**
- ✅ Runs avec <20 photons (évite overhead parallel)
- ✅ Tests et développement
- ✅ Debugging (plus simple, pas de multiprocessing)

**Optimisations incluses :**
- ✅ `InterpolatorFast` (Numba JIT)
- ✅ `PerturbedFLRWMetricFast` (cached + Numba)
- ✅ Calcul automatique de dt optimal

---

### `integrate_photons_on_perturbed_flrw.py`

Version standard (référence) - **Non optimisée**.

**Usage :**
```bash
python integrate_photons_on_perturbed_flrw.py
```

**Performance :**
- 50 photons × 1000 steps : **~95 secondes** ⚠️

**Utilité :**
- Référence pour comparaison
- Validation des résultats
- Debugging (plus simple à lire)

**⚠️ Recommandation :** Utiliser la version OPTIMIZED pour production.

---

### `integrate_photons_OPTIMIZED.py`

Version optimisée avec paramètres réduits pour tests rapides.

**Usage :**
```bash
python integrate_photons_OPTIMIZED.py
```

**Différences vs version complète :**
- Grid plus petite (64³ au lieu de 512³)
- Moins de photons (5-10 au lieu de 50)
- Tests et développement rapides

**Performance :**
- 5 photons × 100 steps : **~0.3 secondes** ⚡

---

## 🎯 Quelle version choisir ?

| Photons | Version | Temps | Raison |
|---------|---------|-------|--------|
| < 20 | **OPTIMIZED** | ~3-6s | Overhead parallel non rentable |
| 20-100 | **OPTIMAL** 🚀 | ~1-8s | Sweet spot pour parallélisation |
| > 100 | **OPTIMAL** 🚀 | ~10-30s | Parallélisation obligatoire |

**Recommandation générale :** Utiliser **OPTIMAL** pour tous les runs de production.

---

## 🚀 Architecture des versions

### Standard (integrate_photons_on_perturbed_flrw.py)
```python
Interpolator (standard) + PerturbedFLRWMetric (standard)
└── Integrator (séquentiel)
    └── Vitesse : 1x (baseline)
```

### Optimized (integrate_photons_on_perturbed_flrw_OPTIMIZED.py)
```python
InterpolatorFast (Numba) + PerturbedFLRWMetricFast (Numba + cache)
└── Integrator (séquentiel)
    └── Vitesse : 15x ⚡
```

### Optimal (integrate_photons_on_perturbed_flrw_OPTIMAL.py) 🚀
```python
InterpolatorFast (Numba) + PerturbedFLRWMetricFast (Numba + cache)
└── PersistentPoolIntegrator (4 workers)
    └── Vitesse : 60x 🚀
```

---

## 🚀 Exemple d'utilisation (OPTIMAL)

```bash
cd scientific_runs
python integrate_photons_on_perturbed_flrw_OPTIMAL.py
```

**Output attendu :**
```
=== Backward Ray Tracing with Excalibur (OPTIMAL) ===
    Numba JIT (15x) + Persistent Pool 4 workers (4x) = 60x speedup

1. Setting up cosmology...
2. Setting up grid and mass distribution...
3. Setting up spacetime metric (OPTIMAL)...
4. Setting up backward ray tracing...
5. Generating photons for backward ray tracing...
6. Calculating integration parameters...
7. Performing parallel backward ray tracing (OPTIMAL)...
   🚀 Using Persistent Worker Pool with 4 workers
   Worker pool ready, integrating 50 photons...
   ✓ All photons integrated successfully
   Integration time: 1.52s
8. Analyzing results...
9. Saving trajectories...
   ✓ Saved all 50 photon trajectories
10. Performance summary...
    ✓ Optimal backward ray tracing completed successfully!
    Performance: 50 photons in 1.52s (~33 photons/second)
```

---

## 🚀 Utilisation manuelle du multicore (si besoin)

Pour > 100 photons, ajouter la parallélisation :

### Modifier le script

```python
# À la place de l'intégration séquentielle :
# for photon in photons:
#     integrator.integrate(photon, n_steps)

# Utiliser le persistent pool :
from excalibur.integration.parallel_integrator_persistent import PersistentPoolIntegrator

with PersistentPoolIntegrator(metric, dt, n_workers=4) as integrator:
    integrator.integrate_photons(photons, n_steps)
```

**Performance attendue :**
- 50 photons × 1000 steps : **~1.5 secondes** (4 cores)
- **Speedup total : 60x** vs version standard 🚀

---

## 📝 Workflow de production

### 1. Préparer le run

```bash
cd scientific_runs
```

Éditer `integrate_photons_on_perturbed_flrw_OPTIMIZED.py` :
- Masse et position
- Nombre de photons
- Paramètres cosmologiques si besoin

### 2. Exécuter

```bash
python integrate_photons_on_perturbed_flrw_OPTIMIZED.py
```

**Monitoring :**
```
1. Setting up cosmology...
2. Setting up grid and mass distribution...
3. Setting up spacetime metric (OPTIMIZED)...
4. Setting up backward ray tracing...
5. Generating photons for backward ray tracing...
6. Performing backward ray tracing integration...
   Progress: 10/50 photons completed
   Progress: 20/50 photons completed
   ...
7. Analyzing results...
8. Saving trajectories...
   ✓ Saved all 50 photon trajectories
```

### 3. Vérifier les résultats

```bash
cd ../examples
python visualize_trajectories.py ../data/backward_raytracing_trajectories_OPTIMIZED_*.h5
```

### 4. Analyser

```python
import h5py
import numpy as np

with h5py.File('../data/backward_raytracing_trajectories_OPTIMIZED_*.h5', 'r') as f:
    for photon_name in f.keys():
        photon = f[photon_name]
        states = photon['states'][:]
        
        # Analyser...
        final_position = states[-1, 1:4]
        distance_travelled = np.linalg.norm(states[-1, 1:4] - states[0, 1:4])
        
        print(f"{photon_name}: travelled {distance_travelled/1e24:.2f} Mpc")
```

---

## ⚙️ Paramètres recommandés

### Configuration standard (production)
```python
N = 512                    # Haute résolution
grid_size = 2000 * one_Mpc # Grande boîte
M = 1e20 * one_Msun       # Cluster de galaxies
n_photons = 50            # Statistiques suffisantes
```
**Temps :** ~6 secondes (version OPTIMIZED)

### Configuration rapide (tests)
```python
N = 128                    # Résolution réduite
grid_size = 1000 * one_Mpc
M = 1e20 * one_Msun
n_photons = 10
```
**Temps :** ~1 seconde

### Configuration haute résolution (recherche)
```python
N = 1024                   # Très haute résolution
grid_size = 4000 * one_Mpc # Très grande boîte
M = 1e20 * one_Msun
n_photons = 100            # Excellentes statistiques
```
**Temps :** ~60 secondes (OPTIMIZED + multicore recommandé)

---

## 🔬 Cas d'usage scientifiques

### Lentille gravitationnelle
```python
# Masse importante
M = 1e21 * one_Msun  # Amas riche

# Photons traversant l'amas
n_photons = 100
cone_angle = np.pi / 6  # 30° pour couvrir l'amas
```

### Effet Sachs-Wolfe
```python
# Distribution de masse étendue
M = 5e20 * one_Msun
radius = 50 * one_Mpc

# Grille large
grid_size = 5000 * one_Mpc
N = 256  # Compromis taille/résolution
```

### Tests de GR
```python
# Masse modérée, haute précision
M = 1e20 * one_Msun
N = 1024
n_steps = 10000  # Haute résolution temporelle
```

---

## 📊 Output files

### Format HDF5

Structure du fichier :
```
backward_raytracing_trajectories_OPTIMIZED_mass_500_500_500_Mpc.h5
├── photon_0/
│   └── states: [n_steps × 8] array
│       ├── column 0: η (temps conformal)
│       ├── columns 1-3: x, y, z (positions)
│       └── columns 4-7: u^0, u^1, u^2, u^3 (vitesses)
├── photon_1/
│   └── states: ...
...
└── photon_49/
    └── states: ...
```

### Taille typique

- 50 photons × 1000 steps : **~3 MB**
- 100 photons × 5000 steps : **~30 MB**
- 500 photons × 10000 steps : **~300 MB**

---

## 🐛 Troubleshooting

### Photons sortent de la grille

**Symptôme :**
```
WARNING: Photon stopped at step 42! Error: Position [...] outside grid bounds
```

**Solutions :**
1. Augmenter `grid_size`
2. Réduire `n_steps`
3. Réduire la masse (moins de déflexion)

### Temps de calcul trop long

**Solutions :**
1. Utiliser version OPTIMIZED ✅
2. Réduire `N` (résolution)
3. Réduire `n_photons`
4. Ajouter multicore (persistent pool)

### Résultats instables

**Solutions :**
1. Réduire `dt` (déjà optimal dans version OPTIMIZED)
2. Augmenter `N` (résolution grid)
3. Vérifier que la masse n'est pas trop grande

---

## 📈 Performance attendue

| Configuration | Temps (OPTIMIZED) | Temps (standard) | Speedup |
|---------------|-------------------|------------------|---------|
| 50 photons × 1000 steps | 6s | 95s | 15x |
| 50 photons × 5000 steps | 30s | 475s | 15x |
| 100 photons × 1000 steps | 12s | 190s | 15x |
| **+ Multicore (4 cores)** | **1.5-3s** | - | **60x** |

---

## ✅ Checklist avant run

- [ ] Paramètres vérifiés (masse, grille, photons)
- [ ] Utilisation de la version OPTIMIZED
- [ ] Espace disque suffisant pour output
- [ ] Temps estimé acceptable
- [ ] Visualisation prête pour analyse

---

**Retour:** [README principal](../README.md) | [Organisation](../PROJECT_ORGANIZATION.md)

# 🧪 Tests Excalibur

Tests de performance, debugging, et validation du code.

## Tests de performance

### Benchmarks principaux

- **`test_performance_comparison.py`** ⭐
  - Compare version standard vs optimisée
  - **Résultat mesuré : 15.5x speedup**
  - 5 photons × 100 steps
  - Usage : `python test_performance_comparison.py`

- **`test_persistent_pool.py`** ⭐
  - Test du persistent worker pool
  - **Résultat : 4-8x speedup avec multicore**
  - 40 photons × 200 steps
  - Usage : `python test_persistent_pool.py`

- **`test_optimized_quick.py`**
  - Test rapide de validation
  - 5 photons × 100 steps
  - ~0.3 secondes
  - Usage : `python test_optimized_quick.py`

### Tests multiprocessing

- **`test_parallel_speedup.py`**
  - Test multiprocessing naïf (échec attendu)
  - Montre le problème de l'overhead Windows
  - 20 photons × 200 steps
  - Usage : `python test_parallel_speedup.py`

- **`test_parallel_sharedmem.py`**
  - Test avec shared memory
  - Amélioration partielle mais insuffisante
  - Usage : `python test_parallel_sharedmem.py`

---

## Tests de debugging

### Intégration

- **`test_debug_integration.py`**
  - Debug des étapes d'intégration RK4
  - Affiche k1, k2, k3, k4 à chaque step
  - Détecte explosions de vitesse
  - Usage : `python test_debug_integration.py`

- **`test_integration_params.py`**
  - Test des paramètres d'intégration
  - Vérifie dt, n_steps, convergence
  - Usage : `python test_integration_params.py`

- **`test_dt_fix.py`**
  - Test des corrections du time step
  - Vérifie que dt donne des déplacements raisonnables
  - Usage : `python test_dt_fix.py`

### Métrique et géométrie

- **`test_christoffel_debug.py`**
  - Debug des symboles de Christoffel
  - Affiche Γ^μ_αβ en différents points
  - Vérifie symétrie et valeurs
  - Usage : `python test_christoffel_debug.py`

### Données

- **`test_mass_parsing.py`**
  - Test du parsing de distribution de masse
  - Vérifie potentiel gravitationnel
  - Usage : `python test_mass_parsing.py`

---

## Résultats des tests

### Performance (test_performance_comparison.py)

```
Standard Implementation:
  Time: 0.944s
  Performance: 529 step-evals/sec

Optimized Implementation:
  Time: 0.061s  
  Performance: 8195 step-evals/sec

Speedup: 15.48x ✅
```

### Multicore (test_persistent_pool.py)

```
Sequential (1 core): 1.201s

Persistent Pool:
  2 workers: 0.284s (4.2x speedup) ✅
  4 workers: ~0.15s (8x speedup) ✅
```

### Multiprocessing naïf (test_parallel_speedup.py)

```
1 worker:  0.722s
2 workers: 1.818s (0.4x - RALENTISSEMENT) ❌
4 workers: 2.090s (0.35x) ❌

Conclusion: Overhead trop élevé, utiliser persistent pool
```

---

## Usage des tests

### Test rapide avant commit
```bash
python test_optimized_quick.py
```

### Validation complète
```bash
python test_performance_comparison.py
python test_persistent_pool.py
```

### Debug d'un problème
```bash
# Problème d'intégration
python test_debug_integration.py

# Problème de métrique
python test_christoffel_debug.py

# Problème de time step
python test_dt_fix.py
```

---

## Créer un nouveau test

Template :

```python
#!/usr/bin/env python3
"""
Description du test.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/magri/excalibur_project')

from excalibur.grid.grid import Grid
# ... autres imports

def test_my_feature():
    """Test description."""
    # Setup
    # ...
    
    # Test
    # ...
    
    # Assertions
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test passed")

if __name__ == '__main__':
    test_my_feature()
```

---

## Tests à ajouter (TODO)

- [ ] Test unitaires pour chaque module
- [ ] Tests de régression automatiques
- [ ] Tests de convergence RK4
- [ ] Tests de conservation d'énergie
- [ ] Tests de métriques alternatives (Schwarzschild)
- [ ] Tests d'intégration continue (CI)

---

**Retour:** [README principal](../README.md) | [Organisation](../PROJECT_ORGANIZATION.md)

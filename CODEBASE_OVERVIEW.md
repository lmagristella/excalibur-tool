# Excalibur — Documentation de la codebase

> Document d'orientation destiné à être donné à un agent (Claude) pour qu'il
> comprenne l'état du projet avant d'y intégrer / implémenter de nouvelles idées.
> Dernière mise à jour : juin 2026.

---

## 1. Ce qu'est Excalibur

Excalibur est une **toolkit de ray-tracing relativiste pour le lentillage
gravitationnel (gravitational lensing)**. On y propage des photons (géodésiques
nulles) à travers des espaces-temps courbes — principalement un **FLRW perturbé**
(jauge newtonienne conforme) et **Schwarzschild** — et on en extrait des
observables : déviation, redshift (décomposé en composantes), retard temporel,
et surtout les **scalaires optiques de Sachs** (convergence κ, cisaillement γ)
via la propagation de la **matrice de Jacobi**.

Cas d'usage typique : tirer un cône de photons « backward » depuis un
observateur, les faire traverser un halo (NFW sphérique ou triaxial, masse
sphérique uniforme, profils composites), et reconstruire les cartes de
lentillage / le ciel lentillé.

**Langue :** commentaires et docstrings mêlent français et anglais. Les
constantes physiques et conventions sont rigoureusement documentées dans le
code (signature métrique `(-,+,+,+)`).

**Pas de `CLAUDE.md` ni de `README` racine** pour l'instant — d'où ce document.

---

## 2. Structure de haut niveau

```
excalibur/              ← le package Python installable (cœur)
  core/                 ← constantes, unités, cosmologie, coordonnées
  metrics/              ← métriques (FLRW perturbé, Schwarzschild) + variantes "fast"/jax
  objects/              ← profils de masse (NFW, NFW triaxial, masse sphérique, composites)
  grid/                 ← grilles, AMR, interpolateurs (tri-linéaire → tricubique), bypass analytique
  photon/               ← état photon, collections, historique, I/O trajectoires
  integration/          ← intégrateurs (RK4, Leapfrog4, RKF45, DOPRI5) + backends Numba/JAX + parallélisme
  observables/          ← Riemann, base de Sachs, matrice tidale optique, Jacobi, redshift, plots
  io/                   ← nommage de runs, visualisation, helpers fichiers
  jax_backend/          ← backend JAX alternatif (géodésiques, intégrateurs, pipeline batch)

_excalibur_runs/        ← scripts de production (pipelines end-to-end, un "main" par étude)
_postprocessing/        ← analyse + apps interactives (ciel lentillé, cartes redshift, time-delay)
_tests/                 ← tests de validation physique + benchmarks de perf (pas pytest pur)
_examples/              ← exemples d'usage minimaux
_bench/                 ← micro-benchmarks d'intégration
results*/, _data/       ← sorties (.npz, .h5, .csv, .mp4, .gif) — artefacts, pas du code
_packaging/             ← build AppImage de l'app interactive
```

`pyproject.toml` : package `excalibur-tool`, setuptools, `requires-python >=3.9`.
Dépendances de fait (non figées dans pyproject) : `numpy`, `scipy`, `numba`,
`matplotlib`, `h5py` (optionnel, I/O trajectoires), `jax` (optionnel, backend).

---

## 3. Le cœur physique (`excalibur/`)

### 3.1 `core/` — fondations

- **`constants.py`** : définit le **système d'unités**, choisi par la variable
  globale `unit_system` (`"si"` par défaut, `"cosmo"` = Msun/Gyr/Gpc, `"natural"`
  c=G=1). ⚠️ **Point de friction connu** : tout le code lit `c`, `G`, `one_Mpc`,
  `one_Msun`… depuis ce module via `from ...constants import *`. Changer
  `unit_system` rejaillit globalement.
- **`cosmology.py`** : `LCDM_Cosmology(H0, Ω_m, Ω_r, Ω_Λ, Ω_k)`. Calcule
  `a(η)`, `ȧ(η)`, le Hubble conforme, les distances comobiles. Fournit des
  interpolateurs Numba (`a_interp_numba`) pour un accès chaud rapide à `a(η)`.
- **`coordinates.py`** : conversions cartésien ↔ sphérique (positions et
  vitesses).

### 3.2 `metrics/` — espaces-temps

Interface abstraite `base_metric.Metric` : `metric_tensor`, `christoffel`,
`geodesic_equations`, `metric_physical_quantities`.

- **`perturbed_flrw_metric.py`** : version de référence, lisible.
  `ds² = a²(η)[-(1+2Ψ)c²dη² + (1-2Φ)δ_ij dxⁱdxʲ]`, avec Ψ=Φ (pas de stress
  anisotrope). Lit Φ et ses dérivées via l'interpolateur.
- **`perturbed_flrw_metric_fast.py`** ⭐ : version de production. Fonctions
  `@njit` (`compute_tensorial_acceleration`, `compute_analytical_acceleration`),
  cache de `a(η)`, `geodesic_equations_extended` (état 24-composantes incluant
  Sachs + Jacobi). C'est la métrique utilisée par les runs de lensing.
- **`perturbed_flrw_metric_jax.py`** / `_adimensionaltest.py` : variantes.
- **`schwarzschild_metric.py`** (+ `_fast`, `_cartesian`) : Schwarzschild en
  sphérique, conversion cartésien↔sphérique à l'entrée. Sert de **banc d'essai
  analytique** (déviation exacte connue) pour valider le solveur.

### 3.3 `objects/` — distributions de masse

Génèrent le potentiel Φ (et gradient/Hessien) déposé sur la grille, **ou**
exposés analytiquement (cf. bypass).

- **`nfw_halo.py`** : `NFWHalo` (sphérique) et `TriaxialNFWHalo`. Fournit le
  potentiel 3D pour le ray-tracing **et** les quantités projetées analytiques
  `Σ(b), κ(b), γ(b)` (formules Bartelmann 1996 / Wright-Brainerd 2000) pour
  validation.
- **`spherical_mass.py`** : masse sphérique uniforme (potentiel intérieur/extérieur).
- **`equivalent_mass_profiles.py`** : `TriaxialProfileSpec`, `ComponentSpec`,
  `CompositeAnalyticalSource`, `LabeledAnalyticalSource` — composer plusieurs
  profils et comparer des « masses équivalentes ».

### 3.4 `grid/` — champs & interpolation

Le potentiel vit sur une grille ; l'interpolateur fournit `valeur, gradient,
Hessien, ∂_t` au point du photon.

- **`grid.py`** : `Grid` (champs nommés, support `shared_memory` pour le
  multiprocessing).
- **`interpolator.py`** → **`interpolator_fast.py`** → **`interpolator_4d_fast.py`**
  (`InterpolatorFast`) : montée en gamme tri-linéaire → **tricubique** Numba, 4D
  (3 espace + 1 temps). C'est l'interpolateur de production.
- **`amr_grid.py`** : AMR par patchs (style Berger-Colella simplifié).
  `AMRGrid.from_field(...)`, `AMRInterpolator` (drop-in de `InterpolatorFast`,
  dispatch transparent vers le patch le plus fin couvrant le point).
- **`analytical_bypass.py`** : `AnalyticalBypassInterpolator` — dans une sphère
  autour de l'objet, **remplace** la valeur grille par la formule analytique
  exacte (gradient, Hessien). Motivation : une grille finie ne résout pas le cusp
  NFW (κ_sim plafonne) ; le bypass restaure κ>1 et les images multiples. Outil
  clé pour **mesurer les effets de grille**.

### 3.5 `photon/` — état & collections

- **`photon.py`** : `Photon(position, direction, weight, record_lensing)`. État =
  `[xᵘ(4), uᵘ(4)]`, plus `D_flat`/`P_flat` (Jacobi) si `record_lensing`.
  Diagnostics de condition nulle (`null_condition`, erreur **relative** — en SI
  `gᵤᵥuᵘuᵛ ~ 10⁵²`, il faut raisonner en relatif).
- **`photons.py`** : `Photons` — collections, génération de cônes
  (`generate_cone_random`, bases de direction), I/O trajectoires HDF5 (h5py
  optionnel). `photons2.py` est une variante.
- **`photon_history.py`** : enregistrement des états le long de la trajectoire.

### 3.6 `observables/` — extraction de la physique

C'est là que se fait le lensing « propre » (Sachs/Jacobi) :

- **`riemann_perturbed_flrw.py`** : blocs minimaux du tenseur de Riemann
  (`R_{k00l}, R_{0lki}, R_{kijl}`, tous indices bas) nécessaires à la matrice
  tidale. Formules explicites pour FLRW perturbé, Ψ=Φ.
- **`sachs_basis.py`** : base d'écran de Sachs `e_1, e_2` (plan orthogonal à kᵘ et
  à l'observateur uᵘ). Initialisation + transport projeté sur l'écran.
  ⚠️ Plusieurs **conventions d'écran** (`metric`, `conformal_metric`,
  `physical_metric`, `euclidean_local`) — c'était la source d'un **biais de 2%**
  désormais corrigé (cf. §6).
- **`optical_tidal_matrix.py`** : `R_AB = R_{μανβ} kᵃkᵝ e_Aᵘ e_Bᵛ`, le RHS de
  Jacobi (`dD=P, dP=-R·D`), les scalaires optiques (κ, γ₁+iγ₂), et
  `lensing_from_jacobi` / `angular_diameter_distance_from_jacobi`.
- **`redshift.py`** : décompose `1+z` en homogène (expansion), Doppler, ISW,
  Sachs-Wolfe, vitesse. `RedshiftCalculator`.
- **`redshift_plots.py`** + **`lensing_conventions.py`** : helpers de
  normalisation Σ_cr (comobile/conforme vs physique, facteur `1/(1+z_l)`).

### 3.7 `integration/` — le moteur, et sa zoologie

**Beaucoup de variantes** (héritage de chasses à la performance). Hiérarchie de
lecture :

- **`integrator.py`** : schémas de base lisibles — `RK4`, `Leapfrog4`
  (Forest-Ruth symplectique 4e ordre, pas fixe), classe `Integrator`. Point
  d'entrée pédagogique.
- **`integrator_numba.py`** ⭐ : `NumbaAMRBackend` — **tout le hot-path sous
  `@njit`** : lookup patch AMR, interpolation tricubique, Christoffel, blocs
  Riemann, matrice tidale, RHS Jacobi, et les schémas (RK4 / RKF45 /
  **Dormand-Prince 5(4)**). État 8-comp (géodésique) ou 24-comp (avec
  Sachs+Jacobi). C'est le backend rapide moderne.
- `integrator_numba_specialized.py`, `_lowalloc.py`, `_schemes.py` : variantes
  spécialisées / bas-allocation.
- `integrator_optimized.py`, `integrator_old.py` : versions historiques.
- **Parallélisme** : `parallel_integrator.py` (Pool naïf),
  `parallel_integrator_persistent.py` (pool persistant ⭐), `_sharedmem.py`
  (mémoire partagée), `_analytical.py`, `parallel_workers.py`.
- **`integrator_jax.py`** + `jax_backend/` : backend JAX (vmap/jit), pipeline
  batch — c'est ce qui donne le **x2000** et « photons jusqu'à z=2 en 0.2 ms »
  du commit récent.

### 3.8 `io/`

- **`filename_utils.py`** : `RunNamer` — nommage canonique des runs (encode
  masse/rayon/observateur/grille dans le nom de fichier).
- **`visualization.py`** : tracés de trajectoires/champs.

---

## 4. Les pipelines (`_excalibur_runs/`)

Chaque script est un « main » autonome (insère `..` dans `sys.path`, importe
`excalibur`, configure une étude, écrit un `.npz`/`.h5`, puis un script
`_postprocessing/analyze_*` le lit). Familles notables :

- **`integrate_photons_on_perturbed_flrw_OPTIMAL.py`** (prod, pool persistant),
  **`_OPTIMIZED.py`** (Numba seul), `_integrator.py`, version de base — pipeline
  de propagation FLRW perturbé.
- **`run_lensing_nfw*.py`** : lensing NFW — `.py`, `_analytic.py`,
  `_analytic_test.py`, `_cosmo.py`, `_amr.py`, `_triaxial.py`. Tirent un cône,
  propagent Sachs+Jacobi, reconstruisent κ/γ et comparent à l'analytique NFW.
- **`run_bias_*`** : scans du biais (zl, cnfw, diagnostic) — investigations de la
  précision du solveur.
- **`run_lensing_equivalent_mass_profiles.py`**, `run_lensing_cone.py`,
  `run_kappa_flrw_check.py`, `run_flrw_bg_test.py`.
- **Schwarzschild** : `excalibur_run_schwarzschild.py`,
  `excalibur_run_compare_schwarzschild_vs_flrw*.py`,
  `integrate_photons_on_schwarzschild.py` — validation contre l'analytique.
- **JAX** : `excalibur_run_perturbed_flrw_jax.py`, `demo_jax_backend_quick.py`.

Voir `_excalibur_runs/README.md` pour les perfs annoncées (15x Numba, 60x
+persistent pool).

---

## 5. Post-traitement & apps interactives (`_postprocessing/`)

C'est le **focus actuel** (commits récents). Deux axes :

1. **Cartes / reconstruction** : `make_lensed_sky.py` (Sachs),
   `make_lensed_sky_raytracing.py` (ray-tracing), `analyze_lensing_nfw.py`,
   `analyze_lensing_cone.py`, `compute_redshift_map.py`,
   `compute_time_delay_map.py`, `plot_redshift_*`,
   `fit_equivalent_mass_spherical_nfw_bias.py`.
2. **Apps interactives** (le « ciel lentillé » manipulable) :
   - `lensed_sky_interactive_common.py` ⭐ (le fichier ouvert dans l'IDE) :
     dataclasses `PrecomputedMapping`, `LensProfile` ; sliders/boutons matplotlib
     pour basculer entre profils de lentille précalculés sur **la même** grille
     (`half_view`, `n_fine`) — le viewer ne fait que permuter le champ β.
   - `make_lensed_sky_raytracing_interactive.py`,
     `make_lensed_sky_raytracing_and_sachs_interactive.py`,
     `lensing_app_launcher.py`, `build_profile_caches.py`.
   - **Packaging** : `lensing_app.spec` (PyInstaller), `build_lensing_app.sh/.bat`,
     `LENSING_APP_BUILD.md`, et `_packaging/build_appimage.sh` (AppImage Linux).
   - `blog_interactive_lensing.md` : article/notes (non commité).

---

## 6. Conventions & pièges connus (à lire avant de toucher)

- **Signature métrique** `(-,+,+,+)`. Indice 0 = temps conforme η.
- **Unités globales** via `core/constants.py` (`unit_system`). En SI, les normes
  `gᵤᵥuᵘuᵛ` sont énormes → toujours juger la condition nulle en **erreur
  relative**.
- **Conventions de Sachs / Σ_cr** : distinction *conforme-comobile* vs *physique*
  (facteur `1/(1+z_l)`). Un **biais de 2%** historique venait d'une mauvaise
  initialisation de l'écran de Sachs ; corrigé → biais 0.3% (commits `522ad98`,
  `a17d38c`). κ/γ d'un NFW sphérique statique sont correctement retrouvés en
  frame conforme. **Ne pas réintroduire** la confusion de convention.
- **Effets de grille** : une grille finie lisse le cusp NFW → utiliser
  `AnalyticalBypassInterpolator` pour isoler la physique des artefacts de
  résolution.
- **RK4 et cancellation** : positions ~1e24 m + incréments petits → mise à jour
  position/momentum **séparée** pour limiter l'annulation catastrophique
  (cf. `integrator.py`).
- **Multiprocessing** : overhead non rentable < ~20 photons ; le `Leapfrog4`
  exige un **pas fixe** (pas d'adaptatif).
- **`h5py` et `jax` optionnels** : imports gardés en try/except pour ne pas
  casser le cœur si absents/incompatibles avec la version de numpy.
- **Beaucoup de fichiers redondants** (`integrator_*`, `photons2`,
  `*_OPTIMAL/_OPTIMIZED`) : héritage d'itérations de perf. Préférer les versions
  ⭐ ci-dessus ; les autres sont surtout des références historiques/tests.

---

## 7. Tests & benchmarks (`_tests/`)

Mélange de **validation physique** (condition nulle, NFW analytique vs simulé,
redshift homogène, perturbé vs pur, Riemann/Christoffel par différences finies,
matrice tidale, parallélisme) et de **benchmarks de perf** (souvent exécutés en
`python test_xxx.py`, pas du pytest strict — un `.pytest_cache` existe mais la
suite est hétérogène). Voir `_tests/README.md`. `_bench/` contient des
micro-benchmarks d'intégration FLRW.

Fichiers de validation clés : `test_triaxial_nfw_halo.py`,
`test_riemann_blocks_*`, `test_optical_*`/`diagnostic_sachs_e0.py`,
`test_flrw_regression_kappa.py`, `test_perturbed_vs_pure_validation.py`,
`test_unified_integrator.py`.

---

## 8. Flux de données end-to-end (vue d'ensemble)

```
 LCDM_Cosmology  ──►  a(η), distances
        │
 objects/NFWHalo ──►  Φ(x)  ──►  Grid / AMRGrid  ──►  Interpolator(Fast/AMR/Bypass)
        │                                                   │
        └──────────── analytique κ,γ (validation)           ▼
                                              metrics/PerturbedFLRWMetricFast
 Photons.generate_cone_random ──► état initial (+ Sachs e_A, Jacobi D,P)
        │
        ▼
 integration (RK4 / DOPRI5 ; backend Numba ou JAX ; séquentiel ou pool)
        │   propage  [xᵘ, uᵘ]  (+ R_AB, D, P si lensing)
        ▼
 observables : redshift décomposé,  Jacobi ─► κ, γ,  distances ang.-diam.
        │
        ▼
 io : trajectoires .h5 / résultats .npz  (RunNamer)
        │
        ▼
 _postprocessing : cartes κ/γ/redshift/time-delay,  ciel lentillé,  apps interactives
```

---

## 9. Pour intégrer de nouvelles idées — points d'ancrage

Quand on voudra **étendre** le projet, voici où brancher selon le type d'idée :

- **Nouveau profil de masse / lentille** → `objects/` (suivre l'API
  `potential / gradient / hessian` + projeté analytique pour validation), puis un
  `run_lensing_*` et une `LensProfile` dans l'app interactive.
- **Nouvelle métrique** → sous-classer `metrics/base_metric.Metric` ; prévoir une
  version « fast » `@njit` et idéalement le bloc Riemann correspondant dans
  `observables/`.
- **Nouvel observable** → `observables/` (modèle : `optical_tidal_matrix.py`),
  l'exposer dans `observables/__init__.py`.
- **Performance / nouveau schéma d'intégration** → `integration/integrator_numba.py`
  (hot-path njit) ou `jax_backend/` ; valider contre Schwarzschild analytique.
- **Interactivité / visualisation** → `_postprocessing/` (réutiliser le système
  de caches précalculés `build_profile_caches.py` + `lensed_sky_interactive_common.py`).
- **Dette technique évidente** : consolider la zoologie d'intégrateurs, ajouter
  un `README`/`CLAUDE.md` racine, figer les dépendances dans `pyproject.toml`,
  rendre la suite de tests homogène (pytest).

---

### Annexe — repères chiffrés (commits récents)

- DOPRI5(4) + batching parallèle : **x2000**, photons jusqu'à z=2 en **~0.2 ms**.
- Numba JIT seul : ~15-20x ; + pool persistant : ~60x vs version de base.
- Biais lentillage NFW sphérique statique (frame conforme) : **0.3%** après
  correction de l'écran de Sachs.

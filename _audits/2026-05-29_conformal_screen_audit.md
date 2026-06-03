# Audit physique du pipeline de lensing — convention conforme vs physique

**Date** : 2026-05-29
**Référence** : Fleury, *Light propagation in inhomogeneous and anisotropic cosmologies*, thèse UPMC 2015 (HAL tel-01227964v2)
**Périmètre** : pipeline `excalibur` de lensing weak (kernel python + numba spécialisé)
**Runner principal audité** : `_excalibur_runs/run_lensing_equivalent_mass_profiles.py`

---

## Question initiale

> *Est-ce que les simulations de lensing simulent réellement un univers en expansion ?
> Le code force `a=1, ȧ=0, H=0, H'=0` dans le kernel numba spécialisé ;
> est-ce physiquement correct ou est-ce qu'on perd la cosmologie ?*

**Réponse courte** : **Oui, c'est physiquement correct**, mais via le *conformal trick* de Fleury §5.1.1. L'expansion entre par les distances cosmologiques, le facteur a_S à la source, et le 1/a(η') dans l'intégrale de convergence — **pas par des termes H, H' dans le Riemann projeté**, qui sont rigoureusement nuls dans la base de Sachs conforme.

---

## 1. Cadre théorique (Fleury)

### 1.1 Le tour de passe-passe conforme (§5.1.1, p.80)

Invariance conforme exacte des géodésiques nulles (théorème §1.2.3) :
si `g = a²(η) g̃`, alors les rayons lumineux suivent **les mêmes courbes** dans `g` et `g̃`.

Fleury écrit la métrique perturbée newtonienne comme :
```
g_μν = a²(η) [-(1+2Ψ) dη² + (1-2Φ) δ_ij dx^i dx^j]
     = a²(η) g̃_μν
```
puis travaille **dans `g̃` (sans expansion explicite)** et récupère les observables physiques via le dictionnaire conforme.

### 1.2 Dictionnaire conforme (Table 5.1, p.81)

| Quantité | Relation g ↔ g̃ |
|---|---|
| Métrique | `g_μν = a² g̃_μν` |
| **Paramètre affine** | `dv = a² dṽ` |
| 4-vecteur d'onde | `k^μ = a⁻² k̃^μ` ; `k_μ = k̃_μ` |
| **Base de Sachs** | `s_A^μ = a⁻¹ s̃_A^μ` |
| **Matrice de Jacobi** | `D(S←O) = a_S a_O · D̃(S←O)` |
| Distance angulaire | `D_A = a_S · D̃_A` |
| **Scalaires γ, ψ, φ** | **invariants** (γ = γ̃) |

**Conséquence cruciale** : `κ ≡ 1 - tr(A)/2` avec `A = D/D̄` est invariant conforme (les facteurs `a_S a_O` se simplifient).

### 1.3 Tenseur de marée optique perturbé (§5.2.2, eq 5.43)

Dans le frame conforme :
```
δR̃_AB = -2ω̄² ∂_A^⊥ ∂_B^⊥ Φ  +  O(∂g̃) terms négligeables
```

**Aucun terme en H, H', a, ȧ ne survit**. Seul le Hessien transverse de Φ contribue. C'est une **identité algébrique exacte** dans le frame conforme, pas une approximation.

### 1.4 Solution Green (eq 5.48)

```
D_AB(S←O) = -a_S ω₀⁻¹ f_K(χ) { δ_AB - 2 ∫₀^χ dχ' [f_K(χ') f_K(χ-χ') / f_K(χ)] ∂_A^⊥ ∂_B^⊥ Φ }
```

C'est **la formule de référence** que le code doit reproduire.

### 1.5 Convergence cosmique (eq 5.56)

```
κ = (3/2) H₀² Ω_m0 ∫₀^χ dχ' [f_K(χ') f_K(χ-χ') / f_K(χ)] × δ(η', x') / a(η')
```

Avec Poisson `∇²Φ = 4πG a² ρ̄_0 δ/a³`. L'expansion entre par `1/a(η')` dans l'intégrale.

---

## 2. Anatomie du code excalibur — trois branches

| Branche | Backend | Sachs convention | Christoffel | Riemann | Cohérence |
|---|---|---|---|---|---|
| **(A)** | Python | `metric` | Γ physique | Full FLRW + Φ | ✓ |
| **(B)** | Python | `conformal_metric` | Γ physique | Full FLRW + Φ | **✗ MÉLANGE** |
| **(C)** | Numba spécialisé | `conformal_metric` | Γ̃ (a=1) | Φ uniquement | ✓ |

Le runner par défaut utilise **(C)** — kernel numba spécialisé en mode conforme.

---

## 3. Audit fichier par fichier

### 3.1 `excalibur/observables/riemann_perturbed_flrw.py:69-78`

Formule du code pour `R_{k00l}` diagonale :
```
H'(1 - 2Ψ/c²) + Ψ''/c² + H(Φ' + Ψ')/c²
```
✓ Cohérent avec Mukhanov / Durrer pour métrique perturbée FLRW newtonienne.

`slow_roll=True` zéroe Ψ', Ψ'' — **justifié** par Fleury §5.2.1 et note sous (5.27) : en ère de matière linéaire (Ω_m ≈ 1), `δ ∝ a` ⇒ `∂_η Φ = 0`. Néglige uniquement ISW intégré, qui contribue < 1% pour weak lensing par halo.

### 3.2 `excalibur/integration/integrator_numba_lowalloc.py:354-359`

```python
geo_a = 1.0
geo_adot = 0.0
optic_a = 1.0
optic_H_conf = 0.0
optic_H_prime = 0.0
```
Activé pour tout `screen_mode != METRIC`. **C'est l'application directe du dictionnaire de Fleury** — pas une approximation, une identité.

### 3.3 `excalibur/observables/sachs_basis.py:164-195`

```python
if convention == "conformal_metric":
    screen_g_mu_nu = g_mu_nu / (a * a)
    ...
```
Base orthonormalisée dans `g̃ = g/a²` ⇒ le `e1, e2` produit est **s̃_A** au sens de Fleury. ✓

### 3.4 `excalibur/observables/optical_tidal_matrix.py:392-429` — `lensing_from_jacobi`

```python
convergence = 1.0 - 0.5 * tr_D    # tr_D = tr(D_flat / lambda_S)
```

⚠️ Cette formule **donne κ_Fleury uniquement si `D̄/λ = 1₂` dans le fond non-perturbé**.

- En mode **conforme + K=0** : `D̄ = ω̃₀⁻¹ χ_S · 1₂`, `λ = ṽ_S = χ_S` ⇒ `D̄/λ = 1₂` ✓
- En mode **physique** : `D̄ = a_S ω̃₀⁻¹ χ_S · 1₂`, `λ = v_S = ∫ a² dṽ ≠ χ_S` ⇒ `D̄/λ ≠ 1₂` ⚠️

**Conclusion** : `lensing_from_jacobi` est correct en convention conforme, **biaisé en convention physique** (facteur constant `a_S × ṽ_S / v_S`).

### 3.5 `excalibur/observables/lensing_conventions.py:21-32` — Σ_cr

```python
sigma_cr_comoving = (c² / 4πG) * d_s / (d_l * d_ls)   # d_X COMOVING
sigma_cr_physical = sigma_cr_comoving / (1 + z_l)
```

⚠️ **Suspect**. La formule standard `Σ_cr_phys = (c²/4πG) D_A^s / (D_A^l × D_A^ls)` avec `D_A = D_C/(1+z)` donne :
```
Σ_cr_phys = Σ_cr_comoving × (1 + z_l)
```
**Le code divise au lieu de multiplier**. À auditer avec test dédié (voir §5).

---

## 4. Explication numérique du test 3-branches

Test : `_tests/test_screen_convention_equivalence.py`, NFW (M=2×10¹⁵ M☉, c=7), z_s=1, 6 impact parameters de 0.05 à 5 Mpc.

### 4.1 Trajectoires

| | |A-B| final pos | |C-A| final pos |
|---|---|---|
| Tous les b | < 5×10⁻¹³ Mpc | **~1834 Mpc** |

- A vs B : géodésique identique (le choix de Sachs basis n'affecte pas la géodésique) ✓
- C vs A : trajectoire différente, car λ_C = ṽ ≠ v_A. **Pas un bug** — c'est la conséquence du changement de paramétrage affine. Le chemin spatial est le même, seul le paramètre d'arrêt change.

### 4.2 κ et |γ| — facteurs constants

| | κ_A / κ_B | κ_C / κ_B |
|---|---|---|
| Tout b | **1.448** | **1.055** |

**Le facteur 1.448** s'explique par :
```
κ_A/κ_B = (D̄_B/λ_B) / (D̄_A/λ_A) = v_S / (a_S · ṽ_S)
```
Pour z_s=1 en ΛCDM : `a_S = 0.5`, `<a²> ≈ 0.725` ⇒ `v_S/(a_S ṽ_S) ≈ 0.725/0.5 = 1.45`. ✓

**Le facteur 1.055** est l'artefact de l'incohérence de (B) — transport de s̃_A avec Γ physique au lieu de Γ̃. Terme parasite `~H/c × λ` accumulé sur la trajectoire.

---

## 5. Verdict final

### ✅ Ce qui est correct

1. **Branche (C) — kernel numba spécialisé en mode conforme** est physiquement correcte, cohérente avec Fleury.
2. **L'expansion EST simulée**, via : χ(z) pour les distances, `a_S` à la source, `1/a(η')` dans l'intégrale de κ.
3. **Géodésique conforme = géodésique physique** par invariance exacte. Pas une approximation.
4. **κ, γ invariants conformes** — c'est la magie de Fleury.

### ⚠️ Ce qui est suspect ou faux

5. **Branche (B) — python en mode conforme** est mathématiquement incohérente. Biais ~5%. À éviter en production.
6. **Branche (A) — python en mode physique** est physiquement correcte mais `lensing_from_jacobi` ne renormalise pas par `D̄_FL` ⇒ κ biaisé d'un facteur ~1.45. **Si on l'utilise, il faut le multiplier par 1/(a_S × <a²>/<a>) explicitement après.**
7. **Σ_cr_physical = Σ_cr_comoving / (1+z_l)** — formule a priori opposée à la dérivation standard. Probablement origine du commit `a17d38c "Bias identified - convention issue between physical and conformal"`. **À auditer.**

### Recommandations

1. ✅ Continuer kernel numba spécialisé `conformal_metric` pour la production.
2. ⚠️ Auditer Σ_cr → **fait, voir §6**.
3. ⚠️ Ne pas utiliser modes (A), (B) sans renormalisation.
4. 💡 Test absolu Born → **fait, voir §7**.

---

## 6. AUDIT Σ_cr — Résultats numériques

### 6.1 Test : `_audits/test_sigma_cr_convention.py`

Géométrie : z_l = 0.3, z_s = 1.0, ΛCDM (H0=70, Ωm=0.3).
Référence : Bartelmann & Schneider 2001, eq 11 (distances ANGULAIRES physiques).

```
Sigma_cr_BS_physical (référence)                = 2.835e+15 Msun/Mpc^2
Sigma_cr_comoving (code, D_C substitué)          = 2.181e+15 Msun/Mpc^2  = 0.7692 × BS = BS / (1+z_l)
Sigma_cr_physical (code, comoving/(1+z_l))      = 1.677e+15 Msun/Mpc^2  = 0.5917 × BS = BS / (1+z_l)²
Sigma_cr_correct (= comoving × (1+z_l))         = 2.835e+15 Msun/Mpc^2  = BS ✓
```

### 6.2 Verdict Σ_cr

⚠️ **La formule `Σ_cr_physical = Σ_cr_comoving / (1+z_l)` du code est FAUSSE** par rapport à la référence Bartelmann-Schneider. La formule correcte est `× (1+z_l)`, pas `÷ (1+z_l)`.

⚠️ **La convention "comoving" du code n'est pas Σ_cr_physique** : elle vaut `Σ_cr_BS / (1+z_l)`. C'est juste la formule de BS où on a substitué naïvement des distances comoving à la place des distances angulaires, sans corriger pour les facteurs (1+z) qui apparaissent dans D_A = D_C/(1+z).

---

## 7. TEST BORN — κ_simulation vs κ_analytique NFW

### 7.1 Test : `_audits/test_born_kappa_vs_analytic.py`

Setup : NFW M_200 = 5×10¹⁴ M☉, c=5, lens à z_l = 0.3, source à z_s = 1.0.
Pipeline production (numba spécialisé, conformal_metric, dopri5, rtol=1e-9).
b_test = [1, 2, 3, 5, 8, 15] Mpc (tous > r_s, régime weak Born).

### 7.2 Résultats numériques

```
 b (Mpc) |     k_sim/k_BS  k_sim/k_comov  k_sim/k_phys_code
    1.00 |         1.3000         1.0000             0.7693
    2.00 |         1.3000         1.0000             0.7692
    3.00 |         1.3000         1.0000             0.7692
    5.00 |         1.2999         0.9999             0.7692
    8.00 |         1.2998         0.9999             0.7691
   15.00 |         1.2996         0.9997             0.7690
    mean |         1.2999         0.9999             0.7692
```

À 4 chiffres significatifs, **constant pour tous les b**.

### 7.3 Interprétation rigoureuse

| Observation | Interprétation |
|---|---|
| κ_sim / κ_comoving_code = 1.0000 | ✓ Cohérence interne parfaite : le code calcule exactement `κ = Σ/Σ_cr_comoving` |
| κ_sim / κ_BS = 1+z_l = 1.3 | ⚠️ Le κ_sim est `(1+z_l) × κ_physique_observable` — il n'est PAS l'invariant conforme de Fleury |
| κ_sim / κ_phys_code = 1/(1+z_l) | Conséquence directe : la formule `Σ_phys = Σ_co/(1+z_l)` du code donne un κ_analytic encore plus éloigné de κ_sim |

### 7.4 Diagnostic du bug

**Conclusion ferme :**

Le `κ_code` calculé par la simulation n'est **PAS** l'observable physique κ que prédit Fleury (eq 5.48 + dictionnaire conforme). Il diffère d'un facteur global `(1+z_l)`.

**Origine probable** : la normalisation de la matrice de Jacobi `D_norm = D_flat / λ_total` utilise `λ_total = D_C_s / c` (paramètre affine conforme), mais l'attendu pour donner directement κ_invariant est `λ̄ = a_S × χ_S × ω̃_0⁻¹` (ou similaire). Le manque du facteur `a_S = 1/(1+z_s)` dans la normalisation introduirait... mais on a (1+z_l), pas (1+z_s).

Plus précisément : `(1+z_l) = a_O/a_l`. Cela suggère que c'est le facteur d'échelle **au plan de la lentille**, pas à la source, qui manque/intervient. Cohérent avec le fait que `Σ_cr` est intrinsèquement liée au point d'évaluation `z_l`.

**Hypothèse mathématique** : la matrice optique de marée du code en frame conforme produit
$$\widetilde{\mathcal R}_{AB}^{\rm code} \stackrel{?}{=} -2 \tilde\omega^2 \partial_A^\perp \partial_B^\perp \Phi^{\rm comoving}$$
où `∂^⊥_comoving` utilise des dérivées en coordonnées comoving (alors que la prédiction de Fleury utilise `∂^⊥` au sens des coordonnées physiques transverses). Cela introduirait un facteur `a` pour chaque dérivée transverse → `a²` net → 1/(1+z_l)² au plan de la lentille... mais on observe `(1+z_l)¹`, pas `²`. Reste à clarifier.

### 7.5 Impact pratique

✅ **Pour des études DIFFÉRENTIELLES** (biais relatif entre formes de halo, ratios de κ entre simulations) : aucun impact. Le facteur (1+z_l) est commun à tous les runs et se simplifie.

⚠️ **Pour des comparaisons ABSOLUES à des prédictions analytiques NFW** : il faut soit :
  - **(option A)** comparer `κ_sim` à `κ_analytic = Σ/Σ_cr_comoving_code` (déjà fait par défaut dans les runners) — la comparaison est COHÉRENTE EN INTERNE.
  - **(option B)** diviser `κ_sim` par `(1+z_l)` pour récupérer le κ observable (Fleury invariant) — nécessaire si on veut comparer à des observations ou à la littérature.

⚠️ **Pour comparaisons à des observations** (catalogues weak-lensing, profils mesurés) : **DIVISER κ_sim par (1+z_l)**. Sinon, on a un facteur 1.3 d'erreur pour z_l = 0.3, jusqu'à 2× pour z_l = 1.

---

## 8. Verdict consolidé final

| Question | Réponse |
|---|---|
| L'expansion est-elle simulée ? | **Oui**, via le formalisme conforme de Fleury (§1) |
| Les géodésiques sont-elles physiques ? | **Oui**, par invariance conforme exacte |
| Le Riemann projeté est-il correct ? | **Oui**, eq 5.43 de Fleury rigoureusement appliquée |
| κ_simulé = invariant conforme observable ? | **NON** — facteur `(1+z_l)` parasite |
| Le code est-il cohérent en interne ? | **Oui** : κ_sim = Σ / Σ_cr_comoving exactement |
| La formule Σ_cr_physical du code est-elle correcte ? | **NON** — devrait être `× (1+z_l)`, pas `÷ (1+z_l)` |

**TL;DR** : la simulation est physiquement consistante (géodésique nulle perturbée correcte, Riemann correct, transport correct). Mais la **normalisation finale de κ contient un facteur (1+z_l) parasite par rapport à l'invariant observable**. Pour des études de biais relatifs (le cas usuel), c'est sans conséquence. Pour des comparaisons absolues à des observations, il faut diviser par (1+z_l).

---

## 9. VALIDATION DU FIX — multi-z_l, κ ET γ

### 9.1 Test : `_audits/test_fixed_kappa_gamma_multi_zl.py`

Scan de z_l ∈ {0.1, 0.2, 0.3, 0.5, 0.8}, z_s = 1.5, NFW (M=3×10¹⁴ M☉, c=5), b/r_s ∈ {1.5, 2.5, 4, 6.5, 10} (régime Born propre).

Le fix testé :
```python
kappa_fixed = kappa_sim / (1 + z_l)
gamma_fixed = gamma_sim / (1 + z_l)
```

### 9.2 Résultats numériques

```
  z_l |  (1+z_l) |  mean k_sim/k_BS  mean k_fix/k_BS  mean g_fix/g_BS |  max|k_fix-1|  max|g_fix-1|
 0.10 |   1.1000 |          1.1000          1.0000          1.0000  |    3.29e-05      4.26e-05  [OK]
 0.20 |   1.2000 |          1.2000          1.0000          1.0000  |    3.37e-05      3.96e-05  [OK]
 0.30 |   1.3000 |          1.3000          1.0000          1.0000  |    3.41e-05      3.94e-05  [OK]
 0.50 |   1.5000 |          1.5000          1.0000          1.0000  |    3.44e-05      3.95e-05  [OK]
 0.80 |   1.8000 |          1.8000          1.0000          1.0000  |    3.45e-05      3.96e-05  [OK]
```

### 9.3 Interprétation

✅ **Le facteur (1+z_l) est rigoureusement universel** : suivi exactement par la simulation à 4 chiffres pour toutes les z_l testées.

✅ **`κ_fixed = κ_sim/(1+z_l)` reproduit le κ analytique BS-2001** à mieux que 0.004% (limites de l'approximation Born + précision dopri5).

✅ **Le même fix marche pour γ** avec la même précision. Cohérent avec Fleury Table 5.1 : κ et γ sont tous deux des invariants conformes, donc subissent la même renormalisation parasite (1+z_l).

### 9.4 Le fix complet

**Module** `_audits/fixed_lensing_conventions.py` :

```python
def sigma_cr_conventions_fixed(d_l, d_s, d_ls, z_l):
    sigma_cr_comoving = (c**2 / (4*np.pi*G)) * d_s / (d_l * d_ls)
    sigma_cr_physical = sigma_cr_comoving * (1.0 + z_l)   # WAS / (1+z_l), now * (1+z_l)
    return sigma_cr_comoving, sigma_cr_physical

def physical_kappa_from_sim(kappa_sim, z_l):
    return kappa_sim / (1.0 + z_l)

def physical_gamma_from_sim(gamma_sim, z_l):
    return gamma_sim / (1.0 + z_l)
```

### 9.5 Verdict définitif

**Avec le fix `/(1+z_l)` en post-processing sur κ et γ, le pipeline excalibur produit des observables de lensing PHYSIQUEMENT CORRECTS au sens de Bartelmann-Schneider 2001, à <0.01% près.**

Le pipeline simule donc bien :
- Géodésiques nulles dans la cosmologie ΛCDM (via le tour de passe-passe conforme exact)
- Tenseur de marée optique perturbé (eq 5.43 de Fleury)
- Matrice de Jacobi (eq 5.48 de Fleury)
- Observables κ et γ — à un facteur global (1+z_l) près trivialement corrigible

**Recommandation pour la production** :
1. Corriger `excalibur/observables/lensing_conventions.py` : changer `/(1+z_l)` en `*(1+z_l)` dans `sigma_cr_conventions`.
2. Appliquer `κ_phys = κ_sim/(1+z_l)`, `γ_phys = γ_sim/(1+z_l)` en sortie de simulation, partout où les observables physiques sont attendues.
3. Mettre à jour les runners pour utiliser `Sigma_cr_physical` (corrigé) au lieu de `Sigma_cr_comoving` comme référence par défaut pour la comparaison NFW.

---

## 10. CAUSE PROFONDE & FIX BARDEEN (kernel-level)

### 10.1 Origine théorique du facteur (1+z_l)

Fleury **eq 4.69** (page 73) donne l'équation de Poisson pour le potentiel de Bardeen en coordonnées comoving :

$$\Delta_{\rm co} \Phi_{\rm Bardeen} = 4\pi G \bar\rho a^2 \delta$$

où Δ_co est le Laplacien comoving et ρ̄ la densité physique de fond.

Pour une lentille mince statique de densité physique Σ_phys :
- δρ_phys = Σ_phys × δ_Dirac_phys(z_phys) = Σ_phys × δ_Dirac_co(χ)/a_l (transformation comoving)
- ∫ ∇²_⊥_co Φ_Bardeen dχ_co = 4πG × **a_l** × Σ_phys

Or le code utilisait `Φ_NFW(x,y,z)` (Newton statique avec r_s **physique**) en passant des coords **comoving** (interprétées comme physiques). L'intégrale :

$$\int \nabla^2_\perp \Phi_{\rm NFW}(r_{\rm co}) \, d\chi_{\rm co} = 4\pi G \Sigma_{\rm phys}$$

manque exactement le facteur **a_l = 1/(1+z_l)** → biais constant (1+z_l) sur κ et γ.

### 10.2 Fix à la source

Le bon potentiel Bardeen pour des coords comoving est :
$$\Phi_{\rm Bardeen}(r_{\rm co}) = \Phi_{\rm NFW}(a_l \cdot r_{\rm co})$$

**Implémentation sans toucher au kernel JIT** : on rescale les paramètres NFW passés au kernel :
- `r_s_eff = r_s × (1+z_l)`
- `ρ_s_eff = ρ_s / (1+z_l)²`

(dérivation par identification : `Φ_kernel(r_co; r_s_eff, ρ_s_eff) = Φ_NFW(a_l × r_co; r_s, ρ_s)` algébriquement)

La masse `M_200_eff = M_200_phys / a_l = (1+z_l) × M_200` n'est pas conservée — c'est attendu, le halo effectif représente un POTENTIEL en coords comoving, pas un objet physique.

### 10.3 API

Activation via le paramètre `bardeen_a_lens` du `NumbaAMRBackend` :

```python
backend = NumbaAMRBackend(
    ..., analytical_source=halo, bypass_radius=...,
    bardeen_a_lens=1.0 / (1 + z_l),  # NEW: activates Bardeen rescaling
)
```

Si `bardeen_a_lens=None` (défaut) : comportement legacy (buggy) préservé pour backward compat.

### 10.4 Sémantique des positions

**Important** : avec le fix Bardeen, les positions de la simulation sont strictement **comoving**. Le photon à target `b_co` correspond physiquement à `b_phys = a_l × b_co` au plan de la lentille. Pour comparer à une prédiction analytique standard (qui utilise b_phys), évaluer `κ_BS` à `b_phys`.

Implémenté dans `run_lensing_equivalent_mass_profiles.py` :
- Le runner calcule `a_l` depuis la cosmologie + position du halo
- Active Bardeen automatiquement (`bardeen_a_lens=a_l`)
- La référence analytique passe automatiquement à `κ_BS(b_phys = a_l × b_co, Σ_cr_BS)`

### 10.5 Validation

Test `_audits/test_bardeen_kernel_fix.py` (z_l ∈ {0.1, 0.3, 0.5, 0.8}, b/r_s ∈ {2, 4, 8}) :

```
  z_l |    a_l |  max |k_fix/k_BS(a_l*b)-1|  max |g_fix/g_BS(a_l*b)-1|
 0.10 | 0.9091 |                   3.20e-05                   4.16e-05  [OK]
 0.30 | 0.7692 |                   3.34e-05                   3.96e-05  [OK]
 0.50 | 0.6667 |                   3.31e-05                   3.99e-05  [OK]
 0.80 | 0.5556 |                   3.12e-05                   3.97e-05  [OK]
```

**κ_sim_fix matche κ_BS(b_phys) à 3×10⁻⁵** à toutes les redshifts. **Le bug est corrigé à la source.**

### 10.6 Nettoyage : retrait du post-process empirique

Le post-process `÷(1+z_l)` (anciennement `physical_kappa_from_sim`/`physical_gamma_from_sim` dans `lensing_conventions.py`) est **retiré** : il devient inutile et incorrect avec le fix Bardeen.

Les helpers `Sigma_cr_comoving` / `Sigma_cr_physical` restent disponibles dans `sigma_cr_conventions()` pour compatibilité.

### 10.7 Non-régression

- 65/65 tests existants passent (`test_lensing_pipeline`, `test_nfw_*`, `test_numba_nfw_screen_convention`)
- Smoke-test du runner principal avec Bardeen active : OK (output Σ_cr correct = physique)

---

## Annexes

### Tests créés pour cet audit

- `_tests/test_screen_convention_equivalence.py` — comparaison 3 branches (metric python, conformal python, conformal numba)
- `_audits/test_sigma_cr_convention.py` — audit dimensionnel/numérique de Σ_cr vs Bartelmann-Schneider
- `_audits/test_born_kappa_vs_analytic.py` — test Born direct : κ_sim vs κ_NFW_analytique en 3 conventions
- `_audits/test_bardeen_kernel_fix.py` — **validation du fix Bardeen kernel** : κ et γ à 4 z_l × 3 b (3-4e-5 précision)
- `_audits/diagnose_z_end.py` — diagnostic du z_end overshoot (branche B mixte python)

### Fichiers du code audités

- `excalibur/observables/riemann_perturbed_flrw.py`
- `excalibur/metrics/perturbed_flrw_metric_fast.py`
- `excalibur/integration/integrator_numba_lowalloc.py`
- `excalibur/integration/integrator_numba_specialized.py`
- `excalibur/observables/sachs_basis.py`
- `excalibur/observables/optical_tidal_matrix.py`
- `excalibur/observables/lensing_conventions.py`

# Szekeres dans EXCALIBUR — explication physique des résultats

> Document de synthèse physique : ce qui a été implémenté, **ce que chaque
> résultat signifie physiquement**, et pourquoi chaque validation prouve que la
> physique est correcte. Référence : M.-N. Célérier, *Precision cosmology with
> exact inhomogeneous solutions of GR: the Szekeres models*, PRD **110**, 123526
> (2024), arXiv:2407.04452. Conventions EXCALIBUR : signature `(−,+,+,+)`, unités
> **cosmo** (Msun/Gyr/Gpc) pour la courbure/distances. 26 tests, tous au vert.

---

## 0. De quoi parle-t-on

Le modèle de Szekeres est une **solution exacte des équations d'Einstein** pour
un fluide de poussière (+ Λ) **sans aucune symétrie**. C'est la généralisation
inhomogène la plus large de FLRW restant analytiquement maîtrisable : il contient
FLRW et LTB comme cas limites. Physiquement, c'est un empilement de **sphères
comobiles non concentriques en expansion**, chacune portant une **distribution de
masse en dipôle** superposée à un monopôle. L'intérêt cosmologique : tester si
des inhomogénéités exactes (pas perturbatives) peuvent imiter une partie de ce
qu'on attribue à l'énergie noire, et fournir un cadre exact pour le lentillage et
les distances.

Cas traité : **quasi-sphérique** (`ε=+1`, QSS).

```
ds² = −c² dt² + (Φ_,r − Φ E_,r/E)²/(ε−k) dr²  +  Φ²/E² (dp² + dq²)
```

`t` = temps cosmique, `(r,p,q)` comobiles. `Φ(t,r)` = **rayon aréolaire**
(physiquement : l'aire d'une 2-sphère `{t,r}=cst` est `4πΦ²/...`). `E(r,p,q)` =
fonction de dipôle. Six fonctions libres de `r` : `M, k, S, P, Q, t_B` (+ `ε, Λ`).

| Fonction | Sens physique |
|---|---|
| `M(r)` | masse gravitationnelle active dans la coquille `r` (monopôle) |
| `k(r)` | courbure spatiale locale (`<0` non-lié/ouvert, `>0` lié/recollapse) |
| `S,P,Q` | **dipôle** (force `S`, orientation `P,Q`) — l'unique source du caractère « Szekeres » via `E_,r/E` |
| `t_B(r)` | temps de big-bang local (non simultané si non constant) |

---

## 1. Dynamique du fond — `Φ(t,r)`

Les équations d'Einstein se réduisent à deux relations (éqs 3-4 du papier) :

```
(Φ_,t)² = 2GM/Φ − k c² + (Λc²/3) Φ²            (Friedmann–LTB, dynamique)
4πρ      = (M_,r − 3M E_,r/E) / [Φ²(Φ_,r − Φ E_,r/E)]   (densité)
```

**Physique** : la première est *exactement* l'équation de Friedmann, mais **par
coquille `r`** — chaque coquille évolue avec sa propre énergie (`M`), courbure
(`k`) et constante cosmologique. La seconde dit `dM = ρ dV` : la masse est
l'intégrale de la densité sur le volume — c'est pourquoi elle est **sans `G`**
(relation purement géométrique masse↔densité, vérifiée dimensionnellement).

**Résultats & validation** (unités cosmo) :
- Cas **Einstein–de Sitter** (poussière plate, dipôle éteint), forme close exacte
  `Φ = r·t^(2/3)` : retrouvée à **3.9e-5**. C'est `a(t)=t^(2/3)`, l'expansion
  décélérée d'un univers dominé par la matière.
- `Φ_,t` (vitesse d'expansion de la coquille) = Friedmann `W` : **2e-5**.
- Densité EdS `ρ = 1/(6πG t²)` : retrouvée à **4.2e-5** — c'est le résultat de
  manuel pour la densité critique d'un univers EdS (`ρ ∝ a⁻³ ∝ t⁻²`).
- **Homogénéité** : dipôle éteint ⟹ `Φ/r` indépendant de `r`, `Φ_,rr≈0` — la
  géométrie redevient FLRW. Confirmé.

> Pourquoi c'est important : ces tests prouvent que le solveur de fond reproduit
> *exactement* la cosmologie homogène quand on éteint les inhomogénéités. Tout
> écart ultérieur sera donc une vraie inhomogénéité, pas un artefact.

---

## 2. Géodésiques nulles — propagation des photons

État `[t,r,p,q, kᵗ,kʳ,kᵖ,kᵠ]`, `kᵘ=dxᵘ/ds`. La métrique étant **diagonale**, les
Christoffels sont assemblés directement de la métrique (les facteurs `c`
tombent automatiquement).

**Résultats & validation :**
- RHS géodésique reconstruit = **éqs 7-10 du papier à 2.0e-16** (précision
  machine). C'est la preuve la plus forte : la propagation des photons est
  *exactement* celle de Célérier 2024.
- Christoffels analytiques = différences finies de la métrique : **2e-5**.
- **Condition nulle** `g_μν kᵘkᵛ=0` préservée le long du rayon : **5.5e-13**
  (radial), **1.6e-7** (non-radial avec dipôle). Physiquement : le photon reste
  sur le cône de lumière — l'énergie est conservée correctement.

> ⚠️ **Piège physique central** (corrigé) : en Szekeres les géodésiques nulles
> **ne sont pas radiales** même si on « vise le centre » — le terme `E_,r/E`
> couple le mouvement radial aux directions `(p,q)`. Fixer `(p,q)` à la main
> (réflexe LTB) donne des résultats faux sans planter. La base de Sachs (§5) gère
> ça via les éqs 85-86.

---

## 3. Redshift

Observateur comobile `uᵘ=(1,0,0,0)`. L'énergie mesurée est `E=−u_α kᵃ = c² kᵗ`,
d'où un **redshift en ratio** (éq. 12) :

```
1 + z = kᵗ_émis / kᵗ_obs
```

**Physique** : à mesure que le photon remonte le temps vers la source, `kᵗ` croît
(le photon était plus « bleu » à l'émission relativement à l'expansion) → `z>0`.
En FLRW c'est exactement `1+z = a_obs/a_émis` (dilution de l'énergie par
l'expansion).

**Validation :**
- Limite EdS : `1+z = (t_obs/t_émis)^(2/3)` retrouvé jusqu'à **z=1.83** à **1.3e-5**.
- **Cross-check Bondi** (méthode indépendante des crêtes d'onde, éq. 20) : accord
  à **7e-4** avec la méthode `kᵗ`. Deux dérivations physiques distinctes du même
  redshift coïncident.

---

## 4. Courbure — focalisation de la lumière

La déformation des faisceaux est pilotée par le tenseur de Riemann. On le sépare
en **Ricci** (convergence, focalisation par la matière) et **Weyl** (cisaillement,
effet de marée).

**Focalisation de Ricci** — résultat physique clé. Pour un photon (`kᵘ` nul) dans
un univers de poussière+Λ :

```
R_αβ kᵃkᵝ = 8πG ρ (kᵗ)²
```

**Physique** : seul ce qui a une **densité de masse `ρ`** focalise un faisceau de
lumière (équation de Raychaudhuri/Sachs). Remarquable : **Λ disparaît** de la
focalisation directe — car `g_αβ kᵃkᵝ=0` pour la lumière. Λ n'agit
qu'indirectement, via la dynamique de `Φ` (l'expansion). Les facteurs `c`
s'annulent exactement entre `T_tt` et `(u_α kᵃ)²`.

**Validation :**
- Courbure **analytique** générée symboliquement (sympy, à partir de la métrique
  — reproduit l'Appendice A du papier). ⚠️ La transcription manuelle de `R_rr`
  était fausse (deux facteurs `H` perdus) ; le codegen l'a corrigée.
- Focalisation analytique vs `8πGρ(kᵗ)²` : **6e-6** (quasi-exacte).
- Tenseur tidal analytique vs Riemann numérique indépendant : **1e-3** (le
  numérique, par différences finies, est le moins précis — l'analytique est
  ~1000× meilleur).

---

## 5. Distances cosmologiques `D_A`, `D_L`

On propage la **carte de Jacobi** `D` (déviation géodésique, éq. 27) le long du
rayon, avec la base d'écran de Sachs `e₁,e₂` (transportée parallèlement) :

```
d²D/dλ² = −R·D ,   R_AB = R_{μανβ} kᵃkᵝ e_Aᵘ e_Bᵛ
D_A = c·kᵗ_o·√|det D|        (distance d'aire, indép. de la normalisation affine)
D_L = (1+z)² D_A             (réciprocité d'Etherington)
```

**Physique** : `D_A` relie une **taille physique** à un **angle observé**
(`δS = D_A² δΩ`). La carte de Jacobi mesure comment un faisceau infinitésimal
s'étale (Ricci) et se distord (Weyl) en se propageant. `D_L` est la même
information vue via le flux (chandelles standard / SNIa).

**Validation :**
- **Oracle FLRW universel** `D_A = a(t_émis)·|Δr|` retrouvé à **1.7e-5** (EdS).
- **Turnover de `D_A` à z=1.25** (EdS) : c'est le **résultat de manuel** — en EdS
  la distance d'aire culmine puis **décroît** à z>1.25 (les objets lointains
  paraissent *plus gros*). Le retrouver exactement valide la **normalisation
  absolue** de la distance.
- Limite **ΛCDM** (Λ≠0) : `D_A` vs oracle à **8.7e-5**, `a(13.8 Gyr)≈1.02`
  (normalisation correcte). Donc la limite FLRW marche **pour tout fond**, pas
  seulement EdS.
- Réciprocité `D_L=(1+z)²D_A` : **2e-16** (machine).

---

## 6. Lentillage — convergence `κ` et cisaillement `γ`

De la carte de Jacobi `D` on extrait (éqs 68-71, relativement au fond FLRW) :

```
κ  = 1 − (D₁₁+D₂₂)/(2 D_A^FLRW)        convergence (grossissement)
γ₁ = (D₂₂−D₁₁)/(2 D_A^FLRW), γ₂ = D₁₂/D_A^FLRW   cisaillement (distorsion)
```

Base d'écran de Sachs : **éqs 85-86 du papier**. ⚠️ J'ai trouvé une **coquille
dans le papier** : l'éq. 86 imprimée donne `E₂·k≠0` et `E₁·E₂≠0`, violant les
conditions d'orthogonalité (84). La forme correcte permute `p↔q`. Vérifié sur le
PDF et numériquement (orthonormalité à 1e-16).

**Validation — la plus subtile et la plus parlante :**
- Écran orthonormé et orthogonal à `k` : **1e-16**.
- **Rayon non-radial dans un modèle homogène (EdS) ⟹ `|γ|=2.6e-6 ≈ 0`** et
  `|ω|=2.6e-18`. *Physiquement crucial* : un univers homogène ne **distord pas**
  les faisceaux (pas de cisaillement, pas de rotation), même pour un rayon
  oblique. Si l'écran ou le tenseur tidal étaient faux, on verrait un cisaillement
  *fantôme*. Sa quasi-nullité prouve que toute la chaîne (écran + transport +
  marée de Weyl) est correcte.
- **Modèle dipôle ⟹ `|γ|=6.4e-4`** (~240× le plancher) : une vraie inhomogénéité
  cisaille la lumière.

**Signature Szekeres (ce qui le distingue de LTB/FLRW) :**
- `D_A(+kᵖ) ≠ D_A(−kᵖ)` pour le dipôle (**asymétrie 3.2e-4**), alors que FLRW est
  **exactement isotrope** (asymétrie 0). La carte de ciel
  (`run_szekeres_lensing_map.py`) montre une **anisotropie dipolaire de `D_A` de
  0.2% crête-à-crête** : la relation distance-redshift **dépend de la direction**.
  C'est l'empreinte observationnelle propre au dipôle de Szekeres.

---

## 7. Cosmographie — décélération vs accélération

Développement bas-`z` de la distance de luminosité :

```
D_L(z) = (c/H₀)[ z + ½(1−q₀) z² + … ]
```

`q₀` = **paramètre de décélération** (`q₀>0` : expansion qui ralentit ;
`q₀<0` : expansion **accélérée**). On l'extrait par un fit cubique de `D_L(z)`.

**Résultats :**
| Modèle | `H₀` (récup./attendu) | `q₀` (récup./attendu) | Interprétation |
|---|---|---|---|
| EdS | 0.0370 / 0.0370 | **+0.509 / +0.500** | univers **décélérant** (matière) |
| ΛCDM (Ωm=0.3, ΩΛ=0.7) | 0.0708 / 0.0715 | **−0.566 / −0.550** | univers **accéléré** (`q₀<0`) |

**Physique** : retrouver `q₀<0` pour ΛCDM signifie que le pipeline capture
correctement l'**accélération cosmique** (le résultat des SNIa, Nobel 2011),
purement à partir de la propagation exacte des photons + la dynamique du fond.
Pour EdS, `q₀=+½` est la décélération exacte d'un univers de poussière. C'est le
lien direct avec l'observable des chandelles standard.

---

## 8. Une subtilité numérique = une leçon physique : les unités

La courbure de Szekeres est **mal conditionnée en SI** : avec `c=3e8`, des
distances ~1e25 m et un facteur d'échelle non normalisé ~1e11, les composantes
`H~O(1)` et `F~1e25` diffèrent de ~25 ordres de grandeur, et `R_rr` (résidu
d'annulation de termes ~1e22) perd toute précision en float64. **Ce n'est pas un
bug** : c'est que le SI est inadapté à la cosmologie. Les unités **cosmo**
(Gpc/Gyr/Msun, `c≈0.31`, `a~1`) — équivalent des unités géométriques `G=c=1` du
papier — rendent toutes les quantités `O(1)` et bien conditionnées. C'est le
système naturel du problème. (`EXCALIBUR_UNITS=cosmo` ; défaut SI inchangé pour
le pipeline FLRW perturbé existant.)

---

## 9. Tableau récapitulatif des validations

| Domaine | Test physique | Précision | Ce que ça prouve |
|---|---|---|---|
| Fond | `Φ=r t^(2/3)`, `ρ=1/(6πGt²)` (EdS) | 4e-5 | Friedmann par coquille exact |
| Géodésiques | RHS = éqs 7-10 papier | **2e-16** | propagation photon = papier |
| Géodésiques | condition nulle préservée | 5e-13 | cône de lumière conservé |
| Redshift | `1+z=(t_o/t_e)^2/3` + Bondi | 1e-5 / 7e-4 | dilution par expansion |
| Courbure | `R_αβkᵃkᵝ=8πGρ(kᵗ)²` | **6e-6** | focalisation par la matière |
| Distance | turnover `D_A` à z=1.25 (EdS) | 2e-5 | normalisation absolue |
| Distance | limite ΛCDM (Λ≠0) | 9e-5 | limite FLRW générale |
| Distance | `D_L=(1+z)²D_A` | 2e-16 | réciprocité d'Etherington |
| Lentillage | homogène ⟹ `|γ|≈0` | 3e-6 | pas de distorsion fantôme |
| Lentillage | dipôle ⟹ anisotropie `D_A` | — | **signature Szekeres** |
| Cosmographie | `q₀`: +0.5 (EdS), −0.55 (ΛCDM) | 3% | décélération / **accélération** |

---

## 10. Périmètre et suites

**Fait & validé (v1 du brief)** : modèle + fond, métrique + géodésiques (=papier),
redshift (2 méthodes), courbure analytique (Ricci+Weyl, codegen), `D_A`/`D_L`
(Sachs/Jacobi), `κ,γ` + écran non-radial, limites EdS & ΛCDM, cartes de
lentillage directionnelles, cosmographie `q₀`.

**Hors périmètre / suites** : ajustement des 6 fonctions libres aux **vraies
données** (SNIa/BAO/CMB) — c'est l'objectif final du programme du papier ;
sous-cas axisymétrique + redshift drift ; portage **Numba** du hot-path
(performance, Phase 2) ; modèles Swiss-cheese multi-patchs.

> Les fonctions `M,k,S,P,Q,t_B` utilisées ici sont des **jouets lisses de
> validation**, pas un modèle contraint. Le code accepte n'importe quels callables
> (changement de modèle = une ligne) ; la physique ci-dessus est indépendante de
> ces choix particuliers.

# Time Delay Map Script

## Description

Ce script calcule une **carte 2D de délai temporel** pour les trajectoires de photons défléchis par un champ gravitationnel.

## Principe Physique

### Délai Temporel Gravitationnel

Lorsqu'un photon passe près d'une masse, deux effets contribuent au délai temporel :

1. **Délai géométrique** : Le chemin défléchi est plus long que la ligne droite
2. **Délai de Shapiro** : Le temps propre s'écoule plus lentement en champ gravitationnel fort

Le délai total est calculé comme :
```
Δt = t_défléchi - t_ligne_droite
```

### Calcul

Pour chaque photon :
- **Temps défléchi** : Intégration le long de la géodésique réelle `∫ dη`
- **Temps ligne droite** : Distance euclidienne divisée par c : `|r_final - r_initial| / c`

## Utilisation

### Syntaxe de base
```bash
python _postprocessing/compute_time_delay_map.py <fichier_trajectoires.h5>
```

### Exemple
```bash
python _postprocessing/compute_time_delay_map.py backward_raytracing_trajectories_OPTIMAL_mass_500_500_500_Mpc.h5
```

### Sortie

Le script produit :
1. **Statistiques** : Délais min/max/moyen, différences de chemin
2. **Carte 2D interpolée** : Visualisation continue du délai en fonction de la position dans le ciel
3. **Fichier PNG** : Sauvegardé automatiquement avec le même nom + `_time_delay_map.png`

## Interprétation des Résultats

### Carte de Délai Positif (Δt > 0)
- Le chemin défléchi est **plus long** temporellement
- Effet combiné : géométrie + Shapiro delay
- Typique pour photons passant **près** de la masse

### Carte de Délai Négatif (Δt < 0)
- Peut arriver en backward tracing si le temps conformal est calculé différemment
- Vérifier les signes de `dt` dans l'intégration

### Symétries
- La carte devrait montrer une **symétrie azimutale** autour de la direction vers la masse
- Maximum de délai : photons avec le plus petit paramètre d'impact
- Délai décroît avec l'angle par rapport à la ligne de visée

## Applications

### Lentilles Gravitationnelles
- Détection de matière noire via délais temporels dans les quasars multiples
- Mesure de H₀ avec la méthode des délais temporels

### Cosmologie
- Test de la Relativité Générale à grandes échelles
- Contraintes sur les modèles de matière noire

### Validation du Code
- Vérifier que les délais sont cohérents avec la théorie d'Einstein
- Comparer avec la formule approximative : `Δt ≈ (4GM/c³) ln(r_lens/r_source)`

## Structure du Code

### Fonctions Principales

1. **`load_trajectories(filename)`**
   - Charge les trajectoires depuis HDF5
   - Retourne liste de tableaux [η, x, y, z, u⁰, u¹, u², u³, ...]

2. **`compute_travel_time_deflected(trajectory)`**
   - Calcule Δη le long de la géodésique
   - Distance parcourue

3. **`compute_travel_time_straight(pos_i, pos_f)`**
   - Calcule le temps pour un photon non-défléchi
   - Distance euclidienne / c

4. **`compute_time_delays(trajectories)`**
   - Calcule Δt pour tous les photons
   - Extrait positions dans le ciel (θ, φ)

5. **`create_2d_map(x, y, delays)`**
   - Interpolation cubique sur grille régulière
   - Retourne X_grid, Y_grid, delay_grid

6. **`plot_time_delay_map(...)`**
   - Visualisation avec matplotlib
   - Colormap divergente centrée sur zéro
   - Contours + points de données

## Améliorations Possibles

### Court Terme
- [ ] Ajouter calcul du délai de Shapiro séparément
- [ ] Supporter des formats de sortie additionnels (FITS, CSV)
- [ ] Permettre différentes méthodes d'interpolation

### Long Terme
- [ ] Calculer les images multiples (strong lensing)
- [ ] Intégrer avec des catalogues de masses réalistes
- [ ] Animation temporelle de l'évolution du délai

## Exemples de Résultats

### Cas Typique (Amas de Galaxies, M = 10¹⁶ M☉)
```
Mean time delay: ~10⁻⁶ s (microseconds)
Path difference: ~10⁻³ Mpc
Fractional difference: ~10⁻⁶ %
```

### Strong Lensing (Masse compacte, petit impact parameter)
```
Mean time delay: ~jours à mois
Path difference: significative
Multiples images possibles
```

## Dépendances

- `numpy` : Calculs numériques
- `h5py` : Lecture des trajectoires
- `matplotlib` : Visualisation
- `scipy` : Interpolation (griddata)

## Auteur et Contact

Partie du projet **Excalibur** - Simulation de ray tracing en Relativité Générale.

Pour bugs ou suggestions : ouvrir une issue sur le repo Git.

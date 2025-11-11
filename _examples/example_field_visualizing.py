#!/usr/bin/env python3
"""
Exemple de visualisation de champs utilisant le module excalibur.

Ce script démontre comment:
- Créer et configurer une grille 3D
- Générer différents types de champs physiques
- Utiliser les objets masses sphériques
- Visualiser les champs avec matplotlib
- Interpoler et analyser les données de grille
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Ajouter le projet à la path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.core.constants import *

def create_test_fields():
    """Crée des champs de test pour la visualisation."""
    
    print("=== Création des champs de test ===\n")
    
    # 1. Configuration de la grille
    print("1. Configuration de la grille...")
    N = 256  # Résolution de grille
    box_size = 100 * one_Mpc  # Taille de la boîte en mètres
    dx = dy = dz = box_size / N
    
    shape = (N, N, N)
    spacing = (dx, dy, dz)
    origin = (0, 0, 0)
    grid = Grid(shape, spacing, origin)
    
    print(f"   Grille: {N}³ cellules")
    print(f"   Taille: {box_size/one_Mpc:.1f} Mpc")
    print(f"   Espacement: {dx/one_Mpc:.2f} Mpc/cellule")
    
    # 2. Création des coordonnées
    print("\n2. Génération des coordonnées...")
    x = np.linspace(0, box_size, N)
    y = np.linspace(0, box_size, N)
    z = np.linspace(0, box_size, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 3. Création de masses sphériques
    print("\n3. Création de masses sphériques...")
    
    # Masse centrale
    M1 = 10**15 * one_Msun
    R1 = 10 * one_Mpc
    center1 = np.array([0.3, 0.5, 0.5]) * box_size
    mass1 = spherical_mass(M1, R1, center1)
    
    # Masse secondaire
    M2 = 10**15 * one_Msun
    R2 = 2 * one_Mpc
    center2 = np.array([0.7, 0.3, 0.5]) * box_size
    mass2 = spherical_mass(M2, R2, center2)
    
    print(f"   Masse 1: {M1/one_Msun:.1e} M☉ à {center1/one_Mpc}")
    print(f"   Masse 2: {M2/one_Msun:.1e} M☉ à {center2/one_Mpc}")
    
    # 4. Calcul des champs
    print("\n4. Calcul des champs physiques...")
    
    # Potentiel gravitationnel
    phi1 = mass1.potential(X, Y, Z)
    phi2 = mass2.potential(X, Y, Z)
    phi_total = phi1 + phi2
    
    # Densité de masse
    rho1 = mass1.mass_enclosed(X, Y, Z)
    rho2 = mass2.mass_enclosed(X, Y, Z)
    rho_total = rho1 + rho2
    
    # Champ de température artificiel (pour la démonstration)
    kx, ky, kz = 2*np.pi/box_size, 2*np.pi/box_size, 2*np.pi/box_size
    temperature = 2.7 + 0.5 * (np.sin(2*kx*X) * np.cos(3*ky*Y) * np.sin(kz*Z))
    
    # Ajout des champs à la grille
    grid.add_field("Phi", phi_total)
    grid.add_field("Rho", rho_total)
    grid.add_field("Temperature", temperature)
    
    print(f"   Potentiel: [{phi_total.min():.2e}, {phi_total.max():.2e}] m²/s²")
    print(f"   Densité: [{rho_total.min():.2e}, {rho_total.max():.2e}] kg/m³")
    print(f"   Température: [{temperature.min():.2f}, {temperature.max():.2f}] K")
    
    return grid, (X, Y, Z), (mass1, mass2)

def plot_2d_slices(grid, coordinates, masses):
    """Crée des visualisations 2D en tranches."""
    
    print("\n=== Création des visualisations 2D ===")
    
    X, Y, Z = coordinates
    N = X.shape[0]
    
    # Tranche au milieu en Z
    z_slice = N // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Potentiel gravitationnel
    ax = axes[0, 0]
    phi_slice = grid.fields["Phi"][:, :, z_slice]
    im1 = ax.imshow(phi_slice.T, origin='lower', extent=[0, X.max()/one_Mpc, 0, Y.max()/one_Mpc],
                    cmap='viridis', aspect='equal')
    ax.set_title('Potentiel gravitationnel Φ')
    ax.set_xlabel('X [Mpc]')
    ax.set_ylabel('Y [Mpc]')
    plt.colorbar(im1, ax=ax, label='Φ [m²/s²]')
    
    # Marquer les positions des masses
    for i, mass in enumerate(masses):
        center_mpc = mass.center / one_Mpc
        ax.plot(center_mpc[0], center_mpc[1], 'r*', markersize=15, 
                label=f'Masse {i+1}')
    ax.legend()
    
    # 2. Densité de masse
    ax = axes[0, 1]
    rho_slice = grid.fields["Rho"][:, :, z_slice]
    im2 = ax.imshow(rho_slice.T, origin='lower', extent=[0, X.max()/one_Mpc, 0, Y.max()/one_Mpc],
                    cmap='plasma', aspect='equal')
    ax.set_title('Densité de masse ρ')
    ax.set_xlabel('X [Mpc]')
    ax.set_ylabel('Y [Mpc]')
    plt.colorbar(im2, ax=ax, label='ρ [kg/m³]')
    
    # 3. Température
    ax = axes[1, 0]
    temp_slice = grid.fields["Temperature"][:, :, z_slice]
    im3 = ax.imshow(temp_slice.T, origin='lower', extent=[0, X.max()/one_Mpc, 0, Y.max()/one_Mpc],
                    cmap='coolwarm', aspect='equal')
    ax.set_title('Champ de température T')
    ax.set_xlabel('X [Mpc]')
    ax.set_ylabel('Y [Mpc]')
    plt.colorbar(im3, ax=ax, label='T [K]')
    
    # 4. Contours du potentiel
    ax = axes[1, 1]
    contours = ax.contour(X[:, :, z_slice]/one_Mpc, Y[:, :, z_slice]/one_Mpc, 
                         phi_slice, levels=20, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_title('Contours du potentiel')
    ax.set_xlabel('X [Mpc]')
    ax.set_ylabel('Y [Mpc]')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('field_visualization_2d.png', dpi=300, bbox_inches='tight')
    print("   Sauvegardé: field_visualization_2d.png")
    plt.show()

def plot_3d_isosurfaces(grid, coordinates):
    """Crée des visualisations 3D avec isosurfaces."""
    
    print("\n=== Création des visualisations 3D ===")
    
    X, Y, Z = coordinates
    
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Isosurface du potentiel
    ax1 = fig.add_subplot(131, projection='3d')
    phi = grid.fields["Phi"]
    
    # Échantillonnage pour la visualisation 3D (réduction de résolution)
    step = 4
    X_sub = X[::step, ::step, ::step]
    Y_sub = Y[::step, ::step, ::step]
    Z_sub = Z[::step, ::step, ::step]
    phi_sub = phi[::step, ::step, ::step]
    
    # Scatter plot avec couleurs basées sur le potentiel
    scatter = ax1.scatter(X_sub.flatten()/one_Mpc, 
                         Y_sub.flatten()/one_Mpc, 
                         Z_sub.flatten()/one_Mpc,
                         c=phi_sub.flatten(), 
                         cmap='viridis', 
                         alpha=0.6, s=1)
    
    ax1.set_xlabel('X [Mpc]')
    ax1.set_ylabel('Y [Mpc]')
    ax1.set_zlabel('Z [Mpc]')
    ax1.set_title('Potentiel 3D')
    
    # 2. Densité 3D
    ax2 = fig.add_subplot(132, projection='3d')
    rho = grid.fields["Rho"]
    rho_sub = rho[::step, ::step, ::step]
    
    # Seuiller pour ne montrer que les hautes densités
    threshold = np.percentile(rho_sub, 95)
    mask = rho_sub > threshold
    
    ax2.scatter(X_sub[mask]/one_Mpc, 
               Y_sub[mask]/one_Mpc, 
               Z_sub[mask]/one_Mpc,
               c=rho_sub[mask], 
               cmap='plasma', 
               alpha=0.8, s=5)
    
    ax2.set_xlabel('X [Mpc]')
    ax2.set_ylabel('Y [Mpc]')
    ax2.set_zlabel('Z [Mpc]')
    ax2.set_title('Densité 3D (seuillée)')
    
    # 3. Température 3D
    ax3 = fig.add_subplot(133, projection='3d')
    temp = grid.fields["Temperature"]
    temp_sub = temp[::step, ::step, ::step]
    
    ax3.scatter(X_sub.flatten()/one_Mpc, 
               Y_sub.flatten()/one_Mpc, 
               Z_sub.flatten()/one_Mpc,
               c=temp_sub.flatten(), 
               cmap='coolwarm', 
               alpha=0.6, s=1)
    
    ax3.set_xlabel('X [Mpc]')
    ax3.set_ylabel('Y [Mpc]')
    ax3.set_zlabel('Z [Mpc]')
    ax3.set_title('Température 3D')
    
    plt.tight_layout()
    plt.savefig('field_visualization_3d.png', dpi=300, bbox_inches='tight')
    print("   Sauvegardé: field_visualization_3d.png")
    plt.show()

def test_interpolation(grid):
    """Teste l'interpolation sur la grille."""
    
    print("\n=== Test de l'interpolation ===")
    
    interpolator = Interpolator(grid)
    
    # Points de test pour l'interpolation
    test_points = [
        np.array([30*one_Mpc, 50*one_Mpc, 50*one_Mpc]),  # 30, 50, 50 Mpc
        np.array([70*one_Mpc, 30*one_Mpc, 70*one_Mpc]),  # 70, 30, 70 Mpc
        np.array([50*one_Mpc, 50*one_Mpc, 20*one_Mpc])   # 50, 50, 20 Mpc
    ]
    
    print("   Tests d'interpolation aux positions:")
    for i, pos in enumerate(test_points):
        print(f"\n   Point {i+1}: [{pos[0]/one_Mpc:.1f}, {pos[1]/one_Mpc:.1f}, {pos[2]/one_Mpc:.1f}] Mpc")
        
        # Interpolation des champs
        phi_val = interpolator.interpolate(pos, "Phi")
        rho_val = interpolator.interpolate(pos, "Rho")
        temp_val = interpolator.interpolate(pos, "Temperature")
        
        print(f"     Φ = {phi_val:.2e} m²/s²")
        print(f"     ρ = {rho_val:.2e} kg/m³")
        print(f"     T = {temp_val:.2f} K")
        
        # Gradients
        phi_grad = interpolator.gradient(pos, "Phi")
        print(f"     ∇Φ = [{phi_grad[0]:.2e}, {phi_grad[1]:.2e}, {phi_grad[2]:.2e}] m/s²")

def plot_field_profiles(grid, coordinates, masses):
    """Trace des profils 1D des champs."""
    
    print("\n=== Création des profils 1D ===")
    
    X, Y, Z = coordinates
    N = X.shape[0]
    
    # Profil le long de l'axe X (y=N/2, z=N/2)
    y_mid = N // 2
    z_mid = N // 2
    
    x_profile = X[:, y_mid, z_mid]
    phi_profile = grid.fields["Phi"][:, y_mid, z_mid]
    rho_profile = grid.fields["Rho"][:, y_mid, z_mid]
    temp_profile = grid.fields["Temperature"][:, y_mid, z_mid]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Potentiel
    ax = axes[0]
    ax.plot(x_profile/one_Mpc, phi_profile, 'b-', linewidth=2)
    ax.set_ylabel('Φ [m²/s²]')
    ax.set_title('Profils des champs le long de l\'axe X')
    ax.grid(True, alpha=0.3)
    
    # Marquer les positions des masses
    for i, mass in enumerate(masses):
        if abs(mass.center[1] - Y[0, y_mid, z_mid]) < 5*one_Mpc and \
           abs(mass.center[2] - Z[0, y_mid, z_mid]) < 5*one_Mpc:
            ax.axvline(mass.center[0]/one_Mpc, color='red', linestyle='--', 
                      label=f'Masse {i+1}')
    ax.legend()
    
    # Densité
    ax = axes[1]
    ax.plot(x_profile/one_Mpc, rho_profile, 'r-', linewidth=2)
    ax.set_ylabel('ρ [kg/m³]')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Température
    ax = axes[2]
    ax.plot(x_profile/one_Mpc, temp_profile, 'g-', linewidth=2)
    ax.set_xlabel('X [Mpc]')
    ax.set_ylabel('T [K]')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('field_profiles_1d.png', dpi=300, bbox_inches='tight')
    print("   Sauvegardé: field_profiles_1d.png")
    plt.show()

def main():
    """Fonction principale de démonstration."""
    
    print("=== Exemple de visualisation de champs avec excalibur ===\n")
    
    # 1. Créer les champs de test
    grid, coordinates, masses = create_test_fields()
    
    # 2. Visualisations 2D
    plot_2d_slices(grid, coordinates, masses)
    
    # 3. Visualisations 3D
    plot_3d_isosurfaces(grid, coordinates)
    
    # 4. Test d'interpolation
    test_interpolation(grid)
    
    # 5. Profils 1D
    plot_field_profiles(grid, coordinates, masses)
    
    print("\n" + "="*60)
    print("VISUALISATION TERMINÉE")
    print("="*60)
    print("Fichiers générés:")
    print("  - field_visualization_2d.png : Tranches 2D des champs")
    print("  - field_visualization_3d.png : Visualisation 3D")
    print("  - field_profiles_1d.png     : Profils 1D des champs")
    print("\nToutes les visualisations ont été créées avec succès!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test de diagnostic pour les unités dans la simulation Schwarzschild.
Vérifie l'initialisation des photons et les premières étapes d'intégration.
"""

import sys
import os
sys.path.insert(0, '/home/magri/excalibur_project')

import numpy as np
from excalibur.metrics.schwarzschild_metric_cartesian import SchwarzschildMetricCartesian
from excalibur.photon.photons import Photons
from excalibur.integration.integrator_old import Integrator
from excalibur.core.constants import *

def test_schwarzschild_units():
    print("=== TEST DIAGNOSTIC SCHWARZSCHILD ===\n")
    
    # Paramètres physiques en unités cohérentes
    mass_msun = 1e15
    mass_kg = mass_msun * one_Msun
    
    # Positions en Mpc, converties en mètres pour la métrique
    mass_position_mpc = np.array([500, 500, 500])
    observer_position_mpc = np.array([1, 1, 1])
    
    mass_position_m = mass_position_mpc * one_Mpc
    observer_position_m = observer_position_mpc * one_Mpc
    
    print(f"1. Paramètres physiques:")
    print(f"   Masse: {mass_msun:.1e} M☉ = {mass_kg:.2e} kg")
    print(f"   Position masse: {mass_position_mpc} Mpc = [{mass_position_m[0]/1e24:.2f}, {mass_position_m[1]/1e24:.2f}, {mass_position_m[2]/1e24:.2f}] x10^24 m")
    print(f"   Position observateur: {observer_position_mpc} Mpc = [{observer_position_m[0]/1e22:.2f}, {observer_position_m[1]/1e22:.2f}, {observer_position_m[2]/1e22:.2f}] x10^22 m")
    
    # Création de la métrique
    metric = SchwarzschildMetricCartesian(
        mass=mass_kg,
        radius=5*one_Mpc,
        center=mass_position_m
    )
    print(f"\n2. Métrique Schwarzschild créée")
    print(f"   Centre: {metric.center/one_Mpc} Mpc")
    print(f"   Rayon Schwarzschild: {metric.r_s/one_Mpc:.2e} Mpc")
    
    # Position de l'observateur en 4D
    observer_pos_4d = np.array([0.0, *observer_position_m])
    print(f"\n3. Position observateur 4D:")
    print(f"   [t, x, y, z] = {observer_pos_4d}")
    print(f"   En Mpc: [0, {observer_pos_4d[1]/one_Mpc:.3f}, {observer_pos_4d[2]/one_Mpc:.3f}, {observer_pos_4d[3]/one_Mpc:.3f}]")
    
    # Test de la métrique à la position de l'observateur
    g_obs = metric.metric_tensor(observer_pos_4d)
    print(f"\n4. Métrique à la position de l'observateur:")
    print(f"   g00 = {g_obs[0,0]:.3e}")
    print(f"   g11 = {g_obs[1,1]:.3e}")
    print(f"   g22 = {g_obs[2,2]:.3e}")
    print(f"   g33 = {g_obs[3,3]:.3e}")
    
    # Génération d'un photon de test
    photons_manager = Photons(metric)
    
    direction_to_mass = mass_position_m - observer_position_m
    direction_to_mass = direction_to_mass / np.linalg.norm(direction_to_mass)
    
    print(f"\n5. Génération d'un photon de test:")
    print(f"   Direction vers la masse: {direction_to_mass}")
    
    # Génération manuelle pour debug
    photons_manager.generate_cone_random(
        n_photons=1,
        origin=observer_pos_4d,
        central_direction=direction_to_mass,
        cone_angle=0.01  # Petit cône pour test
    )
    
    if len(photons_manager.photons) > 0:
        photon = photons_manager.photons[0]
        print(f"\n6. Photon généré:")
        print(f"   Position x = {photon.x}")
        print(f"   Position (en Mpc) = {photon.x[1:]/one_Mpc}")
        print(f"   Vitesse u = {photon.u}")
        print(f"   Vitesse spatiale (en c) = {photon.u[1:]/c}")
        
        # Test condition nulle
        null_cond = photon.null_condition(metric)
        print(f"   Condition nulle: u.u = {null_cond:.3e}")
        
        # Test quelques étapes d'intégration
        integrator = Integrator(metric, dt=-1e13)  # Pas de temps négatif pour remonter le temps
        
        print(f"\n7. Test intégration (10 étapes):")
        print(f"   Position initiale: {photon.x[1:]/one_Mpc}")
        
        # Sauvegarde état initial
        initial_pos = photon.x.copy()
        
        # Intégration de quelques étapes
        try:
            integrator.integrate(photon, 10000)
            
            print(f"   Position finale: {photon.x[1:]/one_Mpc}")
            print(f"   Nombre d'états enregistrés: {len(photon.history.states)}")
            
            # Distance parcourue
            if len(photon.history.states) > 1:
                final_pos = photon.x[1:]
                distance_total = np.linalg.norm(final_pos - initial_pos[1:])
                print(f"   Distance totale parcourue: {distance_total/one_Mpc:.6f} Mpc")
                print(f"   Distance totale parcourue: {distance_total:.2e} m")
                
                # Vérifier si la distance est raisonnable
                if distance_total/one_Mpc > 1000:  # Plus de 1000 Mpc
                    print(f"   ⚠️  Distance ABERRANTE détectée!")
                    return False
                else:
                    print(f"   ✅ Distance raisonnable")
                    return True
            else:
                print(f"   ❌ Pas d'historique enregistré")
                return False
                
        except Exception as e:
            print(f"   ❌ Erreur d'intégration: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("❌ Aucun photon généré!")
        return False
    
    return True

if __name__ == "__main__":
    success = test_schwarzschild_units()
    if not success:
        print("\n❌ TEST ÉCHOUÉ - Problème détecté dans l'intégration Schwarzschild")
        sys.exit(1)
    else:
        print("\n✅ TEST RÉUSSI - Intégration Schwarzschild normale")
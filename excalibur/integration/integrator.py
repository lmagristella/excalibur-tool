# integration/integrator.py
import numpy as np

class Integrator:
    """Intègre la géodésique d’un photon dans une métrique donnée."""
    def __init__(self, metric, dt=1e-3):
        self.metric = metric
        self.dt = dt

    def integrate(self, photon, steps):
        # Use only position and velocity (first 8 components)
        state = np.concatenate([photon.x, photon.u])
        i = 0
        while i < steps:
            try:
                k1 = self.metric.geodesic_equations(state)
                k2 = self.metric.geodesic_equations(state + 0.5*self.dt*k1)
                k3 = self.metric.geodesic_equations(state + 0.5*self.dt*k2)
                k4 = self.metric.geodesic_equations(state + self.dt*k3)
                state += (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
                photon.x = state[:4]
                photon.u = state[4:]
                photon.state_quantities(self.metric.metric_physical_quantities)
                photon.record()
            except (ValueError, IndexError, RuntimeError) as e:
                # Photon left the grid or encountered an error, stop integration
                if i == 0:
                    print(f"WARNING: Photon stopped at step 0! Error: {e}")
                    print(f"  Initial position: {state[:4]}")
                    print(f"  Initial velocity: {state[4:]}")
                    import traceback
                    traceback.print_exc()
                break
            i += 1
# grid/interpolator.py
import numpy as np

class Interpolator:
    """
    Interpolation trilineaire + dérivées spatiales par différences finies.
    Support pour grilles 4D avec interpolation temporelle.
    """
    def __init__(self, grid):
        self.grid = grid
        self.is_4d = len(grid.shape) == 4

    def _index_and_weights(self, x, t=None):
        if self.is_4d and t is not None:
            # Position spatiale
            rel_spatial = (x - self.grid.origin[:3]) / self.grid.spacing[:3]
            
            # Check bounds
            if np.any(rel_spatial < 0) or np.any(rel_spatial >= self.grid.shape[:3] - 1):
                raise ValueError(f"Position {x} outside grid bounds")
            
            i0_spatial = np.floor(rel_spatial).astype(int)
            di_spatial = rel_spatial - i0_spatial
            i0_spatial = np.clip(i0_spatial, 0, self.grid.shape[:3] - 2)
            
            # Position temporelle
            rel_temporal = (t - self.grid.origin[3]) / self.grid.spacing[3]
            i0_temporal = np.floor(rel_temporal).astype(int)
            di_temporal = rel_temporal - i0_temporal
            i0_temporal = np.clip(i0_temporal, 0, self.grid.shape[3] - 2)
            
            i0 = np.concatenate([i0_spatial, [i0_temporal]])
            di = np.concatenate([di_spatial, [di_temporal]])
        else:
            rel = (x - self.grid.origin[:3]) / self.grid.spacing[:3]
            
            # Check bounds
            if np.any(rel < 0) or np.any(rel >= self.grid.shape[:3] - 1):
                raise ValueError(f"Position {x} outside grid bounds")
            
            i0 = np.floor(rel).astype(int)
            di = rel - i0
            i0 = np.clip(i0, 0, self.grid.shape[:3] - 2)
        return i0, di

    def interpolate(self, x, field, t=None):
        if self.is_4d and t is not None:
            return self._interpolate_4d(x, field, t)
        else:
            return self._interpolate_3d(x, field)

    def _interpolate_3d(self, x, field):
        i, di = self._index_and_weights(np.array(x))
        f = self.grid.fields[field]
        # 8 points du cube
        vals = f[i[0]:i[0]+2, i[1]:i[1]+2, i[2]:i[2]+2]
        wx, wy, wz = di
        # interpolation trilineaire
        interp = (
            vals[0,0,0]*(1-wx)*(1-wy)*(1-wz)
            + vals[1,0,0]*wx*(1-wy)*(1-wz)
            + vals[0,1,0]*(1-wx)*wy*(1-wz)
            + vals[0,0,1]*(1-wx)*(1-wy)*wz
            + vals[1,1,0]*wx*wy*(1-wz)
            + vals[1,0,1]*wx*(1-wy)*wz
            + vals[0,1,1]*(1-wx)*wy*wz
            + vals[1,1,1]*wx*wy*wz
        )
        return interp

    def _interpolate_4d(self, x, field, t):
        i, di = self._index_and_weights(np.array(x), t)
        f = self.grid.fields[field]
        # 16 points de l'hypercube 4D
        vals = f[i[0]:i[0]+2, i[1]:i[1]+2, i[2]:i[2]+2, i[3]:i[3]+2]
        wx, wy, wz, wt = di
        
        # Interpolation quadrilineaire
        interp = 0
        # Vectorized 4D interpolation
        w = np.array([1-wx, wx])[:, None, None, None] * \
            np.array([1-wy, wy])[None, :, None, None] * \
            np.array([1-wz, wz])[None, None, :, None] * \
            np.array([1-wt, wt])[None, None, None, :]
        interp = np.sum(vals * w)
        return interp

    def gradient(self, x, field, t=None):
        """Gradient spatial par interpolation trilinéaire CIC."""
        f = self.grid.fields[field]
        
        if self.is_4d and t is not None:
            # Position temporelle pour l'indexation
            t_pos = (t - self.grid.origin[3]) / self.grid.spacing[3]
            it = np.clip(int(np.round(t_pos)), 0, self.grid.shape[3] - 1)
            f_slice = f[:, :, :, it]
        else:
            f_slice = f
        
        # Position relative dans la grille
        pos = (np.array(x) - self.grid.origin[:3]) / self.grid.spacing[:3]
        i0 = np.floor(pos).astype(int)
        di = pos - i0
        
        # Assurer qu'on reste dans les limites pour le gradient (besoin de ±1)
        i0 = np.clip(i0, 1, np.array(self.grid.shape[:3]) - 2)
        
        # Poids d'interpolation trilinéaire
        wx, wy, wz = di
        
        # Les 8 coins du cube centré sur la position d'interpolation
        corners = [
            (i0[0], i0[1], i0[2]),        # (0,0,0)
            (i0[0]+1, i0[1], i0[2]),      # (1,0,0)  
            (i0[0], i0[1]+1, i0[2]),      # (0,1,0)
            (i0[0], i0[1], i0[2]+1),      # (0,0,1)
            (i0[0]+1, i0[1]+1, i0[2]),    # (1,1,0)
            (i0[0]+1, i0[1], i0[2]+1),    # (1,0,1)
            (i0[0], i0[1]+1, i0[2]+1),    # (0,1,1)
            (i0[0]+1, i0[1]+1, i0[2]+1)   # (1,1,1)
        ]
        
        # Poids d'interpolation correspondants
        weights = [
            (1-wx) * (1-wy) * (1-wz),  # w000
            wx * (1-wy) * (1-wz),      # w100
            (1-wx) * wy * (1-wz),      # w010
            (1-wx) * (1-wy) * wz,      # w001
            wx * wy * (1-wz),          # w110
            wx * (1-wy) * wz,          # w101
            (1-wx) * wy * wz,          # w011
            wx * wy * wz               # w111
        ]
        
        grad = np.zeros(3)
        
        # Calculer le gradient pour chaque composante
        for d in range(3):  # d = 0,1,2 pour x,y,z
            grad_at_corners = np.zeros(8)
            
            # Gradient par différences finies centrées à chaque coin
            for i, (ix, iy, iz) in enumerate(corners):
                # Ensure indices are within bounds for finite differences
                ix_safe = np.clip(ix, 1, self.grid.shape[0] - 2)
                iy_safe = np.clip(iy, 1, self.grid.shape[1] - 2)
                iz_safe = np.clip(iz, 1, self.grid.shape[2] - 2)
                
                if d == 0:  # Gradient en X
                    grad_at_corners[i] = (f_slice[ix_safe+1, iy_safe, iz_safe] - f_slice[ix_safe-1, iy_safe, iz_safe]) / (2 * self.grid.spacing[0])
                elif d == 1:  # Gradient en Y
                    grad_at_corners[i] = (f_slice[ix_safe, iy_safe+1, iz_safe] - f_slice[ix_safe, iy_safe-1, iz_safe]) / (2 * self.grid.spacing[1])
                else:  # Gradient en Z (d == 2)
                    grad_at_corners[i] = (f_slice[ix_safe, iy_safe, iz_safe+1] - f_slice[ix_safe, iy_safe, iz_safe-1]) / (2 * self.grid.spacing[2])
            
            # Interpolation trilinéaire du gradient à la position exacte
            grad[d] = np.sum(grad_at_corners * weights)
                   
        return grad

    def time_derivative(self, x, field, t):
        """Approximation de la dérivée temporelle via différences finies."""
        if not self.is_4d:
            raise ValueError("Time derivative only available for 4D grids")
            
        f = self.grid.fields[field]
        t_pos = (t - self.grid.origin[3]) / self.grid.spacing[3]
        it = np.clip(np.floor(t_pos).astype(int), 1, self.grid.shape[3] - 2)
        
        # Interpolation spatiale aux temps t-dt et t+dt
        val_prev = self._interpolate_3d_at_time(x, field, it - 1)
        val_next = self._interpolate_3d_at_time(x, field, it + 1)
        
        dt_deriv = (val_next - val_prev) / (2 * self.grid.spacing[3])
        return dt_deriv

    def _interpolate_3d_at_time(self, x, field, it):
        """Interpolation spatiale 3D à un temps fixé."""
        i, di = self._index_and_weights(np.array(x))
        f = self.grid.fields[field]
        vals = f[i[0]:i[0]+2, i[1]:i[1]+2, i[2]:i[2]+2, it]
        wx, wy, wz = di
        
        interp = (
            vals[0,0,0]*(1-wx)*(1-wy)*(1-wz)
            + vals[1,0,0]*wx*(1-wy)*(1-wz)
            + vals[0,1,0]*(1-wx)*wy*(1-wz)
            + vals[0,0,1]*(1-wx)*(1-wy)*wz
            + vals[1,1,0]*wx*wy*(1-wz)
            + vals[1,0,1]*wx*(1-wy)*wz
            + vals[0,1,1]*(1-wx)*wy*wz
            + vals[1,1,1]*wx*wy*wz
        )
        return interp

    def value_and_gradient(self, x, field, t=None):
        return self.interpolate(x, field, t), self.gradient(x, field, t)

    def value_gradient_and_time_derivative(self, x, field, t):
        """Retourne valeur, gradient spatial et dérivée temporelle."""
        if not self.is_4d:
            return (self.interpolate(x, field), 
                    self.gradient(x, field), 
                    0.0)
        return (self.interpolate(x, field, t), 
                self.gradient(x, field, t), 
                self.time_derivative(x, field, t))

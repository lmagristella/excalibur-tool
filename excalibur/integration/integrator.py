import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

###############################################################
#  VARIOUS INTEGRATORS
###############################################################

class RK4:
    def step(self, metric, state, dt, *_):
        k1 = metric.geodesic_equations(state)
        k2 = metric.geodesic_equations(state + 0.5 * dt * k1)
        k3 = metric.geodesic_equations(state + 0.5 * dt * k2)
        k4 = metric.geodesic_equations(state + dt * k3) 
        # IMPORTANT: when positions are enormous (e.g. ~1e24 m) and dt*u is small
        # (e.g. ~1e2..1e10), the increment can be swallowed by float64 rounding.
        # Update position and momentum separately to reduce catastrophic cancellation.
        incr = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        new_state = state.copy()
        new_state[:4] = (state[:4] + incr[:4])
        new_state[4:] = (state[4:] + incr[4:])
        return new_state, dt, True 

class Leapfrog4:
    """
    4th-order Forest–Ruth symplectic integrator (analytical coefficients).

    Assumes state = [x(4), u(4)], where the geodesic equations return:
        d/dλ [x, u] = [u, a(x,u)].

    This scheme is symplectic and time-reversible.
    It must be used with a FIXED timestep (not adaptive).
    """

    # Forest–Ruth analytical coefficients
    w0 = - 2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))

    c1 = c4 = w1 / 2.0
    c2 = c3 = (w0 + w1) / 2.0

    d1 = d3 = w1
    d2 = w0

    def step(self, metric, state, dt, *_):
        """
        Perform one Forest–Ruth 4th order symplectic integration step.

        Parameters
        ----------
        metric : object
            Must provide geodesic_equations(state) -> [u, a]
        state : ndarray
            Size-8 vector [x, u]
        dt : float
            Fixed timestep
        *_ : ignored (keeps interface consistent)

        Returns
        -------
        new_state : ndarray
            Updated [x, u]
        dt : float
            Unchanged
        True : bool
            Always True (compatible with RK45 interface)
        """

        x = state[:4].copy()
        u = state[4:].copy()

        def accel(x, u):
            su = np.hstack([x, u])
            return metric.geodesic_equations(su)[4:]
        

        # Step 1
        x += self.c1 * dt * u
        u += self.d1 * dt * accel(x, u)
        # Step 2
        x += self.c2 * dt * u
        u += self.d2 * dt * accel(x, u)
        # Step 3
        x += self.c3 * dt * u
        u += self.d3 * dt * accel(x, u)
        # Step 4
        x += self.c4 * dt * u

        new_state = np.hstack([x, u])
        return new_state, dt, True


class RK45Adaptive:
    """
    Vectorized adaptive RK45 integrator (Fehlberg)

    Notes:
    - metric.geodesic_equations(state) is still called 6 times per trial
      step (unavoidable), but all combination arithmetic is vectorized.
    - returns (new_state, new_dt, accepted)
    """

    def __init__(self):
        # Butcher tableau as full arrays for vectorized dot products
        # B is lower-triangular coefficients with zeros where unused
        self.B = np.zeros((6, 6), dtype=float)
        self.B[1, 0] = 1/4
        self.B[2, 0:2] = [3/32, 9/32]
        self.B[3, 0:3] = [1932/2197, -7200/2197, 7296/2197]
        self.B[4, 0:4] = [439/216, -8, 3680/513, -845/4104]
        self.B[5, 0:5] = [-8/27, 2, -3544/2565, 1859/4104, -11/40]

        # embedded formulas
        self.C5 = np.array([16/135, 0, 665/1287, 28561/56430, -9/50, 2/55])
        self.C4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])

    def step(self, metric, state, dt, rtol, atol):
        # state is 1D array (size N)
        # We'll compute k[i] = f(state + dt * sum_j B[i,j] * k[j]) in sequence

        K = np.zeros((6, state.size), dtype=float)
        # k1
        K[0] = metric.geodesic_equations(state)

        # Compute k2..k6 — only 5 iterations at Python level
        for i in range(1, 6):
            # compute linear combination sum_j B[i,j] * K[j]
            # B[i, :i] dot K[:i] gives a vector of shape (state.size,)
            coeffs = self.B[i, :i]
            # use tensordot for efficiency
            incr = np.tensordot(coeffs, K[:i], axes=(0, 0))
            K[i] = metric.geodesic_equations(state + dt * incr)

        # Combine k's for 4th and 5th order estimates using vectorized dot
        y5 = state + dt * np.tensordot(self.C5, K, axes=(0, 0))
        y4 = state + dt * np.tensordot(self.C4, K, axes=(0, 0))

        # error estimate and acceptance
        err = np.abs(y5 - y4)
        tol = atol + rtol * np.maximum(np.abs(state), np.abs(y5))
        # handle zero tolerance entries
        tol = np.where(tol == 0.0, atol, tol)
        err_ratio = np.max(err / tol)

        if err_ratio <= 1.0:
            dt_new = dt * min(5.0, 0.9 * err_ratio ** -0.2 if err_ratio > 0 else 5.0)
            return y5, dt_new, True
        else:
            dt_new = dt * max(0.1, 0.9 * err_ratio ** -0.25)
            return state, dt_new, False


###############################################################
#  INTEGRATOR WITH MULTIPLE STOP CONDITIONS
###############################################################
#dt_min and dt_max should be coherent physically with cosmological scales, in seconds. 
#therefore dt_min should be around a few million years in seconds, and dt_max should be around a few billion years in seconds.
#dt_min = 1e14 # ~3 million years in seconds 
#dt_max = 1e17  # ~3 billion years in seconds
class Integrator:

    STOP_MODES = {"steps", "redshift", "a", "chi"}
    INTEGRATORS = {"rk45": RK45Adaptive, "rk4": RK4, "leapfrog4": Leapfrog4}

    def __init__(
        self,
        metric,
        dt=1e-3,
        mode="sequential",
        integrator="rk45",
        rtol=1e-6,
        atol=1e-13,
        dt_min=1e14,
        dt_max=1e17,
        n_workers=None,
        chunk_size=50,
    ):
        self.metric = metric
        self.dt = dt
        self.mode = mode
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.chunk_size = chunk_size
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)

        # Select the proper integrator based on user choice
        if integrator.lower() not in self.INTEGRATORS:
            raise ValueError(f"Integrator '{integrator}' not supported. Available: {list(self.INTEGRATORS.keys())}")

        # CRITICAL FIX: Actually use the requested integrator!
        integrator_class = self.INTEGRATORS[integrator.lower()]
        self.integrator = integrator_class()

    ###############################################################
    # STOPPING CONDITIONS
    ###############################################################
    def _should_stop(self, photon, stop_mode, stop_value):
        if stop_mode == "steps":
            return False  # handled externally
        elif stop_mode == "redshift":
            # assume photon.z increases when integrating backward/forward as in your project
            return photon.z >= stop_value
        elif stop_mode == "a":
            # factor of scale decreases when going backward in time: stop when a <= value
            return photon.a <= stop_value
        elif stop_mode == "chi":
            return photon.comoving_distance >= stop_value
        else:
            raise ValueError(f"Unknown stop mode {stop_mode}")

    ###############################################################
    # SEQUENTIAL SINGLE PHOTON (optimized loops)
    ###############################################################
    def integrate_single(
        self,
        photon,
        stop_mode="steps",
        stop_value=100,
        *,
        record_every: int = 1,
        trace_norm: bool = False,
        renormalize_every: int = 0,
    ):
        
        state = np.concatenate([photon.x, photon.u])

        # If the metric works internally in spherical coordinates, convert the state.
        # Only do this if a conversion helper exists.
        # Convert to internal coords if needed
        internal = getattr(self.metric, "internal_coords", None)
        input_coords = getattr(self.metric, "input_coords", None)

        if internal == "spherical" and input_coords == "cartesian":
            # photon state is cartesian -> convert to spherical internal
            if hasattr(self.metric, "cartesian_state_to_spherical"):
                state = self.metric.cartesian_state_to_spherical(state)
            else:
                raise ValueError("Metric declares internal spherical but has no cartesian_state_to_spherical()")

        dt = float(self.dt)
        steps = 0

        # Safety counter to avoid infinite loops if stop condition never met
        max_iter = int(1e4) 
        it = 0

        # OPTIMIZATION: Pre-allocate history for better performance
        if not hasattr(photon, 'history'):
            photon.history = []
        
        # OPTIMIZATION: Cache metric physical quantities function to avoid lookups
        metric_quantities_func = self.metric.metric_physical_quantities

    # Recording control:
        #   record_every=1  -> record every accepted step (legacy behavior)
        #   record_every=0  -> do not record intermediate steps (only final record)
        #   record_every=N  -> record every N accepted steps
        if record_every < 0:
            raise ValueError("record_every must be >= 0")

        if renormalize_every < 0:
            raise ValueError("renormalize_every must be >= 0")

        # Optional norm tracing (relative null error, scale-free).
        # Stored on the photon instance for callers that want it.
        if trace_norm and not hasattr(photon, "norm_history"):
            photon.norm_history = []

        def _compute_relative_null_error_from_state(st: np.ndarray) -> float:
            # st = [x(4), u(4), ...]
            x_ = st[:4]
            u_ = st[4:8]
            g = self.metric.metric_tensor(x_)
            # norm = u^T g u
            norm = float(np.einsum("i,ij,j->", u_, g, u_))
            g00_term = abs(g[0, 0] * u_[0] ** 2)
            g_spatial_terms = abs(g[1, 1] * u_[1] ** 2) + abs(g[2, 2] * u_[2] ** 2) + abs(g[3, 3] * u_[3] ** 2)
            denom = g00_term + g_spatial_terms
            return abs(norm) / denom if denom > 0 else abs(norm)

        def _renormalize_state_to_null(st: np.ndarray) -> np.ndarray:
            """Project the state back onto the null cone by solving for u0.

            Keeps x and u_spatial fixed, recomputes u0 from:
                g00 u0^2 + 2 g0i u0 ui + gij ui uj = 0

            Returns a new state (copy) if successful; otherwise returns st unchanged.
            """
            x_ = st[:4]
            u_ = st[4:8].copy()
            g = self.metric.metric_tensor(x_)

            ui = u_[1:4]
            # Quadratic: A u0^2 + B u0 + C = 0
            A = float(g[0, 0])
            B = 2.0 * float(g[0, 1:4] @ ui)
            C = float(ui @ (g[1:4, 1:4] @ ui))

            # If A is ~0, fallback to leaving unchanged (shouldn't happen for sane metrics)
            if abs(A) < 1e-300:
                return st

            disc = B * B - 4.0 * A * C
            if disc < 0:
                return st

            sqrt_disc = float(np.sqrt(disc))
            u0a = (-B + sqrt_disc) / (2.0 * A)
            u0b = (-B - sqrt_disc) / (2.0 * A)

            # Choose the root closest to current u0 (preserves time orientation).
            u0_cur = float(u_[0])
            u0_new = u0a if abs(u0a - u0_cur) <= abs(u0b - u0_cur) else u0b
            u_[0] = u0_new

            st2 = st.copy()
            st2[4:8] = u_
            return st2

        # Ensure we always have an initial record for analysis/debugging.
        # (Even when record_every=0 which means "final only".)
        if hasattr(photon, "record"):
            photon.record()
        else:
            photon.history.append(state)

        if trace_norm:
            photon.norm_history.append(_compute_relative_null_error_from_state(state))

        while True:
            if it >= max_iter:
                break

            # external stop conditions
            if stop_mode == "steps" and steps >= stop_value:
                break
            if stop_mode != "steps" and self._should_stop(photon, stop_mode, stop_value):
                break

            # enforce dt limits - CRITICAL FIX: handle negative dt properly for backward tracing
            if dt < 0:
                # For backward tracing (negative dt), clip the absolute value then restore sign
                dt = -float(np.clip(abs(dt), self.dt_min, self.dt_max))
            else:
                # For forward tracing (positive dt)
                dt = float(np.clip(dt, self.dt_min, self.dt_max))
                
            # attempt step
            new_state, dt_new, accepted = self.integrator.step(
                self.metric, state, dt, self.rtol, self.atol
            )
            
            if accepted:
                state = new_state

                # Optional renormalization to control drift.
                # Apply after accepting the step (so we don't fight adaptive dt logic).
                if renormalize_every and (steps % renormalize_every) == 0:
                    state = _renormalize_state_to_null(state)

                # Keep photon.x/u consistent with the metric coordinate system.
                # If the metric integrates in spherical coords, photon.x becomes spherical.
                photon.x = state[:4]
                photon.u = state[4:]
                photon.state_quantities(metric_quantities_func)

                if trace_norm:
                    photon.norm_history.append(_compute_relative_null_error_from_state(state))

                if record_every == 1:
                    photon.record()
                elif record_every > 1:
                    if (steps % record_every) == 0:
                        photon.record()
                
                steps += 1

            dt = dt_new
            it += 1
            
        # OPTIMIZATION: Ensure final state is always recorded
        photon.state_quantities(metric_quantities_func)
        photon.record()

        if trace_norm:
            photon.norm_history.append(_compute_relative_null_error_from_state(state))

        return photon

    ###############################################################
    # PARALLEL HELPERS
    ###############################################################
    def _worker_single(self, args):
        photon, stop_mode, stop_value = args
        try:
            return self.integrate_single(photon, stop_mode, stop_value), True
        except Exception:
            return photon, False

    def _worker_chunk(self, photons, stop_mode, stop_value):
        results = []
        for p in photons:
            try:
                integrated_photon = self.integrate_single(p, stop_mode, stop_value)
                results.append((integrated_photon, True))
            except Exception:
                results.append((p, False))
        return results

    ###############################################################
    # MAIN ENTRY POINT
    ###############################################################
    def integrate(self, photons, stop_mode="steps", stop_value=100, verbose=True):
        if stop_mode not in self.STOP_MODES:
            raise ValueError(f"stop_mode must be one of {self.STOP_MODES}")

        if self.mode == "sequential":
            for p in photons:
                self.integrate_single(p, stop_mode, stop_value)
            return len(photons)

        elif self.mode == "parallel":
            if verbose:
                print(f"Parallel integration with {self.n_workers} workers...")

            args = [(p, stop_mode, stop_value) for p in photons]
            with Pool(processes=self.n_workers) as pool:
                results = pool.map(self._worker_single, args)
            return sum(1 for _, ok in results if ok)

        elif self.mode == "chunked":
            # chunked mode: group photons to reduce multiprocessing overhead
            chunks = [photons[i:i+self.chunk_size] for i in range(0, len(photons), self.chunk_size)]
            if verbose:
                print(f"Chunked mode: {len(chunks)} chunks of size {self.chunk_size}")

            partial_func = partial(self._worker_chunk, stop_mode=stop_mode, stop_value=stop_value)
            with Pool(processes=self.n_workers) as pool:
                chunk_results = pool.map(partial_func, chunks)

            # Reconstruct photons list with integrated results
            success_count = 0
            photon_index = 0
            for chunk_result in chunk_results:
                for integrated_photon, success in chunk_result:
                    if success:
                        # Replace original photon with integrated version
                        photons.photons[photon_index] = integrated_photon
                        success_count += 1
                    photon_index += 1

            return success_count

        else:
            raise ValueError("Unknown mode. Use 'sequential', 'parallel', or 'chunked'.")


import numpy as np
from multiprocessing import Pool, cpu_count
import scipy.integrate as scint

###############################################################
#  VARIOUS INTEGRATORS
###############################################################

class RK4:
    dt_actual = 0.0  # step size actually used (always == dt for fixed-step)

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
        self.dt_actual = dt
        return new_state, dt, True

class Leapfrog4:
    """
    4th-order Forest-Ruth symplectic integrator (analytical coefficients).

    Assumes state = [x(4), u(4)], where the geodesic equations return:
        d/dlambda [x, u] = [u, a(x,u)].

    This scheme is symplectic and time-reversible.
    It must be used with a FIXED timestep (not adaptive).
    """

    # Forest-Ruth analytical coefficients
    w0 = - 2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))

    c1 = c4 = w1 / 2.0
    c2 = c3 = (w0 + w1) / 2.0

    d1 = d3 = w1
    d2 = w0

    dt_actual = 0.0  # step size actually used (always == dt for fixed-step)

    def step(self, metric, state, dt, *_):
        """
        Perform one Forest-Ruth 4th order symplectic integration step.

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

        if state.shape[0] != 8:
            raise ValueError(
                f"Leapfrog4 only supports 8-component geodesic states, "
                f"got {state.shape[0]}.  Use RK4 or RK45 for lensing (24-comp)."
            )

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
        self.dt_actual = dt
        return new_state, dt, True


class RK45Adaptive:
    """
    Adaptive RK45 integrator (Dormand-Prince, FSAL variant).

    Notes
    -----
    - Uses the **Dormand-Prince** Butcher tableau (same as MATLAB's ode45
      and scipy's RK45), which is generally superior to Fehlberg for
      adaptive stepping because the 5th-order solution is propagated.
    - 6 function evaluations per *accepted* step (FSAL: the last evaluation
      of an accepted step is reused as the first of the next step).
    - Returns ``(new_state, new_dt, accepted)``.
    """

    # Dormand-Prince Butcher tableau  (a_i, b_i, b*_i)
    # -- nodes --
    _c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=float)

    # -- RK matrix (lower-triangular, row i uses stages 0..i-1) --
    _A1 = np.array([1/5])
    _A2 = np.array([3/40, 9/40])
    _A3 = np.array([44/45, -56/15, 32/9])
    _A4 = np.array([19372/6561, -25360/2187, 64448/6561, -212/729])
    _A5 = np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656])
    _A6 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])

    # -- 5th-order weights (propagated solution) --
    _b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
                  dtype=float)

    # -- 4th-order weights (error estimator) --
    _bs = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200,
                    187/2100, 1/40], dtype=float)

    # -- error coefficients  e = b - b*  (pre-computed) --
    _e = _b - _bs

    def __init__(self):
        # Pre-allocate nothing; all temporaries are per-call.
        self.dt_actual = 0.0  # step size actually used

    def step(self, metric, state, dt, rtol, atol):
        f = metric.geodesic_equations
        n = state.size

        # -- stages --
        k0 = f(state)
        k1 = f(state + dt * self._A1[0] * k0)
        k2 = f(state + dt * (self._A2[0] * k0 + self._A2[1] * k1))
        k3 = f(state + dt * (self._A3[0] * k0 + self._A3[1] * k1
                            + self._A3[2] * k2))
        k4 = f(state + dt * (self._A4[0] * k0 + self._A4[1] * k1
                            + self._A4[2] * k2 + self._A4[3] * k3))
        k5 = f(state + dt * (self._A5[0] * k0 + self._A5[1] * k1
                            + self._A5[2] * k2 + self._A5[3] * k3
                            + self._A5[4] * k4))
        k6 = f(state + dt * (self._A6[0] * k0 + self._A6[2] * k2
                            + self._A6[3] * k3 + self._A6[4] * k4
                            + self._A6[5] * k5))  # A6[1]=0

        # -- 5th-order solution --
        y5 = state + dt * (self._b[0] * k0 + self._b[2] * k2
                         + self._b[3] * k3 + self._b[4] * k4
                         + self._b[5] * k5)   # b[1]=b[6]=0

        # -- error estimate  (difference between 5th and 4th order) --
        err = dt * (self._e[0] * k0 + self._e[2] * k2 + self._e[3] * k3
                  + self._e[4] * k4 + self._e[5] * k5 + self._e[6] * k6)

        # -- tolerance (component-wise, mixed absolute/relative) --
        sc = atol + rtol * np.maximum(np.abs(state), np.abs(y5))

        # -- weighted RMS error norm --
        err_norm = np.sqrt(np.mean((err / sc) ** 2))

        # -- step-size control (PI controller, standard safety factors) --
        safety = 0.9
        min_factor = 0.2    # never shrink by more than 5x
        max_factor = 5.0    # never grow by more than 5x

        if err_norm <= 1.0:
            # accepted
            if err_norm < 1e-30:
                factor = max_factor
            else:
                factor = min(max_factor, safety * err_norm ** (-0.2))
            dt_new = dt * factor
            self.dt_actual = dt      # accepted step used exactly dt
            return y5, dt_new, True
        else:
            # rejected  --  shrink
            factor = max(min_factor, safety * err_norm ** (-0.25))
            dt_new = dt * factor
            self.dt_actual = 0.0     # rejected: no step taken
            return state, dt_new, False


class DOP853:
    """Wrapper around scipy's DOP853 (Dormand-Prince 8(5,3)) integrator.

    Unlike the old implementation that created a *new* ``scipy.integrate.DOP853``
    object at every single ``step()`` call (losing all internal state, FSAL
    information and step-size history), this version creates the scipy object
    **once** and reuses it across successive calls.

    The scipy integrator internally takes *one* adaptive sub-step per
    ``.step()`` call and updates its own ``t``, ``y``, ``step_size`` etc.
    We simply relay the result.
    """

    def __init__(self):
        self._ode = None          # lazily created on first step()
        self._t = 0.0             # current "time" (= cumulative affine lambda)
        self.dt_actual = 0.0      # step size actually used by scipy

    def step(self, metric, state, dt, rtol, atol):
        def f(t, y):
            return metric.geodesic_equations(y)

        # (Re-)create the scipy integrator when:
        #  - first call ever
        #  - state size changed (shouldn't happen, but safety)
        #  - the caller replaced the state under us (e.g. renormalization)
        need_init = (
            self._ode is None
            or self._ode.y.shape != state.shape
            or not np.allclose(self._ode.y, state, rtol=0, atol=1e-30)
        )
        if need_init:
            self._t = 0.0
            # t_bound must be far away so scipy doesn't declare "finished"
            # after a single step.  1e30 is effectively infinite.
            # Floor rtol to machine epsilon to avoid scipy warnings.
            safe_rtol = max(rtol, 2.3e-14)
            self._ode = scint.DOP853(
                f, self._t, state, t_bound=np.sign(dt) * 1e30,
                rtol=safe_rtol, atol=atol,
                first_step=abs(dt),
                max_step=abs(dt) * 50,
            )

        # Take ONE adaptive sub-step.
        self._ode.step()

        if self._ode.status == "failed":
            # integrator signalled an unrecoverable error
            self.dt_actual = 0.0
            return state, abs(dt) * 0.5, False

        new_state = self._ode.y
        # The step size actually used by scipy:
        dt_used = self._ode.t - self._t
        self._t = self._ode.t
        self.dt_actual = dt_used   # <-- for lambda accumulation

        # Propose the next step size as what scipy chose internally.
        dt_new = self._ode.step_size
        if dt < 0:
            dt_new = -abs(dt_new)

        return new_state, dt_new, True

###############################################################
#  INTEGRATOR WITH MULTIPLE STOP CONDITIONS
###############################################################
class Integrator:
    """High-level integrator wrapping RK4 / RK45 / Leapfrog4 / DOP853.

    Supports three execution modes:
    - ``"sequential"`` : integrate photons one by one in the main process.
    - ``"parallel"``   : persistent worker pool with one photon per task.
    - ``"parallel_chunked"`` : persistent pool with grouped photon chunks
      for reduced task-submission overhead.

    For parallel modes the class automatically detects the metric type:
    - **Grid-based** metrics (``PerturbedFLRWMetric*``): the grid's Phi
      field is placed in shared memory so workers attach without copying.
    - **Analytical** metrics (``Schwarzschild*``): lightweight parameter
      dicts are passed to each worker.

    Use as a context manager so the pool is created once and reused:

        >>> with Integrator(metric, dt=-1e15, mode="parallel", n_workers=4) as integ:
        ...     integ.integrate(photons, stop_mode="steps", stop_value=1000)

    For sequential mode, context manager is a no-op (also works without ``with``).

    Parameters
    ----------
    dt_min, dt_max : float or None
        Hard bounds on the step size.  For **adaptive** integrators the
        defaults are derived automatically from the initial ``dt``:

            dt_min = |dt| / 1000      (allow the scheme to shrink freely)
            dt_max = |dt| * 50

        For **fixed-step** integrators (RK4, Leapfrog4) these limits are
        irrelevant because dt never changes.

        You can still override them explicitly by passing a value.
    """

    STOP_MODES = {"steps", "redshift", "a", "chi", "affine"}
    INTEGRATORS = {"rk45": RK45Adaptive, "rk4": RK4, "leapfrog4": Leapfrog4, "dop853": DOP853}

    _ADAPTIVE = {"rk45", "dop853"}   # integrators that adapt dt

    # Metric classes that use a grid (and therefore need shared memory)
    _GRID_METRIC_CLASSES = {
        "PerturbedFLRWMetricFast",
        "PerturbedFLRWMetric",
        "PerturbedFLRWMetricJAX",
    }

    # Analytical metric classes with their constructor parameter names
    _ANALYTICAL_METRIC_PARAMS = {
        "SchwarzschildMetricCartesian": ("mass", "radius", "center"),
        "SchwarzschildMetric": ("mass", "radius", "center"),
        "SchwarzschildMetricFast": ("mass", "radius", "center"),
    }

    def __init__(
        self,
        metric,
        dt=1e-3,
        mode="sequential",
        integrator="rk45",
        rtol=1e-9,
        atol=1e-13,
        dt_min=None,
        dt_max=None,
        n_workers=None,
        chunk_size=50,
    ):
        self.metric = metric
        self.dt = dt
        self.mode = mode
        self.rtol = rtol
        self.atol = atol
        self.chunk_size = chunk_size
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)

        # Pool state (created lazily on first parallel integrate call)
        self._pool = None
        self._shm = None

        # Select the proper integrator based on user choice
        integ_key = integrator.lower()
        if integ_key not in self.INTEGRATORS:
            raise ValueError(f"Integrator '{integrator}' not supported. "
                             f"Available: {list(self.INTEGRATORS.keys())}")

        integrator_class = self.INTEGRATORS[integ_key]
        self.integrator = integrator_class()
        self._is_adaptive = integ_key in self._ADAPTIVE

        # -- sensible dt bounds ----------------------------------
        abs_dt = abs(float(dt))
        if dt_min is not None:
            self.dt_min = float(dt_min)
        else:
            self.dt_min = abs_dt / 1000.0 if self._is_adaptive else abs_dt

        if dt_max is not None:
            self.dt_max = float(dt_max)
        else:
            self.dt_max = abs_dt * 50.0 if self._is_adaptive else abs_dt

    # ==============================================================
    # Context manager (pool lifecycle)
    # ==============================================================
    def __enter__(self):
        if self.mode in ("parallel", "parallel_chunked"):
            self._create_pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close worker pool and free shared memory."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None

    # ==============================================================
    # Pool creation (auto-detect metric type)
    # ==============================================================
    def _is_grid_metric(self):
        return self.metric.__class__.__name__ in self._GRID_METRIC_CLASSES

    def _create_pool(self):
        """Create the persistent worker pool, auto-detecting metric type."""
        if self._pool is not None:
            return  # already created

        import time as _time
        start = _time.time()

        if self._is_grid_metric():
            self._create_pool_grid()
        else:
            self._create_pool_analytical()

        elapsed = _time.time() - start
        print(f"  Pool ready ({self.n_workers} workers, {elapsed:.2f}s)")

    def _create_pool_grid(self):
        """Create pool for grid-based metrics with shared memory."""
        from multiprocessing import shared_memory
        from excalibur.integration.parallel_workers import init_worker_grid

        grid = self.metric.grid
        phi_field = grid.fields["Phi"]

        # Create shared memory and copy Phi into it
        self._shm = shared_memory.SharedMemory(
            create=True, size=phi_field.nbytes
        )
        shared_array = np.ndarray(
            phi_field.shape, dtype=phi_field.dtype, buffer=self._shm.buf
        )
        np.copyto(shared_array, phi_field)

        grid_params = {
            'shape': grid.shape,
            'spacing': grid.spacing,
            'origin': grid.origin,
        }
        metric_params = {
            'a_of_eta': self.metric.a_of_eta,
        }

        print(f"  Creating parallel pool (grid-based, shared memory: "
              f"{phi_field.nbytes / 1e6:.1f} MB)...")

        self._pool = Pool(
            processes=self.n_workers,
            initializer=init_worker_grid,
            initargs=(
                self._shm.name,
                phi_field.shape,
                phi_field.dtype.str,
                grid_params,
                self.metric.__class__.__name__,
                metric_params,
                self.dt,
            ),
        )

    def _create_pool_analytical(self):
        """Create pool for analytical metrics (lightweight)."""
        from excalibur.integration.parallel_workers import init_worker_analytical

        metric_class = self.metric.__class__.__name__
        param_names = self._ANALYTICAL_METRIC_PARAMS.get(metric_class)
        if param_names is None:
            raise ValueError(
                f"Metric class '{metric_class}' is not supported for parallel mode. "
                f"Supported: {list(self._GRID_METRIC_CLASSES | set(self._ANALYTICAL_METRIC_PARAMS))}"
            )

        metric_params = {}
        for name in param_names:
            val = getattr(self.metric, name)
            metric_params[name] = val.copy() if hasattr(val, 'copy') else val

        print(f"  Creating parallel pool (analytical: {metric_class})...")

        self._pool = Pool(
            processes=self.n_workers,
            initializer=init_worker_analytical,
            initargs=(metric_class, metric_params, self.dt),
        )

    # ==============================================================
    # Result copy helper
    # ==============================================================
    @staticmethod
    def _copy_results_to_photons(photons, results):
        """Copy worker results back to original photon objects."""
        for photon, (success, final_x, final_u, history_states) in zip(photons, results):
            photon.x = final_x
            photon.u = final_u
            photon.history.states = []
            for state in history_states:
                photon.history.append(state)

    ###############################################################
    # STOPPING CONDITIONS
    ###############################################################
    def _should_stop(self, photon, stop_mode, stop_value):
        if stop_mode == "steps":
            return False  # handled externally
        elif stop_mode == "affine":
            return False  # handled externally via lambda_current
        elif stop_mode == "redshift":
            return photon.z >= stop_value
        elif stop_mode == "a":
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

        # If lensing is enabled and the photon carries Sachs/Jacobi IC,
        # extend the state to 24 components.
        enable_lensing = getattr(self.metric, "enable_lensing", False)
        if enable_lensing and hasattr(photon, "e1"):
            e1 = np.asarray(photon.e1, dtype=float)
            e2 = np.asarray(photon.e2, dtype=float)
            D_flat = np.asarray(getattr(photon, "D_flat", [0, 0, 0, 0]), dtype=float)
            P_flat = np.asarray(getattr(photon, "P_flat", [1, 0, 0, 1]), dtype=float)
            state = np.concatenate([state, e1, e2, D_flat, P_flat])

        atol = self.atol
        if np.isscalar(atol):
            atol_vec = np.empty_like(state)
            atol_vec[0]   = 1e4          # eta (conformal time) in seconds
            atol_vec[1:4] = 1e10         # positions in metres
            atol_vec[4:8] = 1e-6         # 4-momentum components
            if state.shape[0] > 8:
                atol_vec[8:16]  = 1e-14  # Sachs vectors e1, e2
                atol_vec[16:24] = 1e-14  # Jacobi map D and velocity P
            atol = atol_vec
        else:
            atol = np.asarray(atol, dtype=float)

        # Convert to internal coords if needed
        internal = getattr(self.metric, "internal_coords", None)
        input_coords = getattr(self.metric, "input_coords", None)

        if internal == "spherical" and input_coords == "cartesian":
            if hasattr(self.metric, "cartesian_state_to_spherical"):
                state = self.metric.cartesian_state_to_spherical(state)
            else:
                raise ValueError("Metric declares internal spherical but has no cartesian_state_to_spherical()")

        dt = float(self.dt)
        steps = 0

        if stop_mode == "steps":
            max_iter = max(int(1e5), int(stop_value) * 10)
        else:
            max_iter = int(1e7)
        it = 0
        rejected_consecutive = 0

        if not hasattr(photon, 'history'):
            photon.history = []

        metric_quantities_func = self.metric.metric_physical_quantities

        if record_every < 0:
            raise ValueError("record_every must be >= 0")

        if renormalize_every < 0:
            raise ValueError("renormalize_every must be >= 0")

        if trace_norm and not hasattr(photon, "norm_history"):
            photon.norm_history = []

        def _compute_relative_null_error_from_state(st: np.ndarray) -> float:
            x_ = st[:4]
            u_ = st[4:8]
            g = self.metric.metric_tensor(x_)
            norm = float(np.einsum("i,ij,j->", u_, g, u_))
            g00_term = abs(g[0, 0] * u_[0] ** 2)
            g_spatial_terms = abs(g[1, 1] * u_[1] ** 2) + abs(g[2, 2] * u_[2] ** 2) + abs(g[3, 3] * u_[3] ** 2)
            denom = g00_term + g_spatial_terms
            return abs(norm) / denom if denom > 0 else abs(norm)

        def _renormalize_state_to_null(st: np.ndarray) -> np.ndarray:
            """Project the state back onto the null cone by solving for u0."""
            x_ = st[:4]
            u_ = st[4:8].copy()
            g = self.metric.metric_tensor(x_)

            ui = u_[1:4]
            A = float(g[0, 0])
            B = 2.0 * float(g[0, 1:4] @ ui)
            C = float(ui @ (g[1:4, 1:4] @ ui))

            if abs(A) < 1e-300:
                return st

            disc = B * B - 4.0 * A * C
            if disc < 0:
                return st

            sqrt_disc = float(np.sqrt(disc))
            u0a = (-B + sqrt_disc) / (2.0 * A)
            u0b = (-B - sqrt_disc) / (2.0 * A)

            u0_cur = float(u_[0])
            u0_new = u0a if abs(u0a - u0_cur) <= abs(u0b - u0_cur) else u0b
            u_[0] = u0_new

            st2 = st.copy()
            st2[4:8] = u_
            return st2

        # Initial record
        if hasattr(photon, "record"):
            photon.record()
        else:
            photon.history.append(state)

        if trace_norm:
            photon.norm_history.append(_compute_relative_null_error_from_state(state))

        lambda_current = 0.0

        while True:

            if it >= max_iter:
                break

            if stop_mode == "steps" and steps >= stop_value:
                break
            if stop_mode == "affine" and abs(lambda_current) >= abs(stop_value):
                break
            if stop_mode not in ("steps", "affine") and self._should_stop(photon, stop_mode, stop_value):
                break

            abs_dt = abs(dt)
            abs_dt = max(self.dt_min, min(abs_dt, self.dt_max))
            dt = abs_dt if dt >= 0 else -abs_dt

            new_state, dt_new, accepted = self.integrator.step(
                self.metric, state, dt, self.rtol, atol
            )

            if accepted:
                rejected_consecutive = 0
                lambda_current += self.integrator.dt_actual
                state = new_state

                if renormalize_every and (steps % renormalize_every) == 0:
                    state = _renormalize_state_to_null(state)

                photon.x = state[:4]
                photon.u = state[4:8]

                if state.shape[0] > 8:
                    photon.e1 = state[8:12]
                    photon.e2 = state[12:16]
                    photon.D_flat = state[16:20]
                    photon.P_flat = state[20:24]

                if trace_norm:
                    photon.norm_history.append(_compute_relative_null_error_from_state(state))

                if record_every == 1:
                    photon.record()
                    photon.state_quantities(metric_quantities_func)

                elif record_every > 1:
                    if (steps % record_every) == 0:
                        photon.record()
                        photon.state_quantities(metric_quantities_func)


                steps += 1
            else:
                rejected_consecutive += 1
                if rejected_consecutive > 200:
                    import warnings
                    warnings.warn(
                        f"Integrator: {rejected_consecutive} consecutive "
                        f"rejected steps at it={it}, dt={dt:.3e}. "
                        f"Breaking out.  Increase rtol/atol or check RHS.",
                        RuntimeWarning, stacklevel=2,
                    )
                    break

            dt = dt_new
            it += 1

        # Record final state only if not already recorded
        needs_final_record = (record_every == 0) or (record_every > 1 and (steps - 1) % record_every != 0)
        if needs_final_record:
            photon.state_quantities(metric_quantities_func)
            photon.record()

        photon.lambda_affine = lambda_current

        if trace_norm:
            photon.norm_history.append(_compute_relative_null_error_from_state(state))

        return photon

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
            return self._integrate_parallel(photons, stop_mode, stop_value, verbose)

        elif self.mode == "parallel_chunked":
            return self._integrate_parallel_chunked(photons, stop_mode, stop_value, verbose)

        else:
            raise ValueError(
                "Unknown mode. Use 'sequential', 'parallel', or 'parallel_chunked'."
            )

    # ==============================================================
    # Parallel integration
    # ==============================================================
    def _integrate_parallel(self, photons, stop_mode, stop_value, verbose):
        """Integrate photons in parallel (one task per photon)."""
        from excalibur.integration.parallel_workers import integrate_photon_worker

        if self._pool is None:
            self._create_pool()

        n_steps = stop_value if stop_mode == "steps" else 10000

        if verbose:
            print(f"  Parallel integration: {len(photons)} photons, "
                  f"{self.n_workers} workers...")

        # Prepare lightweight photon data
        photon_data = [
            (p.x.copy(), p.u.copy(), n_steps)
            for p in photons
        ]

        import time as _time
        t0 = _time.time()
        results = self._pool.map(integrate_photon_worker, photon_data)
        elapsed = _time.time() - t0

        # Copy results back
        self._copy_results_to_photons(photons, results)

        n_success = sum(1 for r in results if r[0])
        if verbose:
            rate = len(photons) * n_steps / elapsed if elapsed > 0 else 0
            print(f"  Done in {elapsed:.2f}s  |  {n_success}/{len(photons)} success  "
                  f"|  {rate:.0f} step-evals/s")

        return n_success

    def _integrate_parallel_chunked(self, photons, stop_mode, stop_value, verbose):
        """Integrate photons in parallel with chunked task grouping."""
        from excalibur.integration.parallel_workers import integrate_chunk_worker

        if self._pool is None:
            self._create_pool()

        n_steps = stop_value if stop_mode == "steps" else 10000
        chunk_size = self.chunk_size or max(1, len(photons) // (self.n_workers * 3))

        if verbose:
            print(f"  Parallel chunked integration: {len(photons)} photons, "
                  f"chunks of {chunk_size}, {self.n_workers} workers...")

        # Build chunk arguments
        chunk_args = []
        for i in range(0, len(photons), chunk_size):
            chunk_photons = photons[i:i+chunk_size]
            chunk_data = [(p.x.copy(), p.u.copy()) for p in chunk_photons]
            chunk_args.append((chunk_data, n_steps))

        import time as _time
        t0 = _time.time()
        chunk_results = self._pool.map(integrate_chunk_worker, chunk_args)
        elapsed = _time.time() - t0

        # Flatten and copy back
        all_results = []
        for cr in chunk_results:
            all_results.extend(cr)

        self._copy_results_to_photons(photons, all_results)

        n_success = sum(1 for r in all_results if r[0])
        if verbose:
            rate = len(photons) * n_steps / elapsed if elapsed > 0 else 0
            print(f"  Done in {elapsed:.2f}s  |  {n_success}/{len(photons)} success  "
                  f"|  {rate:.0f} step-evals/s")

        return n_success

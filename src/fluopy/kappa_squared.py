"""
Module kappa_squared
"""

import numpy as np
from scipy.special import expit, logit
from scipy.stats import gaussian_kde


def random_unit_vector(size=1, seed=None):
    """
    Generate random 3D unit vectors.

    Parameters
    ----------
    size : int
        The number of random unit vectors to generate.
    seed : int
        Seed.

    Returns
    -------
    np.ndarray
        An array of shape (size, 3) containing random unit vectors.
    """
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(size, 3))
    norm = np.linalg.norm(vec, axis=1).reshape(-1, 1)

    return vec / norm


def rotational_diffusion_step(v, dt, tau_rot, seed=None):
    """
    Apply a random rotation to vector(s) v.

    Parameters
    ----------
    v : np.ndarray
        The vector(s) to be rotated. Can be 1D (shape: (3,)) or 2D (shape: (N, 3)).
    dt : float
        The time step for the simulation in s.
    tau_rot : float
        The rotational diffusion time constant in s.
    seed : int, optional
        Seed for the random number generator for reproducibility (default is None).

    Returns
    -------
    np.ndarray
        The rotated vector(s), normalized to be unit vector(s).
    """
    rng = np.random.default_rng(seed)

    if v.ndim == 1:
        v = v.reshape(1, -1)

    n_vectors = v.shape[0]
    angles = rng.normal(0, np.sqrt(2 * dt / tau_rot), size=n_vectors)
    axes = random_unit_vector(size=n_vectors, seed=rng)

    cos_angles = np.cos(angles).reshape(-1, 1)
    sin_angles = np.sin(angles).reshape(-1, 1)
    one_minus_cos = (1 - np.cos(angles)).reshape(-1, 1)
    axis_dot_v = np.sum(axes * v, axis=1).reshape(-1, 1)
    axis_cross_v = np.cross(axes, v)
    v_rot = (
        v * cos_angles + axis_cross_v * sin_angles + axes * axis_dot_v * one_minus_cos
    )
    norms = np.linalg.norm(v_rot, axis=1).reshape(-1, 1)
    v_rot = v_rot / norms

    return v_rot


def simulate_rotational_motion(tau_rot, tau_life, dt=1e-12, seed=None):
    """
    Simulate rotational motion and return dipole orientations over the lifetime.

    Parameters
    ----------
    tau_rot : float
        The rotational diffusion time constant in s.
    tau_life : float
        The lifetime of the dipole in s.
    dt : float, optional
        The time step for the simulation in s.
    seed : int, optional
        Seed for the random number generator for reproducibility (default is None).

    Returns
    -------
    tuple of np.ndarray
        Two arrays containing the dipole orientations over time for two dipoles.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(tau_life / dt)
    v1 = random_unit_vector(size=1, seed=rng)[0]  # Get 1D vector
    v2 = random_unit_vector(size=1, seed=rng)[0]  # Get 1D vector
    traj1 = [v1]
    traj2 = [v2]
    for _ in range(n_steps):
        v1 = rotational_diffusion_step(v1, dt, tau_rot, seed=rng)[0]
        v2 = rotational_diffusion_step(v2, dt, tau_rot, seed=rng)[0]
        traj1.append(v1)
        traj2.append(v2)

    return np.array(traj1), np.array(traj2)


def kappa_squared(d, a, r):
    """
    Calculate dipole orientation factor κ² for arrays of vectors.

    Parameters
    ----------
    d : np.ndarray
        Donor dipole vectors.
    a : np.ndarray
        Acceptor dipole vectors.
    r : np.ndarray
        Unit vector from donor to acceptor.

    Returns
    -------
    np.ndarray
        The value of κ² calculated from the input vectors.
    """
    dot_d_r = np.sum(d * r, axis=1)
    dot_a_r = np.sum(a * r, axis=1)
    dot_d_a = np.sum(d * a, axis=1)

    k2 = (dot_d_a - 3 * dot_d_r * dot_a_r) ** 2

    return k2


def integral_kappa_squared(traj1, traj2, dt, r=None):
    """
    Calculate the time-averaged κ² using integration.

    Parameters
    ----------
    traj1 : np.ndarray
        Array of dipole orientations for the first dipole.
    traj2 : np.ndarray
        Array of dipole orientations for the second dipole.
    dt : float
        The time step for the simulation.
    r : np.ndarray, optional
        Unit vector from donor to acceptor. If None, assumes z-axis [0, 0, 1].

    Returns
    -------
    float
        The time-averaged value of κ².
    """
    if r is None:
        r = np.array([0, 0, 1])

    r_expanded = np.tile(r, (len(traj1), 1))
    kappas = kappa_squared(traj1, traj2, r_expanded)
    return np.trapezoid(kappas, dx=dt) / (len(kappas) * dt)


def sample_kappa_squared_distribution(k2_values, size=100, seed=None):
    """
    Sample from the distribution of κ² values utilizing Gaussian kernel-density
    estimation and logit transformation.

    Parameters
    ----------
    k2_values : np.ndarray
        Array of κ² values.
    size : int, optional
        Number of samples to generate.
    seed : int, optional
        Seed.

    Returns
    -------
    np.ndarray
        Samples from the κ² distribution.
    """
    rng = np.random.default_rng(seed)

    k2_values_scaled = k2_values / 4
    k2_values_scaled_log = logit(np.clip(k2_values_scaled, 1e-5, 1 - 1e-5))
    kde_logit = gaussian_kde(k2_values_scaled_log, bw_method="silverman")
    samples_logit = kde_logit.resample(size, seed=rng)[0]
    samples_scaled = expit(samples_logit)
    samples_kappa2 = samples_scaled * 4

    return samples_kappa2

"""
Module kappa_squared
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from scipy.special import expit, logit
from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    from .fluopy_types import RandomGeneratorSeed


def random_unit_vector(
    size: int = 1, seed: RandomGeneratorSeed = None
) -> npt.NDArray[np.float64]:
    """
    Generate random 3D unit vectors.

    Parameters
    ----------
    size
        The number of random unit vectors to generate.
    seed
        Seed.

    Returns
    -------
    npt.NDArray[np.float64]
        An array of shape (size, 3) containing random unit vectors.
    """
    rotations = Rotation.random(size, random_state=seed)
    unit_vectors = rotations.apply([1, 0, 0])

    if size == 1:
        return unit_vectors.reshape(1, -1)
    return unit_vectors


def rotational_diffusion_step(
    v: npt.ArrayLike, dt: float, tau_rot: float, seed: RandomGeneratorSeed = None
) -> npt.NDArray[np.float64]:
    """
    Apply a random rotation to vector(s) v.

    Parameters
    ----------
    v
        The vector(s) to be rotated. Can be 1D (shape: (3,)) or 2D (shape: (N, 3)).
    dt
        The time step for the simulation in s.
    tau_rot
        The rotational diffusion time constant in s.
    seed
        Seed.

    Returns
    -------
    npt.NDArray[np.float64]
        The rotated vector(s), normalized to be unit vector(s).
    """
    rng = np.random.default_rng(seed)

    if v.ndim == 1:
        v = v.reshape(1, -1)

    n_vectors = v.shape[0]
    angles = rng.normal(0, np.sqrt(dt / (3 * tau_rot)), size=n_vectors)
    axes = random_unit_vector(size=n_vectors, seed=rng)
    rotation_vectors = axes * angles.reshape(-1, 1)
    rotations = Rotation.from_rotvec(rotation_vectors)
    v_rot = rotations.apply(v)
    norms = np.linalg.norm(v_rot, axis=1).reshape(-1, 1)
    v_rot = v_rot / norms

    return v_rot


def simulate_rotational_motion(
    tau_rot: float,
    tau_life: float,
    dt: float | None = 1e-12,
    seed: RandomGeneratorSeed = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Simulate rotational motion and return dipole orientations over the lifetime.

    Parameters
    ----------
    tau_rot
        The rotational diffusion time constant in s.
    tau_life
        The lifetime of the dipole in s.
    dt
        The time step for the simulation in s.
    seed
        Seed.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
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


def kappa_squared(
    d: npt.ArrayLike, a: npt.ArrayLike, r: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """
    Calculate dipole orientation factor κ² for arrays of vectors.

    Parameters
    ----------
    d
        Donor dipole vectors.
    a
        Acceptor dipole vectors.
    r
        Unit vector from donor to acceptor.

    Returns
    -------
    npt.NDArray[np.float64]
        The value of κ² calculated from the input vectors.
    """
    dot_d_r = np.sum(d * r, axis=1)
    dot_a_r = np.sum(a * r, axis=1)
    dot_d_a = np.sum(d * a, axis=1)

    k2 = (dot_d_a - 3 * dot_d_r * dot_a_r) ** 2

    return k2


def integral_kappa_squared(
    traj1: npt.ArrayLike,
    traj2: npt.ArrayLike,
    dt: float,
    r: npt.ArrayLike | None = None,
) -> float:
    """
    Calculate the time-averaged κ² using integration.

    Parameters
    ----------
    traj1
        Array of dipole orientations for the first dipole.
    traj2
        Array of dipole orientations for the second dipole.
    dt
        The time step for the simulation.
    r
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


def sample_kappa_squared_distribution(
    k2_values: npt.ArrayLike, size: int | None = 100, seed: RandomGeneratorSeed = None
) -> npt.NDArray[np.float64]:
    """
    Sample from the distribution of κ² values utilizing Gaussian kernel-density
    estimation and logit transformation.

    Parameters
    ----------
    k2_values
        Array of κ² values.
    size
        Number of samples to generate.
    seed
        Seed.

    Returns
    -------
    npt.NDArray[np.float64]
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

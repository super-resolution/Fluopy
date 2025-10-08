import numpy as np

from fluopy import kappa_squared as kappa_sq


def test_random_unit_vector():
    rng = np.random.default_rng(42)
    v = kappa_sq.random_unit_vector(size=1, seed=rng)
    assert v.shape == (1, 3)
    assert np.allclose(np.linalg.norm(v, axis=1), 1)
    v = kappa_sq.random_unit_vector(size=10, seed=rng)
    assert v.shape == (10, 3)
    assert np.allclose(np.linalg.norm(v, axis=1), 1)


def test_rotational_diffusion_step():
    rng = np.random.default_rng(42)
    v = np.array([1, 0, 0])
    dt = 1.0
    tau_rot = 10.0
    rotated_v = kappa_sq.rotational_diffusion_step(v, dt, tau_rot, seed=rng)
    assert rotated_v.shape == (1, 3)
    assert np.allclose(np.linalg.norm(rotated_v, axis=1), 1)

    v = np.array([[1, 0, 0], [0, 1, 0]])
    rotated_v = kappa_sq.rotational_diffusion_step(v, dt, tau_rot, seed=rng)
    assert rotated_v.shape == (2, 3)
    assert np.allclose(np.linalg.norm(rotated_v, axis=1), 1)


def test_simulate_rotational_motion():
    rng = np.random.default_rng(42)
    tau_rot = 1e-6
    tau_life = 4e-9
    dt = 1e-12

    traj1, traj2 = kappa_sq.simulate_rotational_motion(tau_rot, tau_life, dt, seed=rng)

    # two trajectories
    assert isinstance(traj1, np.ndarray)
    assert isinstance(traj2, np.ndarray)

    expected_steps = int(tau_life / dt) + 1  # +1 for initial position
    assert traj1.shape == (expected_steps, 3)
    assert traj2.shape == (expected_steps, 3)

    # all vectors are unit vectors
    assert np.allclose(np.linalg.norm(traj1, axis=1), 1)
    assert np.allclose(np.linalg.norm(traj2, axis=1), 1)

    # vectors change over time (not static)
    assert not np.allclose(traj1[0], traj1[-1])
    assert not np.allclose(traj2[0], traj2[-1])


def test_kappa_squared():
    d = np.array([[1, 0, 0], [0, 1, 0]])
    a = np.array([[0, 1, 0], [1, 0, 0]])
    r = np.array([[0, 0, 1], [0, 0, 1]])

    k2 = kappa_sq.kappa_squared(d, a, r)

    assert k2.shape == (2,)
    assert isinstance(k2, np.ndarray)
    assert np.all(k2 >= 0)

    # perpendicular dipoles with z-axis separation
    # κ² = (0 - 3*0*0)² = 0
    d_perp = np.array([[1, 0, 0]])
    a_perp = np.array([[0, 1, 0]])
    r_z = np.array([[0, 0, 1]])
    k2_perp = kappa_sq.kappa_squared(d_perp, a_perp, r_z)
    assert np.allclose(k2_perp, 0)


def test_integral_kappa_squared():
    rng = np.random.default_rng(42)
    traj1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    traj2 = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]])
    dt = 0.001

    avg_k2 = kappa_sq.integral_kappa_squared(traj1, traj2, dt)
    assert isinstance(avg_k2, (float, np.floating))
    assert avg_k2 >= 0

    # custom r
    r = np.array([1, 0, 0])
    avg_k2_custom = kappa_sq.integral_kappa_squared(traj1, traj2, dt, r=r)
    assert isinstance(avg_k2_custom, (float, np.floating))
    assert avg_k2_custom >= 0

    tau_rot = 1e-6
    tau_life = 1e-9
    dt = 1e-11
    traj1_long, traj2_long = kappa_sq.simulate_rotational_motion(
        tau_rot, tau_life, dt, seed=rng
    )
    avg_k2_long = kappa_sq.integral_kappa_squared(traj1_long, traj2_long, dt)
    assert isinstance(avg_k2_long, (float, np.floating))
    assert avg_k2_long >= 0
    assert avg_k2_long <= 4  # κ² is bounded between 0 and 4


def test_sample_kappa_squared_distribution():
    k2_values = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    size = 50
    samples = kappa_sq.sample_kappa_squared_distribution(k2_values, size=size, seed=1)
    assert samples.shape == (size,)
    assert isinstance(samples, np.ndarray)
    assert np.all(samples >= 0)
    assert np.all(samples <= 4)

    samples_repeat = kappa_sq.sample_kappa_squared_distribution(
        k2_values, size=size, seed=1
    )
    assert np.allclose(samples, samples_repeat)
    samples_diff_seed = kappa_sq.sample_kappa_squared_distribution(
        k2_values, size=size, seed=2
    )
    assert not np.allclose(samples, samples_diff_seed)


def test_kappa_squared_edge_cases():
    # parallel dipoles along r-direction
    d_parallel = np.array([[0, 0, 1]])
    a_parallel = np.array([[0, 0, 1]])
    r_z = np.array([[0, 0, 1]])
    k2_parallel = kappa_sq.kappa_squared(d_parallel, a_parallel, r_z)
    # κ² = (1 - 3*1*1)² = 4 (maximum value)
    assert np.allclose(k2_parallel, 4)

    # antiparallel dipoles along r-direction
    d_anti = np.array([[0, 0, 1]])
    a_anti = np.array([[0, 0, -1]])
    r_z = np.array([[0, 0, 1]])
    k2_anti = kappa_sq.kappa_squared(d_anti, a_anti, r_z)
    # κ² = (-1 - 3*1*(-1))² = 4
    assert np.allclose(k2_anti, 4)

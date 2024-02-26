import itertools

import jax
import numpy as np
import pytest
import sympy as sp
from utils import rotation

import quadrotor_dynamics_jax as quad


def symbolic_quadrotor_dynamics(x, u):
    attitude = x[3:7, 0]
    velocity = x[7:10, 0]

    body_rates = u[1:4, 0]

    gravity_vector = sp.Matrix([0.0, 0.0, -9.81])
    return sp.Matrix(
        [
            velocity,
            rotation.quaternion_product(
                attitude, (body_rates / 2.0).col_join(sp.Matrix.zeros(1))
            ),
            rotation.quaternion_rotate_point(attitude, sp.Matrix([0.0, 0.0, u[0]]))
            + gravity_vector,
        ]
    )


x_sym = sp.Matrix(sp.MatrixSymbol("x", 10, 1))
u_sym = sp.Matrix(sp.MatrixSymbol("u", 4, 1))

dx_sym = symbolic_quadrotor_dynamics(x_sym, u_sym)

quadrotor_dynamics = sp.lambdify((x_sym, u_sym), dx_sym)
tx_sym = sp.Matrix(sp.MatrixSymbol("tx", 10, 1))
tu_sym = sp.Matrix(sp.MatrixSymbol("tu", 4, 1))

fjac_sym = dx_sym.jacobian(x_sym)
quadrotor_fjac = sp.lambdify((x_sym, u_sym), fjac_sym)
gjac_sym = dx_sym.jacobian(u_sym)
quadrotor_gjac = sp.lambdify((x_sym, u_sym), gjac_sym)

jvp_sym = fjac_sym @ tx_sym + gjac_sym @ tu_sym

quadrotor_pushforward = sp.lambdify((x_sym, u_sym, tx_sym, tu_sym), jvp_sym)


def random_quaternion(rng):
    u1, u2, u3 = rng.uniform(
        np.zeros(3),
        np.array([1.0, 2 * np.pi, 2 * np.pi]),
        (3,),
    )
    a = np.sqrt(1.0 - u1)
    b = np.sqrt(u1)
    return np.array([a * np.cos(u2), b * np.sin(u3), b * np.cos(u3), a * np.sin(u2)])


MAX_POSITION = np.full([3], 10.0)
MAX_VELOCITY = np.full([3], 1.0)
MAX_THRUST = 80.0
MAX_BODY_RATES = np.full([3], 8.0)

NUM_TRIALS = 10000
devices = list(itertools.chain.from_iterable(map(jax.devices, ["gpu", "cpu"])))


@pytest.fixture(name="test_data")
def _():
    rng = np.random.default_rng(100)
    position = rng.uniform(-MAX_POSITION, MAX_POSITION, size=(NUM_TRIALS, 3))
    attitude = np.array([random_quaternion(rng) for _ in range(NUM_TRIALS)])
    velocity = rng.uniform(-MAX_VELOCITY, MAX_VELOCITY, size=(NUM_TRIALS, 3))
    x = np.hstack([position, attitude, velocity])

    thrust = rng.uniform(0.0, MAX_THRUST, size=[NUM_TRIALS, 1])
    body_rates = rng.uniform(-MAX_BODY_RATES, MAX_BODY_RATES, size=(NUM_TRIALS, 3))
    u = np.hstack([thrust, body_rates])

    tx = rng.uniform(-10.0, 10.0, size=(NUM_TRIALS, 10))
    tu = rng.uniform(-10.0, 10.0, size=(NUM_TRIALS, 4))
    return x, u, tx, tu


@pytest.mark.parametrize("dev", devices, ids=map(str, devices))
def test_quadrotor_dynamics(test_data, dev):
    x, u = test_data[:2]

    x_dev, u_dev = (jax.device_put(it, dev) for it in test_data[:2])

    result = jax.vmap(quad.quadrotor_dynamics)(x_dev, u_dev)
    expected = np.array([quadrotor_dynamics(*args).squeeze() for args in zip(x, u)])

    assert result == pytest.approx(expected, rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("dev", devices, ids=map(str, devices))
def test_quadrotor_pushforward(test_data, dev):
    quadrotor_state = test_data[:2]
    tangents = test_data[2:]

    quadrotor_state_dev = [jax.device_put(it, dev) for it in quadrotor_state]
    tangents_dev = [jax.device_put(it, dev) for it in tangents]

    result = jax.jvp(quad.quadrotor_dynamics, quadrotor_state_dev, tangents_dev)[1]
    expected = np.array(
        [quadrotor_pushforward(*args).squeeze() for args in zip(*test_data)]
    )

    assert result == pytest.approx(expected, rel=5e-3, abs=1e-5)


@pytest.mark.parametrize("dev", devices, ids=map(str, devices))
def test_quadrotor_jacobian(test_data, dev):

    # TODO(Hs293Go): Fix vmapping Jacobian and remove slicing
    x, u = (it[:100, :] for it in test_data[:2])

    x_dev, u_dev = (jax.device_put(it[:100, :], dev) for it in test_data[:2])
    result = np.stack(
        [jax.jacfwd(quad.quadrotor_dynamics)(*args) for args in zip(x_dev, u_dev)]
    )
    expected = np.array([quadrotor_fjac(*args).squeeze() for args in zip(x, u)])

    assert result == pytest.approx(expected, rel=5e-3, abs=1e-5)

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

jvp_sym = dx_sym.jacobian(x_sym) @ tx_sym + dx_sym.jacobian(u_sym) @ tu_sym

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


MAX_POSITION = np.full([3], 100.0)
MAX_VELOCITY = np.full([3], 10.0)
MAX_THRUST = 80.0
MAX_BODY_RATES = np.full([3], 8.0)

NUM_TRIALS = 10000


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


def test_quadrotor_dynamics(test_data):
    quadrotor_state = test_data[:2]

    dx_result = jax.vmap(quad.quadrotor_dynamics)(*quadrotor_state)
    dx_expected = np.array(
        [quadrotor_dynamics(*args).squeeze() for args in zip(*quadrotor_state)]
    )

    assert dx_expected == pytest.approx(dx_result, rel=1e-4, abs=1e-5)


def test_quadrotor_pushforward(test_data):
    quadrotor_state = test_data[:2]
    tangents = test_data[2:]

    jvp_result = jax.jvp(quad.quadrotor_dynamics, quadrotor_state, tangents)[1]
    jvp_expected = np.array(
        [quadrotor_pushforward(*args).squeeze() for args in zip(*test_data)]
    )

    assert jvp_expected == pytest.approx(jvp_result, rel=5e-3, abs=1e-5)

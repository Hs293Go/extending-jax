import numpy as np
import sympy as sp

import quadrotor_dynamics_jax as quad


def quaternion_product(lhs, rhs):
    return sp.Matrix(
        [
            lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[3] * rhs[1] + lhs[1] * rhs[3] + lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[3] * rhs[2] + lhs[2] * rhs[3] + lhs[0] * rhs[1] - lhs[1] * rhs[0],
            lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2],
        ]
    )


def quaternion_rotate_point(quaternion, point):
    vec = quaternion[0:3]
    uv = vec.cross(point)
    uv += uv
    k = quaternion @ quaternion
    return k * point + quaternion[3] * uv + vec.cross(uv)


def symbolic_quadrotor_dynamics(x, u):
    return sp.Matrix(
        [
            x[7:10],
            quaternion_product(x[3:7], (u[1:4] / 2).row_join(sp.Matrix.zeros(1))),
            quaternion_rotate_point(x[3:7], sp.Matrix([0.0, 0.0, u[0]]))
            + sp.Matrix([0.0, 0.0, -9.81]),
        ]
    )


x_sym = sp.Matrix(sp.MatrixSymbol("x", 10, 1))
u_sym = sp.Matrix(sp.MatrixSymbol("u", 4, 1))

dx_sym = symbolic_quadrotor_dynamics(x_sym, u_sym)

quadrotor_dynamics = sp.lambdify((x_sym, u_sym), dx_sym)

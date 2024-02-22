import sympy as sp


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
    vec = quaternion[0:3, 0]
    uv = vec.cross(point)
    uv += uv
    k = quaternion.dot(quaternion)
    return k * point + quaternion[3] * uv + vec.cross(uv)

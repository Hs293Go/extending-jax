# -*- coding: utf-8 -*-

__all__ = ["quadrotor_dynamics"]

import functools
import itertools

import jax
import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jax_geometry import rotation as R
from jaxlib import hlo_helpers

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")

# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


_quadrotor_dynamics_p = core.Primitive("quadrotor_dynamics")
_quadrotor_dynamics_p.multiple_results = False
_quadrotor_dynamics_p.def_impl(
    functools.partial(xla.apply_primitive, _quadrotor_dynamics_p)
)


# This function exposes the primitive to user code and this is the only
# public-facing function in this module
def quadrotor_dynamics(x, u):
    return _quadrotor_dynamics_p.bind(x, u)


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
@_quadrotor_dynamics_p.def_abstract_eval
def _(x_aval: ShapedArray, u_aval: ShapedArray):
    if x_aval.dtype != u_aval.dtype:
        raise ValueError(f"Dtype mismatch: {x_aval.dtype} vs {u_aval.dtype}")
    return ShapedArray(x_aval.shape, x_aval.dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _quadrotor_dynamics_lowering(ctx, x, u, platform, *args, **kw):

    # Checking that input types and shape agree
    if x.type.element_type != u.type.element_type:
        raise TypeError(
            f"Type mismatch: {x.type.element_type} != {u.type.element_type}"
        )

    # Extract the numpy type of the inputs
    x_aval, u_aval = ctx.avals_in
    np_dtype = np.dtype(x_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(x.type)
    dims = dtype.shape
    ndims = len(dims)
    layout = tuple(range(ndims - 1, -1, -1))

    # The total size of the input is the product across dimensions
    size = dims[0] if ndims > 1 else 1

    # We dispatch a different call depending on the dtype
    if np_dtype not in (np.float32, np.float64):
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")
    op_name = f"{platform}_quadrotor_dynamics_f{np_dtype.itemsize * 8}"

    custom_call = functools.partial(
        hlo_helpers.custom_call, op_name, result_types=[dtype], result_layouts=[layout]
    )

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return custom_call(
            operands=[mlir.ir_constant(size), x, u],
            operand_layouts=[()] + default_layouts(x.type.shape, u.type.shape),
        ).results

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'quadrotor_dynamics_jax' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_descriptor(size)

        return custom_call(
            operands=[x, u],
            operand_layouts=default_layouts(x.type.shape, u.type.shape),
            backend_config=opaque,
        ).results

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _quadrotor_dynamics_batch(args, axes):
    assert axes[0] == axes[1]
    return quadrotor_dynamics(*args), axes[0]


batching.primitive_batchers[_quadrotor_dynamics_p] = _quadrotor_dynamics_batch

_quadrotor_jvp_p = core.Primitive("quadrotor_jvp")
_quadrotor_jvp_p.def_impl(functools.partial(xla.apply_primitive, _quadrotor_jvp_p))


def quadrotor_jvp(primals, tangents):
    (x, u), (tx, tu) = primals, tangents
    dx = quadrotor_dynamics(x, u)
    if isinstance(tu, ad.Zero):
        tu = jax.lax.zeros_like_array(u)
    jvp = _quadrotor_jvp_p.bind(x, u, tx, tu)
    return dx, jvp


ad.primitive_jvps[_quadrotor_dynamics_p] = quadrotor_jvp


@_quadrotor_jvp_p.def_abstract_eval
def _(x, u, tx, tu):
    return ShapedArray(tx.shape, tx.dtype)


def _quadrotor_jvp_lowering(ctx, x, u, tx, tu, platform, *args, **kw):

    # Checking that input types and shape agree
    for l, r in itertools.combinations([x, u, tx, tu], 2):
        if l.type.element_type != r.type.element_type:
            raise TypeError(
                f"Type mismatch: {l.type.element_type} != {r.type.element_type}"
            )

    # Extract the numpy type of the inputs
    x_aval, *_ = ctx.avals_in
    np_dtype = np.dtype(x_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(x.type)
    dims = dtype.shape
    ndims = len(dims)

    # The total size of the input is the product across dimensions
    size = dims[0] if ndims > 1 else 1

    # We dispatch a different call depending on the dtype
    if np_dtype not in (np.float32, np.float64):
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    custom_call = functools.partial(
        hlo_helpers.custom_call,
        f"{platform}_quadrotor_pushforward_f{np_dtype.itemsize * 8}",
        result_types=[dtype],
        result_layouts=default_layouts(x.type.shape),
    )

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return custom_call(
            operands=[mlir.ir_constant(size), x, u, tx, tu],
            operand_layouts=[()]
            + default_layouts(x.type.shape, u.type.shape, tx.type.shape, tu.type.shape),
        ).results

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'quadrotor_dynamics_jax' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_descriptor(size)

        return custom_call(
            operands=[x, u, tx, tu],
            operand_layouts=default_layouts(
                x.type.shape, u.type.shape, tx.type.shape, tu.type.shape
            ),
            backend_config=opaque,
        ).results

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


ad.primitive_jvps[_quadrotor_dynamics_p] = quadrotor_jvp


def _quadrotor_jvp_batch(args, axes):
    batch_len = [it.shape[0] for it, dim in zip(args, axes) if dim is not None]

    args = tuple(
        jnp.moveaxis(arg, dim, 0) if dim is not None else arg
        for arg, dim in zip(args, axes)
    )
    args = tuple(
        (
            jnp.broadcast_to(arg, shape=(batch_len[0],) + arg.shape)
            if dim is None
            else arg
        )
        for arg, dim in zip(args, axes)
    )

    return _quadrotor_jvp_p.bind(*args), 0


batching.primitive_batchers[_quadrotor_jvp_p] = _quadrotor_jvp_batch

# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.


# Connect the XLA translation rules for JIT compilation
for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _quadrotor_dynamics_p,
        functools.partial(_quadrotor_dynamics_lowering, platform=platform),
        platform=platform,
    )

    mlir.register_lowering(
        _quadrotor_jvp_p,
        functools.partial(_quadrotor_jvp_lowering, platform=platform),
        platform=platform,
    )

# Connect the JVP and batching rules

# def _quadrotor_dynamics_jvp(args, tangents):
#     # mean_anom, ecc = args
#     # d_mean_anom, d_ecc = tangents
#     x, u = args
#     tx, tu = tangents

#     # # We use "bind" here because we don't want to mod the mean anomaly again

#     q = x[3:7]
#     f = u[0]

#     if isinstance(tu, ad.Zero):
#         jvp_u = jnp.zeros(10)
#     else:

#         ux_13_q = jnp.append(tu[1:4] / 2.0, 0.0)
#         jvp_u = jnp.concatenate(
#             [
#                 jnp.zeros(3),
#                 quaternion_product(q, ux_13_q),
#                 quaternion_rotate_point(q, jnp.array([0.0, 0.0, 1.0])) * tu[0],
#             ]
#         )
#     if isinstance(tx, ad.Zero):
#         jvp_x = jnp.zeros(10)
#     else:
#         omega_q = jnp.append(u[1:4] / 2.0, 0.0)

#         i3_x_tx_35 = jnp.array([-tx[4], tx[3], 0.0])
#         qv_x_i3 = jnp.array([q[1], -q[0], 0.0])
#         jvp_x = jnp.concatenate(
#             [
#                 tx[7:10],
#                 quaternion_product(tx[3:7], omega_q),
#                 2.0
#                 * f
#                 * (
#                     q[2] * tx[3:6]
#                     - jnp.cross(qv_x_i3, tx[3:6])
#                     - q[3] * i3_x_tx_35
#                     + (qv_x_i3 + jnp.array([0.0, 0.0, q[3]])) * tx[6]
#                 ),
#             ]
#         )

#     primals = _quadrotor_dynamics_p.bind(*args)
#     return primals, jvp_x + jvp

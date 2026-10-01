# SPDX-License-Identifier: Apache-2.0
"""Activation-weighted MLX affine packing, shared by text and diffusion."""

import mlx.core as mx


def _affine_minmax_params(grouped, bits: int):
    n_bins = mx.array((1 << bits) - 1, mx.float32)
    eps = mx.array(1e-7, mx.float32)
    zero = mx.array(0.0, mx.float32)
    w_max = mx.max(grouped, axis=-1, keepdims=True).astype(mx.float32)
    w_min = mx.min(grouped, axis=-1, keepdims=True).astype(mx.float32)
    mask = mx.abs(w_min) > mx.abs(w_max)
    scales = mx.maximum((w_max - w_min) / n_bins, eps)
    scales = mx.where(mask, scales, -scales)
    edge = mx.where(mask, w_min, w_max)
    q0 = mx.round(edge / scales)
    scales = mx.where(q0 != zero, edge / q0, scales)
    biases = mx.where(q0 == zero, zero, edge)
    return scales, biases


def _pack_affine_codes(w, scales, biases, group_size: int, bits: int):
    orig = tuple(w.shape)
    grouped = w.reshape(-1, orig[-1] // group_size, group_size)
    n_bins = mx.array((1 << bits) - 1, mx.float32)
    codes = mx.clip(mx.round((grouped - biases) / scales), 0, n_bins).astype(mx.uint32)

    el_per_int = 32 // bits
    if bits in (2, 4, 8):
        shifts = mx.power(
            mx.array(2, mx.uint32),
            mx.arange(0, 32, bits, dtype=mx.uint32),
        )
        packed = codes.reshape(codes.shape[0], -1, el_per_int)
        packed = mx.sum(packed * shifts, axis=2)
    else:
        bits_arange = mx.arange(bits, dtype=mx.uint32)
        bit_values = mx.bitwise_and(mx.right_shift(codes[..., None], bits_arange), 1)
        bit_values = bit_values.reshape(codes.shape[0], -1, 32)
        shifts = mx.arange(32, dtype=mx.uint32)
        packed = mx.sum(mx.left_shift(bit_values, shifts), axis=-1)

    packed_shape = (*orig[:-1], orig[-1] * bits // 32)
    return packed.reshape(packed_shape)


def weighted_affine_quantize(w, group_size: int, bits: int, importance):
    """Quantize with a small imatrix-weighted clipping search.

    The output layout intentionally matches ``mx.quantize(..., mode="affine")``.
    """
    orig = tuple(w.shape)
    grouped = w.reshape(-1, orig[-1] // group_size, group_size)
    grouped_f = grouped.astype(mx.float32)

    imp = importance
    if not isinstance(imp, mx.array):
        imp = mx.array(imp)
    if tuple(imp.shape) == (orig[-1],):
        imp = mx.broadcast_to(imp, orig)
    elif len(orig) >= 3 and tuple(imp.shape) == (orig[0], orig[-1]):
        imp = mx.broadcast_to(imp[:, None, :], orig)
    else:
        imp = mx.broadcast_to(imp, orig)
    imp = imp.reshape(grouped.shape).astype(mx.float32)
    imp = mx.maximum(imp, mx.array(1e-8, mx.float32))

    base_scales, base_biases = _affine_minmax_params(grouped_f, bits)
    best_scales = base_scales
    best_biases = base_biases
    best_codes = mx.clip(
        mx.round((grouped_f - base_biases) / base_scales),
        0,
        mx.array((1 << bits) - 1, mx.float32),
    )
    best_err = mx.sum(
        imp * (grouped_f - (best_codes * base_scales + base_biases)) ** 2,
        axis=-1,
        keepdims=True,
    )

    n_bins = mx.array((1 << bits) - 1, mx.float32)
    eps = mx.array(1e-7, mx.float32)
    w_min = mx.min(grouped_f, axis=-1, keepdims=True)
    w_max = mx.max(grouped_f, axis=-1, keepdims=True)
    for edge, opposite, sign in ((w_max, w_min, -1.0), (w_min, w_max, 1.0)):
        raw = mx.maximum(mx.abs(edge - opposite) / n_bins, eps) * sign
        q0 = mx.round(edge / raw)
        scale0 = mx.where(q0 != 0, edge / q0, raw)
        bias0 = mx.where(q0 == 0, mx.array(0.0, mx.float32), edge)
        for factor in (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25):
            scales = scale0 * mx.array(factor, mx.float32)
            biases = bias0
            codes = mx.clip(mx.round((grouped_f - biases) / scales), 0, n_bins)
            recon = codes * scales + biases
            err = mx.sum(imp * (grouped_f - recon) ** 2, axis=-1, keepdims=True)
            take = err < best_err
            best_err = mx.where(take, err, best_err)
            best_scales = mx.where(take, scales, best_scales)
            best_biases = mx.where(take, biases, best_biases)

    packed = _pack_affine_codes(
        grouped_f.reshape(orig), best_scales, best_biases, group_size, bits
    )
    scale_shape = (*orig[:-1], orig[-1] // group_size)
    scales = best_scales.reshape(scale_shape).astype(w.dtype)
    biases = best_biases.reshape(scale_shape).astype(w.dtype)
    mx.eval(packed, scales, biases)
    return packed, scales, biases

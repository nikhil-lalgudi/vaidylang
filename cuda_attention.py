"""Tiled BF16 attention for one decode token."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(Input, cols, cos, sin, WIDTH: tl.constexpr):
    values = tl.load(Input + cols).to(tl.float32)
    paired = tl.load(Input + (cols + WIDTH // 2) % WIDTH).to(tl.float32)
    return (values * cos + paired * sin).to(tl.bfloat16)


@triton.jit
def _attention_tiles(
    Q, NewK, NewV, CacheK, CacheV, Position, Cos, Sin, Partial, Stats,
    KV_HEADS: tl.constexpr, GROUP: tl.constexpr, WIDTH: tl.constexpr,
    TILES: tl.constexpr, TILE: tl.constexpr, SCALE: tl.constexpr,
):
    head, tile = tl.program_id(0), tl.program_id(1)
    pos = tl.load(Position)
    start = tile * TILE
    if start <= pos:
        kv_head = head // GROUP
        cols = tl.arange(0, WIDTH)
        positions = start + tl.arange(0, TILE)
        rope = pos * (WIDTH // 2) + cols % (WIDTH // 2)
        cos = tl.load(Cos + rope).to(tl.float32)
        sin = tl.load(Sin + rope).to(tl.float32) * tl.where(cols < WIDTH // 2, -1, 1)
        q = _rotate(Q + head * WIDTH, cols, cos, sin, WIDTH)
        k = _rotate(NewK + kv_head * WIDTH, cols, cos, sin, WIDTH)
        v = tl.load(NewV + kv_head * WIDTH + cols)

        # Each query head consumes the new K/V directly, so cache appends need no
        # cross-program synchronization. One query head writes each KV head.
        offsets = (positions[:, None] * KV_HEADS + kv_head) * WIDTH + cols[None, :]
        past = positions < pos
        current = positions == pos
        keys = tl.load(CacheK + offsets, past[:, None], 0)
        keys = tl.where(current[:, None], k[None, :], keys)
        values = tl.load(CacheV + offsets, past[:, None], 0)
        values = tl.where(current[:, None], v[None, :], values)
        if head % GROUP == 0 and start + TILE > pos:
            append = (pos * KV_HEADS + kv_head) * WIDTH + cols
            tl.store(CacheK + append, k)
            tl.store(CacheV + append, v)

        scores = tl.sum(keys.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * SCALE
        scores = tl.where(positions <= pos, scores, float("-inf"))
        maximum = tl.max(scores, axis=0)
        probs = tl.exp(scores - maximum)
        normalizer = tl.sum(probs, axis=0)
        probs = probs.to(tl.bfloat16).to(tl.float32)
        partial = tl.sum(probs[:, None] * values.to(tl.float32), axis=0)
        index = head * TILES + tile
        tl.store(Partial + index * WIDTH + cols, partial)
        tl.store(Stats + index * 2, maximum)
        tl.store(Stats + index * 2 + 1, normalizer)


@triton.jit
def _combine_tiles(
    Partial, Stats, Position, Out, TILES: tl.constexpr, WIDTH: tl.constexpr,
    TILE: tl.constexpr, BLOCK_TILES: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    head = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    pos = tl.load(Position)
    valid = (tiles < TILES) & (tiles * TILE <= pos)
    index = head * TILES + tiles
    maximum = tl.load(Stats + index * 2, valid, float("-inf"))
    normalizer = tl.load(Stats + index * 2 + 1, valid, 0.0)
    correction = tl.exp(maximum - tl.max(maximum, axis=0))
    total = tl.sum(normalizer * correction, axis=0)
    values = tl.load(Partial + index[:, None] * WIDTH + cols[None, :], valid[:, None], 0.0)
    output = tl.sum(values * correction[:, None], axis=0) / total
    tl.store(Out + head * WIDTH + cols, output)


def tiled_decode_attention(query, keys, values, key, value, position, cos, sin):
    """Attend to a contiguous batch-one KV cache and append the current K/V."""
    capacity, kv_heads, width = keys.shape[1:]
    heads = query.shape[-2]
    tile = 32 if capacity <= 256 else 64
    tiles = triton.cdiv(capacity, tile)
    block_tiles = triton.next_power_of_2(tiles)
    block_cols = min(width, max(16, 16384 // block_tiles))
    partial = torch.empty((heads, tiles, width), device=query.device, dtype=torch.float32)
    stats = torch.empty((heads, tiles, 2), device=query.device, dtype=torch.float32)
    output = torch.empty_like(query)
    _attention_tiles[(heads, tiles)](
        query, key, value, keys, values, position, cos, sin, partial, stats,
        kv_heads, heads // kv_heads, width, tiles, tile, width ** -0.5, num_warps=4,
    )
    _combine_tiles[(heads, width // block_cols)](
        partial, stats, position, output, tiles, width, tile, block_tiles, block_cols, num_warps=4,
    )
    return output

"""Tiled BF16 attention for batch-one decode and block verification."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(Input, cols, cos, sin, WIDTH: tl.constexpr):
    values = tl.load(Input + cols).to(tl.float32)
    paired = tl.load(Input + (cols + WIDTH // 2) % WIDTH).to(tl.float32)
    return (values * cos + paired * sin).to(tl.bfloat16)


@triton.jit
def _append_block(
    Q, NewK, NewV, CacheK, CacheV, Position, Cos, Sin, RotatedQ,
    HEADS: tl.constexpr, KV_HEADS: tl.constexpr, WIDTH: tl.constexpr, V_STRIDE: tl.constexpr,
):
    head, row = tl.program_id(0), tl.program_id(1)
    pos = tl.load(Position) + row
    cols = tl.arange(0, WIDTH)
    rope = pos * (WIDTH // 2) + cols % (WIDTH // 2)
    cos = tl.load(Cos + rope).to(tl.float32)
    sin = tl.load(Sin + rope).to(tl.float32) * tl.where(cols < WIDTH // 2, -1, 1)
    if head < HEADS:
        query_offset = (row * HEADS + head) * WIDTH
        tl.store(RotatedQ + query_offset + cols, _rotate(Q + query_offset, cols, cos, sin, WIDTH))
    else:
        kv_head = head - HEADS
        key = _rotate(NewK + (row * KV_HEADS + kv_head) * WIDTH, cols, cos, sin, WIDTH)
        value = tl.load(NewV + row * V_STRIDE + kv_head * WIDTH + cols)
        cache_offset = (pos * KV_HEADS + kv_head) * WIDTH + cols
        tl.store(CacheK + cache_offset, key)
        tl.store(CacheV + cache_offset, value)


@triton.jit
def _attention_tiles(
    Q, NewK, NewV, CacheK, CacheV, Position, Cos, Sin, Partial, Stats,
    KV_HEADS: tl.constexpr, GROUP: tl.constexpr, WIDTH: tl.constexpr,
    TILES: tl.constexpr, TILE: tl.constexpr, SCALE: tl.constexpr, VERIFY: tl.constexpr,
):
    head, tile = tl.program_id(0), tl.program_id(1)
    row = tl.program_id(2) if VERIFY else 0
    query_head = row * KV_HEADS * GROUP + head
    pos = tl.load(Position) + row
    start = tile * TILE
    if start <= pos:
        kv_head = head // GROUP
        cols = tl.arange(0, WIDTH)
        positions = start + tl.arange(0, TILE)
        offsets = (positions[:, None] * KV_HEADS + kv_head) * WIDTH + cols[None, :]
        if VERIFY:
            q = tl.load(Q + query_head * WIDTH + cols)
            valid = positions <= pos
            keys = tl.load(CacheK + offsets, valid[:, None], 0)
            values = tl.load(CacheV + offsets, valid[:, None], 0)
        else:
            rope = pos * (WIDTH // 2) + cols % (WIDTH // 2)
            cos = tl.load(Cos + rope).to(tl.float32)
            sin = tl.load(Sin + rope).to(tl.float32) * tl.where(cols < WIDTH // 2, -1, 1)
            q = _rotate(Q + head * WIDTH, cols, cos, sin, WIDTH)
            k = _rotate(NewK + kv_head * WIDTH, cols, cos, sin, WIDTH)
            v = tl.load(NewV + kv_head * WIDTH + cols)
            # Each query head consumes the new K/V directly. One head writes it.
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
        index = query_head * TILES + tile
        tl.store(Partial + index * WIDTH + cols, partial)
        tl.store(Stats + index * 2, maximum)
        tl.store(Stats + index * 2 + 1, normalizer)


@triton.jit
def _combine_tiles(
    Partial, Stats, Position, Out, TILES: tl.constexpr, WIDTH: tl.constexpr,
    TILE: tl.constexpr, BLOCK_TILES: tl.constexpr, BLOCK_COLS: tl.constexpr,
    HEADS: tl.constexpr, VERIFY: tl.constexpr,
):
    row = tl.program_id(2) if VERIFY else 0
    head = row * HEADS + tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    pos = tl.load(Position) + row
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
    """Append K/V and apply causal attention to one or more decode rows."""
    capacity, kv_heads, width = keys.shape[1:]
    rows, heads = query.shape[:2]
    verify = rows > 1
    tile = 32 if capacity <= 256 else 64
    tiles = triton.cdiv(capacity, tile)
    block_tiles = triton.next_power_of_2(tiles)
    block_cols = min(width, max(16, 16384 // block_tiles))
    partial = torch.empty((rows, heads, tiles, width), device=query.device, dtype=torch.float32)
    stats = torch.empty((rows, heads, tiles, 2), device=query.device, dtype=torch.float32)
    output = torch.empty_like(query)
    if verify:
        rotated = torch.empty_like(query)
        _append_block[(heads + kv_heads, rows)](
            query, key, value, keys, values, position, cos, sin, rotated,
            heads, kv_heads, width, value.stride(0), num_warps=4,
        )
        query = rotated
    _attention_tiles[(heads, tiles, rows)](
        query, key, value, keys, values, position, cos, sin, partial, stats,
        kv_heads, heads // kv_heads, width, tiles, tile, width ** -0.5, verify, num_warps=4,
    )
    _combine_tiles[(heads, width // block_cols, rows)](
        partial, stats, position, output, tiles, width, tile, block_tiles, block_cols, heads, verify,
        num_warps=4,
    )
    return output

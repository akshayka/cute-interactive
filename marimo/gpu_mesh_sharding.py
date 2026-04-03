import marimo

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "tensor-layouts",
# ]
# ///

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Representing GPU Mesh Sharding with CuTe Layouts

    On GPUs, device IDs typically follow a simple linear order (no topology permutation).
    This makes the mesh layout a plain row-major `Layout` — no `Swizzle` needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The setup

    8 GPUs arranged as a 4×2 mesh, sharding an 8×4 array with `P('a', 'b')`.
    Each GPU gets a contiguous 2×2 tile.

    ```
    ┌──────────┬──────────┐
    │  GPU 0   │  GPU 1   │
    ├──────────┼──────────┤
    │  GPU 2   │  GPU 3   │
    ├──────────┼──────────┤
    │  GPU 4   │  GPU 5   │
    ├──────────┼──────────┤
    │  GPU 6   │  GPU 7   │
    └──────────┴──────────┘
    ```

    Device IDs are in plain row-major order — no swizzling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: The device mesh

    A 4×2 row-major layout directly maps mesh coordinates to GPU IDs: `(i,j) → 2i + j`.
    """)
    return


@app.cell
def _():
    from tensor_layouts import Layout, compose, blocked_product
    from tensor_layouts.viz import draw_layout
    device_mesh = Layout((4, 2), (2, 1))
    print(device_mesh)
    for _i in range(4):
        row = [f'GPU {device_mesh(_i, j)}' for j in range(2)]
        print(f'  row {_i}: {row}')
    return Layout, blocked_product, device_mesh, draw_layout


@app.cell
def _(device_mesh, draw_layout):
    draw_layout(device_mesh, colorize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Sharding an 8×4 array with contiguous local chunks

    Each GPU gets a 2×2 row-major tile (PyTorch convention: biggest stride on the left).
    `blocked_product` tiles the local chunk across device positions, pairing corresponding
    modes: row with row, column with column.
    """)
    return


@app.cell
def _(Layout, blocked_product, device_mesh):
    # Local tile: 2x2 row-major (contiguous 4 elements per device)
    local_tile = Layout((2, 2), (2, 1))
    full = blocked_product(local_tile, device_mesh)
    # blocked_product pairs modes: ((local_row, device_row), (local_col, device_col))
    print('Layout:', full)
    print()
    for _i in range(8):
        offsets = [f'{full(_i, j):2d}' for j in range(4)]
    # Print the 8x4 grid with device assignments
        devices = [f'd{full(_i, j) // 4}' for j in range(4)]
        print(f"  row {_i}: offsets [{', '.join(offsets)}]  devices [{', '.join(devices)}]")
    return (full,)


@app.cell
def _(Layout, blocked_product, device_mesh, draw_layout, full):
    # Color by device ID
    color = blocked_product(Layout((2, 2), (0, 0)), device_mesh)

    draw_layout(full, flatten_hierarchical=False, color_layout=color, colorize=True, num_colors=8)
    return (color,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each cell shows the hierarchical coordinates from `blocked_product`'s shape `((2, 4), (2, 2))`:

    - **row** = `(local_row, device_row)` — first component is position within the 2×2 tile, second is which device row (0–3)
    - **col** = `(local_col, device_col)` — first component is position within the tile, second is which device column (0–1)
    - **offset** = linearized physical offset

    Without the topology swizzle, offsets are simply `device_id * 4 + local_offset` in plain order.

    ## Step 3: Top-down — partitioning a global array with `zipped_divide`

    `blocked_product` works **bottom-up**: given a local tile and a device mesh, it builds the global layout.
    But what if we start with an existing global array and want to partition it into tiles?

    `zipped_divide(global_array, tile_shape)` splits the array into `(tile, grid)`:
    the first mode indexes within a tile, the second mode selects which tile (i.e., which device).

    The key difference: `blocked_product` **creates** a layout where each device's chunk is contiguous,
    while `zipped_divide` **preserves** the original memory layout (here, row-major).
    """)
    return


@app.cell
def _(Layout):
    from tensor_layouts import zipped_divide

    # Start with an 8x4 row-major global array
    global_array = Layout((8, 4), (4, 1))
    print("Global array:", global_array)
    print()

    # Partition into 2x2 tiles: result is ((tile), (grid))
    tiled = zipped_divide(global_array, Layout((2, 2)))
    print("zipped_divide result:", tiled)
    print("  Shape:", tiled.shape)
    print("  Mode 0 (tile):  shape", tiled.shape[0])
    print("  Mode 1 (grid):  shape", tiled.shape[1])
    print()

    # Show what each grid position (device) gets
    for g in range(8):
        tile_offsets = sorted([tiled(t, g) for t in range(4)])
        print(f"  grid {g} (device {g}): offsets {tile_offsets}")
    return tiled, zipped_divide


@app.cell
def _(Layout, color, draw_layout, tiled, zipped_divide):
    # Visualize: same 8x4 grid, but now from zipped_divide
    color_td = zipped_divide(color, Layout((2, 2)))
    draw_layout(tiled, flatten_hierarchical=False, color_layout=color_td, colorize=True, num_colors=8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing the two approaches

    | | `blocked_product` (bottom-up) | `zipped_divide` (top-down) |
    |---|---|---|
    | **Input** | local tile + device mesh | global array + tile shape |
    | **Memory layout** | Each device's chunk is contiguous (offsets 0–3, 4–7, ...) | Preserves original row-major layout (a 2×2 tile spans rows 4 apart) |
    | **Use case** | Designing a distributed memory layout | Partitioning an existing array across devices |

    Both produce the same logical tiling (same 2×2 blocks assigned to same devices), but the physical offsets differ because the memory layouts are different.
    """)
    return


if __name__ == "__main__":
    app.run()

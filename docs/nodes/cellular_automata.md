# Cellular Automata (`CompositorNodeLabCellularAutomata`, menu Simulate)

A grid of cells advanced by Life-like and Generations rules, one step per frame. A stateful node: it
shares the state machinery of the other Simulate nodes (see
[lib/README.md](../../addons/compositor_lab/lib/README.md#stateful-nodes)): Reset button, scrub-back
cache, catch-up after jumps, Max Catch-up and Cached Frames in the sidebar.

## Inputs

| Socket | Default | Meaning |
|---|---|---|
| Seed | unlinked | Image seeding the grid on the first frame: a cell is alive where the luminance (Rec. 709) at its centre pixel is greater than Threshold. Unlinked: random cells |
| Inject | 0 | Painted mask: every step, cells where its luminance is > 0.5 become alive (a dying or dead cell is revived with age 0, alive cells are untouched). Animate a mask to paint into the simulation. A single value above 0.5 fills the grid |
| Threshold | 0.5 | Seed luminance threshold |
| Density | 0.35 | Fraction of alive cells when Seed is unlinked |
| Random Seed | 0 | Seed of the shared PCG hash (`lab_rand`) that decides which cells start alive: cell (x, y) is alive when `rand(x, y, seed) < Density`, identical on CPU and GPU |
| Generations per Frame | 1 | Generations run each frame (0 only injects) |
| Cell Size | 2 | Pixels per cell. The grid has `ceil(width / size) x ceil(height / size)` cells and covers the image; changing it restarts the state |

## Properties

* **Rule**: a preset (Life `B3/S23`, HighLife `B36/S23`, Seeds `B2/S`, Day & Night `B3678/S34678`,
  Brian's Brain `B2/S/3`, Star Wars `B2/S345/4`, Maze `B3/S12345`) or **Custom** with a rule string.
  Choosing a preset copies its string into the field, so it can be edited as Custom.
* **Rule string**: `B<digits>/S<digits>` with an optional third part for the number of states C
  (`/3`, `/C4` or `/G4`; Generations). Case and spaces are ignored, the digits are neighbour counts
  0-8 (counts above 4 never occur with von Neumann), either part may be empty, the parts may come as
  `S../B..`. C is 2-256 (default 2). Anything else (`B3/X23`, `B9/S1`, `B3/S23/1`...) is an error:
  the node shows the message and outputs default values (the field turns red in the node).
* **Neighbourhood**: Moore (8 neighbours, default) or von Neumann (4). **Edges**: Wrap (torus,
  default) or Dead.
* **Pre-roll**: generations run on the seed when the state starts (the stateless fallback: a single
  still render shows an evolved pattern; 0 = the first frame is the seed).
* **Age Range** (20): the age that maps to the end of the Age output and of the colour gradient.
  **Color A / B / C** (newborn / middle / old) and **Background**.

## Semantics

A cell has state `s` in 0..C-1: 0 dead, 1 alive, 2..C-1 dying. Only state-1 cells count as
neighbours.

* dead: becomes alive if its alive-neighbour count is in B;
* alive: stays alive if the count is in S, otherwise starts dying (state 2), or dies when C = 2;
* dying: `s + 1` each generation, dead again after C - 1.

**Age** is the number of generations since birth (0 at birth, +1 per generation while not dead, 0 when
dead; saturates at 65535). One frame = inject, then Generations per Frame generations. The first
frame (and any frame at or before the scene start) is the seed (after the injection and Pre-roll).
Row 0 is the bottom: a glider `.O. / ..O / OOO` listed as rows y = 0, 1, 2 moves (+1, +1) in
(x, y) every 4 generations.

State: `(uint8 states, uint16 ages)` arrays on the CPU, one RG32F texture (r state, g age) on the GPU,
at grid size (memory scales with the grid, not the image). All of it is integer logic: the CPU and
GPU results are bit-identical.

## Outputs

* **Cells** (float): alive 1, dying `(C - s) / (C - 1)` (down to `1 / (C - 1)`), dead 0.
* **Age** (float): `min(age / Age Range, 1)`.
* **Color**: alive cells take the gradient Color A (age 0) - Color B (half of Age Range) - Color C
  (Age Range) on age; dying cells are Color C faded to the Background by the square of their
  Cells shade (so the fade reads evenly after the display transform); dead cells are the Background.

All outputs repeat each cell over its Cell Size block (nearest).

## Tests

`tests/lab/test_cellular_automata.py`: blinker, block, glider and one Brian's Brain step; an independent
reference (plain loops, written in the test) compared **exactly** with the CPU and GPU node for 20+
generations on random 64x48 grids for Life, HighLife, Seeds, Day & Night, Brian's Brain, Star Wars,
Maze, Moore / von Neumann, wrap / dead and custom Generations rules, states and ages; rule parser
accept / reject cases; seeding, inject, cell size, colour formulas; the stateful semantics (sequential,
re-render, rule tweak, scrub back, catch-up, hold, start frame, Reset, duplicate, pre-roll, memory
caps); CPU vs GPU on Cells / Age exactly and Color within 1e-5. Seed / Inject luminance is computed in
float32 on the CPU and as a GLSL `dot` on the GPU, so an image exactly on the threshold could differ
by one ulp (the tests use values clear of it).

## Ranges

Socket / property | soft range | clamp

* Inject: 0..1, none (luminance > 0.5 paints)
* Threshold: 0..1, none
* Density: 0..1, clamped to 0..1
* Random Seed: 0..1000, none
* Generations per Frame: 0..16, clamped to 0..1024
* Cell Size: 1..32, clamped to 1..256
* Pre-roll: hard 0..4096, soft max 256; Age Range: hard 1..65535, soft max 1000

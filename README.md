
## Paper figures (proposal deck)

Every figure of the "Python model / AO vs LJ" deck can be rebuilt from the run folders:

- In the app: **Output type → Paper figures**, set the BD runs folder (`BD runs v2`, with `ARF/` and `NAF/`)
  and the COMSOL csv folder (for figures 0 and 2). Each figure has a download button; a custom
  seed-band plot and an end-of-run table are in the other tabs.
- Command line: `python paper_figures.py --bd "<BD runs v2>" --comsol "<raw csv data>" --out figures`

Figures 6 / 6a (two-particle B₂ and AO force check) come from `simulation/verify_ao.py`.
The simulations themselves: `simulation/airpods_bd.py` (plans `validation`, `revision`, `ao`).

## Figure builder (editable paper figures)

**Output type → Paper figures → Figure builder.** Every figure is a recipe (JSON in `<Data>/_figures/`):
panels (a measure each, optionally only one d_n / A / F₀) and lines (which runs: regime, Python/COMSOL,
LJ/AO, A, d_n, F₀, length, q, φ_p, name tag; colour, line style, marker, seed band). Two kinds: over time, or
the end value / break-up time against a run parameter. *Add starter recipes* writes editable versions of the
deck figures. *Re-make all figures* redraws every saved recipe from the current runs, and so does

    python figure_builder.py --data "D:/Work/AIRPODS Paper 2627/Data" --out figures

The fixed deck figures (B₂* explainer, seeds, state map, g(r), snapshots, ...) are in the second tab.

## Run simulation (GUI for the Python Brownian dynamics)

**Output type → Run simulation** runs `simulation/airpods_bd.py` from the app:

- interaction: LJ (A, cut-off), AO (q, φ_p, or φ_p matched to an LJ depth by B₂*), or core only; core type and Z
- acoustic field on/off, d_n, and the force law (F₀, k)
- start: square grid, random with minimum distance, the COMSOL AAF band (regime III), any frame of an existing run, or a CSV
- run length, time step (automatic per potential), saved frames, seeds, runs in parallel, regime folder, name tag
- physics constants (box, temperature, viscosity, radius, force cap)
- live previews of U(r) with B₂*, the acoustic force and the start positions, a runtime estimate, and the run names
- a Jobs tab with progress per seed, time left and a stop button; runs continue if the page is closed

Every run folder gets `sim_config.json`; "Load settings from an existing run" puts those settings back into the form.
Without the app: `python simulation/sim_runner.py <job>/config.json`.

### Batch runs

**Run simulation → Batch**: pick a scenario (Regime I–IV), lists of A values and d_n (Regime II) or F₀
(Regime IV), LJ and/or AO (q list; AO is matched to each A by B₂*), run length, seeds and how many runs at the
same time. The tab lists every run (existing ones are skipped) with a time estimate, starts the whole batch as
one job (progress in the Jobs tab), and can save it to `<Data>/_batches/`. A saved batch also runs without the
app: `python simulation/sim_runner.py "<Data>/_batches/<name>.json"`. Keep "runs at the same time" below your
number of cores and don't run two jobs at once; each run is its own Python process.

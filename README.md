
## Paper figures (proposal deck)

Every figure of the "Python model / AO vs LJ" deck can be rebuilt from the run folders:

- In the app: **Output type → Paper figures**, set the BD runs folder (`BD runs v2`, with `ARF/` and `NAF/`)
  and the COMSOL csv folder (for figures 0 and 2). Each figure has a download button; a custom
  seed-band plot and an end-of-run table are in the other tabs.
- Command line: `python paper_figures.py --bd "<BD runs v2>" --comsol "<raw csv data>" --out figures`

Figures 6 / 6a (two-particle B₂ and AO force check) come from `simulation/verify_ao.py`.
The simulations themselves: `simulation/airpods_bd.py` (plans `validation`, `revision`, `ao`).

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

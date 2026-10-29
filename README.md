
## Paper figures (proposal deck)

Every figure of the "Python model / AO vs LJ" deck can be rebuilt from the run folders:

- In the app: **Output type → Paper figures**, set the BD runs folder (`BD runs v2`, with `ARF/` and `NAF/`)
  and the COMSOL csv folder (for figures 0 and 2). Each figure has a download button; a custom
  seed-band plot and an end-of-run table are in the other tabs.
- Command line: `python paper_figures.py --bd "<BD runs v2>" --comsol "<raw csv data>" --out figures`

Figures 6 / 6a (two-particle B₂ and AO force check) come from `simulation/verify_ao.py`.
The simulations themselves: `simulation/airpods_bd.py` (plans `validation`, `revision`, `ao`).

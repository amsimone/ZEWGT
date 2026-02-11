# zewgt

A command-line reweighter for neutral Drell-Yan LHE event files using GRIFFIN electroweak virtual corrections. It reads events from an input LHE file, reconstructs the underlying `qqbar -> Z` Born projection (using `# uub` when available, otherwise `#rwgt` + `FlavRegList` or incoming-parton fallback), computes an EW virtual weight per event, and writes the result into the output LHE as an additional weight entry.

Requirements:

- A local GRIFFIN build/source tree (headers + `libgriffin`) accessible on your machine.
- CMake >= 3.8 and a C++11 compiler.

Build:

```bash
cmake -S . -B build -DGRIFFIN_DIR=/path/to/GRIFFIN
cmake --build build -j
```

Run:

```bash
./build/zewgt \
  --input pwgevents.lhe \
  --output pwgevents_zewgt.lhe \
  --card /path/to/GRIFFIN/examples/ewvirt_card.dat \
  --flavreglist /path/to/POWHEG-BOX-V2/Zj/ZjMiNNLO/suggested-run/FlavRegList
```

Notes:

- If `# uub` lines are present in events, they are used as exact MiNNLO projection to `qqbar -> Z`.
- Without `# uub`, the code uses `#rwgt` + `FlavRegList` to map NLO (`type=1`) and NNLO-like (`type=2/3`) events to the underlying Born flavor flow.
- As a final fallback (when mapping is unavailable), it infers quark direction/flavor from incoming partons.
- Default behavior is strict (abort on unresolved events). Use `--skip-unresolved` to keep running and write weight `1.0` for unresolved events.

References:

- GRIFFIN: Y. Chen and A. Freitas, *GRIFFIN: a program for weak virtual corrections to Drell-Yan-like production of heavy bosons*, SciPost Phys. Codebases 18 (2023), [arXiv:2211.16272](https://arxiv.org/abs/2211.16272), [DOI:10.21468/SciPostPhysCodeb.18](https://doi.org/10.21468/SciPostPhysCodeb.18), [GitHub](https://github.com/lisongc/GRIFFIN) ([releases](https://github.com/lisongc/GRIFFIN/releases)).
- POWHEG Z: S. Alioli, P. Nason, C. Oleari, and E. Re, *NLO vector-boson production matched with shower in POWHEG*, JHEP 07 (2008) 060, [arXiv:0805.4802](https://arxiv.org/abs/0805.4802).
- POWHEG Z_ew: L. Chiesa, S. Dittmaier, A. Huss, M. Schulte, and C. Schwinn, *On electroweak corrections to neutral current Drell-Yan with the POWHEG BOX*, Eur. Phys. J. C 84, 539 (2024), [arXiv:2402.14659](https://arxiv.org/abs/2402.14659).
- Zj_MiNNLO / MiNNLOPS: P. F. Monni, E. Re, and P. Torrielli, *MiNNLOPS: a new method to match NNLO QCD to parton showers*, JHEP 05 (2020) 143, [arXiv:1908.06987](https://arxiv.org/abs/1908.06987); P. F. Monni, E. Re, and M. Wiesemann, *MiNNLOPS: optimizing 2 -> 1 hadronic processes*, Eur. Phys. J. C 80, 1075 (2020), [arXiv:2006.04133](https://arxiv.org/abs/2006.04133).

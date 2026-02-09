# LHE EW Virtual Reweighter (GRIFFIN)

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
./build/lhe_add_griffin_ewvirt \
  --input pwgevents.lhe \
  --output pwgevents_ewvirt.lhe \
  --card /path/to/GRIFFIN/examples/ewvirt_card.dat \
  --flavreglist /path/to/POWHEG-BOX-V2/Zj/ZjMiNNLO/suggested-run/FlavRegList
```

Notes:

- If `# uub` lines are present in events, they are used as exact MiNNLO projection to `qqbar -> Z`.
- Without `# uub`, the code uses `#rwgt` + `FlavRegList` to map NLO (`type=1`) and NNLO-like (`type=2/3`) events to the underlying Born flavor flow.
- As a final fallback (when mapping is unavailable), it infers quark direction/flavor from incoming partons.
- Default behavior is strict (abort on unresolved events). Use `--skip-unresolved` to keep running and write weight `1.0` for unresolved events.

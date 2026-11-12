# UI reference and design fidelity

Figma prototype supplied by project owner:

https://www.figma.com/proto/sSicep0Vpc6jLp0Rw68ORP/Untitled?node-id=0-1&t=Gd2MLFbA01aZctCm-1

**Reference-access limitation:** The prototype could not be fetched or inspected in the build environment (Figma blocked the remote fetch). The interface in this repository is a **new, usable component-based UI** informed by the established LightBrain requirements, **not** a claim of 1:1 pixel matching to this Figma file. No colors, spacing, layout, typography, or screen order were extracted from Figma. To align it to Figma, provide exported screenshots or inspect-mode specifications (per-frame dimensions, typography, tokens and assets).

## Implemented interface

- Overview / source mode selector: Demo, native BLE, sender-compatible BLE, BLED112 COM port.
- Calibration Studio: original SSVEP 30 s, EOG natural 20 s, EOG intentional 20 s; progress, validation, sensitivity, ARM, reset and REST test.
- Live Monitor: EOG raw/filtered, SSVEP raw/filtered, Welch PSD, PSDA z-scores, classifier outputs, visual three-command states, STOP.
- Event & Safety Log: human-readable events with engine-side CSV persistent logging.
- Custom `BrainVisual`: animated, illustrative vector brain with labeled Fp1/O1 markers, **not anatomically precise**.
- Qt stylesheet theme tokens, high-DPI scaling, scrollable pages, reusable controls.

## Component map

```
main.py                       # application/CLI entrypoint
acquisition.py                # original previous merge acquisition bridge, unchanged
hybrid_engine.py              # previous-merge control, preserved
ssvep_original.py             # original copied algorithms, unchanged
eog_original.py               # original copied algorithms, unchanged
desktop_output.py             # optional desktop W/A/D, unchanged
ui/
  main_window.py              # application integration and QTimer update loop
  styles/theme.py             # design tokens and global QSS
  components/common.py        # Panel, Button, StepCard, StatusPill, MetricCard
  components/signal_chart.py  # reusable PyQtGraph
  components/brain_visual.py  # custom Qt-painted animated schematic
  pages/overview.py           # launch/acquisition and command overview
  pages/calibration.py        # sequential baselines
  pages/monitor.py            # live monitoring and safety
  pages/logs.py               # logs
original_sources/             # supplied source files, unchanged
tests/                        # original behavioral/AST tests + UI architecture tests
```

## Visual adjustments

The main theme lives at `ui/styles/theme.py`, and component layouts are under `ui/pages/`.
After receiving reference screenshots, adjust **only UI files**. Do not modify EOG/SSVEP core or original command gate definitions to match a visual design.

# LightBrain Hybrid BCI — Modular Qt Interface

This is a complete **Python/PyQt5 desktop source project** with a redesigned component-based UI around the existing research-grade Hybrid BCI engine. No web frontend was introduced: the application remains compatible with its original OpenBCI Ganglion / BrainFlow signal source architecture.

> **Figma fidelity:** The supplied Figma prototype could not be inspected from this environment. The interface in this package is a newly authored responsive dark dashboard based on the previously established LightBrain layout and workflows, **not verified to match Figma 1:1**. See `docs/UI_REFERENCE.md`. Provide exported frames/assets for exact matching.
>
> **Safety:** This is a research simulator. It is not certified to operate a powered wheelchair. There is no motor/actuator interface. Optional Windows W/A/D keyboard control is designed exclusively for a **desktop STIMULATOR simulation**, and defaults off. No real Ganglion or Windows GUI session was available in the build environment.

## Screens

1. **Overview** — source selector (Demo / Native BLE / sender-compatible / BLED112), illustrated animated brain with Fp1 & O1 (schematic), status cards and control mapping.
2. **Calibration Studio** — SSVEP Null 30 s → EOG Natural 20 s → EOG Intentional 20 s. Live filter previews, EOG sensitivity, validate, rest-screening and arm.
3. **Live Control Monitor** — EOG and EEG raw/filtered traces, Welch PSD, PSDA Z-scores, current command, original Gate state, demo injection, always-available STOP.
4. **Session Logs** — events from acquisition, calibration, classification and safety; underlying engine also writes CSV to `logs/`.

## Run on Windows

Use Python **3.11 or 3.12** (Python 3.14 might not be supported by the exact versions of PyQt/BrainFlow).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

`python main.py` opens the Overview selector without starting the signal source; choose Demo and click **CONNECT SOURCE**.

Direct demo mode, preserving old CLI behavior:

```powershell
.\.venv\Scripts\python.exe main.py --mode demo
```

To use OpenBCI Ganglion native BLE, install BrainFlow:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_hardware.txt
.\.venv\Scripts\python.exe main.py --mode native --eog-ch 1 --ssvep-ch 3
```

For BLED112 dongle:

```powershell
.\.venv\Scripts\python.exe main.py --mode dongle --port COM3 --eog-ch 1 --ssvep-ch 3
```

To use the original SSVEP raw BrainFlow array row `SELECTED_CH=2` instead of mapped EEG CH3, use `--legacy-ssvep-row2`. **The names Fp1/O1 denote intended electrode placement, not verified physical wiring.** Check electrode, reference/ground positions and actual BrainFlow units before collecting participant data. The original SSVEP source's `SCALE_FACTOR=0.02235` is preserved rather than silently corrected.

Run unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Or double-click `start_demo_windows.bat` / `test_windows.bat` after setting up the environment.

## Original logic preservation

All original algorithm modules and the previous merge's hybrid engine remain **byte-for-byte unchanged**:

- `eog_original.py` — `StreamingFilter`, `natural_reference`, `extract_events`, `PersonalBlinkModel`, `SingleBlinkGate` and its calibration model.
- `ssvep_original.py` — original Elliptic/Notch filter, Welch PSD (`nperseg=800`, `noverlap=700`), SSVEP PSDA, harmonic/null z-score decision, artifact filtering and `CommandGate`.
- `hybrid_engine.py` — original-logic orchestration and interlock from the prior merged package.
- `acquisition.py`, `desktop_output.py` — preserved from the prior merged package.

The EOG W/forward command is triggered by a classified **intentional** single blink and uses the original 1 s hold / 2 s cooldown behavior. SSVEP 9 Hz maps to LEFT/A and 15 Hz to RIGHT/D with **5 s non-overlapping decisions** and the original command gate. Shared source, cross-modal interlock and STOP are the previous merge's unavoidable new behavior, not identical to running two original processes independently.

Only `main.py` (thin launcher) and `ui/` are introduced/rewritten. UI components **do not perform EOG feature extraction, SSVEP classification or command arbitration**.

## Using the program

1. Open the Overview and connect a source. Wait for a valid signal and original filter warmup.
2. Open Calibration Studio and record SSVEP Null 30 s while looking away from the targets.
3. Record Natural EOG 20 s and Intentional EOG 20 s. Demo mode generates training blinks automatically.
4. Press **VALIDATE**, then **ARM SIMULATOR**. Invalid/missing recordings never allow ARM.
5. On Live Monitor, inspect signals and use the Demo 9/15 Hz or blink injectors **only when in Demo**.
6. Click **EMERGENCY STOP** or press Space in the application. Reset E-STOP before rearming. Source loss, bad samples and queue overflow are safety stops.
7. Optional `--keyboard` or Live Monitor checkbox for W/A/D output to an external **Windows desktop STIMULATOR only**. Keyboard output cannot safely operate motor hardware.

The app logs events in UTF-8 CSV under `logs/`, and records baseline EOG data as CSV. To understand study validity, inspect the baseline data, electrode references, real sampling rate and unintended-trigger rate; passing synthetic tests is not clinical or operational validation.

## Development environment and verified checks

- `python -m compileall` passed for all authored files.
- `python -m unittest discover -s tests -v` passed the original algorithm-preservation, synthetic baseline/SSVEP and interlock tests, plus UI-structure checks.
- **GUI couldn't be launched here** because `PyQt5` and `pyqtgraph` are not installed and the environment cannot download them; actual Windows visual testing and hardware testing remain necessary.

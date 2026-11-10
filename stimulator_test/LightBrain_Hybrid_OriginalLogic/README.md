# LightBrain Hybrid BCI — ORIGINAL ALGORITHMS PRESERVED

**Research build · November 2026 · Two independent signal paths, one Ganglion session.**

> **Scope:** the original signal-processing / feature-extraction / classification functions, their constants, and **both original command gate classes** are copied without modifying their bodies. A test compares their complete Python ASTs against the exact supplied source files. The **full merged application's behavior cannot be identical to running both old programs independently**, because one shared acquisition connection, a cross-modal command interlock, and a common STOP mechanism are necessary. All added behavior is isolated in `acquisition.py`, `hybrid_engine.py`, `desktop_output.py` and `main.py`.
>
> **No direct motor or real-wheelchair interface is included.** Outputs go to an internal simulator. Optional Windows W/A/D is for a **desktop STIMULATOR application only**, never a physical actuator. This is not a certified assistive-device controller.

## Command mapping — same as supplied sources

| Original component | Recognition | Command | Original Gate |
|---|---|---|---|
| EOG / CH1 / Fp1 | 1 **intentional** blink; natural/artifact rejected | `FORWARD` / W | `SingleBlinkGate`: W 1.0 s, cooldown **2.0 s after release**, refractory 0.38 s |
| SSVEP / EEG channel assigned to O1 | 9 Hz, 18 Hz, 27 Hz harmonics | `LEFT` / A | `CommandGate`: 1.0 s hold, 2.0 s lockout after start |
| SSVEP / EEG channel assigned to O1 | 15 Hz, 30 Hz harmonics | `RIGHT` / D | Same original `CommandGate` |
| Any channel / app | no accepted command, expired gate, invalid samples, timeout, manual/Space STOP | **STOP** | Releases both modalities / desktop keys |

The source EOG mapping is **forward**, not blink confirmation for SSVEP. The hybrid combines both signal types *in parallel* and does not reinterpret their classifications.

## Exact original algorithms preserved

The following algorithms are copied directly, with no changes inside their definitions:

- **EOG:** `StreamingFilter` (0.1–10 Hz), `robust_median_mad`, `natural_reference`, `waveform`, `extract_events`, `_ln_features`, `_corr`, `PersonalBlinkModel` (`predict` and `model_diagnostic`), `SingleBlinkGate` and `VirtualWheelchair`.
- **SSVEP:** `SignalProcessor` (Elliptic 1–40 Hz + 50 Hz notch), Welch PSD (`nperseg=800`, `noverlap=700`), `SSVEPDetector` (original PSDA harmonic scoring / null calibration / z-score / thresholds), `DecisionFilter`, `CommandGate` and `_artifact_reason`.
- **SSVEP decision schedule:** **5.0-second epoch** and **5.0-second decision step**, **not 1 second**, with original artifact rejection and per-target PSDA thresholds.
- **SSVEP baseline:** 30 seconds, PSD-frame aggregation followed by PSDA null calibration. **EOG:** natural 20 s + intentional 20 s (the durations in the supplied EOG file; not 30 s).
- Default EOG sensitivity 3.0 (BALANCED). Other original settings (2.0 SENSITIVE, 4.5 STRICT) are optional.
- **Original EOG rest-screening** of 60 s is available without output.

To audit preservation:

```powershell
python -m unittest discover -s tests -v
```

`tests/test_preserve_original.py` compares the original function/class ASTs; it will fail if a copied algorithm body is changed. Original files remain unchanged under `original_sources/`.

## Running on Windows

Recommended: **Python 3.11 or 3.12** in a virtual environment (BrainFlow and GUI wheels may not yet work with 3.14).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --mode demo
```

If the virtual environment activation is blocked by PowerShell policy, run `.\.venv\Scripts\python.exe main.py --mode demo` without activating it.

Install BrainFlow **only when connecting real EEG/EOG hardware**:

```powershell
python -m pip install -r requirements_hardware.txt
python main.py --mode native --eog-ch 1 --ssvep-ch 3
```

Other hardware connection modes:

```powershell
python main.py --mode dongle --port COM3 --eog-ch 1 --ssvep-ch 3
python main.py --mode sender --port COM3 --eog-ch 1 --ssvep-ch 3
```

`--sender` is *the original EOG sender-compatible native BLE mode*, not a separate UDP forwarding process.

### Important: source-channel selection / 100% fidelity detail

- The EOG source's CH1 is `BoardShim.get_eeg_channels(board_id)[0]` and is assumed to have an Fp1 electrode connected.
- This merged program **defaults to EEG channel CH3 for the SSVEP/O1 electrode** using `BoardShim.get_eeg_channels(board_id)[2]`. **EEG CH3 is not automatically O1**; ensure the actual electrode is connected to the selected channel.
- The supplied SSVEP program uses `SELECTED_CH = 2`, which is **raw BrainFlow array row 2** (this is not necessarily EEG CH3). To reproduce that **exact original data selection** instead, run with `--legacy-ssvep-row2`. The two choices may read different physical signals; verify your original wiring.
- Original SSVEP BrainFlow path multiplies the raw channel by `SCALE_FACTOR=0.02235`. This value is preserved, though modern BrainFlow EEG values are often already in µV, and you may need to verify the actual unit handling experimentally. EOG is **not double-scaled**.

```powershell
python main.py --mode native --eog-ch 1 --ssvep-ch 3 --legacy-ssvep-row2
```

## Calibration and demo workflow

1. Launch `python main.py --mode demo`; check both signals update. Wait for 2 s filter warm-up.
2. Click **SSVEP Null 30 s** (do not look at 9 Hz or 15 Hz target). The demo target is automatically set to NEUTRAL.
3. Click **EOG Natural 20 s**; during demo, spontaneous blinks are synthesized automatically.
4. Click **EOG Intentional 20 s**; during demo, deliberate blinks are synthesized automatically.
5. Click **VALIDATE**, then **ARM SIMULATOR**. An EOG model must have at least 2 intentional candidates, as specified in its original source.
6. Press **INTENT** to simulate an EOG blink → W for 1 s. Press **9 Hz LEFT** or **15 Hz RIGHT** to inject the corresponding SSVEP for long enough to fill a **5 s** analysis epoch.
7. STOP by clicking **EMERGENCY STOP** or pressing **Space** while the GUI is focused. E-STOP latches, releases commands, and requires **RESET E-STOP** and another ARM.
8. Run **REST TEST 60 s** with output blocked to assess apparent false intentional blinks. This is **not a safety certification**.

**Windows desktop STIMULATOR keyboard (optional, OFF by default):**

```powershell
python main.py --mode demo --keyboard
```

- Enables Windows `SendInput` for W, A, D to the focused **desktop application**. Requires a compatible Windows environment. Do not use this to drive a physical wheelchair.
- Key release happens when the original gate expires or STOP is triggered. A separate watchdog attempts key release if the GUI stops updating.
- The keyboard output adapter is newly implemented for three keys; **the feature/classification/gate logic itself is unchanged**. It is not bit-for-bit identical to the two old independent keyboard adapters.

## Hybrid arbitration: the only new decision layer

- Both original classifiers always have their own filters, buffers, thresholds and gates.
- The shared output can have **only one active motion**. If EOG/W is active, an SSVEP command candidate is ignored by the output interlock; if SSVEP A/D is active, an intentional EOG command is suppressed. No motion is queued across the other modality's hold. The original recognition result is still logged.
- No detection means **no newly accepted motion**, and after any original 1-s hold finishes the simulator goes to STOP. An original gate's lockout still applies to its **own** modality; an EOG cooldown does not alter the SSVEP classifier.
- A stale signal (**1.2 s**, original EOG timeout), invalid/non-finite data or acquisition buffer overrun forces STOP and latches E-STOP. Resetting E-STOP leaves the app disarmed.

**These coordination and safety behaviors do not exist when running two independent original programs, so complete end-to-end equivalence cannot be claimed.** This design preserves algorithm and per-modality command interpretation exactly while making concurrency explicit and auditable.

## Test status / limitations

Core tests (`python -m unittest discover -s tests -v`) passed **17/17** using Python 3.13 with NumPy and SciPy; this includes a complete synthetic 30/20/20 calibration followed by 9/15 Hz recognition, exact source AST comparisons, timing and safety checks. Python `py_compile` also passes for all project modules. This build environment lacks PyQt5/pyqtgraph/BrainFlow, so **Windows GUI startup and live Ganglion acquisition have not been tested** and must be validated on the user's machine. Synthetic success does not establish real-world accuracy, false activation rate, or suitability for wheelchair use.

## Structure

```text
LightBrain_Hybrid_OriginalLogic/
├── main.py                    # Qt GUI / real-time dashboard
├── acquisition.py             # One Ganglion session, paired EEG+EOG, demo
├── eog_original.py            # Exact original EOG functions/classes
├── ssvep_original.py          # Exact original SSVEP functions/classes
├── hybrid_engine.py           # Baseline coordination and arbitration ONLY
├── desktop_output.py          # Optional desktop STIMULATOR keyboard / watchdog
├── requirements.txt
├── requirements_hardware.txt
├── original_sources/          # Both unmodified files supplied by user
├── tests/                     # Original source integrity and integration tests
└── logs/                      # Event CSV and EOG calibration CSV saved here
```

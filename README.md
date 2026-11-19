# AI Test-Fit Planner

Streamlit-based architectural test-fit planner using OR-Tools CP-SAT for the primary room-placement solver, a deterministic compact-site fallback, and compact dataset-derived priors from RPLAN and CubiCasa5K.

## Run the existing interface

From this folder, run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The current Streamlit interface is preserved. Use the same site presets or select **Custom dimensions**. The primary optimizer keeps the requested room sizes whenever they fit. If CP-SAT cannot find a layout, an adaptive fallback tries smaller room footprints and alternate floor distributions. It checks room/site bounds and collisions, keeps attached bedroom/bathroom pairs together, and preserves the Kitchen-to-Utility connection when feasible. On compact plots, the fallback may reduce parking from two cars to one and/or omit an optional room; the result message reports any such adjustment. These are conceptual test-fits, not construction-ready plans.

The 3D view hides graph axes, ticks, and grid lines only in the 3D visualization. The 2D plan axes remain unchanged. Bathroom-to-bedroom door symbols are matched by room number, so Bathroom 2 does not receive a direct door to Bedroom 3.

## Dataset use: RPLAN + CubiCasa5K

The project uses **RPLAN-derived room-count/connectivity rules** in `data/knowledge/architectural_rules.json` and `data/priors/resplan_priors.json`. It also loads the compact **CubiCasa5K inspection/adjacency summary** in `data/inspection/cubicasa_inspection.json` as a weak adjacency prior for the fallback placement scorer. Dataset statistics are preferences, not hard building-code constraints.

Raw RPLAN and CubiCasa5K data are not included in this ZIP. Keep your licensed/local copies outside this project. `tools/dataset/` contains scripts for inspection and extracting compact priors; the existing CubiCasa training subset is small and is not a replacement for the full dataset. The live planner is a constraint-based generator guided by dataset priors, not a neural network that claims to have trained on every raw dataset sample.

To regenerate **120 varied plan examples** (one ground-floor PNG and a JSON record containing all floors per example), run:

```powershell
python scripts/generate_120_plans.py
```

Outputs are written to `generated_plans_120/`, which is created only when the script runs and is not bundled in the source ZIP. You can change the solver time limit with `--time-limit 1.0` or the output directory with `--output my_plans`.

## Project contents

- `app.py` — unchanged Streamlit interface and application flow.
- `testfit/` — CP-SAT optimizer, compact fallback, dataset-prior reader, room planning and 2D/3D visualization.
- `data/knowledge`, `data/priors`, `data/inspection`, `data/legends` — compact aggregate dataset artifacts only.
- `data/training/ai_baseline` and `cubicasa5k_candidates_100.jsonl` — small CubiCasa5K-derived sample used by the optional model-training pipeline.
- `ai/` — optional sample-based PyTorch training/evaluation code.
- `tools/dataset/` — scripts to inspect local raw datasets and rebuild compact aggregate files.
- `scripts/generate_120_plans.py` — repeatable 120-case generation script.
- `tests/` — regression tests.

## Test

```powershell
python -m unittest discover -s tests -v
```

For offline dataset inspection, install `requirements-dataset.txt`. For the optional PyTorch sample-training pipeline, install `requirements-training.txt`. The default planner only requires `requirements.txt`.

## 📸 Project Screenshots

### 🏠 Home Page
![Home Page](outputs/home%20page.png)

### 🏢 Ground Floor Plan
![Ground Floor](outputs/ground%20floor.png)

### 🏢 First Floor Plan
![First Floor](outputs/first%20floor.png)

### 🧊 3D Model Visualization
![3D Model](outputs/3d%20model.png)

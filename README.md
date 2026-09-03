# Higher-order alleys

Research code and artifacts for Jacobian-Lens experiments using Qwen3.6-27B.

## Repository layout

| Path | Purpose |
| --- | --- |
| `experiments/higher_v_readout_global/` | Runnable global-steering experiment and CPU tests |
| `dataset/metacognition.csv` | The 90-row experiment dataset |
| `assets/` | Parity exports and completed run artifacts |
| `docs/` | Research context, migration notes, next-stage plan, and results |

## Setup

Run commands from this repository root:

```powershell
uv sync
```

The model phases require a CUDA host with sufficient VRAM. Static validation
and the unit tests do not download or load the model.

## Verify the project

```powershell
uv run python -m unittest experiments.higher_v_readout_global.test_global_experiment
uv run python -m experiments.higher_v_readout_global.runner --phase validate
```

## Run the experiment

See `experiments/higher_v_readout_global/README.md` for parity, pilot, full-run,
resume, control, output-root, and analysis commands. Runs default to
`experiments/higher_v_readout_global/results/`; override this with
`--output-root PATH`.

The documents in `docs/` include historical predecessor context. Current
operational paths and commands are defined by this README and the experiment
README.

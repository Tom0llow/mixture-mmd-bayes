# mixture mmd-bayes

GMM estimation — Quick start

This repository contains code for estimating Gaussian mixture models using an MMD-based Bayesian approach. The following steps describe the minimal flow to run a numerical simulation for GMM estimation.

Prerequisites
- Python 3.10+ and typical Python tooling (pip, virtualenv, etc.).

Workflow
1. Install `uv` (the project's runner utility). For example:

```bash
pip install uv
```

2. Synchronize project files (run from the repository root):

```bash
uv sync
```

3. Run the GMM estimation simulation (run from the repository root):

```bash
uv run python gmm_estimatioin/src/main.py
```

Notes
- The commands above assume you run them in the repository root where this `README.md` lives.
- If your environment or `uv` installation differs, adapt the install/run commands accordingly.

If you want, I can add a more detailed setup (virtualenv, deps, examples) or fix the path spelling if `gmm_estimatioin` is a typo.
# edge-of-stability

Code for studying the "edge of stability" phenomenon in neural network training
under different optimizers (plain/momentum GD, SignGD, Muon, block-coordinate GD, Adam).

## Layout

- `src/` — all source code.
  - `methods.py` — custom optimizer implementations (`MyAdam`, `MySGD`, `MySignSGD`,
    `MyBlockSCD`, `SGDM`, `Muon`) with built-in tracking of the local sharpness
    (`direct_smooth`) along the optimization trajectory.
  - `utilities.py` — shared helpers: directory naming, loss/accuracy functions,
    Hessian-vector products, Lanczos top-eigenvalue estimation, and the
    Newton-Schulz / PolarExpress matrix-sign iterations used by Muon.
  - `archs.py`, `vgg.py`, `resnet_cifar.py` — network architectures.
  - `data.py`, `cifar.py`, `synthetic.py` — dataset loading (CIFAR-10 subsets,
    synthetic Chebyshev/linear regression data).
  - Training entrypoints (one per optimizer), each runnable as a script:
    - `gd.py` — full-batch gradient descent.
    - `sgd.py` — full-batch gradient descent with momentum + injected noise (`SGDM`).
    - `adam.py` — Adam.
    - `signgd.py` — sign gradient descent.
    - `signgd_fw.py` — sign gradient descent with an additional Frank-Wolfe-based
      probe of the edge-of-stability direction.
    - `muon.py` — Muon (orthogonalized momentum via PolarExpress).
    - `blockgd.py` — block-coordinate gradient descent (updates only the
      highest-gradient-norm layer each step).
    - `flow.py` — (approximate) gradient flow, for comparison with discrete-time GD.

## Running

Each training script is a standalone CLI, e.g.:

```bash
python src/gd.py cifar10-5k fc-tanh mse 0.01 1000 --eig_freq 50 --neigs 2
```

Run `python src/<script>.py --help` for the full list of arguments. Results
(eigenvalues, losses, accuracies, iterates) are saved under `results/`, and
downloaded datasets are cached under `data/` — neither is tracked in git.

## Requirements

PyTorch, NumPy, SciPy, torchvision, and Weights & Biases (`wandb`) for logging.

import torch
from runners.run import run

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run(
        dim=1,
        K=4,
        separation=3.0,
        sigma=1.0,
        n_train=1000,
        n_test=2000,
        steps=1200,
        M=32,
        S=32,
        gamma_scale=1e-4,
        R=100,
        seed=None,
        device=device,
        dtype=torch.float64,
    )
    for method, stats in results.items():
        mean_ed2 = stats["mean_ED2"]
        se_ed2 = stats["stderr"]
        print(f"{method}: Mean ED^2 = {mean_ed2:.4f} ± {se_ed2:.4f}")

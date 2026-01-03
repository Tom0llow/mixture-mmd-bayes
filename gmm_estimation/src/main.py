import argparse

import torch
from runners.run_parallel import run_parallel
from runners.run_sequantial import run_sequantial

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GMM estimation experiments")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "parallel"],
        default="sequential",
        help="Execution mode: 'sequential' or 'parallel' (default: sequential)",
    )
    parser.add_argument("--dim", type=int, default=1, help="Dimension (default: 1)")
    parser.add_argument(
        "--K", type=int, default=4, help="Number of mixture components (default: 4)"
    )
    parser.add_argument(
        "--separation", type=float, default=3.0, help="Separation (default: 3.0)"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="Standard deviation (default: 1.0)"
    )
    parser.add_argument(
        "--n-train", type=int, default=1000, help="Training samples (default: 1000)"
    )
    parser.add_argument(
        "--n-test", type=int, default=2000, help="Test samples (default: 2000)"
    )
    parser.add_argument(
        "--steps", type=int, default=1200, help="Optimization steps (default: 1200)"
    )
    parser.add_argument(
        "--M", type=int, default=32, help="Particle count (default: 32)"
    )
    parser.add_argument(
        "--S", type=int, default=32, help="Subsample count (default: 32)"
    )
    parser.add_argument(
        "--gamma-scale", type=float, default=1e-4, help="Gamma scale (default: 1e-4)"
    )
    parser.add_argument(
        "--R", type=int, default=100, help="Number of runs (default: 100)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="GPU IDs for parallel mode: comma-separated or 'auto' (default: auto)",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "sequential":
        print("Running SEQUENTIAL mode...")
        results = run_sequantial(
            dim=args.dim,
            K=args.K,
            separation=args.separation,
            sigma=args.sigma,
            n_train=args.n_train,
            n_test=args.n_test,
            steps=args.steps,
            M=args.M,
            S=args.S,
            gamma_scale=args.gamma_scale,
            R=args.R,
            seed=args.seed,
            device=device,
            dtype=torch.float64,
        )
    elif args.mode == "parallel":
        print("Running PARALLEL mode...")
        if args.devices.lower() == "auto":
            devices = (
                list(range(torch.cuda.device_count()))
                if torch.cuda.is_available()
                else []
            )
        else:
            devices = [int(x) for x in args.devices.split(",") if x.strip()]

        if len(devices) < 2:
            raise RuntimeError(
                f"Parallel mode requires >=2 GPUs, but only {len(devices)} available."
            )

        results = run_parallel(
            devices=devices,
            dim=args.dim,
            K=args.K,
            separation=args.separation,
            sigma=args.sigma,
            n_train=args.n_train,
            n_test=args.n_test,
            steps=args.steps,
            M=args.M,
            S=args.S,
            gamma_scale=args.gamma_scale,
            R=args.R,
            base_seed=args.seed if args.seed is not None else 0,
        )

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for method, stats in results.items():
        mean_ed2 = stats["mean_ED2"]
        se_ed2 = stats["stderr"]
        runs = stats["runs"]
        print(f"{method}: Mean ED^2 = {mean_ed2:.4f} ± {se_ed2:.4f} (runs={runs})")

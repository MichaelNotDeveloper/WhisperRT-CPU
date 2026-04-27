import gc
import time

import matplotlib.pyplot as plt
import torch

DEVICE = "cpu"
DTYPE = torch.float32

DIMS = [256, 512, 1024, 2048]
SPARSITIES = [0.85, 0.90, 0.92, 0.94, 0.95, 0.97, 0.98, 0.99]
REPEATS = 50
WARMUP = 10


def prune_random(w: torch.Tensor, sparsity: float) -> torch.Tensor:
    if sparsity == 0:
        return w

    mask = torch.rand_like(w) > sparsity
    return w * mask


def tensor_memory_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / 1024**2


def sparse_memory_mb(s: torch.Tensor) -> float:
    # COO: values + indices
    s = s.coalesce()
    values = s.values()
    indices = s.indices()
    return (
        values.numel() * values.element_size() + indices.numel() * indices.element_size()
    ) / 1024**2


def bench_matmul(x, w, repeats=REPEATS, warmup=WARMUP):
    with torch.inference_mode():
        for _ in range(warmup):
            _ = x @ w.T

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = x @ w.T
        t1 = time.perf_counter()

    return (t1 - t0) / repeats


def bench_sparse_matmul(x, w_sparse, repeats=REPEATS, warmup=WARMUP):
    # x @ W.T = (W @ x.T).T
    with torch.inference_mode():
        for _ in range(warmup):
            _ = torch.sparse.mm(w_sparse, x.T).T

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = torch.sparse.mm(w_sparse, x.T).T
        t1 = time.perf_counter()

    return (t1 - t0) / repeats


def run_experiment(name, shape_fn, batch_size=1):
    rows = []

    for dim in DIMS:
        out_dim, in_dim = shape_fn(dim)

        x = torch.randn(batch_size, in_dim, device=DEVICE, dtype=DTYPE)
        w_base = torch.randn(out_dim, in_dim, device=DEVICE, dtype=DTYPE)

        for sparsity in SPARSITIES:
            gc.collect()

            w_dense = prune_random(w_base, sparsity).contiguous()
            w_sparse = w_dense.to_sparse_coo().coalesce()

            dense_time = bench_matmul(x, w_dense, REPEATS, WARMUP)
            sparse_time = bench_sparse_matmul(x, w_sparse, REPEATS, WARMUP)

            dense_mem = tensor_memory_mb(w_dense)
            sparse_mem = sparse_memory_mb(w_sparse)

            rows.append(
                {
                    "experiment": name,
                    "dim": dim,
                    "out_dim": out_dim,
                    "in_dim": in_dim,
                    "sparsity": sparsity,
                    "pruned_percent": sparsity * 100,
                    "dense_time_ms": dense_time * 1000,
                    "sparse_time_ms": sparse_time * 1000,
                    "speedup_sparse_vs_dense": dense_time / sparse_time,
                    "dense_memory_mb": dense_mem,
                    "sparse_memory_mb": sparse_mem,
                    "memory_ratio_sparse_vs_dense": sparse_mem / dense_mem,
                    "nnz": w_sparse._nnz(),
                }
            )

            print(
                f"{name:8s} dim={dim:4d} sparsity={sparsity:4.0%} "
                f"dense={dense_time * 1000:8.3f} ms "
                f"sparse={sparse_time * 1000:8.3f} ms "
                f"speedup={dense_time / sparse_time:6.2f}x "
                f"mem_ratio={sparse_mem / dense_mem:6.2f}"
            )

    return rows


def plot_metric(rows, experiment, metric, ylabel):
    plt.figure(figsize=(9, 6))

    for dim in DIMS:
        xs = []
        ys = []
        for r in rows:
            if r["experiment"] == experiment and r["dim"] == dim:
                xs.append(r["pruned_percent"])
                ys.append(r[metric])

        plt.plot(xs, ys, marker="o", label=f"dim={dim}")

    plt.xlabel("Pruned percent, %")
    plt.ylabel(ylabel)
    plt.title(f"{experiment}: {ylabel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    rows = []
    rows += run_experiment("square", shape_fn=lambda dim: (dim, dim), batch_size=1)
    rows += run_experiment("ffn", shape_fn=lambda dim: (dim * 8, dim), batch_size=1)

    for experiment in ["square", "ffn"]:
        plot_metric(
            rows,
            experiment,
            "speedup_sparse_vs_dense",
            "Sparse speedup vs dense, x",
        )
        plot_metric(
            rows,
            experiment,
            "memory_ratio_sparse_vs_dense",
            "Sparse memory / dense memory",
        )
        plot_metric(
            rows,
            experiment,
            "dense_time_ms",
            "Dense matmul time, ms",
        )
        plot_metric(
            rows,
            experiment,
            "sparse_time_ms",
            "Sparse matmul time, ms",
        )


if __name__ == "__main__":
    main()

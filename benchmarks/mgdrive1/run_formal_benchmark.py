"""Run the resumable, block-interleaved formal benchmark suite."""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from .run_slim_benchmark import write_records
from .run_spatial_benchmark import run_mgdrive, write_natal_records
from .slim_panmmictic import (
    benchmark_natal_panmmictic,
    benchmark_slim,
)
from .spatial_benchmark import benchmark_natal

Task = Callable[[Path, int, int], None]


def _completed(path: Path, *, expected_rows: int) -> bool:
    """Return whether a block result exists with the expected row count.

    Args:
        path: Candidate block CSV.
        expected_rows: Required number of data rows.

    Returns:
        Whether the file is a complete normalized block result.
    """
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        len(rows) == expected_rows
        and all("block" in row for row in rows)
        and all(int(row["repeat"]) >= 1 for row in rows)
    )


def _normalize_block(
    source: Path,
    destination: Path,
    *,
    block: int,
    repeat_offset: int,
    expected_rows: int,
) -> None:
    """Add block and global replicate identifiers to one task result.

    Args:
        source: Newly written task CSV.
        destination: Durable normalized block CSV.
        block: One-based block identifier.
        repeat_offset: Offset added to local replicate identifiers.
        expected_rows: Required number of data rows.

    Raises:
        RuntimeError: If the source has an invalid schema or row count.
    """
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None or "repeat" not in fieldnames:
        raise RuntimeError(f"{source} lacks a repeat column")
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"{source} has {len(rows)} rows; expected {expected_rows}"
        )
    normalized_fields = [
        *fieldnames[: fieldnames.index("repeat")],
        "block",
        *fieldnames[fieldnames.index("repeat") :],
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=normalized_fields)
        writer.writeheader()
        for row in rows:
            row["block"] = block
            row["repeat"] = int(row["repeat"]) + repeat_offset
            writer.writerow(row)
    source.unlink()


def _aggregate(
    block_paths: Sequence[Path],
    destination: Path,
    *,
    expected_rows: int,
) -> None:
    """Concatenate normalized block files into one formal result.

    Args:
        block_paths: Ordered complete block results.
        destination: Aggregate CSV path.
        expected_rows: Required aggregate row count.

    Raises:
        RuntimeError: If schemas differ or row count is incorrect.
    """
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for path in block_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError(f"{path} lacks a CSV header")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            elif reader.fieldnames != fieldnames:
                raise RuntimeError(f"{path} has a mismatched CSV schema")
            rows.extend(reader)
    if fieldnames is None or len(rows) != expected_rows:
        raise RuntimeError(
            f"{destination.name} has {len(rows)} rows; expected {expected_rows}"
        )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_tasks(
    *,
    rows: int,
    cols: int,
    n_days: int,
    panmictic_per_block: int,
    spatial_per_block: int,
) -> tuple[tuple[str, int, Task], ...]:
    """Build the seven engine/scenario tasks run in every block.

    Args:
        rows: Hex-grid rows.
        cols: Hex-grid columns.
        n_days: Simulated days per replicate.
        panmictic_per_block: Panmictic replicates per block.
        spatial_per_block: Spatial replicates per block.

    Returns:
        Task name, block size, and task callable tuples.
    """

    def natal_pan(path: Path, repeats: int, seed: int) -> None:
        """Write one block of NATAL panmictic records."""
        records = benchmark_natal_panmmictic(
            repeats=repeats,
            n_days=n_days,
            seed=seed,
        )
        write_records(path, records)

    def slim_pan(path: Path, repeats: int, seed: int) -> None:
        """Write one block of SLiM panmictic records."""
        records = benchmark_slim(
            repeats=repeats,
            n_days=n_days,
            seed=seed,
        )
        write_records(path, records)

    def mgdrive_pan(path: Path, repeats: int, seed: int) -> None:
        """Write one block of MGDrivE1 panmictic records."""
        run_mgdrive(
            mode="stochastic",
            rows=1,
            cols=1,
            n_days=n_days,
            repeats=repeats,
            seed=seed,
            output_path=path,
        )

    def natal_spatial(
        path: Path,
        repeats: int,
        *,
        stochastic: bool,
    ) -> None:
        """Write one seeded-independent NATAL spatial block."""
        records = benchmark_natal(
            stochastic=stochastic,
            rows=rows,
            cols=cols,
            n_days=n_days,
            repeats=repeats,
        )
        write_natal_records(
            path,
            records,
            rows=rows,
            cols=cols,
            n_days=n_days,
        )

    def mgdrive_spatial(
        path: Path,
        repeats: int,
        seed: int,
        *,
        mode: str,
    ) -> None:
        """Write one block of seeded MGDrivE1 spatial records."""
        run_mgdrive(
            mode=mode,
            rows=rows,
            cols=cols,
            n_days=n_days,
            repeats=repeats,
            seed=seed,
            output_path=path,
        )

    return (
        ("natal-panmmictic", panmictic_per_block, natal_pan),
        ("mgdrive-panmmictic", panmictic_per_block, mgdrive_pan),
        ("slim-panmmictic", panmictic_per_block, slim_pan),
        (
            "natal-spatial-deterministic",
            spatial_per_block,
            lambda path, repeats, _seed: natal_spatial(
                path,
                repeats,
                stochastic=False,
            ),
        ),
        (
            "mgdrive-spatial-deterministic",
            spatial_per_block,
            lambda path, repeats, seed: mgdrive_spatial(
                path,
                repeats,
                seed,
                mode="deterministic",
            ),
        ),
        (
            "natal-spatial-stochastic",
            spatial_per_block,
            lambda path, repeats, _seed: natal_spatial(
                path,
                repeats,
                stochastic=True,
            ),
        ),
        (
            "mgdrive-spatial-stochastic",
            spatial_per_block,
            lambda path, repeats, seed: mgdrive_spatial(
                path,
                repeats,
                seed,
                mode="stochastic",
            ),
        ),
    )


def main() -> None:
    """Run all formal benchmark blocks and write aggregate CSV files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=15)
    parser.add_argument("--cols", type=int, default=15)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--panmictic-repeats", type=int, default=1000)
    parser.add_argument("--spatial-repeats", type=int, default=100)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results/formal"),
    )
    parser.add_argument(
        "--mgdrive-r-lib",
        type=Path,
        default=Path("/private/tmp/natal-r-lib"),
    )
    args = parser.parse_args()
    if args.blocks < 1:
        raise ValueError("blocks must be positive")
    if args.panmictic_repeats % args.blocks != 0:
        raise ValueError("panmictic repeats must be divisible by blocks")
    if args.spatial_repeats % args.blocks != 0:
        raise ValueError("spatial repeats must be divisible by blocks")
    if not args.mgdrive_r_lib.is_dir():
        raise FileNotFoundError(args.mgdrive_r_lib)
    os.environ["MGDRIVE_R_LIB"] = str(args.mgdrive_r_lib)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir = args.output_dir / "blocks"
    blocks_dir.mkdir(exist_ok=True)

    panmictic_per_block = args.panmictic_repeats // args.blocks
    spatial_per_block = args.spatial_repeats // args.blocks
    tasks = _build_tasks(
        rows=args.rows,
        cols=args.cols,
        n_days=args.days,
        panmictic_per_block=panmictic_per_block,
        spatial_per_block=spatial_per_block,
    )
    for block_index in range(args.blocks):
        block = block_index + 1
        block_dir = blocks_dir / f"block-{block:02d}"
        block_dir.mkdir(exist_ok=True)
        # Rotate the first task so each engine occupies different thermal slots.
        ordered_tasks = tasks[block_index % len(tasks) :] + tasks[
            : block_index % len(tasks)
        ]
        for name, repeats, task in ordered_tasks:
            destination = block_dir / f"{name}.csv"
            if _completed(destination, expected_rows=repeats):
                print(f"SKIP block={block:02d} task={name}", flush=True)
                continue
            partial = block_dir / f"{name}.partial.csv"
            seed = args.seed + block_index * repeats
            repeat_offset = block_index * repeats
            print(
                f"START block={block:02d} task={name} repeats={repeats}",
                flush=True,
            )
            task(partial, repeats, seed)
            _normalize_block(
                partial,
                destination,
                block=block,
                repeat_offset=repeat_offset,
                expected_rows=repeats,
            )
            print(f"DONE block={block:02d} task={name}", flush=True)

    for name, repeats, _ in tasks:
        block_paths = tuple(
            blocks_dir / f"block-{block:02d}" / f"{name}.csv"
            for block in range(1, args.blocks + 1)
        )
        _aggregate(
            block_paths,
            args.output_dir / f"{name}.csv",
            expected_rows=repeats * args.blocks,
        )
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

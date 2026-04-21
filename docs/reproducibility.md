# Reproducibility

This repository provides a public inference-and-evaluation path for PRISM that matches the same-distribution slice-level data format used during training.

## Recommended Public Checkpoint

- Recommended checkpoint: `quality_5fold_fold_4_exp3_no_mmd/dual_modal_best.pth`
- Public inference mode: `self_attn`

The public `inference.py` entry point keeps only the `self_attn` inference path to avoid user-facing mode confusion.

## Expected Sample Data Layout

The sample data package should unpack into a folder shaped like:

```text
official_8case_demo/
  channel_0/
    imgs/
      TRUS_Prostate_case000065-000.npy
      ...
    gts/
      TRUS_Prostate_case000065-000.npy
      ...
    TRUS_Prostate_case000065.npz
    ...
```

The important part is that:

- `imgs/*.npy` contains `256x256x3` float arrays in `[0, 1]`
- `gts/*.npy` contains the matching slice-level labels
- case-level `.npz` files provide case grouping and spacing metadata for evaluation output

## Reproduction Commands

### Windows

```powershell
./scripts/reproduce_demo.ps1 `
  -DataRoot ./official_8case_demo/channel_0 `
  -Checkpoint ./checkpoints/dual_modal_best.pth `
  -OutputRoot ./outputs/reproduce_demo `
  -PythonExe python `
  -Device cuda:0
```

### Linux / macOS

```bash
./scripts/reproduce_demo.sh \
  ./official_8case_demo/channel_0 \
  ./checkpoints/dual_modal_best.pth \
  ./outputs/reproduce_demo \
  python \
  cuda:0
```

## What Gets Produced

- Case-level `.npz` prediction files under `outputs/reproduce_demo/predictions`
- A single evaluation CSV at `outputs/reproduce_demo/evaluation_results.csv`

## External Assets

For public release, large assets should be hosted outside GitHub:

- checkpoint package
- 8-case sample-data package

The repository should link to those downloads from the main README.

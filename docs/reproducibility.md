# Reproducibility

This repo keeps a compact public reproduction path for PRISM.

## Recommended Public Checkpoint

- `quality_5fold_fold_4_exp3_no_mmd/dual_modal_best.pth`
- Public inference mode: `self_attn`

## Reproduction Commands

```powershell
./scripts/reproduce_demo.ps1 `
  -DataRoot ./official_8case_demo/channel_0 `
  -Checkpoint ./checkpoints/dual_modal_best.pth `
  -OutputRoot ./outputs/reproduce_demo `
  -PythonExe python `
  -Device cuda:0
```

```bash
./scripts/reproduce_demo.sh \
  ./official_8case_demo/channel_0 \
  ./checkpoints/dual_modal_best.pth \
  ./outputs/reproduce_demo \
  python \
  cuda:0
```

## External Assets

- checkpoint package
- 8-case sample-data package

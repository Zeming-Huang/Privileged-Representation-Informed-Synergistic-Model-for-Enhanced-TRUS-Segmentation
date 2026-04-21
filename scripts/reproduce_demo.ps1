param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [string]$OutputRoot = "./outputs/reproduce_demo",

    [string]$PythonExe = "python",

    [string]$Device = "cuda:0"
)

$predDir = Join-Path $OutputRoot "predictions"
$csvPath = Join-Path $OutputRoot "evaluation_results.csv"

New-Item -ItemType Directory -Force $OutputRoot | Out-Null

& $PythonExe inference.py `
    -data_root $DataRoot `
    -checkpoint $Checkpoint `
    -pred_save_dir $predDir `
    -device $Device

& $PythonExe evaluate.py `
    -pred_dir $predDir `
    -gt_dir $DataRoot `
    -output_csv $csvPath

Write-Host "Prediction directory: $predDir"
Write-Host "Evaluation CSV: $csvPath"

param(
    [string[]]$Targets = @('E1', 'E2', 'E3')
)

$ErrorActionPreference = 'Stop'

$ProjectDir = if ($env:PROJECT_DIR) { $env:PROJECT_DIR } else { (Get-Location).Path }
Set-Location $ProjectDir

if (-not $env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES = '0' }
$PytorchGpu = if ($env:PYTORCH_GPU) { $env:PYTORCH_GPU } else { '0' }

$BaseCfg = if ($env:BASE_CFG) { $env:BASE_CFG } else { 'configs/dynamic/transformer_levir_cc_aux_mask_conf_semantic.yaml' }
$Anno = if ($env:ANNO) { $env:ANNO } else { './Levir-CC/levir_cc_captions_reformat.json' }
$ExpRoot = if ($env:EXP_ROOT) { $env:EXP_ROOT } else { './outputs' }
$ResultsRoot = if ($env:RESULTS_ROOT) { $env:RESULTS_ROOT } else { './results' }
$MaxIter = if ($env:MAX_ITER) { $env:MAX_ITER } else { '10000' }
$SnapshotInterval = if ($env:SNAPSHOT_INTERVAL) { $env:SNAPSHOT_INTERVAL } else { '1000' }
$MinCkpt = if ($env:MIN_CKPT) { $env:MIN_CKPT } else { '1000' }
$MaxCkpt = if ($env:MAX_CKPT) { $env:MAX_CKPT } else { '10000' }
$StableWindow = if ($env:STABLE_WINDOW) { $env:STABLE_WINDOW } else { '1' }
$AllowExisting = if ($env:ALLOW_EXISTING) { $env:ALLOW_EXISTING } else { '0' }

function Join-Command {
    param([string[]]$Command)
    return ($Command | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' '
}

function Prepare-OutputDirs {
    param([string]$ExpName)

    $OutDir = Join-Path $ExpRoot $ExpName
    $ResultDir = Join-Path $ResultsRoot $ExpName
    if ($AllowExisting -ne '1' -and ((Test-Path $OutDir) -or (Test-Path $ResultDir))) {
        throw "Refusing to overwrite existing output for $ExpName. Set ALLOW_EXISTING=1 only if you intentionally want to reuse the directory."
    }

    New-Item -ItemType Directory -Force -Path $OutDir, $ResultDir | Out-Null
}

function Write-ArgsFile {
    param(
        [string]$Path,
        [string]$Tag,
        [string]$ExpName,
        [string]$OutDir,
        [string]$SemanticLateStart,
        [string]$SemanticStartIter,
        [string]$UseSemanticDetach,
        [string[]]$TrainCommand
    )

    @(
        "tag: $Tag",
        "exp_name: $ExpName",
        "output_dir: $OutDir",
        "lambda_mask: 0.02",
        "lambda_semantic: 0.10",
        "use_mask_aux: True",
        "use_semantic_aux: True",
        "semantic_late_start: $SemanticLateStart",
        "semantic_start_iter: $SemanticStartIter",
        "use_semantic_warmup: False",
        "use_semantic_detach: $UseSemanticDetach",
        "",
        "train_command:",
        "  $(Join-Command $TrainCommand)"
    ) | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Write-TestTemplate {
    param(
        [string]$Path,
        [string]$ExpName,
        [string]$BestCkpt,
        [string]$UseSemanticDetach
    )

    @"
# Optional test command for the selected validation snapshot.
# Do not use test scores for checkpoint selection.
python test_card_spot.py `
  --cfg "$BaseCfg" `
  --snapshot "$BestCkpt" `
  --gpu "$PytorchGpu" `
  exp_dir "$ExpRoot" `
  exp_name "$ExpName" `
  model.enable_aux_mask True `
  train.use_semantic_aux True `
  train.lambda_mask 0.02 `
  train.lambda_semantic 0.10 `
  train.use_semantic_warmup False `
  train.use_semantic_detach "$UseSemanticDetach"

python evaluate_spot.py `
  --results_dir "$ExpRoot/$ExpName/test_output/captions" `
  --anno "$Anno"
"@ | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-LoggedCommand {
    param(
        [string[]]$Command,
        [string]$LogPath
    )

    $Exe = $Command[0]
    $Argv = $Command[1..($Command.Count - 1)]
    & $Exe @Argv 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $(Join-Command $Command)"
    }
}

function Run-One {
    param(
        [string]$Tag,
        [string]$ExpName,
        [string]$SemanticLateStart,
        [string]$SemanticStartIter,
        [string]$UseSemanticDetach
    )

    Prepare-OutputDirs $ExpName

    $OutDir = Join-Path $ExpRoot $ExpName
    $ResultDir = Join-Path $ResultsRoot $ExpName
    $EvalDir = Join-Path $OutDir 'eval_sents'
    $SelectDir = Join-Path $ResultDir 'snapshot_selection'

    $TrainCmd = @(
        'python', 'train_card_spot.py',
        '--cfg', $BaseCfg,
        '--exp_name', $ExpName,
        '--output_dir', $OutDir,
        '--use_mask_aux',
        '--use_semantic_aux',
        '--lambda_mask', '0.02',
        '--lambda_semantic', '0.10'
    )
    if ($SemanticLateStart -eq 'True') {
        $TrainCmd += @('--semantic_late_start', '--semantic_start_iter', $SemanticStartIter)
    }
    if ($UseSemanticDetach -eq 'True') {
        $TrainCmd += @('--use_semantic_detach')
    }
    $TrainCmd += @(
        'gpu_id', "[$PytorchGpu]",
        'train.max_iter', $MaxIter,
        'train.snapshot_interval', $SnapshotInterval,
        'train.use_mask_warmup', 'False',
        'train.use_semantic_warmup', 'False',
        'train.semantic_late_start', $SemanticLateStart,
        'train.semantic_start_iter', $SemanticStartIter,
        'train.use_semantic_detach', $UseSemanticDetach
    )

    Write-ArgsFile `
        -Path (Join-Path $OutDir 'args.txt') `
        -Tag $Tag `
        -ExpName $ExpName `
        -OutDir $OutDir `
        -SemanticLateStart $SemanticLateStart `
        -SemanticStartIter $SemanticStartIter `
        -UseSemanticDetach $UseSemanticDetach `
        -TrainCommand $TrainCmd

    Write-Host "========== [$Tag] train $ExpName =========="
    Invoke-LoggedCommand -Command $TrainCmd -LogPath (Join-Path $OutDir 'train.log')

    Write-Host "========== [$Tag] eval validation snapshots =========="
    $EvalCmd = @('python', 'evaluate_spot.py', '--results_dir', $EvalDir, '--anno', $Anno)
    Invoke-LoggedCommand -Command $EvalCmd -LogPath (Join-Path $ResultDir 'eval.log')
    Copy-Item -Force -LiteralPath (Join-Path $EvalDir 'eval_results.txt') -Destination (Join-Path $ResultDir 'eval_results.txt')

    Write-Host "========== [$Tag] select best snapshot from validation =========="
    $SelectCmd = @(
        'python', 'scripts/select_best_snapshot_from_eval_txt.py',
        '--input', (Join-Path $ResultDir 'eval_results.txt'),
        '--output_dir', $SelectDir,
        '--config_name', $ExpName,
        '--min_ckpt', $MinCkpt,
        '--max_ckpt', $MaxCkpt,
        '--stable_window', $StableWindow,
        '--save_all'
    )
    Invoke-LoggedCommand -Command $SelectCmd -LogPath (Join-Path $ResultDir 'select_snapshot.log')

    $BestRow = Import-Csv -LiteralPath (Join-Path $SelectDir 'best_checkpoint.csv') | Select-Object -First 1
    $BestCkpt = $BestRow.selected_checkpoint
    if (-not $BestCkpt) {
        throw "Failed to read selected checkpoint from $SelectDir/best_checkpoint.csv"
    }

    $TemplatePath = Join-Path $ResultDir 'test_command_template.txt'
    Write-TestTemplate $TemplatePath $ExpName $BestCkpt $UseSemanticDetach
    Add-Content -LiteralPath (Join-Path $ResultsRoot 'three_semantic_experiments_summary.csv') -Value "$Tag,$ExpName,$BestCkpt,$SelectDir/best_checkpoint.md,$TemplatePath"
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
Set-Content -LiteralPath (Join-Path $ResultsRoot 'three_semantic_experiments_summary.csv') -Value 'tag,exp_name,best_ckpt,selection_md,test_command_template' -Encoding UTF8

foreach ($Target in $Targets) {
    switch ($Target) {
        { $_ -in @('E1', 'latestart4000') } {
            Run-One 'E1' 'lmask002_lsem01_latestart4000' 'True' '4000' 'False'
            break
        }
        { $_ -in @('E2', 'latestart6000') } {
            Run-One 'E2' 'lmask002_lsem01_latestart6000' 'True' '6000' 'False'
            break
        }
        { $_ -in @('E3', 'semantic_detach') } {
            Run-One 'E3' 'lmask002_lsem01_semantic_detach' 'False' '5000' 'True'
            break
        }
        default {
            throw "Unknown experiment target: $Target. Valid targets: E1 E2 E3"
        }
    }
}

Write-Host "Finished requested experiments. Summary: $ResultsRoot/three_semantic_experiments_summary.csv"

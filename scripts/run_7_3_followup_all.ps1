param(
    [string]$OnlyExp = '',
    [switch]$Overwrite,
    [switch]$Force,
    [switch]$SkipCheck,
    [switch]$SkipTrain,
    [switch]$SkipReselect,
    [switch]$SkipTest,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)

$bashArgs = @('scripts/run_7_3_followup_all.sh')
if ($OnlyExp) { $bashArgs += @('--only_exp', $OnlyExp) }
if ($Overwrite) { $bashArgs += '--overwrite' }
if ($Force) { $bashArgs += '--force' }
if ($SkipCheck) { $bashArgs += '--skip_check' }
if ($SkipTrain) { $bashArgs += '--skip_train' }
if ($SkipReselect) { $bashArgs += '--skip_reselect' }
if ($SkipTest) { $bashArgs += '--skip_test' }
if ($DryRun) { $bashArgs += '--dry_run' }
$bashArgs += $ExtraArgs

& bash @bashArgs
exit $LASTEXITCODE
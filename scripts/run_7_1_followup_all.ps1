param(
    [string]$OnlyExp = '',
    [switch]$Overwrite,
    [switch]$Force,
    [switch]$SkipTrain,
    [switch]$SkipTest,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)

$bashArgs = @('scripts/run_7_1_followup_all.sh')
if ($OnlyExp) { $bashArgs += @('--only_exp', $OnlyExp) }
if ($Overwrite) { $bashArgs += '--overwrite' }
if ($Force) { $bashArgs += '--force' }
if ($SkipTrain) { $bashArgs += '--skip_train' }
if ($SkipTest) { $bashArgs += '--skip_test' }
if ($DryRun) { $bashArgs += '--dry_run' }
$bashArgs += $ExtraArgs

& bash @bashArgs
exit $LASTEXITCODE
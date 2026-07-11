param(
    [string]$OnlyExp = '',
    [switch]$Overwrite,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)

$bashArgs = @('scripts/run_7_4_followup_train.sh')
if ($OnlyExp) { $bashArgs += @('--only_exp', $OnlyExp) }
if ($Overwrite) { $bashArgs += '--overwrite' }
if ($DryRun) { $bashArgs += '--dry_run' }
$bashArgs += $ExtraArgs

& bash @bashArgs
exit $LASTEXITCODE
param(
    [string]$OnlyExp = '',
    [switch]$Force,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)

$bashArgs = @('scripts/run_7_3_followup_test_and_summary.sh')
if ($OnlyExp) { $bashArgs += @('--only_exp', $OnlyExp) }
if ($Force) { $bashArgs += '--force' }
if ($DryRun) { $bashArgs += '--dry_run' }
$bashArgs += $ExtraArgs

& bash @bashArgs
exit $LASTEXITCODE
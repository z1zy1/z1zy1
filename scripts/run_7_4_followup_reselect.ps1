param(
    [string]$OnlyExp = '',
    [switch]$Force,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs
)

$bashArgs = @('scripts/run_7_4_followup_reselect.sh')
if ($OnlyExp) { $bashArgs += @('--only_exp', $OnlyExp) }
if ($Force) { $bashArgs += '--force' }
if ($DryRun) { $bashArgs += '--dry_run' }
$bashArgs += $ExtraArgs

& bash @bashArgs
exit $LASTEXITCODE
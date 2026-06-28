param(
    [string]$ExpRoot = "./experiments",
    [string]$BaselineMetrics = ""
)
$experiments = @(
    "levir_mci_card_baseline",
    "levir_mci_card_mask",
    "levir_mci_card_semantic",
    "levir_mci_card_mask_semantic",
    "levir_mci_card_mask_semantic_pd05",
    "levir_mci_card_mask_semantic_pd05_noreweight",
    "levir_mci_card_mask_semantic_pd05_reweight",
    "levir_mci_wcsg_card_final",
    "second_cc_card_rgb_baseline",
    "second_cc_semantic_aux",
    "second_cc_semantic_crossattn",
    "second_cc_semantic_hard_gate",
    "second_cc_wcsg_card_final"
)
foreach ($exp in $experiments) {
    $expDir = Join-Path $ExpRoot $exp
    if (-not (Test-Path $expDir)) {
        Write-Host "Skipping missing experiment directory: $expDir"
        continue
    }
    $cmdArgs = @("scripts/select_best_checkpoint.py", "--exp_dir", $expDir, "--strategy", "spice_constrained_balanced")
    if ($BaselineMetrics) { $cmdArgs += @("--baseline_metrics", $BaselineMetrics) }
    python @cmdArgs
}
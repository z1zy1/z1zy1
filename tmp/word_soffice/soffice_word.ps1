param(
    [Parameter(Mandatory = $true)][string]$ConvertTo,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [Parameter(Mandatory = $true)][string]$InputPath
)

$ErrorActionPreference = 'Stop'
if ($ConvertTo -ne 'pdf') {
    Write-Error "Word compatibility shim only supports PDF conversion, got: $ConvertTo"
    exit 2
}

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$resolvedOutDir = [System.IO.Path]::GetFullPath($OutDir)
[System.IO.Directory]::CreateDirectory($resolvedOutDir) | Out-Null
$outputName = [System.IO.Path]::GetFileNameWithoutExtension($resolvedInput) + '.pdf'
$outputPath = Join-Path -Path $resolvedOutDir -ChildPath $outputName

$preconvertedPdf = $env:CODEX_PRECONVERTED_PDF
if ($preconvertedPdf -and (Test-Path -LiteralPath $preconvertedPdf)) {
    Copy-Item -LiteralPath $preconvertedPdf -Destination $outputPath -Force
    Write-Output "Copied preconverted PDF $preconvertedPdf -> $outputPath"
    exit 0
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($resolvedInput, $false, $true)
    # SaveAs2 with wdFormatPDF is more reliable than ExportAsFixedFormat in
    # unattended Word sessions on this workstation.
    $document.SaveAs2($outputPath, 17)
    Write-Output "Converted $resolvedInput -> $outputPath"
}
finally {
    if ($document -ne $null) {
        $document.Close($false)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
    }
    if ($word -ne $null) {
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

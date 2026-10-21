param(
    [string]$LandmarksDir = 'data/landmarks/minds_libras_school8',
    [string]$Manifest = 'Modelos salvos/transformer_minds_8_classes_pose_face/data_split/manifest.json',
    [string]$Splits = 'Modelos salvos/transformer_minds_8_classes_pose_face/data_split/splits.json',
    [string]$OutputDir = 'reports/hand_coverage_school8'
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Read-HandMaskCounts([string]$NpzPath) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($NpzPath)
    try {
        $entry = $zip.GetEntry('mask.npy')
        if ($null -eq $entry) { throw "mask.npy ausente: $NpzPath" }
        $stream = $entry.Open()
        $memory = New-Object System.IO.MemoryStream
        try {
            $stream.CopyTo($memory)
            $bytes = $memory.ToArray()
        } finally {
            $memory.Dispose()
            $stream.Dispose()
        }
    } finally {
        $zip.Dispose()
    }

    if ($bytes.Length -lt 12 -or [System.Text.Encoding]::ASCII.GetString($bytes, 1, 5) -ne 'NUMPY' -or $bytes[0] -ne 147) {
        throw "Cabecalho NPY invalido: $NpzPath"
    }
    $version = [int]$bytes[6]
    if ($version -eq 1) {
        $headerLength = [int][System.BitConverter]::ToUInt16($bytes, 8)
        $offset = 10
    } elseif ($version -eq 2 -or $version -eq 3) {
        $headerLength = [int][System.BitConverter]::ToUInt32($bytes, 8)
        $offset = 12
    } else {
        throw "Versao NPY nao suportada: $version em $NpzPath"
    }
    if ($bytes.Length -lt $offset + $headerLength) { throw "Cabecalho NPY truncado: $NpzPath" }
    $header = [System.Text.Encoding]::ASCII.GetString($bytes, $offset, $headerLength)
    if ($header -notmatch "'descr'\s*:\s*'\|b1'" -or $header -notmatch "'fortran_order'\s*:\s*False") {
        throw "Formato da mascara diferente de booleano C-order: $NpzPath ($header)"
    }
    if ($header -notmatch "'shape'\s*:\s*\(\s*(\d+)\s*,\s*4\s*\)") {
        throw "Mascara nao tem shape (T, 4): $NpzPath ($header)"
    }
    $frames = [int]$Matches[1]
    $start = $offset + $headerLength
    if ($frames -le 0 -or $bytes.Length -ne $start + $frames * 4) {
        throw "Tamanho da mascara inconsistente: $NpzPath"
    }

    $left = 0; $right = 0; $both = 0; $none = 0; $leftOnly = 0; $rightOnly = 0
    for ($i = 0; $i -lt $frames; $i++) {
        $l = [int]$bytes[$start + 4 * $i]
        $r = [int]$bytes[$start + 4 * $i + 1]
        if ($l -gt 1 -or $r -gt 1) { throw "Valor nao booleano na mascara: $NpzPath" }
        if ($l -eq 1) { $left++ }
        if ($r -eq 1) { $right++ }
        if ($l -eq 1 -and $r -eq 1) { $both++ }
        elseif ($l -eq 1) { $leftOnly++ }
        elseif ($r -eq 1) { $rightOnly++ }
        else { $none++ }
    }
    if ($both + $leftOnly + $rightOnly + $none -ne $frames) { throw "Contagem inconsistente: $NpzPath" }
    return [pscustomobject]@{
        frames = $frames; left = $left; right = $right; both = $both
        left_only = $leftOnly; right_only = $rightOnly; none = $none
    }
}

function Get-GroupSummary([object[]]$Records, [string]$Property) {
    foreach ($item in ($Records | Group-Object -Property $Property | Sort-Object Name)) {
        $totals = @{}
        foreach ($field in @('frames', 'left', 'right', 'both', 'left_only', 'right_only', 'none')) {
            $totals[$field] = [int](($item.Group | Measure-Object -Property $field -Sum).Sum)
        }
        $f = $totals['frames']
        $halfBoth = @($item.Group | Where-Object { $_.both / $_.frames -ge 0.5 }).Count
        $meanVideoBoth = ($item.Group | ForEach-Object { 100.0 * $_.both / $_.frames } | Measure-Object -Average).Average
        [pscustomobject]@{
            group = $item.Name
            videos = $item.Count
            frames = $f
            left_pct = [math]::Round(100.0 * $totals['left'] / $f, 2)
            right_pct = [math]::Round(100.0 * $totals['right'] / $f, 2)
            both_pct = [math]::Round(100.0 * $totals['both'] / $f, 2)
            any_pct = [math]::Round(100.0 * ($f - $totals['none']) / $f, 2)
            none_pct = [math]::Round(100.0 * $totals['none'] / $f, 2)
            left_only_pct = [math]::Round(100.0 * $totals['left_only'] / $f, 2)
            right_only_pct = [math]::Round(100.0 * $totals['right_only'] / $f, 2)
            mean_video_both_pct = [math]::Round($meanVideoBoth, 2)
            videos_at_least_half_both = $halfBoth
            videos_without_any_hand = @($item.Group | Where-Object { $_.left -eq 0 -and $_.right -eq 0 }).Count
        }
    }
}

$landmarksRoot = [System.IO.Path]::GetFullPath($LandmarksDir)
$manifestRows = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
$splitData = Get-Content -LiteralPath $Splits -Raw -Encoding UTF8 | ConvertFrom-Json
$splitByIndex = @{}
foreach ($split in @('train', 'val', 'test')) {
    foreach ($index in $splitData.$split) {
        $key = [int]$index
        if ($splitByIndex.ContainsKey($key)) { throw "Indice repetido entre splits: $key" }
        $splitByIndex[$key] = $split
    }
}
if ($splitByIndex.Count -ne $manifestRows.Count) {
    throw "Splits cobrem $($splitByIndex.Count) de $($manifestRows.Count) entradas do manifesto"
}
$rows = New-Object System.Collections.Generic.List[object]
$seenPaths = @{}
for ($index = 0; $index -lt $manifestRows.Count; $index++) {
    $sample = $manifestRows[$index]
    if (-not $splitByIndex.ContainsKey($index)) { throw "Indice $index sem split" }
    $path = [System.IO.Path]::GetFullPath((Join-Path $landmarksRoot $sample.path))
    if (-not $path.StartsWith($landmarksRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Caminho fora da pasta de landmarks: $($sample.path)"
    }
    if ($seenPaths.ContainsKey($path)) { throw "Arquivo duplicado no manifesto: $path" }
    $seenPaths[$path] = $true
    if (-not [System.IO.File]::Exists($path)) { throw "Arquivo ausente: $path" }
    $counts = Read-HandMaskCounts $path
    $split = $splitByIndex[$index]
    $rows.Add([pscustomobject]@{
        path = [string]$sample.path
        label = [string]$sample.label
        signer = [string]$sample.signer
        split = $split
        dataset = 'all'
        class_signer = "$($sample.label)|$($sample.signer)"
        split_class = "$split|$($sample.label)"
        frames = $counts.frames
        left = $counts.left
        right = $counts.right
        both = $counts.both
        left_only = $counts.left_only
        right_only = $counts.right_only
        none = $counts.none
        both_pct = [math]::Round(100.0 * $counts.both / $counts.frames, 2)
    })
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$outputRoot = [System.IO.Path]::GetFullPath($OutputDir)
$perVideo = $rows.ToArray()
$overall = @(Get-GroupSummary $perVideo 'split')
$total = @(Get-GroupSummary $perVideo 'dataset')
$byClass = @(Get-GroupSummary $perVideo 'label')
$bySigner = @(Get-GroupSummary $perVideo 'signer')
$bySplitClass = @(Get-GroupSummary $perVideo 'split_class')
$byClassSigner = @(Get-GroupSummary $perVideo 'class_signer')
$perVideo | Export-Csv -LiteralPath (Join-Path $outputRoot 'per_video.csv') -NoTypeInformation -Encoding UTF8
$overall | Export-Csv -LiteralPath (Join-Path $outputRoot 'by_split.csv') -NoTypeInformation -Encoding UTF8
$byClass | Export-Csv -LiteralPath (Join-Path $outputRoot 'by_class.csv') -NoTypeInformation -Encoding UTF8
$bySigner | Export-Csv -LiteralPath (Join-Path $outputRoot 'by_signer.csv') -NoTypeInformation -Encoding UTF8
$bySplitClass | Export-Csv -LiteralPath (Join-Path $outputRoot 'by_split_class.csv') -NoTypeInformation -Encoding UTF8
$byClassSigner | Export-Csv -LiteralPath (Join-Path $outputRoot 'by_class_signer.csv') -NoTypeInformation -Encoding UTF8
$report = [ordered]@{
    source_manifest = $Manifest
    manifest_sha256 = (Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    source_splits = $Splits
    splits_sha256 = (Get-FileHash -LiteralPath $Splits -Algorithm SHA256).Hash.ToLowerInvariant()
    landmarks_dir = $LandmarksDir
    num_videos = $perVideo.Count
    total = $total
    by_split = $overall
    by_class = $byClass
    by_signer = $bySigner
    by_split_class = $bySplitClass
}
[System.IO.File]::WriteAllText((Join-Path $outputRoot 'summary.json'), ($report | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Auditoria concluida: $($perVideo.Count) videos. Relatorios em $outputRoot"
$byClass | Format-Table group, videos, frames, left_pct, right_pct, any_pct, both_pct, none_pct -AutoSize

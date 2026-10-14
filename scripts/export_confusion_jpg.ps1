param(
    [Parameter(Mandatory = $true)][string]$SvgPath,
    [string]$JpgPath
)

$ErrorActionPreference = 'Stop'
$svg = (Resolve-Path -LiteralPath $SvgPath).Path
if (-not $JpgPath) { $JpgPath = [IO.Path]::ChangeExtension($svg, '.jpg') }
$jpg = [IO.Path]::GetFullPath($JpgPath)
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
if (-not (Test-Path -LiteralPath $edge)) {
    throw 'Microsoft Edge nao encontrado no caminho esperado.'
}

$header = Get-Content -LiteralPath $svg -TotalCount 1
$width = [regex]::Match($header, 'width="(\d+)"').Groups[1].Value
$height = [regex]::Match($header, 'height="(\d+)"').Groups[1].Value
if (-not $width -or -not $height) { throw 'SVG sem largura e altura numericas.' }

$destination = [IO.Path]::GetDirectoryName($jpg)
[IO.Directory]::CreateDirectory($destination) | Out-Null
$png = [IO.Path]::ChangeExtension($jpg, '.png')
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
$profile = Join-Path $tempRoot ('sinaliza-matrix-export-' + [guid]::NewGuid().ToString('N'))
$uri = [System.Uri]::new($svg).AbsoluteUri
$edgeArgs = @(
    '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
    ('--user-data-dir=' + $profile), ('--screenshot=' + $png),
    ('--window-size=' + $width + ',' + $height), $uri
)

try {
    $process = Start-Process -FilePath $edge -ArgumentList $edgeArgs -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $png)) {
        throw "Edge nao conseguiu renderizar o SVG (codigo $($process.ExitCode))."
    }

    Add-Type -AssemblyName System.Drawing
    $image = [System.Drawing.Image]::FromFile($png)
    $encoderParams = [System.Drawing.Imaging.EncoderParameters]::new(1)
    try {
        $jpegCodec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() |
            Where-Object { $_.MimeType -eq 'image/jpeg' } | Select-Object -First 1
        if (-not $jpegCodec) { throw 'Codificador JPEG indisponivel.' }
        $encoderParams.Param[0] = [System.Drawing.Imaging.EncoderParameter]::new(
            [System.Drawing.Imaging.Encoder]::Quality, [long]92)
        $image.Save($jpg, $jpegCodec, $encoderParams)
    } finally {
        $image.Dispose()
        $encoderParams.Dispose()
    }
    Write-Output "JPG salvo: $jpg"
} finally {
    if (Test-Path -LiteralPath $png) { Remove-Item -LiteralPath $png -Force }
    $resolvedProfile = [IO.Path]::GetFullPath($profile)
    if ($resolvedProfile.StartsWith($tempRoot + '\sinaliza-matrix-export-', [StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $resolvedProfile)) {
        Remove-Item -LiteralPath $resolvedProfile -Recurse -Force -ErrorAction SilentlyContinue
    }
}

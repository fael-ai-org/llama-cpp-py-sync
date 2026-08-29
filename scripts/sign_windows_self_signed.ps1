[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Root,

    [Parameter(Mandatory = $true)]
    [string]$CertificateOutput
)

$ErrorActionPreference = 'Stop'

$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
$certificatePath = [System.IO.Path]::GetFullPath($CertificateOutput)
$certificateParent = Split-Path -Parent $certificatePath
New-Item -ItemType Directory -Force -Path $certificateParent | Out-Null

$signtoolCommand = Get-Command signtool.exe -ErrorAction SilentlyContinue
if ($null -ne $signtoolCommand) {
    $signtool = $signtoolCommand.Source
} else {
    $kitsRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    $signtool = Get-ChildItem -LiteralPath $kitsRoot -Filter 'signtool.exe' -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '\\x64\\signtool\.exe$' } |
        Sort-Object FullName -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $signtool) {
    throw 'signtool.exe was not found. Install the Windows SDK on the runner.'
}

# Only sign files owned by the LLaMA/ggml build. Microsoft, CUDA, Vulkan, and
# other redistributable DLLs retain their vendor identity and signatures.
$ownedPrefixes = @('llama', 'libllama', 'ggml', 'libggml', 'mtmd', 'libmtmd')
$signable = Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File |
    Where-Object {
        $extension = $_.Extension.ToLowerInvariant()
        $name = $_.Name.ToLowerInvariant()
        $isNative = $extension -in @('.dll', '.pyd', '.exe')
        $isOwned = $extension -eq '.pyd'
        foreach ($prefix in $ownedPrefixes) {
            if ($name.StartsWith($prefix)) {
                $isOwned = $true
                break
            }
        }
        $isNative -and $isOwned
    } |
    Sort-Object FullName

if (-not $signable) {
    throw "No LLaMA-owned Windows PE files were found under $resolvedRoot."
}

$certificate = New-SelfSignedCertificate `
    -Type CodeSigningCert `
    -Subject 'CN=Ephemeral CI Test Certificate' `
    -CertStoreLocation 'Cert:\CurrentUser\My' `
    -KeyAlgorithm RSA `
    -KeyLength 3072 `
    -HashAlgorithm SHA256 `
    -KeyExportPolicy NonExportable `
    -NotAfter (Get-Date).AddMonths(6)

try {
    Export-Certificate -Cert $certificate -FilePath $certificatePath -Type CERT | Out-Null
    Write-Host "Created ephemeral self-signed certificate: $($certificate.Thumbprint)"

    foreach ($file in $signable) {
        & $signtool sign /fd SHA256 /sha1 $certificate.Thumbprint /s MY /d 'FAEL LLaMA (TEST)' $file.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "signtool failed for $($file.Name)."
        }

        $signature = Get-AuthenticodeSignature -LiteralPath $file.FullName
        if ($null -eq $signature.SignerCertificate -or $signature.Status -eq 'NotSigned') {
            throw "The signed file has no Authenticode signature: $($file.Name)."
        }
        if ($signature.SignerCertificate.Thumbprint -ne $certificate.Thumbprint) {
            throw "The Authenticode signer does not match the test certificate: $($file.Name)."
        }
        Write-Host "Signed: $($file.FullName) [$($signature.Status)]"
    }

    Write-Host "Signed $($signable.Count) LLaMA-owned Windows PE file(s)."
    Write-Host "Public certificate: $certificatePath"
} finally {
    Remove-Item -LiteralPath "Cert:\CurrentUser\My\$($certificate.Thumbprint)" -Force
    Write-Host 'Removed the ephemeral private key from the runner certificate store.'
}

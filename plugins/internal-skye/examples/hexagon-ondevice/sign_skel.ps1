# Creates a self-signed code-signing certificate, trusts it machine-wide, and
# signs the skel catalogue. Requires elevation. Uses only built-in cmdlets -
# no signtool (which ships with the Windows SDK, not the WDK we have).
$ErrorActionPreference = 'Stop'

$dir = 'C:\Users\Public\mojonpu'
$cat = Join-Path $dir 'libmojo_npu.cat'
$subject = 'CN=Modular Mojo NPU Test Signing'

if (-not (Test-Path $cat)) { throw "catalogue missing: $cat" }

# Reuse an existing cert if this script already ran, so repeat runs are safe.
$cert = Get-ChildItem Cert:\LocalMachine\My |
        Where-Object { $_.Subject -eq $subject } |
        Select-Object -First 1

if ($cert) {
    "reusing existing cert : $($cert.Thumbprint)"
} else {
    $cert = New-SelfSignedCertificate `
        -Subject $subject `
        -Type CodeSigningCert `
        -CertStoreLocation 'Cert:\LocalMachine\My' `
        -KeyUsage DigitalSignature `
        -KeyLength 2048 `
        -NotAfter (Get-Date).AddYears(2)
    "created cert          : $($cert.Thumbprint)"
}

# Machine-wide trust: the driver validates against these two stores.
$tmp = Join-Path $env:TEMP 'mojo_npu_cert.cer'
Export-Certificate -Cert $cert -FilePath $tmp -Force | Out-Null
foreach ($store in 'Root', 'TrustedPublisher') {
    Import-Certificate -FilePath $tmp -CertStoreLocation "Cert:\LocalMachine\$store" | Out-Null
    "imported to           : LocalMachine\$store"
}
Remove-Item $tmp -Force

$sig = Set-AuthenticodeSignature -FilePath $cat -Certificate $cert -HashAlgorithm SHA256
"sign status           : $($sig.Status)"

$v = Get-AuthenticodeSignature -FilePath $cat
"verify status         : $($v.Status)"
"verify message        : $($v.StatusMessage)"
"signer                : $($v.SignerCertificate.Subject)"

# Register the catalogue in the system database so WinVerifyTrust can find it
# by hash. Without this a sibling .cat may simply be ignored.
try {
    $r = pnputil /add-driver (Join-Path $dir 'libmojo_npu.inf') 2>&1
    "pnputil               : $($r -join ' | ')"
} catch {
    "pnputil FAILED        : $($_.Exception.Message)"
}

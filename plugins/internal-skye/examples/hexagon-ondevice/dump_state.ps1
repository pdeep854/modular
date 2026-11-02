# Dumps the machine state that distinguishes which provisioning steps were
# actually required. Read-only; changes nothing.
# ASCII only: PowerShell 5.1 reads scripts as ANSI unless a BOM is present,
# so a non-ASCII character here can break string parsing.
$ErrorActionPreference = 'Continue'

"=== BitLocker (was the volume decrypted?) ==="
$vol = Get-BitLockerVolume -MountPoint 'C:' -ErrorAction SilentlyContinue
if ($vol) {
    "VolumeStatus         : " + $vol.VolumeStatus
    "EncryptionPercentage : " + $vol.EncryptionPercentage
    "ProtectionStatus     : " + $vol.ProtectionStatus
    "KeyProtector         : " + ($vol.KeyProtector -join ', ')
} else {
    "unreadable"
}

""
"=== Secure Boot platform state (were platform keys cleared?) ==="
foreach ($name in 'SetupMode', 'AuditMode') {
    try {
        $bytes = (Get-SecureBootUEFI -Name $name).Bytes
        "$name : " + ($bytes -join ',') + "   # 1 means Setup Mode, PK cleared"
    } catch {
        "$name : unavailable"
    }
}
foreach ($name in 'PK', 'KEK', 'db') {
    try {
        $bytes = (Get-SecureBootUEFI -Name $name).Bytes
        "$name size : " + $bytes.Count + " bytes"
    } catch {
        "$name : unavailable, consistent with having been cleared"
    }
}

""
"=== confirm the rest ==="
$sb = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\SecureBoot\State'
"SecureBootEnabled : " + $sb.UEFISecureBootEnabled
$dg = Get-CimInstance -ClassName Win32_DeviceGuard -Namespace root\Microsoft\Windows\DeviceGuard -ErrorAction SilentlyContinue
"VBS status        : " + $dg.VirtualizationBasedSecurityStatus + "  configured={" + ($dg.SecurityServicesConfigured -join ',') + "}  running={" + ($dg.SecurityServicesRunning -join ',') + "}"
$so = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control' -Name SystemStartOptions
"BootOptions       : " + $so.SystemStartOptions

""
"=== BCD elements currently set ==="
bcdedit /enum "{current}" | Select-String -Pattern 'identifier','testsigning','nointegritychecks','hypervisorlaunchtype','vsmlaunchtype'

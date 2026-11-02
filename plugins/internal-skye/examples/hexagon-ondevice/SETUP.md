# Provisioning a Windows-on-ARM laptop for on-device NPU development

One-time machine setup, verified on a Snapdragon X2-class laptop (Insyde
firmware) running Windows 11 ARM64, to the point of a Mojo kernel executing on
the NPU. For building and running once the machine is set up, see `README.md`.

Budget half a day. Most of it is the drive decryption in step 1 and the SDK
download; the rest is firmware menus and reboots.

## The end state you are aiming for

Every value below was measured on the working machine. If `bcdedit` refuses a
change, compare against this table before trying anything clever — the usual
cause is that Secure Boot platform keys have not been cleared.

| Property                         | Required value                             |
|----------------------------------|--------------------------------------------|
| `VolumeStatus`                   | `FullyDecrypted`, `EncryptionPercentage 0` |
| `SetupMode` UEFI variable        | `1` (platform keys cleared)                |
| `PK` / `KEK` / `db`              | unavailable                                |
| `UEFISecureBootEnabled`          | `0`                                        |
| `testsigning`                    | `Yes`                                      |
| `nointegritychecks`              | `Yes`                                      |
| VBS `SecurityServicesConfigured` | `{3}` (Memory Integrity off)               |

`hypervisorlaunchtype` and `vsmlaunchtype` remain `Auto`, and VBS still reports
status `2` with System Guard (service `3`) running. That is expected and
blocks nothing — DRTM is firmware-enforced and does not need to be off.

A caveat worth stating: this is the configuration in which it demonstrably
works. It has not been ablated, so some steps may be sufficient rather than
strictly necessary. Nobody should spend time finding out; follow the sequence.

Dump your own machine's state with `dump_state.ps1` in this directory:

```powershell
sudo cmd /c "powershell -ExecutionPolicy Bypass -File dump_state.ps1"
```

## Step 1 - decrypt the OS volume

Device encryption blocks manipulation of the BCD store. Suspending is **not**
enough; the volume must be fully decrypted.

```powershell
Disable-BitLocker -MountPoint "C:"
```

Watch it, and do not reboot until it finishes:

```powershell
Get-BitLockerVolume -MountPoint "C:" | Select-Object VolumeStatus,EncryptionPercentage
```

Wait for `FullyDecrypted` and `0`. On a ~1 TB volume this takes a while;
leave it plugged in.

## Step 2 - clear the Secure Boot platform keys

**This is the step everything else depends on, and the one most easily
underestimated.** Setting Secure Boot to *Disabled* in the menu is not
sufficient. The platform keys must be cleared from NVRAM so the firmware enters
Setup Mode; until then the BCD store stays locked and `bcdedit` returns
`Access is denied` no matter how elevated the shell is.

1. Shut down. Hold **F2** and tap the **power button** to enter firmware setup.
2. **F7** for Advanced Mode.
3. **Security > Secure Boot**, set **Secure Boot Mode** to **Custom**.
4. Enter **Key Management** and clear the keys - **Delete PK**, or
   **Clear Secure Boot Keys** / **Reset to Setup Mode** if offered.
5. **F10** to save and reboot, then immediately tap **F2** again to re-enter
   setup so the platform reset applies before Windows boots.
6. Confirm the header now reads **Platform Mode: Setup** and
   **Secure Boot: Not Active**.
7. Set **OS Type** to **Other OS**, and **Secure Boot Control** to **Disabled**.
8. **F10** to save and exit.

Verify from Windows, no elevation needed:

```powershell
(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\SecureBoot\State').UEFISecureBootEnabled
```

Must print `0`. `dump_state.ps1` additionally confirms `SetupMode : 1`.

## Step 3 - set the BCD flags from the recovery environment

Windows 11 on ARM refuses these from a booted session - `bcdedit` returns
`Access is denied` regardless of elevation, including under `sudo`. Set them
from WinRE instead, which is on the internal SSD and needs no USB media.

1. Hold **Shift** and click **Restart** from the Windows power menu.
2. **Troubleshoot > Advanced options > Command Prompt.** You land on a WinPE
   prompt at `X:\windows\system32>`.
3. Run:

   ```cmd
   bcdedit /set {default} testsigning on
   bcdedit /set {default} nointegritychecks on
   bcdedit /enum {default}
   ```

4. Confirm the enum lists `testsigning Yes` and `nointegritychecks Yes`.
5. `exit`, then **Continue** to boot back into Windows.

Three details that matter:

- **The `{default}` token is load-bearing.** WinRE has its own boot context, so
  an unqualified `bcdedit /set testsigning on` can target the wrong entry.
- **No `diskpart` and no `/store`.** WinRE scans the drive controllers on
  startup and mounts the primary installation's BCD hive itself, so there is no
  need to assign a letter to the EFI System Partition or point `bcdedit` at it.
- This is a Windows-style prompt, not a firmware EFI Shell. There is no
  `Shell>` prompt and no `FS0:` volume mapping involved.

Reboot. You should get a **Test Mode** watermark bottom-right.

If `bcdedit` still refuses here, step 2 did not take - the platform keys are
almost certainly still present. Re-check `SetupMode` before trying anything
else.

## Step 4 - turn Memory Integrity off

Settings > Privacy & security > Windows Security > Device security >
**Core isolation details**, set **Memory Integrity** to **Off**, reboot.

Confirm `SecurityServicesConfigured` no longer contains `2`. It will still list
`3`; that is System Guard and is fine.

## Step 5 - install the toolchain

1. **Hexagon SDK 6.6.0.0, `arm64-wos` build**, to
   `C:\Qualcomm\Hexagon_SDK\6.6.0.0`. A plain GitHub release from
   [snapdragon-toolchain/hexagon-sdk][sdk], no vendor account needed. There is
   **no Linux-aarch64 build**, so it lives on the Windows side even if you build
   from WSL. Extract with the *Windows* tar rather than pushing gigabytes
   through the WSL filesystem bridge:

   ```bash
   mkdir -p /mnt/c/Qualcomm/Hexagon_SDK
   cd /mnt/c && /mnt/c/Windows/System32/tar.exe -xf \
     "C:\path\to\hexagon-sdk-v6.6.0.0-arm64-wos.tar.xz" \
     -C "C:\Qualcomm\Hexagon_SDK"
   ```

2. **Windows Driver Kit**, for `Inf2Cat`:

   ```powershell
   winget install Microsoft.WindowsWDK.10.0.26100
   ```

   The Windows SDK is not required - `signtool` is replaced by PowerShell's
   built-in `Set-AuthenticodeSignature`. `Inf2Cat` ships x86-only and runs fine
   under emulation; do not hunt for an arm64 build.

3. **Python for Windows ARM64**, if the host driver will be Python:

   ```powershell
   winget install Python.Python.3.13 --scope user
   ```

   Check whether `python.exe` is real first. A bare `python.exe` on `PATH` is
   often the Microsoft Store alias stub, which prints "Python was not found" and
   exits 9009.

[sdk]: https://github.com/snapdragon-toolchain/hexagon-sdk/releases

## Step 6 - build, sign, run

```bash
HEX_VER=81 ./build.sh
```

Then, once per machine, create and trust a signing certificate and sign the
catalogue:

```powershell
cd C:\Users\Public\mojonpu
sudo cmd /c "powershell -ExecutionPolicy Bypass -File sign_skel.ps1"
```

`pnputil` only succeeds once test-signing is active - a self-signed chain is
otherwise rejected with *"the publisher of an Authenticode signed catalog was
not established as trusted"*. That makes it a useful check that step 3 worked.

**Re-run the signing after every build.** `Inf2Cat` regenerates the catalogue
whenever the skel changes, which invalidates the signature. A stale catalogue
shows up as `AEE_EUNABLETOLOAD`.

```powershell
python.exe C:\Users\Public\mojonpu\run_npu.py
```

Expect `handle_open : 0x00000000 SUCCESS` and a `PASS` line.

## Security posture this leaves behind

State it plainly to whoever owns the machine's compliance:

- **Disk encryption: off**, and it cannot be re-enabled while test-signing is
  on, since Device Encryption requires Secure Boot. This is the item most likely
  to need a written exemption.
- **Secure Boot: off, and platform keys cleared.** The firmware is in Setup
  Mode.
- **Test-signing and `nointegritychecks`: on.** Kernel-mode signature
  enforcement is relaxed machine-wide, not only for our skel.
- **Memory Integrity: off.**
- **A self-signed code-signing certificate is trusted machine-wide.** Narrow
  scope - `Code Signing` EKU only, not a CA, so it cannot issue subordinates or
  be used for TLS interception - but its private key is on the machine.
- **Retained:** System Guard still runs, since firmware enforces it.

This is a substantial reduction, appropriate for a dedicated development
machine and not for one holding sensitive data. The exit path is attestation
signing through the Microsoft Partner Center: a Microsoft-rooted signature that
`pnputil` accepts without test mode, letting every item above be restored.

## Reverting

```powershell
sudo cmd /c "bcdedit /set nointegritychecks off"
sudo cmd /c "bcdedit /set testsigning off"
```

Remove the certificate, which achieves nothing once test-signing is off:

```powershell
sudo cmd /c "powershell -Command \"Get-ChildItem Cert:\LocalMachine\Root, Cert:\LocalMachine\TrustedPublisher, Cert:\LocalMachine\My | Where-Object Subject -match 'Mojo NPU Test Signing' | Remove-Item -Force\""
```

Then in firmware: **Key Management > Restore Factory Keys** (or **Install
Default Secure Boot Keys**), **OS Type** back to **Windows UEFI Mode**, and
**Secure Boot Control** to **Enabled**. Finally, back in Windows, re-enable
**Memory Integrity** under Core isolation and turn **Device encryption** back
on.

## Provenance, and what a fresh-machine run should check

Not all of this is equally trustworthy. Where it came from:

- **Verified by measurement** on the working machine: the end-state table, and
  steps 5 and 6 (toolchain, build, sign, run) which were performed directly.
- **Verified by the operator's account**: step 3, including the `{default}`
  token and that neither `diskpart` nor `/store` is needed.
- **Reconstructed, not observed**: the firmware menu navigation in step 2. The
  end state (`SetupMode 1`, keys absent) is measured, but the exact labels —
  "Secure Boot Mode > Custom", "Key Management", "Delete PK", the F2-plus-power
  entry, the re-enter-setup-immediately-after-saving step — come from a guide
  rather than from watching the screens. Insyde builds vary; expect to adapt.

Specific questions a first run on a fresh machine should answer, none of which
we can currently settle:

1. **Is `nointegritychecks` actually required**, or incidental? It is set here,
   but nothing isolated its contribution.
2. **Does decryption have to precede clearing the platform keys**, or is the
   order free? Both are true of the end state; the dependency is unknown.
3. **Is the "re-enter setup immediately after saving" step in 2.5 real**, or
   superstition carried from the source guide?
4. **Does Memory Integrity have to be off**, or was it turned off en route and
   simply left that way? It is off here and test-signing works, but so would a
   configuration where it never mattered.
5. **Which steps need their own reboot?** The doc implies one where each is
   described; the true minimum is not established.
6. **Does anything survive Windows Update** — in particular whether Memory
   Integrity or the platform keys get restored, which would silently break a
   working machine.

Whoever runs this: please record what actually happened at each step, including
the menu labels you saw, and correct the doc in place. A run that succeeds
without answering the questions above is less useful than one that fails while
answering them.

## Gotchas

- **`.ps1` files must be ASCII.** PowerShell 5.1 reads scripts as ANSI unless
  they carry a BOM, so a stray en dash or smart quote mangles and breaks string
  parsing with errors that point at the wrong line.
- **Windows `sudo` hides output** by default, running the command in a window
  that closes immediately. Redirect to a file, or set it inline once with
  `sudo config --enable disableInput`.
- **`HKLM:\BCD00000000\Objects` enumerates to zero entries without elevation.**
  It does not error, so any query over it reports "not set" regardless of the
  truth. Do not read BCD state that way.

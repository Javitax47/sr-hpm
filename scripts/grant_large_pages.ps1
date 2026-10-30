# Grants the current user the "Lock pages in memory" right (SeLockMemoryPrivilege), so SR-HPM can keep the CPU-side
# weights on 2 MB pages (GGML_LARGE_PAGES=1, set by the launcher when the right is present). Works on Windows Home,
# which has no secpol.msc. Run as administrator, then sign out and back in.
#
# Undo: secedit /export /cfg x.inf /areas USER_RIGHTS, remove your account from the SeLockMemoryPrivilege line, then
# secedit /configure /db x.sdb /cfg x.inf /areas USER_RIGHTS.
$ErrorActionPreference = "Stop"
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "run this script as administrator"
}
$user = $env:SRHPM_USER
if (-not $user) {
    # the interactive user (an elevated console can belong to another admin account)
    $user = (Get-CimInstance Win32_ComputerSystem).UserName
}
$sid = (New-Object Security.Principal.NTAccount($user)).Translate([Security.Principal.SecurityIdentifier]).Value
$tmp = Join-Path $env:TEMP "srhpm_lockmem"
$cur = "$tmp.cur.inf"
secedit /export /cfg $cur /areas USER_RIGHTS | Out-Null
$line = Select-String -Path $cur -Pattern "^SeLockMemoryPrivilege\s*=\s*(.*)$" | Select-Object -First 1
$holders = if ($line) { $line.Matches[0].Groups[1].Value.Trim() } else { "" }
if ($holders -match [regex]::Escape($sid) -or $holders -match [regex]::Escape(($user -split '\\')[-1])) {
    Write-Host "$user already has 'Lock pages in memory'"; exit 0
}
$value = if ($holders) { "$holders,*$sid" } else { "*$sid" }
Set-Content "$tmp.inf" -Encoding Unicode -Value @(
    "[Unicode]", "Unicode=yes", "[Version]", 'signature="$CHICAGO$"', "Revision=1",
    "[Privilege Rights]", "SeLockMemoryPrivilege = $value")
secedit /configure /db "$tmp.sdb" /cfg "$tmp.inf" /areas USER_RIGHTS | Out-Null
secedit /export /cfg $cur /areas USER_RIGHTS | Out-Null
Select-String -Path $cur -Pattern "^SeLockMemoryPrivilege"
Write-Host "Granted to $user ($sid). Sign out and back in for it to take effect." -ForegroundColor Green

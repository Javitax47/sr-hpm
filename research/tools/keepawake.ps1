# Keeps the machine from sleeping while a file does not exist (a runtime request, like a media player's: no power
# setting is changed). usage: powershell -File keepawake.ps1 -Until C:\path\DONE
param([string]$Until)
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$k = Add-Type -MemberDefinition $sig -Name K -Namespace W -PassThru
while (-not (Test-Path $Until)) {
    [void]$k::SetThreadExecutionState([uint32]"0x80000001")  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    Start-Sleep -Seconds 30
}
[void]$k::SetThreadExecutionState([uint32]"0x80000000")

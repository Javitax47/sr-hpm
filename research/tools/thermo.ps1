# Samples CPU performance (% of nominal frequency, so >100 = boost) and GPU clock / power / temperature every 2 s
# into a CSV, to see throttling during benchmarks. Stops when the file named by -Stop exists.
# usage: powershell -File f3/thermo.ps1 -Out thermo.csv -Stop thermo.stop
param([string]$Out, [string]$Stop)
"time,cpu_perf_pct,cpu_mhz,gpu_mhz,gpu_w,gpu_c,gpu_util" | Out-File -Encoding ascii $Out
while (-not (Test-Path $Stop)) {
    $c = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'"
    $g = (& nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits) -replace ' ', ''
    "{0},{1},{2},{3}" -f (Get-Date -Format 'HH:mm:ss'), $c.PercentProcessorPerformance, $c.ProcessorFrequency, $g | Out-File -Append -Encoding ascii $Out
    Start-Sleep -Seconds 2
}

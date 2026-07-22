param(
    [string]$Server = "liangshengyu@192.168.106.20",
    [int]$LocalPort = 18765,
    [int]$RemotePort = 18765
)

$ErrorActionPreference = "Stop"
$dashboardUrl = "http://127.0.0.1:$LocalPort"

Write-Host "Opening an SSH tunnel to the private Nemori dashboard."
Write-Host "After authentication, browse to $dashboardUrl"
Start-Process $dashboardUrl
& ssh -o ExitOnForwardFailure=yes -N -L "${LocalPort}:127.0.0.1:${RemotePort}" $Server

$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$targetPath = Join-Path $projectDir 'launch_client.bat'
$iconPath = Join-Path $env:SystemRoot 'System32\SHELL32.dll'
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'Phishing Research App.lnk'

if (-not (Test-Path $targetPath)) {
    throw "Cannot find launch_client.bat in $projectDir"
}

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.WorkingDirectory = $projectDir
$shortcut.WindowStyle = 1
$shortcut.Description = 'Launch Multimodal Phishing Research Web App'
$shortcut.IconLocation = "$iconPath,13"
$shortcut.Save()

Write-Host "Shortcut created: $shortcutPath"

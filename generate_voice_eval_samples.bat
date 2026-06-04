@echo off
setlocal

cd /d "%~dp0"

set "COUNT=10"
if not "%~1"=="" set "COUNT=%~1"

echo [INFO] Creating %COUNT% real and %COUNT% fake voice evaluation samples.

if not exist "datasets\voice\test_samples\legitimate" (
  call generate_voice_test_samples.bat
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$count=[int]'%COUNT%';" ^
  "$realOut='datasets/evaluation_samples/voice_deepfake/real';" ^
  "$fakeOut='datasets/evaluation_samples/voice_deepfake/fake';" ^
  "New-Item -ItemType Directory -Force -Path $realOut,$fakeOut | Out-Null;" ^
  "$realSources=@('datasets/voice/evaluation/real','datasets/voice/real','datasets/voice/test_samples/legitimate');" ^
  "$fakeSources=@('datasets/voice/evaluation/fake','datasets/voice/fake','datasets/voice/test_samples/fake');" ^
  "$realFiles=@(); foreach ($src in $realSources) { if (Test-Path $src) { $realFiles += Get-ChildItem $src -Recurse -File -Include *.wav,*.mp3,*.flac,*.ogg,*.m4a } }" ^
  "$fakeFiles=@(); foreach ($src in $fakeSources) { if (Test-Path $src) { $fakeFiles += Get-ChildItem $src -Recurse -File -Include *.wav,*.mp3,*.flac,*.ogg,*.m4a } }" ^
  "$realFiles=$realFiles | Select-Object -First $count;" ^
  "$fakeFiles=$fakeFiles | Select-Object -First $count;" ^
  "if ($realFiles.Count -lt $count) { Write-Host '[WARN] Fewer real voice samples found than requested.' }" ^
  "if ($fakeFiles.Count -lt $count) { Write-Host '[WARN] Fewer fake voice samples found than requested.' }" ^
  "$i=1; foreach ($file in $realFiles) { $out=Join-Path $realOut ('real_voice_{0:D2}{1}' -f $i,$file.Extension); Copy-Item -LiteralPath $file.FullName -Destination $out -Force; Write-Host '[OK]' $out; $i++ }" ^
  "$i=1; foreach ($file in $fakeFiles) { $out=Join-Path $fakeOut ('fake_voice_{0:D2}{1}' -f $i,$file.Extension); Copy-Item -LiteralPath $file.FullName -Destination $out -Force; Write-Host '[OK]' $out; $i++ }"

if errorlevel 1 goto :error

echo [SUCCESS] Voice evaluation samples created in datasets\evaluation_samples\voice_deepfake
pause
exit /b 0

:error
echo [ERROR] Voice evaluation sample generation failed.
pause
exit /b 1

@echo off
setlocal

cd /d "%~dp0"

set "COUNT=10"
set "MODE=clip"

if not "%~1"=="" set "COUNT=%~1"
if not "%~2"=="" set "MODE=%~2"

echo [INFO] Creating %COUNT% real and %COUNT% fake video deepfake evaluation samples.
echo [INFO] Mode: %MODE%
echo [INFO] Use: generate_video_deepfake_eval_samples.bat 20 frame
echo [INFO] Modes: clip or frame

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$ffmpeg=(Get-Command ffmpeg -ErrorAction SilentlyContinue).Source;" ^
  "if (-not $ffmpeg) { $ffmpeg=(Get-ChildItem \"$env:LOCALAPPDATA\Microsoft\WinGet\Packages\" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName) }" ^
  "if (-not $ffmpeg) { throw 'FFmpeg was not found. Restart your terminal after installation, or install FFmpeg with winget.' }" ^
  "Write-Host '[INFO] Using FFmpeg:' $ffmpeg;" ^
  "$count=[int]'%COUNT%';" ^
  "$mode='%MODE%'.ToLowerInvariant();" ^
  "if ($mode -notin @('clip','frame')) { throw 'Mode must be clip or frame.' }" ^
  "$realSrc='datasets/faceforensics/original';" ^
  "$fakeSrc='datasets/faceforensics/manipulated';" ^
  "$realOut='datasets/evaluation_samples/video_deepfake/real';" ^
  "$fakeOut='datasets/evaluation_samples/video_deepfake/fake';" ^
  "New-Item -ItemType Directory -Force -Path $realOut,$fakeOut | Out-Null;" ^
  "$realFiles=Get-ChildItem $realSrc -Recurse -File -Include *.mp4,*.avi,*.mov,*.mkv,*.webm | Select-Object -First $count;" ^
  "$fakeFiles=Get-ChildItem $fakeSrc -Recurse -File -Include *.mp4,*.avi,*.mov,*.mkv,*.webm | Select-Object -First $count;" ^
  "if ($realFiles.Count -lt $count) { Write-Host '[WARN] Fewer real videos found than requested.' }" ^
  "if ($fakeFiles.Count -lt $count) { Write-Host '[WARN] Fewer fake videos found than requested.' }" ^
  "$i=1; foreach ($file in $realFiles) { $name='real_{0:D2}' -f $i; if ($mode -eq 'frame') { $out=Join-Path $realOut ($name + '.jpg'); & $ffmpeg -y -loglevel error -i $file.FullName -ss 00:00:02 -vframes 1 $out } else { $out=Join-Path $realOut ($name + '.mp4'); & $ffmpeg -y -loglevel error -i $file.FullName -t 5 -vf 'scale=320:-1' -c:v libx264 -crf 28 -an $out }; Write-Host '[OK]' $out; $i++ }" ^
  "$i=1; foreach ($file in $fakeFiles) { $name='fake_{0:D2}' -f $i; if ($mode -eq 'frame') { $out=Join-Path $fakeOut ($name + '.jpg'); & $ffmpeg -y -loglevel error -i $file.FullName -ss 00:00:02 -vframes 1 $out } else { $out=Join-Path $fakeOut ($name + '.mp4'); & $ffmpeg -y -loglevel error -i $file.FullName -t 5 -vf 'scale=320:-1' -c:v libx264 -crf 28 -an $out }; Write-Host '[OK]' $out; $i++ }"

if errorlevel 1 goto :error

echo [SUCCESS] Video deepfake evaluation samples created in datasets\evaluation_samples\video_deepfake
pause
exit /b 0

:error
echo [ERROR] Video evaluation sample generation failed.
pause
exit /b 1

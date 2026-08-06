@echo off
setlocal

cd /d "%~dp0"

echo ========================================
echo AUTOMATIC PATH FIX TOOL
echo Phishing Detection Research App
echo ========================================
echo.

echo This tool will automatically move your project to a shorter path
echo to fix the "filename too long" error.
echo.

echo Current location: %CD%
echo.

REM Check if path is already short enough
set "CURRENT_PATH=%CD%"
set "PATH_LENGTH=0"
for /f "delims=" %%A in ("%CURRENT_PATH%") do set "PATH_LENGTH=%%~zA"
set /a PATH_LENGTH=PATH_LENGTH/2

echo Current path length: %PATH_LENGTH% characters
echo Windows limit: 260 characters
echo.

if %PATH_LENGTH% LSS 200 (
    echo [INFO] Your path is already short enough.
    echo No action needed.
    pause
    exit /b 0
)

echo [WARN] Your path is too long. This will cause installation errors.
echo.

REM Define target path
set "TARGET_PATH=C:\Phising"

echo This tool will move the project to:
echo %TARGET_PATH%
echo.

echo WARNING: This will:
echo   1. Copy the entire project to C:\Phising
echo   2. Delete the .venv folder from the old location
echo   3. Create a shortcut on your desktop
echo   4. Open the new location
echo.

pause

echo [INFO] Starting automatic fix...
echo.

REM Check if target already exists
if exist "%TARGET_PATH%" (
    echo [WARN] C:\Phising already exists.
    echo.
    set /p "CHOICE=Do you want to overwrite it? (Y/N): "
    if /i not "%CHOICE%"=="Y" (
        echo [INFO] Operation cancelled by user.
        pause
        exit /b 1
    )
    echo [INFO] Removing existing C:\Phising...
    rmdir /s /q "%TARGET_PATH%" 2>nul
)

REM Create target directory
echo [INFO] Creating target directory...
mkdir "%TARGET_PATH%" 2>nul

REM Copy project files
echo [INFO] Copying project files (this may take a few minutes)...
xcopy "%CD%" "%TARGET_PATH%\" /E /I /H /Y /Q
if errorlevel 1 (
    echo [ERROR] Failed to copy files.
    pause
    exit /b 1
)

echo [INFO] Files copied successfully.
echo.

REM Delete .venv from old location
echo [INFO] Cleaning up old virtual environment...
if exist ".venv" (
    rmdir /s /q ".venv"
    echo [INFO] Deleted .venv from old location.
)

REM Create desktop shortcut
echo [INFO] Creating desktop shortcut...
set "SHORTCUT_TARGET=%USERPROFILE%\Desktop\Phishing App.lnk"
set "SCRIPT_PATH=%TARGET_PATH%\launch_client.bat"

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT_TARGET%'); $s.TargetPath = '%SCRIPT_PATH%'; $s.WorkingDirectory = '%TARGET_PATH%'; $s.Description = 'Phishing Detection Research App'; $s.Save()"

if exist "%SHORTCUT_TARGET%" (
    echo [INFO] Desktop shortcut created: Phishing App.lnk
) else (
    echo [WARN] Could not create desktop shortcut (PowerShell may be disabled)
)

REM Open new location
echo.
echo ========================================
echo FIX COMPLETE
echo ========================================
echo.
echo Your project has been moved to:
echo %TARGET_PATH%
echo.
echo A desktop shortcut has been created (if possible).
echo.
echo NEXT STEPS:
echo 1. Close this window
echo 2. Double-click the desktop shortcut "Phishing App"
echo 3. Or navigate to C:\Phising and double-click launch_client.bat
echo.
echo The old location will still exist but should not be used.
echo You can delete the old folder after confirming everything works.
echo.

set /p "OPEN=Open the new location now? (Y/N): "
if /i "%OPEN%"=="Y" (
    explorer "%TARGET_PATH%"
)

echo.
echo Press any key to exit...
pause >nul
exit /b 0

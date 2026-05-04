@echo off
setlocal

set "REPO_ROOT=%~dp0"
set "SCRIPT=%REPO_ROOT%scripts\postprocess_gui.py"

if defined SUSPENSION_TELEMETRY_PYTHON if exist "%SUSPENSION_TELEMETRY_PYTHON%" (
    "%SUSPENSION_TELEMETRY_PYTHON%" "%SCRIPT%" %*
    exit /b %errorlevel%
)

if exist "%USERPROFILE%\Python313\python.exe" (
    "%USERPROFILE%\Python313\python.exe" "%SCRIPT%" %*
    exit /b %errorlevel%
)

if exist "%REPO_ROOT%\.python\Python313\python.exe" (
    "%REPO_ROOT%\.python\Python313\python.exe" "%SCRIPT%" %*
    exit /b %errorlevel%
)

where python >nul 2>nul
if %errorlevel%==0 (
    python "%SCRIPT%" %*
    exit /b %errorlevel%
)

echo No usable Python interpreter found.
echo Set SUSPENSION_TELEMETRY_PYTHON to a full python.exe path or install Python at "%USERPROFILE%\Python313\python.exe".
exit /b 1

@echo off
setlocal
set "CONVERT_TO="
set "OUT_DIR="
set "INPUT_PATH="

:parse
if "%~1"=="" goto run
if /I "%~1"=="--convert-to" (
  set "CONVERT_TO=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--outdir" (
  set "OUT_DIR=%~2"
  shift
  shift
  goto parse
)
set "INPUT_PATH=%~1"
shift
goto parse

:run
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0soffice_word.ps1" -ConvertTo "%CONVERT_TO%" -OutDir "%OUT_DIR%" -InputPath "%INPUT_PATH%"
exit /b %ERRORLEVEL%

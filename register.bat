@echo off
:: Register Astronalyze as the default handler for FITS and XISF files.
:: Must be run as Administrator.

net session >nul 2>&1
if errorlevel 1 (
    echo ERROR: This script must be run as Administrator.
    echo Right-click register.bat and choose "Run as administrator".
    pause
    exit /b 1
)

set "EXE=%~dp0Astronalyze.exe"

echo ============================================================
echo  Astronalyze - File Association Registration
echo ============================================================
echo.
echo Registering: %EXE%
echo.

:: ---- helper: register one extension ----
:: Usage: call :register_ext .ext ExtLabel "Friendly Name"

call :register_ext .fits  AstronalyzeFile.fits  "FITS Astronomical Image"
call :register_ext .fit   AstronalyzeFile.fit   "FITS Astronomical Image"
call :register_ext .fts   AstronalyzeFile.fts   "FITS Astronomical Image"
call :register_ext .xisf  AstronalyzeFile.xisf  "XISF Astronomical Image"

echo.
echo Done! You may need to log off and back on (or restart Explorer)
echo for the icon and "Open with" changes to appear.
echo.
pause
exit /b 0


:register_ext
:: %1 = extension (.fits)   %2 = ProgID   %3 = friendly name
set "EXT=%~1"
set "PROGID=%~2"
set "DESC=%~3"

:: Extension → ProgID
reg add "HKCR\%EXT%"                          /ve /d "%PROGID%"              /f >nul
reg add "HKCR\%EXT%"                          /v "Content Type" /d "application/octet-stream" /f >nul

:: ProgID shell open command
reg add "HKCR\%PROGID%"                       /ve /d "%DESC%"                /f >nul
reg add "HKCR\%PROGID%\DefaultIcon"           /ve /d "\"%EXE%\",0"           /f >nul
reg add "HKCR\%PROGID%\shell\open\command"    /ve /d "\"%EXE%\" \"%%1\""     /f >nul

:: Register as a capable application for "Open with"
reg add "HKCR\Applications\Astronalyze.exe\shell\open\command" /ve /d "\"%EXE%\" \"%%1\"" /f >nul
reg add "HKCR\Applications\Astronalyze.exe\SupportedTypes" /v "%EXT%" /d "" /f >nul

:: Notify Explorer
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\%EXT%\OpenWithProgids" /v "%PROGID%" /d "" /f >nul

echo   Registered %EXT%  ^(%DESC%^)
goto :eof

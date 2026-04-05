@echo off
echo ============================================================
echo  Astronalyze - Build Executable
echo ============================================================
echo.

python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

python -c "import send2trash" >nul 2>&1
if errorlevel 1 (
    echo send2trash not found. Installing...
    pip install send2trash
)

echo Building...
python -m PyInstaller --noconfirm --onedir --windowed ^
    --name Astronalyze ^
    --collect-all astropy ^
    --collect-all photutils ^
    --collect-all pyqtgraph ^
    --collect-all xisf ^
    --hidden-import scipy.spatial ^
    --hidden-import scipy.interpolate ^
    --hidden-import scipy.optimize ^
    --hidden-import send2trash ^
    astronalyze.py

if errorlevel 1 (
    echo.
    echo Build FAILED. See errors above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done!
echo  Executable: dist\Astronalyze\Astronalyze.exe
echo  Distribute the entire dist\Astronalyze\ folder.
echo ============================================================
pause

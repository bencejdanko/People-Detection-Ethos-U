@echo off
echo ==========================================================
echo Edge AI People Counting - Model Compilation Helper
echo ==========================================================
echo.
echo Step 1: Downloading the latest best.pt from Modal Volume...
set PYTHONIOENCODING=utf-8
python -m modal volume get yolov8-runs-cache detect_person/train/weights/best.pt best_person.pt --force
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to download best.pt from Modal Volume.
    exit /b %ERRORLEVEL%
)

echo.
echo Step 2: Compiling best_person.pt to MODEL.TFL on Modal...
python -m modal run modal_export.py --pt-path best_person.pt --output-path MODEL.TFL --imgsz 192 --optimise Size
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Model compilation failed.
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================================
echo [SUCCESS] Model successfully compiled and saved to MODEL.TFL
echo ==========================================================

@echo off

REM Check if UV is installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing UV...
    pip install uv
)

REM Create virtual environment
echo Creating virtual environment...
uv venv

REM Install dependencies
echo Installing dependencies...
uv pip install -r requirements.txt

echo.
echo Setup completed successfully!
echo.
echo To activate the virtual environment:
echo     .venv\Scripts\activate 
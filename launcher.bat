@echo off
setlocal
cd /d "%~dp0"

".\.venv\Scripts\python.exe" "inverter_gui_qt.py"

endlocal
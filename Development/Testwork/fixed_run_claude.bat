@echo off
:: Use PowerShell to bypass CMD path parsing bugs with '&' and spaces
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { Set-Location 'leak_sense_twin'; python main_leak_detection_system.py }"
pause

@echo off
rem Runs the Python Brownian-dynamics plans and the AO checks. Double-click this file.
cd /d "%~dp0"
set DATA=D:\Work\AIRPODS Paper 2627\Data
set PY="C:\Users\omare\AppData\Local\Programs\Python\Python310\python.exe"
%PY% verify_ao.py --check match
%PY% verify_ao.py --check force
%PY% airpods_bd.py --plan validation --out "%DATA%" --compare "%DATA%"
%PY% airpods_bd.py --plan revision --out "%DATA%"
%PY% verify_ao.py --check b2
%PY% airpods_bd.py --plan ao --out "%DATA%"
%PY% airpods_bd.py --plan ao_dt --out "%DATA%"
echo.
echo All finished. Figures from the AO checks are in the ao_verification folder.
pause

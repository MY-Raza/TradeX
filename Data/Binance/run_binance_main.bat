@echo off
REM =====================================
REM Activate Conda environment
REM =====================================
CALL C:\Users\Raza\miniconda3\Scripts\activate.bat tradex_env

REM =====================================
REM Move to project directory
REM =====================================
cd /d D:\trading\TradeX

REM =====================================
REM Infinite loop
REM =====================================
:loop
echo =====================================
echo Running main.py at %DATE% %TIME%
echo =====================================

python main.py

echo Waiting 60 seconds before next run...
timeout /t 60 /nobreak

goto loop

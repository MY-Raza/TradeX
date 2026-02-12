@echo off
setlocal EnableDelayedExpansion

:: ================================
:: Activate Conda Environment
:: ================================

set CONDAPATH=%CONDAPATH%
set ENVNAME=tradex_env

if "%ENVNAME%"=="base" (
    set ENVPATH=%CONDAPATH%
) else (
    set ENVPATH=%CONDAPATH%\envs\%ENVNAME%
)

call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

cd /d "%~dp0"

:: ================================
:: Get Current Minute
:: ================================

for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
    set MINUTE=%%b
)

set MINUTE=!MINUTE: =!

set TF=

:: ================================
:: Priority Logic
:: 1h > 15m > 5m
:: ================================

if "!MINUTE!"=="00" (
    set TF=1h
) else (
    set /a MOD15=!MINUTE! %% 15
    if !MOD15!==0 (
        set TF=15m
    ) else (
        set /a MOD5=!MINUTE! %% 5
        if !MOD5!==0 (
            set TF=5m
        )
    )
)

:: ================================
:: Execute if valid
:: ================================

if defined TF (
    echo Running timeframe: !TF!
    python main.py !TF!
) else (
    echo Not a valid execution minute.
)

exit

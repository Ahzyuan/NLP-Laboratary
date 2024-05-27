@echo off
call conda activate my_env
E:
cd E:\AIL\project\NLP-Laboratary

REM setlocal enabledelayedexpansion
REM set "FILES_DIR=E:\AIL\project\NLP-Laboratary\Config\pick_vectorizer"
REM for /R "%FILES_DIR%" %%i in (*) do (
REM     set "FILE_PATH=%%~fi"
REM     python train.py -c !FILE_PATH! -r E:\AIL\project\NLP-Laboratary\Results\pick_vectorizer
REM )
REM endlocal

REM setlocal enabledelayedexpansion
REM set "FILES_DIR=E:\AIL\project\NLP-Laboratary\Config\pick_evalscheme"
REM for /R "%FILES_DIR%" %%i in (*) do (
REM     set "FILE_PATH=%%~fi"
REM     python train.py -c !FILE_PATH! -r E:\AIL\project\NLP-Laboratary\Results\pick_evalscheme
REM )
REM endlocal

REM setlocal enabledelayedexpansion
REM set "FILES_DIR=E:\AIL\project\NLP-Laboratary\Config\pick_model"
REM for /R "%FILES_DIR%" %%i in (*) do (
REM     set "FILE_PATH=%%~fi"
REM     python train.py -c !FILE_PATH! -r E:\AIL\project\NLP-Laboratary\Results\pick_model
REM )
REM endlocal

REM setlocal enabledelayedexpansion
REM set "FILES_DIR=E:\AIL\project\NLP-Laboratary\Config\pick_dropout"
REM for /R "%FILES_DIR%" %%i in (*) do (
REM     set "FILE_PATH=%%~fi"
REM     python train.py -c !FILE_PATH! -r E:\AIL\project\NLP-Laboratary\Results\pick_dropout
REM )
REM endlocal

REM setlocal enabledelayedexpansion
REM set "FILES_DIR=E:\AIL\project\NLP-Laboratary\Config\pick_lr"
REM for /R "%FILES_DIR%" %%i in (*) do (
REM     set "FILE_PATH=%%~fi"
REM     python train.py -c !FILE_PATH! -r E:\AIL\project\NLP-Laboratary\Results\pick_lr
REM )
REM endlocal

REM setlocal enabledelayedexpansion
REM set "FILES_DIR=E:\AIL\project\NLP-Laboratary\Config\pick_loss"
REM for /R "%FILES_DIR%" %%i in (*) do (
REM     set "FILE_PATH=%%~fi"
REM     python train.py -c !FILE_PATH! -r E:\AIL\project\NLP-Laboratary\Results\pick_loss
REM )
REM endlocal

REM setlocal enabledelayedexpansion
REM set "FILES_DIR=E:\AIL\project\NLP-Laboratary\Config\pick_optim"
REM for /R "%FILES_DIR%" %%i in (*) do (
REM     set "FILE_PATH=%%~fi"
REM     python train.py -c !FILE_PATH! -r E:\AIL\project\NLP-Laboratary\Results\pick_optim
REM )
REM endlocal

setlocal enabledelayedexpansion
set "FILES_DIR=E:\AIL\project\NLP-Laboratary\Config\pick_activate"
for /R "%FILES_DIR%" %%i in (*) do (
    set "FILE_PATH=%%~fi"
    python train.py -c !FILE_PATH! -r E:\AIL\project\NLP-Laboratary\Results\pick_activate
)
endlocal
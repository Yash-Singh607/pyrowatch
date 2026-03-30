@echo off
REM PyroWatch - Pipeline runner for Windows
REM
REM Usage:
REM   run.bat demo      full pipeline
REM   run.bat train     train only
REM   run.bat eval      evaluate with trained model
REM   run.bat image PATH
REM   run.bat test      run unit tests

SET MODE=%1
IF "%MODE%"=="" SET MODE=demo

echo.
echo ========================================
echo    PyroWatch -- Wildfire Smoke Detector
echo ========================================
echo.

IF "%MODE%"=="demo" GOTO DEMO
IF "%MODE%"=="train" GOTO TRAIN
IF "%MODE%"=="eval" GOTO EVAL
IF "%MODE%"=="image" GOTO IMAGE
IF "%MODE%"=="test" GOTO TEST
echo Unknown mode: %MODE%
echo Usage: run.bat [demo^|train^|eval^|image ^<path^>^|test]
EXIT /B 1

:DEMO
echo [1/4] Generating 60 sample images...
python generate_samples.py
IF ERRORLEVEL 1 EXIT /B 1

echo.
echo [2/4] Training RandomForest classifier...
python train.py
IF ERRORLEVEL 1 EXIT /B 1

echo.
echo [3/4] Running detector on all samples...
python detector.py --source data\sample_images --model models\rf_classifier.pkl --save --json
IF ERRORLEVEL 1 EXIT /B 1

echo.
echo [4/4] Evaluation metrics + confusion matrix...
python evaluate.py --model models\rf_classifier.pkl --out-dir outputs
IF ERRORLEVEL 1 EXIT /B 1

echo.
echo Done. Check outputs\ for annotated images and eval_report.json
GOTO END

:TRAIN
python train.py
GOTO END

:EVAL
python evaluate.py --model models\rf_classifier.pkl
GOTO END

:IMAGE
IF "%2"=="" (echo Usage: run.bat image ^<path^> & EXIT /B 1)
python detector.py --source %2 --model models\rf_classifier.pkl --save
GOTO END

:TEST
python -m pytest tests/ -v
GOTO END

:END

@echo off
title Git Push Script

:: Configuration - Change these values as needed
set "BranchName=main"
set "CommitMessage="

:: Check if Git is installed
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo Git is not installed. Please install Git for Windows.
    pause
    exit /b 1
)

:: Get commit message from the user
set /p "CommitMessage=Enter commit message: "

:: Check if commit message is empty
if "%CommitMessage%" == "" (
    echo Commit message cannot be empty.
    pause
    exit /b 1
)


:: Push to the remote repository
git add .
git commit -m "%CommitMessage%"
git push origin 


echo Successfully pushed to %BranchName%.
pause

exit /b 0

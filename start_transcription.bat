@echo off
cd /d "%~dp0"

:: Find the latest .wav file in the records folder
set "latest="
for /f "delims=" %%f in ('dir /b /o-d records\*.wav 2^>nul') do (
    set "latest=records\%%f"
    goto :found
)

echo ERROR: No WAV files found in the 'records' folder.
echo Please run start_caption.bat first to record a meeting.
pause
exit /b 1

:found
echo Transcribing: %latest%
python asr_speaker_diarization.py "%latest%" --device cuda
pause

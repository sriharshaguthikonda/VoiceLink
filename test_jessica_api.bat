@echo off
echo Testing Jessica voice with VoiceLink server...

curl.exe -X POST "http://127.0.0.1:7860/v1/tts" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Hello, I am Jessica, and I am now available in VoiceLink!\",\"voice\":\"af_jessica\",\"speed\":1.0,\"format\":\"pcm_24k_16bit\"}" ^
  --output "c:\Windows_software\VoiceLink\jessica_test.pcm"

echo.
if exist "c:\Windows_software\VoiceLink\jessica_test.pcm" (
    echo ✅ Jessica voice test successful!
    echo Audio saved to: jessica_test.pcm
    echo File size: 
    dir "c:\Windows_software\VoiceLink\jessica_test.pcm" | findstr jessica_test.pcm
) else (
    echo ❌ Jessica voice test failed!
)

pause

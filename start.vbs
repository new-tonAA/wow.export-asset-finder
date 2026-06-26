Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get script directory
strDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Launch splash immediately
WshShell.Run "powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & strDir & "\splash.ps1""", 0, False

' Launch server (visible console window)
WshShell.Run "cmd /k cd /d """ & strDir & """ && title WoW Asset Finder && echo Starting... && venv\Scripts\python.exe app.py", 1, False

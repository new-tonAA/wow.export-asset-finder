Set objShell = CreateObject("WScript.Shell")
Set objIE = CreateObject("InternetExplorer.Application")
objIE.Navigate "about:blank"
objIE.Width = 440
objIE.Height = 200
objIE.Left = (objIE.Document.parentWindow.screen.availWidth - 440) / 2
objIE.Top = (objIE.Document.parentWindow.screen.availHeight - 200) / 2
objIE.MenuBar = False
objIE.ToolBar = False
objIE.StatusBar = False
objIE.Resizable = False
objIE.Visible = True

Do While objIE.Busy
    WScript.Sleep 50
Loop

objIE.Document.title = "WoW Asset Finder"
objIE.Document.body.innerHTML = "<div style='background:#1a1a1a;color:white;font-family:Segoe UI;text-align:center;padding-top:40px;height:100%;margin:0'><h1 style='color:#e8a500;margin:0'>WoW Asset Finder</h1><p style='color:#ccc;margin-top:12px;font-size:14px'>Loading CLIP model...</p><p style='color:#888;font-size:12px'>Browser will open when ready</p></div>"
objIE.Document.body.style.margin = "0"
objIE.Document.body.style.overflow = "hidden"
objIE.Document.body.scroll = "no"

' Keep alive until killed by taskkill
Do While True
    WScript.Sleep 1000
Loop

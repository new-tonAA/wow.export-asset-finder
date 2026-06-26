Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Enable DPI awareness for sharp rendering
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DPI {
    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();
}
"@
[DPI]::SetProcessDPIAware()

$form = New-Object System.Windows.Forms.Form
$form.Text = "WoW Asset Finder"
$form.ClientSize = New-Object System.Drawing.Size(500, 200)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "None"
$form.BackColor = [System.Drawing.Color]::FromArgb(26, 26, 26)
$form.TopMost = $true
$form.ShowInTaskbar = $true
$form.DoubleBuffered = $true

# Use a TableLayoutPanel for centering everything
$panel = New-Object System.Windows.Forms.Panel
$panel.Dock = [System.Windows.Forms.DockStyle]::Fill
$form.Controls.Add($panel)

# Title - centered
$title = New-Object System.Windows.Forms.Label
$title.Text = "WoW Asset Finder"
$title.Font = New-Object System.Drawing.Font("Segoe UI", 22, [System.Drawing.FontStyle]::Bold)
$title.ForeColor = [System.Drawing.Color]::FromArgb(232, 165, 0)
$title.AutoSize = $false
$title.Size = New-Object System.Drawing.Size(500, 45)
$title.Location = New-Object System.Drawing.Point(0, 30)
$title.TextAlign = [System.Drawing.ContentAlignment]::MiddleCenter
$panel.Controls.Add($title)

# Message - centered
$msg = New-Object System.Windows.Forms.Label
$msg.Text = "Loading CLIP model..."
$msg.Font = New-Object System.Drawing.Font("Segoe UI", 11)
$msg.ForeColor = [System.Drawing.Color]::FromArgb(200, 200, 200)
$msg.AutoSize = $false
$msg.Size = New-Object System.Drawing.Size(500, 30)
$msg.Location = New-Object System.Drawing.Point(0, 85)
$msg.TextAlign = [System.Drawing.ContentAlignment]::MiddleCenter
$panel.Controls.Add($msg)

# Sub message - centered
$sub = New-Object System.Windows.Forms.Label
$sub.Text = "Browser will open automatically when ready"
$sub.Font = New-Object System.Drawing.Font("Segoe UI", 9)
$sub.ForeColor = [System.Drawing.Color]::FromArgb(120, 120, 120)
$sub.AutoSize = $false
$sub.Size = New-Object System.Drawing.Size(500, 25)
$sub.Location = New-Object System.Drawing.Point(0, 115)
$sub.TextAlign = [System.Drawing.ContentAlignment]::MiddleCenter
$panel.Controls.Add($sub)

# Animated dots - centered
$dots = New-Object System.Windows.Forms.Label
$dots.Text = ""
$dots.Font = New-Object System.Drawing.Font("Segoe UI", 24, [System.Drawing.FontStyle]::Bold)
$dots.ForeColor = [System.Drawing.Color]::FromArgb(232, 165, 0)
$dots.AutoSize = $false
$dots.Size = New-Object System.Drawing.Size(500, 40)
$dots.Location = New-Object System.Drawing.Point(0, 150)
$dots.TextAlign = [System.Drawing.ContentAlignment]::MiddleCenter
$panel.Controls.Add($dots)

$script:dotState = 0
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 400
$timer.Add_Tick({
    $script:dotState = ($script:dotState + 1) % 4
    $dotChars = @("", ". ", ". . ", ". . .")
    $dots.Text = $dotChars[$script:dotState]
})
$timer.Start()

$form.ShowDialog()
$timer.Stop()

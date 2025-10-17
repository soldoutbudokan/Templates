param(
  [string]$Path = 'C:\Users\tbhatt\Documents\Templates\',
  [switch]$Recurse
)

$ErrorActionPreference = 'Stop'

# Launch Excel safely
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false

# Disable macros for files opened programmatically (3 = msoAutomationSecurityForceDisable)
$prevSec = $excel.AutomationSecurity
$excel.AutomationSecurity = 3  # msoAutomationSecurityForceDisable  :contentReference[oaicite:1]{index=1}

try {
  $search = Get-ChildItem -Path $Path -File -Include *.xlsx,*.xls -Recurse:$Recurse

  $xlRDIAll = 99  # Removes all document info (comments, properties, personal info, etc.)  :contentReference[oaicite:2]{index=2}

  foreach ($file in $search) {
    $wb = $null
    try {
      Write-Host "Scrubbing: $($file.FullName)"
      # Open without updating links; read/write
      $wb = $excel.Workbooks.Open($file.FullName, 0, $false)

      # Strip all document info
      $wb.RemoveDocumentInformation($xlRDIAll)

      # Clear common built-in properties explicitly
      try {
        $builtin = $wb.BuiltinDocumentProperties
        $names = 'Author','Last Author','Title','Subject','Category','Keywords','Comments','Manager','Company','Hyperlink Base'
        foreach ($n in $names) {
          try { $p = $builtin.Item($n); if ($p -and $p.Value) { $p.Value = '' } } catch {}
        }
      } catch {}

      # Remove all custom properties
      try {
        $cp = $wb.CustomDocumentProperties
        for ($i = $cp.Count; $i -ge 1; $i--) { try { $cp.Item($i).Delete() } catch {} }
      } catch {}

      $wb.Save()
      $wb.Close($true)
      [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wb)
    }
    catch {
      Write-Warning "Failed: $($file.Name) -> $($_.Exception.Message)"
      try { if ($wb) { $wb.Close($false) } } catch {}
    }
  }
}
finally {
  # Restore macro security and clean up
  $excel.AutomationSecurity = $prevSec
  $excel.Quit()
  [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
  [GC]::Collect()
  [GC]::WaitForPendingFinalizers()
}

Write-Host "Done."

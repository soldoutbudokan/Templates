// Functions to compare two excel files (must be .xlsx)

* Base file
import excel using "FILEPATH.xlsx", sheet("SHEETNAME") cellrange(A1) firstrow clear
	
	tempfile Base
	save `Base'
	
* Audit File
	
import excel "FILEPATH.xlsx", sheet("SHEETNAME") cellrange (A1) firstrow clear
	
	tempfile Audit
	save `Audit'
	
	cf _all using `Audit', all verbose
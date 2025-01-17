/**************************************************
* Project: [Project Name]
* Author: [Author Name]
* Date: [Date]
* Purpose: [Brief description]
***************************************************/

* Clear workspace and set memory
clear all
set more off
capture log close

/***************************************************
* Define paths and directories
***************************************************/

//Project folders
global PATH "\\server\project\Study-Name\Username\ProjectNumber"
global DATA "$PATH\Data"
global OUTPUT "$PATH\Output"
global CODE "$PATH\Code"

//Raw data folders
global RAW "$DATA\Raw"
	global FACILITY "$RAW\Facilities"
	global CLAIMS "$RAW\Claims"
	global PROVIDER "$RAW\Provider"

//Intermediate data
global INTER "$DATA\Intermediate"
	global CLEANED "$INTER\Cleaned"
	global MERGED "$INTER\Merged"

//Analysis folders
global ANALYSIS "$PATH\Analysis"
	global TABLES "$ANALYSIS\Tables"
	global FIGURES "$ANALYSIS\Figures"

//Program name and logging
global PGRM "Project_Name_Analysis"
global LOGDIR "$PATH\Logs"

* Create necessary directories if they don't exist
foreach dir in "$DATA" "$OUTPUT" "$CODE" "$RAW" "$INTER" "$ANALYSIS" "$LOGDIR" {
    capture mkdir "`dir'"
}

* Set working directory
cd "$PATH"

* Start log file
log using "$LOGDIR\\$PGRM - $S_DATE", replace

/**************************************************
* Load and prepare data
***************************************************/

* Import data
use "dataset.dta", clear

* Basic data inspection
describe
summarize
codebook

* Check for missing values
misstable summarize

/**************************************************
* Data cleaning
***************************************************/

* Generate new variables
gen ln_income = ln(income)
label variable ln_income "Log of income"

* Handle missing values
replace age = . if age < 0
drop if missing(education)

/**************************************************
* Summary statistics
***************************************************/

* Generate summary stats table
estpost summarize income education age experience
esttab using "summary_stats.tex", replace ///
    cells("mean(fmt(%9.2f)) sd min max") ///
    label noobs

* Create correlation matrix
pwcorr income education age experience, star(.05)

/**************************************************
* Main analysis
***************************************************/

* Basic OLS regression
reg income education age experience

* Store results
estimates store model1

* Add controls and fixed effects
reghdfe income education age experience, absorb(state_id year) cluster(state_id)
estimates store model2

* Export regression results
esttab model1 model2 using "regression_results.tex", ///
    replace label b(%9.3f) se(%9.3f) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    title("Regression Results") ///
    addnotes("Standard errors in parentheses" "* p<0.10, ** p<0.05, *** p<0.01")

/**************************************************
* Robustness checks
***************************************************/

* Subsample analysis
reg income education age experience if female == 1
estimates store female_sample

reg income education age experience if female == 0
estimates store male_sample

* Alternative specifications
reg ln_income education age experience
estimates store log_spec

/**************************************************
* Figures
***************************************************/

* Histogram of dependent variable
histogram income, normal
graph export "income_dist.pdf", replace

* Scatter plot with fitted line
twoway (scatter income education) (lfit income education), ///
    title("Income vs. Education") ///
    xtitle("Years of Education") ///
    ytitle("Income")
graph export "income_education_scatter.pdf", replace

* Close log file
log close
exit
/**************************************************
* End of file
***************************************************/
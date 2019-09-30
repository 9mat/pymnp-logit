// DHL 12 Sep 2019
// prepare data for estimation

set more off

global datapath D:/Dropbox/work/ethanol/data

use $datapath/individual_data_wide.dta, clear


// drop Rio de Janerio, midgrade ethanol and treatment 3 and 4
drop if dv_rj |  (choice == 4) | inlist(treattype, 3, 4)

// old coding: 2 = midgrade gasoline, 3 = ethanol
// new coding: 2 = ethanol, 3 = midgrad gasoline
replace choice = 93 if choice == 3
replace choice = 3 if choice == 2
replace choice = 2 if choice == 93


gen cons = 1

// impute missing prices
replace pgmidgrade_km_adj = 100000000 if pgmidgrade_km_adj == .

// there should be no missing gasoline and ethanol price
assert pe_km_adj != . & pg_km_adj != .


// relative log prices
gen rel_lpgmidgrade_km_adj = ln(pgmidgrade_km_adj) - ln(pg_km_adj)
gen rel_lpe_km_adj = ln(pe_km_adj) - ln(pg_km_adj)


// dummies for car make
gen dv_make_gm = car_make == "GM"
gen dv_make_vw = car_make == "VW"
gen dv_make_fiat = car_make == "FIAT"
gen dv_make_ford = car_make == "FORD"
gen dv_make_other = ~inlist(car_make, "GM", "VW", "FIAT", "FORD")


// dummies for car class
gen dv_class_compact = car_class == "Compact"
gen dv_class_subcompact = car_class == "Subcompact"
gen dv_class_midsize = car_class == "Midsize"
gen dv_class_other = ~inlist(car_class, "Compact", "Subcompact", "Midsize")


gen car_age = 2012 - car_model_year
gen car_lprice = ln(car_price_adj)


// omitted dummies (for subsample selection purpose)
gen dv_carpriceadj_p0p75 = 1 - dv_carpriceadj_p75p100
gen dv_usageveh_p0p75 = 1 - dv_usageveh_p75p100
gen dv_nocollege = 1 - dv_somecollege

gen p_ratio = pe_lt/pg_lt
gen e_favotabr = p_ratio > 0.705



// generate a running sequence index for fuel-station fixed effects
// unidentifiable fixed effects (stations with no motorist choosing a fuel) are indexed 0

egen fuel_station_idx = group(stationid choice)
forvalues j=1/3 {
  bys stationid: egen nchoice`j' = total(choice==`j')
  bys stationid: gen fuel_station_id`j' = fuel_station_idx if choice == `j'
  bys stationid (fuel_station_id`j'): replace fuel_station_id`j' = fuel_station_id`j'[1]
}


// unidentifiable fuel-station fixed effects (there should be none for gasoline and ethanol choice)
assert nchoice1 > 0 & fuel_station_id1 != .
assert nchoice2 > 0 & fuel_station_id2 != .
assert nchoice3 == 0 if fuel_station_id3 == .
replace fuel_station_id3 = 0 if fuel_station_id3 == .


drop fuel_station_idx nchoice*

compress



preserve
	clear
	foreach city in bh cur rec rj sp {
		append using D:/Dropbox/work/ethanol/data/price/mediancitywideprices_waves1and2_`city'.dta
		cap gen cityname = strupper("`city'")
		cap replace cityname = strupper("`city'") if city == ""
	}

	append using D:/Dropbox/work/ethanol/data/price/mediancitywideprices_wave3_sp.dta
	replace cityname = "SP" if city == ""

	replace cityname = "CTB" if cityname == "CUR"


	gen week = (date - td(2jan2011) - dow(date))/7
	encode cityname, g(citycode)
	bys citycode week: keep if _n == 1

	xtset citycode week

	gen lpedivpg = log(pedivpg_median)
	gen dl_pedivpg = D.lpedivpg
	compress
	tempfile prices
	save `prices'
restore


gen week = (date - td(2jan2011) - dow(date))/7
compress
merge m:1 cityname week using `prices', keep(match master) keepusing(dl_pedivpg) nogen 

cap mkdir $datapath/generated
saveold $datapath/generated/individual_data_wide_prepared.dta, replace

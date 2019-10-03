set specfolder=D:\Dropbox\work\ethanol\spec

for  %%specfile in (spec1hetfe_new, spec2hetfe_new, spec3hetfe_new, spec4hetfe_new) do (
  echo %specfolder/%%specfile.json
  rem python het-probit.py %specfolder%\%%specfile.json > %specfolder%\%%specfile_new_log.txt
)
rem python het-probit.py %specfolder%\spec2hetfe_new.json > %specfolder%\spec2hetfe_new_log.txt
rem python het-probit.py %specfolder%\spec3hetfe_new.json > %specfolder%\spec3hetfe_new_log.txt
rem python het-probit.py %specfolder%\spec4hetfe_new.json > %specfolder%\spec4hetfe_new_log.txt
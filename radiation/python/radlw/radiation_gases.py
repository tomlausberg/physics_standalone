import numpy as np
import xarray as xr
import os
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radphysparam import co2cyc_file, co2gbl_file, co2dat_file

class GasClass():
    VTAGGAS = 'NCEP-Radiation_gases     v5.1  Nov 2012'
    IMXCO2 = 24
    JMXCO2 = 12
    co2vmr_def = 350.0e-6
    MINYEAR = 1957

    def __init__(self, me, iozn, ico2, ictm):
        self.kyrsav = 0
        self.kmonsav = 1

        self.me = me
        self.ioznflg = iozn
        self.ico2flg = ico2
        self.ictmflg = ictm

        if self.ioznflg > 0:
            if self.me == 0:
                print(' - Using interactive ozone distribution')
        else:
            print('Climatological ozone data not implemented')

        #  --- ...  co2 data section

        self.co2_glb = self.co2vmr_def

        if self.ico2flg == 0:
            if me == 0:
                print(f'- Using prescribed co2 global mean value={self.co2vmr_def}')

        else:
            if self.ictmflg == -1:      # input user provided data
                print('ictmflg = -1 is not implemented')

            else:                     # input from observed data
                if self.ico2flg == 1:
                    print('Using observed co2 global annual mean value')

                elif self.ico2flg == 2:
                    if me == 0:
                        print('Using observed co2 monthly 2-d data')

                else:
                    print(f' ICO2={self.ico2flg}, is not a valid selection',
                          ' - Stoped in subroutine gas_init!!!')


            if self.ictmflg == -2:
                file_exist = os.path.isfile(co2cyc_file)
                if not file_exist:
                    if me == 0:
                      print('Can not find seasonal cycle CO2 data: ',
                            f'{co2cyc_file} - Stopped in subroutine gas_init !!')
                else:
                    co2cyc_sav = np.zeros((self.IMXCO2, self.JMXCO2, 12))
                    ds = xr.open_dataset(co2cyc_file)
                    #  --- ...  read in co2 2-d seasonal cycle data
                    cline = ds['cline'].data
                    co2g1 = ds['co2g1'].data
                    co2g2 = ds['co2g2'].data
                    co2dat = ds['co2dat']

                    if me == 0:
                        print(' - Superimpose seasonal cycle to mean CO2 data')
                        print('Opened CO2 climatology seasonal cycle data',
                              f' file: {co2cyc_file}')

                    gco2cyc = ds['gco2cyc']
                    gco2cyc = gco2cyc * 1.0e-6

                    self.co2cyc_sav = co2dat

    def return_initdata(self):
        outdict = {'co2cyc_sav': 0}
        return outdict

    def return_updatedata(self):
        outdict = {'co2vmr_sav': self.co2vmr_sav,
                   'gco2cyc': self.gco2cyc}
        return outdict

    def gas_update(self, iyear, imon, iday, ihour, loz1st, ldoco2, me):
        #  ===================================================================  !
        #                                                                       !
        #  gas_update reads in 2-d monthly co2 data set for a specified year.   !
        #  data are in a 15 degree lat/lon horizontal resolution.               !
        #                                                                       !
        #  inputs:                                               dimemsion      !
        #     iyear   - year of the requested data for fcst         1           !
        #     imon    - month of the year                           1           !
        #     iday    - day of the month                            1           !
        #     ihour   - hour of the day                             1           !
        #     loz1st  - clim ozone 1st time update control flag     1           !
        #     ldoco2  - co2 update control flag                     1           !
        #     me      - print message control flag                  1           !
        #                                                                       !
        #  outputs: (to the module variables)                                   !
        #    ( none )                                                           !
        #                                                                       !
        #  external module variables:  (in physparam)                           !
        #     ico2flg    - co2 data source control flag                         !
        #                   =0: use prescribed co2 global mean value            !
        #                   =1: use input global mean co2 value (co2_glb)       !
        #                   =2: use input 2-d monthly co2 value (co2vmr_sav)    !
        #     ictmflg    - =yyyy#, data ic time/date control flag               !
        #                  =   -2: same as 0, but superimpose seasonal cycle    !
        #                          from climatology data set.                   !
        #                  =   -1: use user provided external data for the fcst !
        #                          time, no extrapolation.                      !
        #                  =    0: use data at initial cond time, if not existed!
        #                          then use latest, without extrapolation.      !
        #                  =    1: use data at the forecast time, if not existed!
        #                          then use latest and extrapolate to fcst time.!
        #                  =yyyy0: use yyyy data for the forecast time, no      !
        #                          further data extrapolation.                  !
        #                  =yyyy1: use yyyy data for the fcst. if needed, do    !
        #                          extrapolation to match the fcst time.        !
        #     ioznflg    - ozone data control flag                              !
        #                   =0: use climatological ozone profile                !
        #                   >0: use interactive ozone profile                   !
        #     ivflip     - vertical profile indexing flag                       !
        #     co2dat_file- external co2 2d monthly obsv data table              !
        #     co2gbl_file- external co2 global annual mean data table           !
        #                                                                       !
        #  internal module variables:                                           !
        #     co2vmr_sav - monthly co2 volume mixing ratio     IMXCO2*JMXCO2*12 !
        #     co2cyc_sav - monthly cycle co2 vol mixing ratio  IMXCO2*JMXCO2*12 !
        #     co2_glb    - global annual mean co2 mixing ratio                  !
        #     gco2cyc    - global monthly mean co2 variation       12           !
        #     k1oz,k2oz,facoz                                                   !
        #                - climatology ozone parameters             1           !
        #                                                                       !
        #  usage:    call gas_update                                            !
        #                                                                       !
        #  subprograms called:  none                                            !
        #                                                                       !
        #  ===================================================================  !
        #

        co2dat = np.zeros((self.IMXCO2, self.JMXCO2))
        co2ann = np.zeros((self.IMXCO2, self.JMXCO2))
        co2vmr_sav = np.zeros((self.IMXCO2, self.JMXCO2, 12))

        midmon = 15
        midm = 15
        midp = 45
        #  ---  number of days in a month
        mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 30]

        #
        #===>  ...  begin here
        #
        # - Ozone data section

        if self.ioznflg == 0:
            midmon = mdays[imon]/2 + 1
            change = loz1st or ( (iday == midmon) and (ihour == 0))

            if change:
                if iday < midmon:
                    k1oz = (imon+10 % 12) + 1
                    midm = mdays[k1oz]/2 + 1
                    k2oz = imon
                    midp = mdays[k1oz] + midmon
                else:
                    k1oz = imon
                    midm = midmon
                    k2oz = (imon % 12) + 1
                    midp = mdays[k2oz]/2 + 1 + mdays[k1oz]

            if iday < midmon:
                id = iday + mdays[k1oz]
            else:
                id = iday

            facoz = float(id - midm) / float(midp - midm)

        # - co2 data section

        if self.ico2flg ==  0:
            return    # use prescribed global mean co2 data
        if self.ictmflg == -1:
            return    # use user provided co2 data
        if not ldoco2:
            return    # no need to update co2 data

        if self.ictmflg < 0:       # use user provided external data
            lextpl = False    # no time extrapolation
            idyr   = iyear    # use the model year
        else:                 # use historically observed data
            lextpl = ((self.ictmflg % 10) == 1 )  # flag for data extrapolation
            idyr   = self.ictmflg // 10            # year of data source used
            if idyr == 0:
                idyr = iyear    # not specified, use model year

        #  --- ...  auto select co2 2-d data table for required year
        self.kmonsav = imon
        if self.kyrsav == iyear:
            return
        self.kyrsav = iyear
        iyr    = iyear

        #  --- ...  for data earlier than MINYEAR (1957), the data are in
        #           the form of semi-yearly global mean values.  otherwise,
        #           data are monthly mean in horizontal 2-d map.

        if idyr < self.MINYEAR and self.ictmflg > 0:
            if me == 0:
             print(f'Requested CO2 data year {iyear} earlier than {self.MINYEAR}')
             print('Which is the earliest monthly observation',
                   ' data available.')
             print('Thus, historical global mean data is used')

            #  --- ... check to see if requested co2 data file existed

            file_exist = os.path.isfile(co2gbl_file)

            if not file_exist:
                print(f'Requested co2 data file "{co2gbl_file}" not found',
                      ' - Stopped in subroutine gas_update!!')
            else:
                ds = xr.open_dataset(co2gbl_file)
                iyr1 = ds['iyr1']
                iyr2 = ds['iyr2']
                cline = ds['cline']

                if me == 0:
                    print(f'Opened co2 data file: {co2gbl_file}')

                if idyr < iyr1:
                    iyr = iyr1

                i = iyr2
                while i >= iyr1:
                    jyr = ds['jyr']
                    co2g1 = ds['co2g1']
                    co2g2 = ds['co2g2']

                    if i == iyr and iyr == jyr:
                        co2_glb = (co2g1+co2g2) * 0.5e-6
                        if self.ico2flg == 2:
                            for j in range(self.JMXCO2):
                                for i in range(self.IMXCO2):
                                    co2vmr_sav[i, j, :6]  = co2g1 * 1.0e-6
                                    co2vmr_sav[i, j, 6:] = co2g2 * 1.0e-6

                        if me == 0:
                            print(f'Co2 data for year {iyear} = {co2_glb}')
                        break
                    else:
                        i -= 1

        else:  # Lab_if_idyr

            #  --- ...  set up input data file name

            cfile1 = co2dat_file
            cfile1 = co2dat_file[:18] + str(idyr) + co2dat_file[22:]

            #  --- ... check to see if requested co2 data file existed

            file_exist = os.path.isfile(cfile1)
            if not file_exist:
                if self.ictmflg  > 10:    # specified year of data not found
                    if me == 0:
                        print(f'Specified co2 data for year {idyr} not found !!')
                        print('Need to change namelist ICTM !!')
                        print('   *** Stopped in subroutine gas_update !!')
                else:   # looking for latest available data
                    if me == 0:
                        print(f'Requested co2 data for year {idyr}',
                              ' not found, check for other available data set')

                    while iyr >= self.MINYEAR:
                        iyr -= 1
                        cfile1 = co2dat_file[:18] + str(iyr) + co2dat_file[22:]

                        file_exist = os.path.isfile(cfile1)
                        if me == 0:
                            print(f'Looking for CO2 file {cfile1}')

                        if file_exist:
                            break

                    if not file_exist:
                        if me == 0:
                            print('   Can not find co2 data source file')
                            print('   *** Stopped in subroutine gas_update !!')

            #  --- ...  read in co2 2-d data for the requested month
            ds = xr.open_dataset(cfile1)
            iyr = ds['iyr'].data 
            cline = ds['cline'].data
            co2g1 = ds['co2g1'].data
            co2g2 = ds['co2g2'].data

            if me == 0:
                print(f'Opened co2 data file: {cfile1}')
                print(f'{iyr}, {cline} {co2g1},  GROWTH RATE = {co2g2}')

            #  --- ...  add growth rate if needed
            if lextpl:
                rate = 2.00  * (iyear - iyr)   # avg rate for recent period
            else:
                rate = 0.0

            co2_glb = (co2g1 + rate) * 1.0e-6
            if me == 0:
                print(f'Global annual mean CO2 data for year {iyear} = {co2_glb}')

            if self.ictmflg == -2:     # need to calc ic time annual mean first
                print('Not implemented')
            else:                  # no need to calc ic time annual mean first
                if self.ico2flg == 2:
                    co2dat = ds['co2dat'].data

                    co2vmr_sav = (co2dat + rate)*1.0e-6

                    if me == 0:
                        print('CHECK: Sample of selected months of CO2 ',
                              f'data used for year: {iyear}')
                        for imo in range(0, 12, 3):
                            print(f'Month = {imo+1}')
                            print(co2vmr_sav[0, :, imo])

                gco2cyc = np.zeros(12)

            self.co2vmr_sav = co2vmr_sav
            self.gco2cyc = gco2cyc
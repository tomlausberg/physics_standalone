from gt4py.gtscript import (
    stencil,
    computation,
    interval,
    PARALLEL,
    FORWARD,
    BACKWARD,
    exp,
    log,
    mod,
)
import sys

sys.path.insert(0, "..")
from phys_const import con_amw, con_amd, con_g, con_avgd, con_amo3
from radlw.radlw_param import (
    nbands,
    nplnk,
    nrates,
    eps,
    ngptlw,
    abssnow0,
    absrain,
    cldmin,
    nspa,
    nspb,
    ngb,
    ng01,
    ng02,
    ng03,
    ng04,
    ng05,
    ng06,
    ng07,
    ng08,
    ng09,
    ng10,
    ng11,
    ng12,
    ng13,
    ng14,
    ng15,
    ng16,
    ns02,
    ns03,
    ns04,
    ns05,
    ns06,
    ns07,
    ns08,
    ns09,
    ns10,
    ns11,
    ns12,
    ns13,
    ns14,
    ns15,
    ns16,
    oneminus,
    bpade,
    fluxfac,
    heatfac,
    ntbl,
    wtdiff,
)
from radphysparam import ilwcice, ilwcliq
from config import *

rebuild = False
validate = True

amdw = con_amd / con_amw
amdo3 = con_amd / con_amo3


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nbands": nbands,
        "maxgas": maxgas,
        "ilwcliq": ilwcliq,
        "ilwrgas": ilwrgas,
        "amdw": amdw,
        "amdo3": amdo3,
        "con_avgd": con_avgd,
        "con_g": con_g,
        "con_amd": con_amd,
        "con_amw": con_amw,
        "eps": eps,
    },
)
def firstloop(
    plyr: FIELD_FLT,
    plvl: FIELD_FLT,
    tlyr: FIELD_FLT,
    tlvl: FIELD_FLT,
    qlyr: FIELD_FLT,
    olyr: FIELD_FLT,
    gasvmr: Field[(DTYPE_FLT, (10,))],
    clouds: Field[(DTYPE_FLT, (9,))],
    icseed: FIELD_INT,
    aerosols: Field[(DTYPE_FLT, (nbands, 3))],
    sfemis: FIELD_2D,
    sfgtmp: FIELD_2D,
    dzlyr: FIELD_FLT,
    delpin: FIELD_FLT,
    de_lgth: FIELD_2D,
    cldfrc: FIELD_FLT,
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    delp: FIELD_FLT,
    dz: FIELD_FLT,
    h2ovmr: FIELD_FLT,
    o3vmr: FIELD_FLT,
    coldry: FIELD_FLT,
    colbrd: FIELD_FLT,
    colamt: Field[type_maxgas],
    wx: Field[type_maxxsec],
    tauaer: Field[type_nbands],
    semiss0: Field[gtscript.IJ, type_nbands],
    semiss: Field[gtscript.IJ, type_nbands],
    tem11: FIELD_FLT,
    tem22: FIELD_FLT,
    tem00: FIELD_2D,
    summol: FIELD_FLT,
    pwvcm: FIELD_2D,
    clwp: FIELD_FLT,
    relw: FIELD_FLT,
    ciwp: FIELD_FLT,
    reiw: FIELD_FLT,
    cda1: FIELD_FLT,
    cda2: FIELD_FLT,
    cda3: FIELD_FLT,
    cda4: FIELD_FLT,
    secdiff: Field[gtscript.IJ, type_nbands],
    a0: Field[type_nbands],
    a1: Field[type_nbands],
    a2: Field[type_nbands],
):
    from __externals__ import (
        nbands,
        ilwcliq,
        ilwrgas,
        maxgas,
        amdw,
        amdo3,
        con_avgd,
        con_amd,
        con_amw,
        con_g,
        eps,
    )

    with computation(FORWARD):
        with interval(0, 1):
            for j0 in range(nbands):
                semiss0[0, 0][j0] = 1.0

            if sfemis[0, 0] > eps and sfemis[0, 0] <= 1.0:
                for j in range(nbands):
                    semiss[0, 0][j] = sfemis[0, 0]
            else:
                for j2 in range(nbands):
                    semiss[0, 0][j2] = semiss0[0, 0][j2]

    with computation(PARALLEL):
        with interval(1, None):
            pavel = plyr
            delp = delpin
            tavel = tlyr
            dz = dzlyr

            tem1 = 100.0 * con_g
            tem2 = 1.0e-20 * 1.0e3 * con_avgd

            h2ovmr = max(0.0, qlyr * amdw / (1.0 - qlyr))  # input specific humidity
            o3vmr = max(0.0, olyr * amdo3)  # input mass mixing ratio

            tem0 = (1.0 - h2ovmr) * con_amd + h2ovmr * con_amw
            coldry = tem2 * delp / (tem1 * tem0 * (1.0 + h2ovmr))
            temcol = 1.0e-12 * coldry

            colamt[0, 0, 0][0] = max(0.0, coldry * h2ovmr)  # h2o
            colamt[0, 0, 0][1] = max(temcol, coldry * gasvmr[0, 0, 0][0])  # co2
            colamt[0, 0, 0][2] = max(temcol, coldry * o3vmr)  # o3

            if ilwrgas > 0:
                colamt[0, 0, 0][3] = max(temcol, coldry * gasvmr[0, 0, 0][1])  # n2o
                colamt[0, 0, 0][4] = max(temcol, coldry * gasvmr[0, 0, 0][2])  # ch4
                colamt[0, 0, 0][5] = max(0.0, coldry * gasvmr[0, 0, 0][3])  # o2
                colamt[0, 0, 0][6] = max(0.0, coldry * gasvmr[0, 0, 0][4])  # co

                wx[0, 0, 0][0] = max(0.0, coldry * gasvmr[0, 0, 0][8])  # ccl4
                wx[0, 0, 0][1] = max(0.0, coldry * gasvmr[0, 0, 0][5])  # cf11
                wx[0, 0, 0][2] = max(0.0, coldry * gasvmr[0, 0, 0][6])  # cf12
                wx[0, 0, 0][3] = max(0.0, coldry * gasvmr[0, 0, 0][7])  # cf22

            else:
                colamt[0, 0, 0][3] = 0.0  # n2o
                colamt[0, 0, 0][4] = 0.0  # ch4
                colamt[0, 0, 0][5] = 0.0  # o2
                colamt[0, 0, 0][6] = 0.0  # co

                wx[0, 0, 0][0] = 0.0
                wx[0, 0, 0][1] = 0.0
                wx[0, 0, 0][2] = 0.0
                wx[0, 0, 0][3] = 0.0

            for j3 in range(nbands):
                tauaer[0, 0, 0][j3] = aerosols[0, 0, 0][j3, 0] * (
                    1.0 - aerosols[0, 0, 0][j3, 1]
                )

    with computation(PARALLEL):
        with interval(1, None):
            cldfrc = clouds[0, 0, 0][0]

    with computation(PARALLEL):
        with interval(1, None):
            # Workaround for variables first referenced inside if statements
            # Can be removed at next gt4py release
            clwp = clwp
            relw = relw
            ciwp = ciwp
            reiw = reiw
            cda1 = cda1
            cda2 = cda2
            cda3 = cda3
            cda4 = cda4
            clouds = clouds
            if ilwcliq > 0:
                clwp = clouds[0, 0, 0][1]
                relw = clouds[0, 0, 0][2]
                ciwp = clouds[0, 0, 0][3]
                reiw = clouds[0, 0, 0][4]
                cda1 = clouds[0, 0, 0][5]
                cda2 = clouds[0, 0, 0][6]
                cda3 = clouds[0, 0, 0][7]
                cda4 = clouds[0, 0, 0][8]
            else:
                cda1 = clouds[0, 0, 0][1]

    with computation(FORWARD):
        with interval(0, 1):
            cldfrc = 1.0
        with interval(1, 2):
            tem11 = coldry[0, 0, 0] + colamt[0, 0, 0][0]
            tem22 = colamt[0, 0, 0][0]

    with computation(FORWARD):
        with interval(2, None):
            #  --- ...  compute precipitable water vapor for diffusivity angle adjustments
            tem11 = tem11[0, 0, -1] + coldry + colamt[0, 0, 0][0]
            tem22 = tem22[0, 0, -1] + colamt[0, 0, 0][0]

    with computation(FORWARD):
        with interval(-1, None):
            tem00 = 10.0 * tem22 / (amdw * tem11 * con_g)
    with computation(FORWARD):
        with interval(0, 1):
            pwvcm[0, 0] = tem00[0, 0] * plvl[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            tem1 = 1.80
            tem2 = 1.50
            for j4 in range(nbands):
                if j4 == 0 or j4 == 3 or j4 == 9:
                    secdiff[0, 0][j4] = 1.66
                else:
                    secdiff[0, 0][j4] = min(
                        tem1,
                        max(
                            tem2,
                            a0[0, 0, 0][j4]
                            + a1[0, 0, 0][j4] * exp(a2[0, 0, 0][j4] * pwvcm),
                        ),
                    )
        with interval(1, None):
            for m in range(1, maxgas):
                summol += colamt[0, 0, 0][m]
            colbrd = coldry - summol


@gtscript.stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nbands": nbands,
        "ilwcliq": ilwcliq,
        "ngptlw": ngptlw,
        "isubclw": isubclw,
    },
)
def cldprop(
    cfrac: FIELD_FLT,
    cliqp: FIELD_FLT,
    reliq: FIELD_FLT,
    cicep: FIELD_FLT,
    reice: FIELD_FLT,
    cdat1: FIELD_FLT,
    cdat2: FIELD_FLT,
    cdat3: FIELD_FLT,
    cdat4: FIELD_FLT,
    dz: FIELD_FLT,
    cldfmc: Field[type_ngptlw],
    taucld: Field[type_nbands],
    cldtau: FIELD_FLT,
    absliq1: Field[(DTYPE_FLT, (58, nbands))],
    absice1: Field[(DTYPE_FLT, (2, 5))],
    absice2: Field[(DTYPE_FLT, (43, nbands))],
    absice3: Field[(DTYPE_FLT, (46, nbands))],
    ipat: Field[(DTYPE_INT, (nbands,))],
    tauliq: Field[type_nbands],
    tauice: Field[type_nbands],
    cldf: FIELD_FLT,
    dgeice: FIELD_FLT,
    factor: FIELD_FLT,
    fint: FIELD_FLT,
    tauran: FIELD_FLT,
    tausnw: FIELD_FLT,
    cldliq: FIELD_FLT,
    refliq: FIELD_FLT,
    cldice: FIELD_FLT,
    refice: FIELD_FLT,
    index: FIELD_INT,
    ia: FIELD_INT,
    lcloudy: Field[(DTYPE_INT, (ngptlw,))],
    cdfunc: Field[type_ngptlw],
    tem1: FIELD_FLT,
    lcf1: FIELD_2DBOOL,
    cldsum: FIELD_FLT,
):
    from __externals__ import nbands, ilwcliq, ngptlw, isubclw

    # Compute flag for whether or not there is cloud in the vertical column
    with computation(FORWARD):
        with interval(0, 1):
            cldsum = cfrac[0, 0, 1]
        with interval(1, -1):
            cldsum = cldsum[0, 0, -1] + cfrac[0, 0, 1]
    with computation(FORWARD), interval(-2, -1):
        lcf1 = cldsum > 0

    with computation(FORWARD), interval(1, None):
        if lcf1:
            if ilwcliq > 0:
                if cfrac > cldmin:
                    tauran = absrain * cdat1
                    if cdat3 > 0.0 and cdat4 > 10.0:
                        tausnw = abssnow0 * 1.05756 * cdat3 / cdat4
                    else:
                        tausnw = 0.0

                    cldliq = cliqp
                    cldice = cicep
                    refliq = reliq
                    refice = reice

                    if cldliq <= 0:
                        for i in range(nbands):
                            tauliq[0, 0, 0][i] = 0.0
                    else:
                        if ilwcliq == 1:
                            factor = refliq - 1.5
                            index = max(1, min(57, factor)) - 1
                            fint = factor - (index + 1)

                            for ib in range(nbands):
                                tmp = cldliq * (
                                    absliq1[0, 0, 0][index, ib]
                                    + fint
                                    * (
                                        absliq1[0, 0, 0][index + 1, ib]
                                        - absliq1[0, 0, 0][index, ib]
                                    )
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauliq[0, 0, 0][ib] = tmp if tmp > 0.0 else 0.0

                    if cldice <= 0.0:
                        for ib2 in range(nbands):
                            tauice[0, 0, 0][ib2] = 0.0
                    else:
                        if ilwcice == 1:
                            refice = min(130.0, max(13.0, refice))

                            for ib3 in range(nbands):
                                ia = ipat[0, 0, 0][ib3] - 1
                                tmp = cldice * (
                                    absice1[0, 0, 0][0, ia]
                                    + absice1[0, 0, 0][1, ia] / refice
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauice[0, 0, 0][ib3] = tmp if tmp > 0.0 else 0.0
                        elif ilwcice == 2:
                            factor = (refice - 2.0) / 3.0
                            index = max(1, min(42, factor)) - 1
                            fint = factor - (index + 1)

                            for ib4 in range(nbands):
                                tmp = cldice * (
                                    absice2[0, 0, 0][index, ib4]
                                    + fint
                                    * (
                                        absice2[0, 0, 0][index + 1, ib4]
                                        - absice2[0, 0, 0][index, ib4]
                                    )
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauice[0, 0, 0][ib4] = tmp if tmp > 0.0 else 0.0

                        elif ilwcice == 3:
                            dgeice = max(5.0, 1.0315 * refice)  # v4.71 value
                            factor = (dgeice - 2.0) / 3.0
                            index = max(1, min(45, factor)) - 1
                            fint = factor - (index + 1)

                            for ib5 in range(nbands):
                                tmp = cldice * (
                                    absice3[0, 0, 0][index, ib5]
                                    + fint
                                    * (
                                        absice3[0, 0, 0][index + 1, ib5]
                                        - absice3[0, 0, 0][index, ib5]
                                    )
                                )
                                # workaround since max doesn't work in for loop in if statement
                                tauice[0, 0, 0][ib5] = tmp if tmp > 0.0 else 0.0

                    for ib6 in range(nbands):
                        taucld[0, 0, 0][ib6] = (
                            tauice[0, 0, 0][ib6]
                            + tauliq[0, 0, 0][ib6]
                            + tauran
                            + tausnw
                        )

            else:
                if cfrac > cldmin:
                    for ib7 in range(nbands):
                        taucld[0, 0, 0][ib7] = cdat1

            if isubclw > 0:
                if cfrac < cldmin:
                    cldf = 0.0
                else:
                    cldf = cfrac

    # This section builds mcica_subcol from the fortran into cldprop.
    # Here I've read in the generated random numbers until we figure out
    # what to do with them. This will definitely need to change in future.
    # Only the iovrlw = 1 option is ported from Fortran
    with computation(PARALLEL), interval(2, None):
        if lcf1:
            tem1 = 1.0 - cldf[0, 0, -1]

            for n in range(ngptlw):
                if cdfunc[0, 0, -1][n] > tem1:
                    cdfunc[0, 0, 0][n] = cdfunc[0, 0, -1][n]
                else:
                    cdfunc[0, 0, 0][n] = cdfunc[0, 0, 0][n] * tem1

    with computation(PARALLEL), interval(1, None):
        if lcf1:
            tem1 = 1.0 - cldf[0, 0, 0]

            for n2 in range(ngptlw):
                if cdfunc[0, 0, 0][n2] >= tem1:
                    lcloudy[0, 0, 0][n2] = 1
                else:
                    lcloudy[0, 0, 0][n2] = 0

            for n3 in range(ngptlw):
                if lcloudy[0, 0, 0][n3] == 1:
                    cldfmc[0, 0, 0][n3] = 1.0
                else:
                    cldfmc[0, 0, 0][n3] = 0.0

            cldtau = taucld[0, 0, 0][6]


stpfac = 296.0 / 1013.0


@stencil(
    backend=backend, rebuild=rebuild, externals={"nbands": nbands, "stpfac": stpfac}
)
def setcoef(
    pavel: FIELD_FLT,
    tavel: FIELD_FLT,
    tz: FIELD_FLT,
    stemp: FIELD_2D,
    h2ovmr: FIELD_FLT,
    colamt: Field[type_maxgas],
    coldry: FIELD_FLT,
    colbrd: FIELD_FLT,
    totplnk: Field[(DTYPE_FLT, (nplnk, nbands))],
    pref: Field[(DTYPE_FLT, (59,))],
    preflog: Field[(DTYPE_FLT, (59,))],
    tref: Field[(DTYPE_FLT, (59,))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    delwave: Field[type_nbands],
    laytrop: Field[bool],
    pklay: Field[type_nbands],
    pklev: Field[type_nbands],
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac: Field[(DTYPE_FLT, (4,))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    scaleminor: FIELD_FLT,
    scaleminorn2: FIELD_FLT,
    indminor: FIELD_INT,
    tzint: FIELD_INT,
    stempint: FIELD_INT,
    tavelint: FIELD_INT,
    indlay: FIELD_INT,
    indlev: FIELD_INT,
    tlyrfr: FIELD_FLT,
    tlvlfr: FIELD_FLT,
    jp1: FIELD_INT,
    plog: FIELD_FLT,
):
    from __externals__ import nbands, stpfac

    with computation(PARALLEL):
        #  --- ...  calculate information needed by the radiative transfer routine
        #           that is specific to this atmosphere, especially some of the
        #           coefficients and indices needed to compute the optical depths
        #           by interpolating data from stored reference atmospheres.
        with interval(0, 1):
            indlay = min(180, max(1, stemp - 159.0))
            indlev = min(180, max(1, tz - 159.0))
            tzint = tz
            stempint = stemp
            tlyrfr = stemp - stempint
            tlvlfr = tz - tzint

            for i0 in range(nbands):
                tem1 = totplnk[0, 0, 0][indlay, i0] - totplnk[0, 0, 0][indlay - 1, i0]
                tem2 = totplnk[0, 0, 0][indlev, i0] - totplnk[0, 0, 0][indlev - 1, i0]
                pklay[0, 0, 0][i0] = delwave[0, 0, 0][i0] * (
                    totplnk[0, 0, 0][indlay - 1, i0] + tlyrfr * tem1
                )
                pklev[0, 0, 0][i0] = delwave[0, 0, 0][i0] * (
                    totplnk[0, 0, 0][indlev - 1, i0] + tlvlfr * tem2
                )

        #           calculate the integrated Planck functions for each band at the
        #           surface, level, and layer temperatures.
        with interval(1, None):
            indlay = min(180, max(1, tavel - 159.0))
            tavelint = tavel
            tlyrfr = tavel - tavelint

            indlev = min(180, max(1, tz - 159.0))
            tzint = tz
            tlvlfr = tz - tzint

            #  --- ...  begin spectral band loop
            for i in range(nbands):
                pklay[0, 0, 0][i] = delwave[0, 0, 0][i] * (
                    totplnk[0, 0, 0][indlay - 1, i]
                    + tlyrfr
                    * (totplnk[0, 0, 0][indlay, i] - totplnk[0, 0, 0][indlay - 1, i])
                )
                pklev[0, 0, 0][i] = delwave[0, 0, 0][i] * (
                    totplnk[0, 0, 0][indlev - 1, i]
                    + tlvlfr
                    * (totplnk[0, 0, 0][indlev, i] - totplnk[0, 0, 0][indlev - 1, i])
                )

            #  --- ...  find the two reference pressures on either side of the
            #           layer pressure. store them in jp and jp1. store in fp the
            #           fraction of the difference (in ln(pressure)) between these
            #           two values that the layer pressure lies.

            plog = log(pavel)
            jp = max(1, min(58, 36.0 - 5.0 * (plog + 0.04))) - 1
            jp1 = jp + 1
            #  --- ...  limit pressure extrapolation at the top
            fp = max(0.0, min(1.0, 5.0 * (preflog[0, 0, 0][jp] - plog)))

            #  --- ...  determine, for each reference pressure (jp and jp1), which
            #           reference temperature (these are different for each
            #           reference pressure) is nearest the layer temperature but does
            #           not exceed it. store these indices in jt and jt1, resp.
            #           store in ft (resp. ft1) the fraction of the way between jt
            #           (jt1) and the next highest reference temperature that the
            #           layer temperature falls.

            tem1 = (tavel - tref[0, 0, 0][jp]) / 15.0
            tem2 = (tavel - tref[0, 0, 0][jp1]) / 15.0
            jt = max(1, min(4, 3.0 + tem1)) - 1
            jt1 = max(1, min(4, 3.0 + tem2)) - 1
            # --- ...  restrict extrapolation ranges by limiting abs(det t) < 37.5 deg
            ft = max(-0.5, min(1.5, tem1 - (jt - 2)))
            ft1 = max(-0.5, min(1.5, tem2 - (jt1 - 2)))

            #  --- ...  we have now isolated the layer ln pressure and temperature,
            #           between two reference pressures and two reference temperatures
            #           (for each reference pressure).  we multiply the pressure
            #           fraction fp with the appropriate temperature fractions to get
            #           the factors that will be needed for the interpolation that yields
            #           the optical depths (performed in routines taugbn for band n)

            tem1 = 1.0 - fp
            fac10 = tem1 * ft
            fac00 = tem1 * (1.0 - ft)
            fac11 = fp * ft1
            fac01 = fp * (1.0 - ft1)

            fac[0,0,0][0] = fac00
            fac[0,0,0][1] = fac01
            fac[0,0,0][2] = fac10
            fac[0,0,0][3] = fac11

            forfac = pavel * stpfac / (tavel * (1.0 + h2ovmr))
            selffac = h2ovmr * forfac

            #  --- ...  set up factors needed to separately include the minor gases
            #           in the calculation of absorption coefficient

            scaleminor = pavel / tavel
            scaleminorn2 = (pavel / tavel) * (colbrd / (coldry + colamt[0, 0, 0][0]))

            tem1 = (tavel - 180.8) / 7.2
            indminor = min(18, max(1, tem1))
            minorfrac = tem1 - indminor

            #  --- ...  if the pressure is less than ~100mb, perform a different
            #           set of species interpolations.

            indfor = indfor
            forfrac = forfrac
            indself = indself
            selffrac = selffrac
            rfrate = rfrate
            chi_mls = chi_mls
            laytrop = laytrop

            if plog > 4.56:

                # compute troposphere mask, True in troposphere, False otherwise
                laytrop = True

                tem1 = (332.0 - tavel) / 36.0
                indfor = min(2, max(1, tem1))
                forfrac = tem1 - indfor

                #  --- ...  set up factors needed to separately include the water vapor
                #           self-continuum in the calculation of absorption coefficient.

                tem1 = (tavel - 188.0) / 7.2
                indself = min(9, max(1, tem1 - 7))
                selffrac = tem1 - (indself + 7)

                #  --- ...  setup reference ratio to be used in calculation of binary
                #           species parameter in lower atmosphere.

                rfrate[0, 0, 0][0, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][0, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )
                rfrate[0, 0, 0][1, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][2, jp]
                )
                rfrate[0, 0, 0][1, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][2, jp + 1]
                )
                rfrate[0, 0, 0][2, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][3, jp]
                )
                rfrate[0, 0, 0][2, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][3, jp + 1]
                )
                rfrate[0, 0, 0][3, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][5, jp]
                )
                rfrate[0, 0, 0][3, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][5, jp + 1]
                )
                rfrate[0, 0, 0][4, 0] = (
                    chi_mls[0, 0, 0][3, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][4, 1] = (
                    chi_mls[0, 0, 0][3, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )

            else:
                laytrop = False

                tem1 = (tavel - 188.0) / 36.0
                indfor = 3
                forfrac = tem1 - 1.0

                indself = 0
                selffrac = 0.0

                #  --- ...  setup reference ratio to be used in calculation of binary
                #           species parameter in upper atmosphere.

                rfrate[0, 0, 0][0, 0] = (
                    chi_mls[0, 0, 0][0, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][0, 1] = (
                    chi_mls[0, 0, 0][0, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )
                rfrate[0, 0, 0][5, 0] = (
                    chi_mls[0, 0, 0][2, jp] / chi_mls[0, 0, 0][1, jp]
                )
                rfrate[0, 0, 0][5, 1] = (
                    chi_mls[0, 0, 0][2, jp + 1] / chi_mls[0, 0, 0][1, jp + 1]
                )

            #  --- ...  rescale selffac and forfac for use in taumol

            selffac = colamt[0, 0, 0][0] * selffac
            forfac = colamt[0, 0, 0][0] * forfac

            #  --- ...  add one to computed indices for compatibility with later
            #           subroutines

            jp += 1
            jt += 1
            jt1 += 1

@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[0],
        "nspb": nspb[0],
        "ng01": ng01,
    },
)
def taugb01(
    laytrop: FIELD_BOOL,
    pavel: FIELD_FLT,
    colamt: Field[type_maxgas],
    colbrd: FIELD_FLT,
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    scaleminorn2: FIELD_FLT,
    indminor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (10, 65))],
    absb: Field[(DTYPE_FLT, (10, 235))],
    selfref: Field[(DTYPE_FLT, (10, 10))],
    forref: Field[(DTYPE_FLT, (10, 4))],
    fracrefa: Field[(DTYPE_FLT, (10,))],
    fracrefb: Field[(DTYPE_FLT, (10,))],
    ka_mn2: Field[(DTYPE_FLT, (10, 19))],
    kb_mn2: Field[(DTYPE_FLT, (10, 19))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    pp: FIELD_FLT,
    corradj: FIELD_FLT,
    scalen2: FIELD_FLT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    taun2: FIELD_FLT,
):

    from __externals__ import nspa, nspb, ng01

    with computation(PARALLEL), interval(1, None):
        # Workaround for bug in gt4py
        jp = jp
        jt = jt
        jt1 = jt1
        indself = indself
        indfor = indfor
        indminor = indminor
        pavel = pavel
        colbrd = colbrd
        scaleminorn2 = scaleminorn2

        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa
            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1

            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1

            pp = pavel
            scalen2 = colbrd * scaleminorn2
            if pp < 250.0:
                corradj = 1.0 - 0.15 * (250.0 - pp) / 154.4
            else:
                corradj = 1.0

            for ig in range(ng01):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                taun2 = scalen2 * (
                    ka_mn2[0, 0, 0][ig, indm]
                    + minorfrac
                    * (ka_mn2[0, 0, 0][ig, indmp] - ka_mn2[0, 0, 0][ig, indm])
                )

                taug[0, 0, 0][ig] = corradj * (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                    + taun2
                )

                fracs[0, 0, 0][ig] = fracrefa[0, 0, 0][ig]

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb
            indf = indfor - 1
            indm = indminor - 1

            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indfp = indf + 1
            indmp = indm + 1

            scalen2 = colbrd * scaleminorn2
            corradj = 1.0 - 0.15 * (pavel / 95.6)

            for ig2 in range(ng01):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )
                taun2 = scalen2 * (
                    kb_mn2[0, 0, 0][ig2, indm]
                    + minorfrac
                    * (kb_mn2[0, 0, 0][ig2, indmp] - kb_mn2[0, 0, 0][ig2, indm])
                )

                taug[0, 0, 0][ig2] = corradj * (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + taufor
                    + taun2
                )

                fracs[0, 0, 0][ig2] = fracrefb[0, 0, 0][ig2]

@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[1],
        "nspb": nspb[1],
        "ng02": ng02,
        "ns02": ns02,
    },
)
def taugb02(
    laytrop: FIELD_BOOL,
    pavel: FIELD_FLT,
    colamt: Field[type_maxgas],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (10, 65))],
    absb: Field[(DTYPE_FLT, (10, 235))],
    selfref: Field[(DTYPE_FLT, (10, 10))],
    forref: Field[(DTYPE_FLT, (10, 4))],
    fracrefa: Field[(DTYPE_FLT, (10,))],
    fracrefb: Field[(DTYPE_FLT, (10,))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    corradj: FIELD_FLT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng02, ns02

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa
            ind1 = (jp * 5 + (jt1 - 1)) * nspa
            inds = indself - 1
            indf = indfor - 1

            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indsp = inds + 1
            indfp = indf + 1

            corradj = 1.0 - 0.05 * (pavel - 100.0) / 900.0

            for ig in range(ng02):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                taug[0, 0, 0][ns02 + ig] = corradj * (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absa[0, 0, 0][ig, ind0]
                        + fac10 * absa[0, 0, 0][ig, ind0p]
                        + fac01 * absa[0, 0, 0][ig, ind1]
                        + fac11 * absa[0, 0, 0][ig, ind1p]
                    )
                    + tauself
                    + taufor
                )

                fracs[0, 0, 0][ns02 + ig] = fracrefa[0, 0, 0][ig]

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb
            indf = indfor - 1

            ind0p = ind0 + 1
            ind1p = ind1 + 1
            indfp = indf + 1

            for ig2 in range(ng02):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )

                taug[0, 0, 0][ns02 + ig2] = (
                    colamt[0, 0, 0][0]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                    + taufor
                )

                fracs[0, 0, 0][ns02 + ig2] = fracrefb[0, 0, 0][ig2]

@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[2],
        "nspb": nspb[2],
        "ng03": ng03,
        "ns03": ns03,
        "oneminus": oneminus,
    },
)
def taubg03a(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac01: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt1: FIELD_INT,
    absa: Field[(DTYPE_FLT, (ng03, 585))],
    absb: Field[(DTYPE_FLT, (ng03, 1175))],
    ind1: FIELD_INT,
    js1: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm1: FIELD_FLT,
    taumajor1: Field[type_ngptlw],
):
    from __externals__ import nspa, nspb, ng03, ns03, oneminus
    
    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1
    
            if specparm1 < 0.125:
                p = fs1 - 1.0
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p = -fs1
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk0 = 1.0 - fs1
                fk1 = fs1
                fk2 = 0.0
                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk0 * fac01
            fac101 = fk1 * fac01
            fac201 = fk2 * fac01
            fac011 = fk0 * fac11
            fac111 = fk1 * fac11
            fac211 = fk2 * fac11

            for ig in range(ng03):
                taumajor1[0, 0, 0][ns03 + ig] = speccomb1 * (
                    fac001 * absa[0, 0, 0][ig, id001]
                    + fac011 * absa[0, 0, 0][ig, id011]
                    + fac101 * absa[0, 0, 0][ig, id101]
                    + fac111 * absa[0, 0, 0][ig, id111]
                    + fac201 * absa[0, 0, 0][ig, id201]
                    + fac211 * absa[0, 0, 0][ig, id211]
                )
        else:
            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 4.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb + js1 - 1

            id001 = ind1
            id011 = ind1 + 5
            id101 = ind1 + 1
            id111 = ind1 + 6

            fk0 = 1.0 - fs1
            fk1 = fs1
            fac001 = fk0 * fac01
            fac011 = fk0 * fac11
            fac101 = fk1 * fac01
            fac111 = fk1 * fac11

            for ig2 in range(ng03):
                taumajor1[0, 0, 0][ns03 + ig2] = speccomb1 * (
                    fac001 * absb[0, 0, 0][ig2, id001]
                    + fac011 * absb[0, 0, 0][ig2, id011]
                    + fac101 * absb[0, 0, 0][ig2, id101]
                    + fac111 * absb[0, 0, 0][ig2, id111]
                )

@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[2],
        "nspb": nspb[2],
        "ng03": ng03,
        "ns03": ns03,
        "oneminus": oneminus,
    },
)
def taubg03b(
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac10: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    minorfrac: FIELD_FLT,
    indminor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng03, 585))],
    absb: Field[(DTYPE_FLT, (ng03, 1175))],
    selfref: Field[(DTYPE_FLT, (ng03, 10))],
    forref: Field[(DTYPE_FLT, (ng03, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng03, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng03, 5))],
    ka_mn2o: Field[(DTYPE_FLT, (ng03, 9, 19))],
    kb_mn2o: Field[(DTYPE_FLT, (ng03, 5, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    jmn2o: FIELD_INT,
    jmn2op: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    specparm: FIELD_FLT,
    ratn2o: FIELD_FLT,
    taumajor1: Field[type_ngptlw],
):
    from __externals__ import nspa, nspb, ng03, ns03, oneminus

    with computation(PARALLEL):
        with interval(...):
            #  --- ...  minor gas mapping levels:
            #     lower - n2o, p = 706.272 mbar, t = 278.94 k
            #     upper - n2o, p = 95.58 mbar, t = 215.7 k

            refrat_planck_a = chi_mls[0, 0, 0][0, 8] / chi_mls[0, 0, 0][1, 8]
            refrat_planck_b = chi_mls[0, 0, 0][0, 12] / chi_mls[0, 0, 0][1, 12]
            refrat_m_a = chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][1, 2]
            refrat_m_b = chi_mls[0, 0, 0][0, 12] / chi_mls[0, 0, 0][1, 12]

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb_mn2o = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][1]
            specparm_mn2o = colamt[0, 0, 0][0] / speccomb_mn2o
            specmult_mn2o = 8.0 * min(specparm_mn2o, oneminus)
            jmn2o = 1 + specmult_mn2o - 1
            fmn2o = mod(specmult_mn2o, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1
            jmn2op = jmn2o + 1
            jplp = jpl + 1

            #  --- ...  in atmospheres where the amount of n2O is too great to be considered
            #           a minor species, adjust the column amount of n2O by an empirical factor
            #           to obtain the proper contribution.

            p = coldry * chi_mls[0, 0, 0][3, jp]
            ratn2o = colamt[0, 0, 0][3] / p
            if ratn2o > 1.5:
                adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
                adjcoln2o = adjfac * p
            else:
                adjcoln2o = colamt[0, 0, 0][3]

            if specparm < 0.125:
                p = fs - 1.0
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p = -fs
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0 * 1
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk0 = 1.0 - fs
                fk1 = fs
                fk2 = 0.0
                id000 = ind0 * 1
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 * 1
                id210 = ind0 * 1

            fac000 = fk0 * fac00
            fac100 = fk1 * fac00
            fac200 = fk2 * fac00
            fac010 = fk0 * fac10
            fac110 = fk1 * fac10
            fac210 = fk2 * fac10

            for ig in range(ng03):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                n2om1 = ka_mn2o[0, 0, 0][ig, jmn2o, indm] + fmn2o * (
                    ka_mn2o[0, 0, 0][ig, jmn2op, indm]
                    - ka_mn2o[0, 0, 0][ig, jmn2o, indm]
                )
                n2om2 = ka_mn2o[0, 0, 0][ig, jmn2o, indmp] + fmn2o * (
                    ka_mn2o[0, 0, 0][ig, jmn2op, indmp]
                    - ka_mn2o[0, 0, 0][ig, jmn2o, indmp]
                )
                absn2o = n2om1 + minorfrac * (n2om2 - n2om1)

                tau_major = speccomb * (
                    fac000 * absa[0, 0, 0][ig, id000]
                    + fac010 * absa[0, 0, 0][ig, id010]
                    + fac100 * absa[0, 0, 0][ig, id100]
                    + fac110 * absa[0, 0, 0][ig, id110]
                    + fac200 * absa[0, 0, 0][ig, id200]
                    + fac210 * absa[0, 0, 0][ig, id210]
                )

                taug[0, 0, 0][ns03 + ig] = (
                    tau_major + taumajor1[0,0,0][ns03 + ig] + tauself + taufor + adjcoln2o * absn2o
                )

                fracs[0, 0, 0][ns03 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:

            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 4.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb + js - 1

            speccomb_mn2o = colamt[0, 0, 0][0] + refrat_m_b * colamt[0, 0, 0][1]
            specparm_mn2o = colamt[0, 0, 0][0] / speccomb_mn2o
            specmult_mn2o = 4.0 * min(specparm_mn2o, oneminus)
            jmn2o = 1 + specmult_mn2o - 1
            fmn2o = mod(specmult_mn2o, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_b * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 4.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            indf = indfor - 1
            indm = indminor - 1
            indfp = indf + 1
            indmp = indm + 1
            jmn2op = jmn2o + 1
            jplp = jpl + 1

            id000 = ind0
            id010 = ind0 + 5
            id100 = ind0 + 1
            id110 = ind0 + 6

            #  --- ...  in atmospheres where the amount of n2o is too great to be considered
            #           a minor species, adjust the column amount of N2O by an empirical factor
            #           to obtain the proper contribution.

            p = coldry * chi_mls[0, 0, 0][3, jp]
            ratn2o = colamt[0, 0, 0][3] / p
            if ratn2o > 1.5:
                adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
                adjcoln2o = adjfac * p
            else:
                adjcoln2o = colamt[0, 0, 0][3]

            fk0 = 1.0 - fs
            fk1 = fs
            fac000 = fk0 * fac00
            fac010 = fk0 * fac10
            fac100 = fk1 * fac00
            fac110 = fk1 * fac10

            for ig2 in range(ng03):
                taufor = forfac * (
                    forref[0, 0, 0][ig2, indf]
                    + forfrac
                    * (forref[0, 0, 0][ig2, indfp] - forref[0, 0, 0][ig2, indf])
                )
                n2om1 = kb_mn2o[0, 0, 0][ig2, jmn2o, indm] + fmn2o * (
                    kb_mn2o[0, 0, 0][ig2, jmn2op, indm]
                    - kb_mn2o[0, 0, 0][ig2, jmn2o, indm]
                )
                n2om2 = kb_mn2o[0, 0, 0][ig2, jmn2o, indmp] + fmn2o * (
                    kb_mn2o[0, 0, 0][ig2, jmn2op, indmp]
                    - kb_mn2o[0, 0, 0][ig2, jmn2o, indmp]
                )
                absn2o = n2om1 + minorfrac * (n2om2 - n2om1)

                tau_major = speccomb * (
                    fac000 * absb[0, 0, 0][ig2, id000]
                    + fac010 * absb[0, 0, 0][ig2, id010]
                    + fac100 * absb[0, 0, 0][ig2, id100]
                    + fac110 * absb[0, 0, 0][ig2, id110]
                )

                taug[0, 0, 0][ns03 + ig2] = (
                    tau_major + taumajor1[0,0,0][ns03 + ig2] + taufor + adjcoln2o * absn2o
                )

                fracs[0, 0, 0][ns03 + ig2] = fracrefb[0, 0, 0][ig2, jpl] + fpl * (
                    fracrefb[0, 0, 0][ig2, jplp] - fracrefb[0, 0, 0][ig2, jpl]
                )




@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[3],
        "nspb": nspb[3],
        "ng04": ng04,
        "ns04": ns04,
        "oneminus": oneminus,
    },
)
def taugb04(
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac: Field[(DTYPE_FLT, (4,))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    absa: Field[(DTYPE_FLT, (ng04, 585))],
    absb: Field[(DTYPE_FLT, (ng04, 1175))],
    selfref: Field[(DTYPE_FLT, (ng04, 10))],
    forref: Field[(DTYPE_FLT, (ng04, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng04, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng04, 5))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    js: FIELD_INT,
    js1: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng04, ns04, oneminus

    with computation(PARALLEL):
        with interval(...):
            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 10] / chi_mls[0, 0, 0][1, 10]
            )  # P = 142.5940 mb
            refrat_planck_b = (
                chi_mls[0, 0, 0][2, 12] / chi_mls[0, 0, 0][1, 12]
            )  # P = 95.58350 mb

    with computation(PARALLEL), interval(1, None):
        refrat_planck_a = refrat_planck_a
        refrat_planck_b = refrat_planck_b
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1
            jplp = jpl + 1

            if specparm < 0.125:
                p = fs - 1.0
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p = -fs
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk0 = 1.0 - fs
                fk1 = fs
                fk2 = 0.0
                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0
                id210 = ind0

            fac000 = fk0 * fac00
            fac100 = fk1 * fac00
            fac200 = fk2 * fac00
            fac010 = fk0 * fac10
            fac110 = fk1 * fac10
            fac210 = fk2 * fac10

            if specparm1 < 0.125:
                p = fs1 - 1.0
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p = -fs1
                p4 = p ** 4
                fk0 = p4
                fk1 = 1.0 - p - 2.0 * p4
                fk2 = p + p4
                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk0 = 1.0 - fs1
                fk1 = fs1
                fk2 = 0.0
                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk0 * fac01
            fac101 = fk1 * fac01
            fac201 = fk2 * fac01
            fac011 = fk0 * fac11
            fac111 = fk1 * fac11
            fac211 = fk2 * fac11

            for ig in range(ng04):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                tau_major = speccomb * (
                    fac000 * absa[0, 0, 0][ig, id000]
                    + fac010 * absa[0, 0, 0][ig, id010]
                    + fac100 * absa[0, 0, 0][ig, id100]
                    + fac110 * absa[0, 0, 0][ig, id110]
                    + fac200 * absa[0, 0, 0][ig, id200]
                    + fac210 * absa[0, 0, 0][ig, id210]
                )

                tau_major1 = speccomb1 * (
                    fac001 * absa[0, 0, 0][ig, id001]
                    + fac011 * absa[0, 0, 0][ig, id011]
                    + fac101 * absa[0, 0, 0][ig, id101]
                    + fac111 * absa[0, 0, 0][ig, id111]
                    + fac201 * absa[0, 0, 0][ig, id201]
                    + fac211 * absa[0, 0, 0][ig, id211]
                )

                taug[0, 0, 0][ns04 + ig] = tau_major + tau_major1 + tauself + taufor

                fracs[0, 0, 0][ns04 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            speccomb = colamt[0, 0, 0][2] + rfrate[0, 0, 0][5, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][2] / speccomb
            specmult = 4.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb + js - 1

            speccomb1 = colamt[0, 0, 0][2] + rfrate[0, 0, 0][5, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][2] / speccomb1
            specmult1 = 4.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb + js1 - 1

            speccomb_planck = colamt[0, 0, 0][2] + refrat_planck_b * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][2] / speccomb_planck
            specmult_planck = 4.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)
            jplp = jpl + 1

            fk0 = 1.0 - fs
            fk1 = fs
            fac000 = fk0 * fac00
            fac010 = fk0 * fac10
            fac100 = fk1 * fac00
            fac110 = fk1 * fac10

            fk0 = 1.0 - fs1
            fk1 = fs1
            fac001 = fk0 * fac01
            fac011 = fk0 * fac11
            fac101 = fk1 * fac01
            fac111 = fk1 * fac11

            for ig2 in range(ng04):
                tau_major = speccomb * (
                    fac000 * absb[0, 0, 0][ig2, ind0]
                    + fac010 * absb[0, 0, 0][ig2, ind0 + 5]
                    + fac100 * absb[0, 0, 0][ig2, ind0 + 1]
                    + fac110 * absb[0, 0, 0][ig2, ind0 + 6]
                )
                tau_major1 = speccomb1 * (
                    fac001 * absb[0, 0, 0][ig2, ind1]
                    + fac011 * absb[0, 0, 0][ig2, ind1 + 5]
                    + fac101 * absb[0, 0, 0][ig2, ind1 + 1]
                    + fac111 * absb[0, 0, 0][ig2, ind1 + 6]
                )

                taug[0, 0, 0][ns04 + ig2] = tau_major + tau_major1

                fracs[0, 0, 0][ns04 + ig2] = fracrefb[0, 0, 0][ig2, jpl] + fpl * (
                    fracrefb[0, 0, 0][ig2, jplp] - fracrefb[0, 0, 0][ig2, jpl]
                )

            taug[0, 0, 0][ns04 + 7] = taug[0, 0, 0][ns04 + 7] * 0.92
            taug[0, 0, 0][ns04 + 8] = taug[0, 0, 0][ns04 + 8] * 0.88
            taug[0, 0, 0][ns04 + 9] = taug[0, 0, 0][ns04 + 9] * 1.07
            taug[0, 0, 0][ns04 + 10] = taug[0, 0, 0][ns04 + 10] * 1.1
            taug[0, 0, 0][ns04 + 11] = taug[0, 0, 0][ns04 + 11] * 0.99
            taug[0, 0, 0][ns04 + 12] = taug[0, 0, 0][ns04 + 12] * 0.88
            taug[0, 0, 0][ns04 + 13] = taug[0, 0, 0][ns04 + 13] * 0.943

@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[3],
        "nspb": nspb[3],
        "ng04": ng04,
        "ns04": ns04,
        "oneminus": oneminus,
    },
)
def taugb04_abs(
    abc: Field[(DTYPE_FLT, (ng04,))],
    abc1: Field[(DTYPE_FLT, (ng04,))],
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    rfrate: Field[(DTYPE_FLT, (nrates, 2))],
    fac00: FIELD_FLT,
    fac01: FIELD_FLT,
    fac10: FIELD_FLT,
    fac11: FIELD_FLT,
    jp: FIELD_INT,
    jt: FIELD_INT,
    jt1: FIELD_INT,
    absa: Field[(DTYPE_FLT, (ng04, 585))],
    absb: Field[(DTYPE_FLT, (ng04, 1175))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    js: FIELD_INT,
    js1: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, nspb, oneminus, ng04
    
    if laytrop:
        speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 0] * colamt[0, 0, 0][1]
        specparm = colamt[0, 0, 0][0] / speccomb
        specmult = 8.0 * min(specparm, oneminus)
        js = 1 + specmult
        fs = mod(specmult, 1.0)
        ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

        speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][0, 1] * colamt[0, 0, 0][1]
        specparm1 = colamt[0, 0, 0][0] / speccomb1
        specmult1 = 8.0 * min(specparm1, oneminus)
        js1 = 1 + specmult1
        fs1 = mod(specmult1, 1.0)
        ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1
        
        if specparm < 0.125:
            p = fs - 1.0
            p4 = p ** 4
            fk0 = p4
            fk1 = 1.0 - p - 2.0 * p4
            fk2 = p + p4
            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 + 10
            id200 = ind0 + 2
            id210 = ind0 + 11
        elif specparm > 0.875:
            p = -fs
            p4 = p ** 4
            fk0 = p4
            fk1 = 1.0 - p - 2.0 * p4
            fk2 = p + p4
            id000 = ind0 + 1
            id010 = ind0 + 10
            id100 = ind0
            id110 = ind0 + 9
            id200 = ind0 - 1
            id210 = ind0 + 8
        else:
            fk0 = 1.0 - fs
            fk1 = fs
            fk2 = 0.0
            id000 = ind0
            id010 = ind0 + 9
            id100 = ind0 + 1
            id110 = ind0 + 10
            id200 = ind0
            id210 = ind0
        fac000 = fk0 * fac00
        fac100 = fk1 * fac00
        fac200 = fk2 * fac00
        fac010 = fk0 * fac10
        fac110 = fk1 * fac10
        fac210 = fk2 * fac10


        if specparm1 < 0.125:
            p = fs1 - 1.0
            p4 = p ** 4
            fk0 = p4
            fk1 = 1.0 - p - 2.0 * p4
            fk2 = p + p4
            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 + 10
            id201 = ind1 + 2
            id211 = ind1 + 11
        elif specparm1 > 0.875:
            p = -fs1
            p4 = p ** 4
            fk0 = p4
            fk1 = 1.0 - p - 2.0 * p4
            fk2 = p + p4
            id001 = ind1 + 1
            id011 = ind1 + 10
            id101 = ind1
            id111 = ind1 + 9
            id201 = ind1 - 1
            id211 = ind1 + 8
        else:
            fk0 = 1.0 - fs1
            fk1 = fs1
            fk2 = 0.0
            id001 = ind1
            id011 = ind1 + 9
            id101 = ind1 + 1
            id111 = ind1 + 10
            id201 = ind1
            id211 = ind1

        fac001 = fk0 * fac01
        fac101 = fk1 * fac01
        fac201 = fk2 * fac01
        fac011 = fk0 * fac11
        fac111 = fk1 * fac11
        fac211 = fk2 * fac11

        for ig in range(ng04):
            abc[0,0,0][ig] = fac000* absa[0, 0, 0][ig, id000] \
                + fac010 * absa[0, 0, 0][ig, id010] \
                + fac100 * absa[0, 0, 0][ig, id100] \
                + fac110 * absa[0, 0, 0][ig, id110] \
                + fac200 * absa[0, 0, 0][ig, id200] \
                + fac210 * absa[0, 0, 0][ig, id210]

            abc1[0,0,0][ig] = fac001 * absa[0, 0, 0][ig, id001] \
                + fac011 * absa[0, 0, 0][ig, id011] \
                + fac101 * absa[0, 0, 0][ig, id101] \
                + fac111 * absa[0, 0, 0][ig, id111] \
                + fac201 * absa[0, 0, 0][ig, id201] \
                + fac211 * absa[0, 0, 0][ig, id211]


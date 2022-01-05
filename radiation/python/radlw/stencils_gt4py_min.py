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

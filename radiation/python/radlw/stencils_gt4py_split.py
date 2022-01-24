from cmath import tau
from gt4py.gtscript import (
    stencil,
    function,
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

@function
def set_ids(specparm,fs,fac0,fac1,ind):
    if specparm < 0.125:
        p = fs - 1.0
        p4 = p ** 4
        fk0 = p4
        fk1 = 1.0 - p - 2.0 * p4
        fk2 = p + p4
        id00 = ind
        id01 = ind + 9
        id10 = ind + 1
        id11 = ind + 10
        id20 = ind + 2
        id21 = ind + 11
    elif specparm > 0.875:
        p = -fs
        p4 = p ** 4
        fk0 = p4
        fk1 = 1.0 - p - 2.0 * p4
        fk2 = p + p4

        id00 = ind + 1
        id01 = ind + 10
        id10 = ind
        id11 = ind + 9
        id20 = ind - 1
        id21 = ind + 8
    else:
        fk0 = 1.0 - fs
        fk1 = fs
        fk2 = 0.0
    
        id00 = ind
        id01 = ind + 9
        id10 = ind + 1
        id11 = ind + 10
        id20 = ind
        id21 = ind

    fac00 = fk0 * fac0
    fac10 = fk1 * fac0
    fac20 = fk2 * fac0
    fac01 = fk0 * fac1
    fac11 = fk1 * fac1
    fac21 = fk2 * fac1

    return fac00, fac10, fac20, fac01, fac11, fac21, id00, id01, id10, id11, id20, id21


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
def taugb04a(
    tau_major: Field[(DTYPE_FLT, (ng04,))],
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
    absb: Field[(DTYPE_FLT, (ng03, 1175))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    js: FIELD_INT,
    js1: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, nspb, oneminus, ng04

    with computation(PARALLEL), interval(1, None):
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
                tau_major[0,0,0][ig] = speccomb * (
                    fac000 * absa[0, 0, 0][ig, id000]
                    + fac010 * absa[0, 0, 0][ig, id010]
                    + fac100 * absa[0, 0, 0][ig, id100]
                    + fac110 * absa[0, 0, 0][ig, id110]
                    + fac200 * absa[0, 0, 0][ig, id200]
                    + fac210 * absa[0, 0, 0][ig, id210]
                ) + speccomb1 * (
                    fac001 * absa[0, 0, 0][ig, id001]
                    + fac011 * absa[0, 0, 0][ig, id011]
                    + fac101 * absa[0, 0, 0][ig, id101]
                    + fac111 * absa[0, 0, 0][ig, id111]
                    + fac201 * absa[0, 0, 0][ig, id201]
                    + fac211 * absa[0, 0, 0][ig, id211]
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

            id000 = ind0
            id010 = ind0 + 5
            id100 = ind0 + 1
            id110 = ind0 + 6
            id001 = ind1
            id011 = ind1 + 5
            id101 = ind1 + 1
            id111 = ind1 + 6

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
                    fac000 * absb[0, 0, 0][ig2, id000]
                    + fac010 * absb[0, 0, 0][ig2, id010]
                    + fac100 * absb[0, 0, 0][ig2, id100]
                    + fac110 * absb[0, 0, 0][ig2, id110]
                ) + speccomb1 * (
                    fac001 * absb[0, 0, 0][ig2, id001]
                    + fac011 * absb[0, 0, 0][ig2, id011]
                    + fac101 * absb[0, 0, 0][ig2, id101]
                    + fac111 * absb[0, 0, 0][ig2, id111]
                )

def taugb04b(
    tau_major: Field[(DTYPE_FLT, (ng04,))],
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    selfref: Field[(DTYPE_FLT, (ng04, 10))],
    forref: Field[(DTYPE_FLT, (ng04, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng04, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng04, 5))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
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

                taug[0, 0, 0][ns04 + ig] = tau_major + tauself + taufor

                fracs[0, 0, 0][ns04 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:

            speccomb_planck = colamt[0, 0, 0][2] + refrat_planck_b * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][2] / speccomb_planck
            specmult_planck = 4.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)
            jplp = jpl + 1

            for ig2 in range(ng04):

                taug[0, 0, 0][ns04 + ig2] = tau_major

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
        "nspa": nspa[4],
        "nspb": nspb[4],
        "ng05": ng05,
        "ns05": ns05,
        "oneminus": oneminus,
    },
)
def taugb05a(
    tau_major: Field[(DTYPE_FLT, (ng05,))],
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
    absa: Field[(DTYPE_FLT, (ng05, 585))],
    absb: Field[(DTYPE_FLT, (ng05, 1175))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    js: FIELD_INT,
    js1: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng05, ns05, oneminus

    with computation(PARALLEL), interval(1, None):
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
                p0 = fs - 1.0
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0 + 2
                id210 = ind0 + 11
            elif specparm > 0.875:
                p0 = -fs
                p40 = p0 ** 4
                fk00 = p40
                fk10 = 1.0 - p0 - 2.0 * p40
                fk20 = p0 + p40

                id000 = ind0 + 1
                id010 = ind0 + 10
                id100 = ind0
                id110 = ind0 + 9
                id200 = ind0 - 1
                id210 = ind0 + 8
            else:
                fk00 = 1.0 - fs
                fk10 = fs
                fk20 = 0.0

                id000 = ind0
                id010 = ind0 + 9
                id100 = ind0 + 1
                id110 = ind0 + 10
                id200 = ind0
                id210 = ind0

            fac000 = fk00 * fac00
            fac100 = fk10 * fac00
            fac200 = fk20 * fac00
            fac010 = fk00 * fac10
            fac110 = fk10 * fac10
            fac210 = fk20 * fac10

            if specparm1 < 0.125:
                p1 = fs1 - 1.0
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1 + 2
                id211 = ind1 + 11
            elif specparm1 > 0.875:
                p1 = -fs1
                p41 = p1 ** 4
                fk01 = p41
                fk11 = 1.0 - p1 - 2.0 * p41
                fk21 = p1 + p41

                id001 = ind1 + 1
                id011 = ind1 + 10
                id101 = ind1
                id111 = ind1 + 9
                id201 = ind1 - 1
                id211 = ind1 + 8
            else:
                fk01 = 1.0 - fs1
                fk11 = fs1
                fk21 = 0.0

                id001 = ind1
                id011 = ind1 + 9
                id101 = ind1 + 1
                id111 = ind1 + 10
                id201 = ind1
                id211 = ind1

            fac001 = fk01 * fac01
            fac101 = fk11 * fac01
            fac201 = fk21 * fac01
            fac011 = fk01 * fac11
            fac111 = fk11 * fac11
            fac211 = fk21 * fac11

            for ig in range(ng05):

                tau_major[0, 0, 0][ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
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

            id000 = ind0
            id010 = ind0 + 5
            id100 = ind0 + 1
            id110 = ind0 + 6
            id001 = ind1
            id011 = ind1 + 5
            id101 = ind1 + 1
            id111 = ind1 + 6

            fk00 = 1.0 - fs
            fk10 = fs

            fk01 = 1.0 - fs1
            fk11 = fs1

            fac000 = fk00 * fac00
            fac010 = fk00 * fac10
            fac100 = fk10 * fac00
            fac110 = fk10 * fac10

            fac001 = fk01 * fac01
            fac011 = fk01 * fac11
            fac101 = fk11 * fac01
            fac111 = fk11 * fac11

            for ig2 in range(ng05):
                tau_major[0, 0, 0][ig2] = (
                    speccomb
                    * (
                        fac000 * absb[0, 0, 0][ig2, id000]
                        + fac010 * absb[0, 0, 0][ig2, id010]
                        + fac100 * absb[0, 0, 0][ig2, id100]
                        + fac110 * absb[0, 0, 0][ig2, id110]
                    )
                    + speccomb1
                    * (
                        fac001 * absb[0, 0, 0][ig2, id001]
                        + fac011 * absb[0, 0, 0][ig2, id011]
                        + fac101 * absb[0, 0, 0][ig2, id101]
                        + fac111 * absb[0, 0, 0][ig2, id111]
                    )
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[4],
        "nspb": nspb[4],
        "ng05": ng05,
        "ns05": ns05,
        "oneminus": oneminus,
    },
)
def taugb05b(
    tau_major: Field[(DTYPE_FLT, (ng05,))],
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    wx: Field[type_maxxsec],
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
    selfref: Field[(DTYPE_FLT, (ng05, 10))],
    forref: Field[(DTYPE_FLT, (ng05, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng05, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng05, 5))],
    ka_mo3: Field[(DTYPE_FLT, (ng05, 9, 19))],
    ccl4: Field[(DTYPE_FLT, (ng05,))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    jmo3: FIELD_INT,
    jmo3p: FIELD_INT,
):
    from __externals__ import nspa, nspb, ng05, ns05, oneminus

    with computation(PARALLEL):
        with interval(...):
            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 4] / chi_mls[0, 0, 0][1, 4]
            )  # P = 142.5940 mb
            refrat_planck_b = (
                chi_mls[0, 0, 0][2, 42] / chi_mls[0, 0, 0][1, 42]
            )  # P = 95.58350 mb
            refrat_m_a = chi_mls[0, 0, 0][0, 6] / chi_mls[0, 0, 0][1, 6]

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb_mo3 = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][1]
            specparm_mo3 = colamt[0, 0, 0][0] / speccomb_mo3
            specmult_mo3 = 8.0 * min(specparm_mo3, oneminus)
            jmo3 = 1 + specmult_mo3 - 1
            fmo3 = mod(specmult_mo3, 1.0)

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
            jplp = jpl + 1
            jmo3p = jmo3 + 1

            for ig in range(ng05):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                o3m1 = ka_mo3[0, 0, 0][ig, jmo3, indm] + fmo3 * (
                    ka_mo3[0, 0, 0][ig, jmo3p, indm] - ka_mo3[0, 0, 0][ig, jmo3, indm]
                )
                o3m2 = ka_mo3[0, 0, 0][ig, jmo3, indmp] + fmo3 * (
                    ka_mo3[0, 0, 0][ig, jmo3p, indmp] - ka_mo3[0, 0, 0][ig, jmo3, indmp]
                )
                abso3 = o3m1 + minorfrac * (o3m2 - o3m1)

                taug[0, 0, 0][ns05 + ig] = (
                    tau_major[0,0,0][ig]
                    + tauself
                    + taufor
                    + abso3 * colamt[0, 0, 0][2]
                    + wx[0, 0, 0][0] * ccl4[0, 0, 0][ig]
                )

                fracs[0, 0, 0][ns05 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            speccomb_planck = colamt[0, 0, 0][2] + refrat_planck_b * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][2] / speccomb_planck
            specmult_planck = 4.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)
            jplp = jpl + 1



            for ig2 in range(ng05):
                taug[0, 0, 0][ns05 + ig2] = tau_major[0,0,0][ig2] + wx[0, 0, 0][0] * ccl4[0, 0, 0][ig2]

                fracs[0, 0, 0][ns05 + ig2] = fracrefb[0, 0, 0][ig2, jpl] + fpl * (
                    fracrefb[0, 0, 0][ig2, jplp] - fracrefb[0, 0, 0][ig2, jpl]
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[6],
        "nspb": nspb[6],
        "ng07": ng07,
        "ns07": ns07,
        "oneminus": oneminus,
    },
)
def taugb07a(
    tau_major: Field[(DTYPE_FLT, (ng07,))],
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
    indminor: FIELD_INT,
    absa: Field[(DTYPE_FLT, (ng07, 585))],
    absb: Field[(DTYPE_FLT, (ng07, 235))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    indm: FIELD_INT,
    js: FIELD_INT,
    js1: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng07, ns07, oneminus


    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][1, 0] * colamt[0, 0, 0][2]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][1, 1] * colamt[0, 0, 0][2]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            fac000,fac100,fac200,fac010,fac110,fac210,id000,id010,id100,id110,id200,id210 = set_ids(specparm,fs,fac00, fac10,ind0)
            
            fac001,fac101,fac201,fac011,fac111,fac211,id001,id011,id101,id111,id201,id211 = set_ids(specparm1,fs1,fac01, fac11,ind1)

            for ig in range(ng07):

                tau_major[0, 0, 0][ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                )

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            indm = indminor - 1
            indmp = indm + 1
            ind0p = ind0 + 1
            ind1p = ind1 + 1

            for ig2 in range(ng07):
                tau_major[0, 0, 0][ig2] = (
                    colamt[0, 0, 0][2]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                )



@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[6],
        "nspb": nspb[6],
        "ng07": ng07,
        "ns07": ns07,
        "oneminus": oneminus,
    },
)
def taugb07b(
    tau_major: Field[(DTYPE_FLT, (ng07,))],
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    jp: FIELD_INT,
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
    selfref: Field[(DTYPE_FLT, (ng07, 10))],
    forref: Field[(DTYPE_FLT, (ng07, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng07, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng07,))],
    ka_mco2: Field[(DTYPE_FLT, (ng07, 9, 19))],
    kb_mco2: Field[(DTYPE_FLT, (ng07, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    jmco2: FIELD_INT,
    jmco2p: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    ratco2: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng07, ns07, oneminus

    with computation(PARALLEL):
        with interval(...):
            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][2, 2]
            )  # P = 706.2620 mb
            refrat_m_a = (
                chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][2, 2]
            )  # P = 706.2720 mb

    with computation(PARALLEL), interval(1, None):
        if laytrop:

            speccomb_mco2 = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][2]
            specparm_mco2 = colamt[0, 0, 0][0] / speccomb_mco2
            specmult_mco2 = 8.0 * min(specparm_mco2, oneminus)
            jmco2 = 1 + specmult_mco2 - 1
            fmco2 = mod(specmult_mco2, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][2]
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
            jplp = jpl + 1
            jmco2p = jmco2 + 1

            temp = coldry * chi_mls[0, 0, 0][1, jp]
            ratco2 = colamt[0, 0, 0][1] / temp
            if ratco2 > 3.0:
                adjfac = 3.0 + (ratco2 - 3.0) ** 0.79
                adjcolco2 = adjfac * temp
            else:
                adjcolco2 = colamt[0, 0, 0][1]

            for ig in range(ng07):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                co2m1 = ka_mco2[0, 0, 0][ig, jmco2, indm] + fmco2 * (
                    ka_mco2[0, 0, 0][ig, jmco2p, indm]
                    - ka_mco2[0, 0, 0][ig, jmco2, indm]
                )
                co2m2 = ka_mco2[0, 0, 0][ig, jmco2, indmp] + fmco2 * (
                    ka_mco2[0, 0, 0][ig, jmco2p, indmp]
                    - ka_mco2[0, 0, 0][ig, jmco2, indmp]
                )
                absco2 = co2m1 + minorfrac * (co2m2 - co2m1)

                taug[0, 0, 0][ns07 + ig] = (
                   tau_major[0,0,0][ig]
                    + tauself
                    + taufor
                    + adjcolco2 * absco2
                )

                fracs[0, 0, 0][ns07 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            temp = coldry * chi_mls[0, 0, 0][1, jp]
            ratco2 = colamt[0, 0, 0][1] / temp
            if ratco2 > 3.0:
                adjfac = 2.0 + (ratco2 - 2.0) ** 0.79
                adjcolco2 = adjfac * temp
            else:
                adjcolco2 = colamt[0, 0, 0][1]

            indm = indminor - 1
            indmp = indm + 1

            for ig2 in range(ng07):
                absco2 = kb_mco2[0, 0, 0][ig2, indm] + minorfrac * (
                    kb_mco2[0, 0, 0][ig2, indmp] - kb_mco2[0, 0, 0][ig2, indm]
                )

                taug[0, 0, 0][ns07 + ig2] = (
                    tau_major[0,0,0][ig2]
                    + adjcolco2 * absco2
                )

                fracs[0, 0, 0][ns07 + ig2] = fracrefb[0, 0, 0][ig2]

            taug[0, 0, 0][ns07 + 5] = taug[0, 0, 0][ns07 + 5] * 0.92
            taug[0, 0, 0][ns07 + 6] = taug[0, 0, 0][ns07 + 6] * 0.88
            taug[0, 0, 0][ns07 + 7] = taug[0, 0, 0][ns07 + 7] * 1.07
            taug[0, 0, 0][ns07 + 8] = taug[0, 0, 0][ns07 + 8] * 1.1
            taug[0, 0, 0][ns07 + 9] = taug[0, 0, 0][ns07 + 9] * 0.99
            taug[0, 0, 0][ns07 + 10] = taug[0, 0, 0][ns07 + 10] * 0.855


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[8],
        "nspb": nspb[8],
        "ng09": ng09,
        "ns09": ns09,
        "oneminus": oneminus,
    },
)
def taugb09a(
    tau_major: Field[(DTYPE_FLT, (ng09,))],
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
    absa: Field[(DTYPE_FLT, (ng09, 585))],
    absb: Field[(DTYPE_FLT, (ng09, 235))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    js: FIELD_INT,
    js1: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng09, ns09, oneminus

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][3, 0] * colamt[0, 0, 0][4]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][3, 1] * colamt[0, 0, 0][4]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            fac000,fac100,fac200,fac010,fac110,fac210,id000,id010,id100,id110,id200,id210 = set_ids(specparm, fs, fac00, fac10,ind0)
            fac001,fac101,fac201,fac011,fac111,fac211,id001,id011,id101,id111,id201,id211 = set_ids(specparm1,fs1,fac01, fac11,ind1)

            for ig in range(ng09):
                tau_major[0, 0, 0][ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                )
        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            ind0p = ind0 + 1
            ind1p = ind1 + 1

            for ig2 in range(ng09):
                tau_major[0, 0, 0][ig2] = (
                    colamt[0, 0, 0][4]
                    * (
                        fac00 * absb[0, 0, 0][ig2, ind0]
                        + fac10 * absb[0, 0, 0][ig2, ind0p]
                        + fac01 * absb[0, 0, 0][ig2, ind1]
                        + fac11 * absb[0, 0, 0][ig2, ind1p]
                    )
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[8],
        "nspb": nspb[8],
        "ng09": ng09,
        "ns09": ns09,
        "oneminus": oneminus,
    },
)
def taugb09b(
    tau_major: Field[(DTYPE_FLT, (ng09,))],
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    jp: FIELD_INT,
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
    selfref: Field[(DTYPE_FLT, (ng09, 10))],
    forref: Field[(DTYPE_FLT, (ng09, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng09, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng09,))],
    ka_mn2o: Field[(DTYPE_FLT, (ng09, 9, 19))],
    kb_mn2o: Field[(DTYPE_FLT, (ng09, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    jmn2o: FIELD_INT,
    jmn2op: FIELD_INT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    ratn2o: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng09, ns09, oneminus

    with computation(PARALLEL):
        with interval(...):
            #  --- ...  calculate reference ratio to be used in calculation of Planck
            #           fraction in lower/upper atmosphere.

            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 8] / chi_mls[0, 0, 0][5, 8]
            )  # P = 212 mb
            refrat_m_a = (
                chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][5, 2]
            )  # P = 706.272 mb

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb_mn2o = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][4]
            specparm_mn2o = colamt[0, 0, 0][0] / speccomb_mn2o
            specmult_mn2o = 8.0 * min(specparm_mn2o, oneminus)
            jmn2o = 1 + specmult_mn2o - 1
            fmn2o = mod(specmult_mn2o, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][4]
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
            jplp = jpl + 1
            jmn2op = jmn2o + 1

            temp = coldry * chi_mls[0, 0, 0][3, jp]
            ratn2o = colamt[0, 0, 0][3] / temp
            if ratn2o > 1.5:
                adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
                adjcoln2o = adjfac * temp
            else:
                adjcoln2o = colamt[0, 0, 0][3]

            for ig in range(ng09):
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

                taug[0, 0, 0][ns09 + ig] = (
                    tau_major[0,0,0][ig]
                    + tauself
                    + taufor
                    + adjcoln2o * absn2o
                )

                fracs[0, 0, 0][ns09 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:

            indm = indminor - 1
            indmp = indm + 1

            temp = coldry * chi_mls[0, 0, 0][3, jp]
            ratn2o = colamt[0, 0, 0][3] / temp
            if ratn2o > 1.5:
                adjfac = 0.5 + (ratn2o - 0.5) ** 0.65
                adjcoln2o = adjfac * temp
            else:
                adjcoln2o = colamt[0, 0, 0][3]

            for ig2 in range(ng09):
                absn2o = kb_mn2o[0, 0, 0][ig2, indm] + minorfrac * (
                    kb_mn2o[0, 0, 0][ig2, indmp] - kb_mn2o[0, 0, 0][ig2, indm]
                )

                taug[0, 0, 0][ns09 + ig2] = (
                    tau_major[0,0,0][ig2]
                    + adjcoln2o * absn2o
                )

                fracs[0, 0, 0][ns09 + ig2] = fracrefb[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[12],
        "nspb": nspb[12],
        "ng13": ng13,
        "ns13": ns13,
        "oneminus": oneminus,
    },
)
def taugb13a(
    tau_major: Field[(DTYPE_FLT, (ng13,))],
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
    absa: Field[(DTYPE_FLT, (ng13, 585))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    js: FIELD_INT,
    js1: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, ng13, ns13, oneminus

    with computation(PARALLEL):
        with interval(...):
            #  --- ...  calculate reference ratio to be used in calculation of Planck
            #           fraction in lower/upper atmosphere.

            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 4] / chi_mls[0, 0, 0][3, 4]
            )  # P = 473.420 mb (Level 5)
            refrat_m_a = (
                chi_mls[0, 0, 0][0, 0] / chi_mls[0, 0, 0][3, 0]
            )  # P = 1053. (Level 1)
            refrat_m_a3 = (
                chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][3, 2]
            )  # P = 706. (Level 3)

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][2, 0] * colamt[0, 0, 0][3]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][2, 1] * colamt[0, 0, 0][3]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            fac000,fac100,fac200,fac010,fac110,fac210,id000,id010,id100,id110,id200,id210 = set_ids(specparm, fs, fac00, fac10,ind0)
            fac001,fac101,fac201,fac011,fac111,fac211,id001,id011,id101,id111,id201,id211 = set_ids(specparm1,fs1,fac01, fac11,ind1)

            for ig in range(ng13):
                tau_major[0, 0, 0][ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[12],
        "nspb": nspb[12],
        "ng13": ng13,
        "ns13": ns13,
        "oneminus": oneminus,
    },
)
def taugb13b(
    tau_major: Field[(DTYPE_FLT, (ng13,))],
    laytrop: FIELD_BOOL,
    coldry: FIELD_FLT,
    colamt: Field[type_maxgas],
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    indminor: FIELD_INT,
    minorfrac: FIELD_FLT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    selfref: Field[(DTYPE_FLT, (ng13, 10))],
    forref: Field[(DTYPE_FLT, (ng13, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng13, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng13,))],
    ka_mco: Field[(DTYPE_FLT, (ng13, 9, 19))],
    ka_mco2: Field[(DTYPE_FLT, (ng13, 9, 19))],
    kb_mo3: Field[(DTYPE_FLT, (ng13, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    jmco: FIELD_INT,
    jmcop: FIELD_INT,
    jmco2: FIELD_INT,
    jmco2p: FIELD_INT,
    ratco2: FIELD_FLT,
):
    from __externals__ import nspa, ng13, ns13, oneminus

    with computation(PARALLEL):
        with interval(...):
            #  --- ...  calculate reference ratio to be used in calculation of Planck
            #           fraction in lower/upper atmosphere.

            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 4] / chi_mls[0, 0, 0][3, 4]
            )  # P = 473.420 mb (Level 5)
            refrat_m_a = (
                chi_mls[0, 0, 0][0, 0] / chi_mls[0, 0, 0][3, 0]
            )  # P = 1053. (Level 1)
            refrat_m_a3 = (
                chi_mls[0, 0, 0][0, 2] / chi_mls[0, 0, 0][3, 2]
            )  # P = 706. (Level 3)

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb_mco2 = colamt[0, 0, 0][0] + refrat_m_a * colamt[0, 0, 0][3]
            specparm_mco2 = colamt[0, 0, 0][0] / speccomb_mco2
            specmult_mco2 = 8.0 * min(specparm_mco2, oneminus)
            jmco2 = 1 + specmult_mco2 - 1
            fmco2 = mod(specmult_mco2, 1.0)

            speccomb_mco = colamt[0, 0, 0][0] + refrat_m_a3 * colamt[0, 0, 0][3]
            specparm_mco = colamt[0, 0, 0][0] / speccomb_mco
            specmult_mco = 8.0 * min(specparm_mco, oneminus)
            jmco = 1 + specmult_mco - 1
            fmco = mod(specmult_mco, 1.0)

            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][3]
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
            jplp = jpl + 1
            jmco2p = jmco2 + 1
            jmcop = jmco + 1

            temp = coldry * 3.55e-4
            ratco2 = colamt[0, 0, 0][1] / temp
            if ratco2 > 3.0:
                adjfac = 2.0 + (ratco2 - 2.0) ** 0.68
                adjcolco2 = adjfac * temp
            else:
                adjcolco2 = colamt[0, 0, 0][1]

            for ig in range(ng13):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                co2m1 = ka_mco2[0, 0, 0][ig, jmco2, indm] + fmco2 * (
                    ka_mco2[0, 0, 0][ig, jmco2p, indm]
                    - ka_mco2[0, 0, 0][ig, jmco2, indm]
                )
                co2m2 = ka_mco2[0, 0, 0][ig, jmco2, indmp] + fmco2 * (
                    ka_mco2[0, 0, 0][ig, jmco2p, indmp]
                    - ka_mco2[0, 0, 0][ig, jmco2, indmp]
                )
                absco2 = co2m1 + minorfrac * (co2m2 - co2m1)
                com1 = ka_mco[0, 0, 0][ig, jmco, indm] + fmco * (
                    ka_mco[0, 0, 0][ig, jmcop, indm] - ka_mco[0, 0, 0][ig, jmco, indm]
                )
                com2 = ka_mco[0, 0, 0][ig, jmco, indmp] + fmco * (
                    ka_mco[0, 0, 0][ig, jmcop, indmp] - ka_mco[0, 0, 0][ig, jmco, indmp]
                )
                absco = com1 + minorfrac * (com2 - com1)

                taug[0, 0, 0][ns13 + ig] = (
                    tau_major[0,0,0][ig]
                    + tauself
                    + taufor
                    + adjcolco2 * absco2
                    + colamt[0, 0, 0][6] * absco
                )

                fracs[0, 0, 0][ns13 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            indm = indminor - 1
            indmp = indm + 1

            for ig2 in range(ng13):
                abso3 = kb_mo3[0, 0, 0][ig2, indm] + minorfrac * (
                    kb_mo3[0, 0, 0][ig2, indmp] - kb_mo3[0, 0, 0][ig2, indm]
                )

                taug[0, 0, 0][ns13 + ig2] = colamt[0, 0, 0][2] * abso3

                fracs[0, 0, 0][ns13 + ig2] = fracrefb[0, 0, 0][ig2]


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[14],
        "nspb": nspb[14],
        "ng15": ng15,
        "ns15": ns15,
        "oneminus": oneminus,
    },
)
def taugb15a(
    tau_major: Field[(DTYPE_FLT, (ng15,))],
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
    absa: Field[(DTYPE_FLT, (ng15, 585))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    js: FIELD_INT,
    js1: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, ng15, ns15, oneminus

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][3] + rfrate[0, 0, 0][4, 0] * colamt[0, 0, 0][1]
            specparm = colamt[0, 0, 0][3] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][3] + rfrate[0, 0, 0][4, 1] * colamt[0, 0, 0][1]
            specparm1 = colamt[0, 0, 0][3] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            fac000,fac100,fac200,fac010,fac110,fac210,id000,id010,id100,id110,id200,id210 = set_ids(specparm, fs, fac00, fac10,ind0)
            fac001,fac101,fac201,fac011,fac111,fac211,id001,id011,id101,id111,id201,id211 = set_ids(specparm1,fs1,fac01, fac11,ind1)


            for ig in range(ng15):
                tau_major[0, 0, 0][ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                )


@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[14],
        "nspb": nspb[14],
        "ng15": ng15,
        "ns15": ns15,
        "oneminus": oneminus,
    },
)
def taugb15b(
    tau_major: Field[(DTYPE_FLT, (ng15,))],
    laytrop: FIELD_BOOL,
    colamt: Field[type_maxgas],
    colbrd: FIELD_FLT,
    selffac: FIELD_FLT,
    selffrac: FIELD_FLT,
    indself: FIELD_INT,
    forfac: FIELD_FLT,
    forfrac: FIELD_FLT,
    indfor: FIELD_INT,
    indminor: FIELD_INT,
    minorfrac: FIELD_FLT,
    scaleminor: FIELD_FLT,
    fracs: Field[type_ngptlw],
    taug: Field[type_ngptlw],
    selfref: Field[(DTYPE_FLT, (ng15, 10))],
    forref: Field[(DTYPE_FLT, (ng15, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng15, 9))],
    ka_mn2: Field[(DTYPE_FLT, (ng15, 9, 19))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    indm: FIELD_INT,
    indmp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    taun2: FIELD_FLT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    jmn2: FIELD_INT,
    jmn2p: FIELD_INT,
    fpl: FIELD_FLT,
):
    from __externals__ import nspa, ng15, ns15, oneminus

    with computation(PARALLEL):
        with interval(...):

            #  --- ...  calculate reference ratio to be used in calculation of Planck
            #           fraction in lower atmosphere.

            refrat_planck_a = (
                chi_mls[0, 0, 0][3, 0] / chi_mls[0, 0, 0][1, 0]
            )  # P = 1053. mb (Level 1)
            refrat_m_a = chi_mls[0, 0, 0][3, 0] / chi_mls[0, 0, 0][1, 0]  # P = 1053. mb

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb_mn2 = colamt[0, 0, 0][3] + refrat_m_a * colamt[0, 0, 0][1]
            specparm_mn2 = colamt[0, 0, 0][3] / speccomb_mn2
            specmult_mn2 = 8.0 * min(specparm_mn2, oneminus)
            jmn2 = 1 + specmult_mn2 - 1
            fmn2 = mod(specmult_mn2, 1.0)

            speccomb_planck = colamt[0, 0, 0][3] + refrat_planck_a * colamt[0, 0, 0][1]
            specparm_planck = colamt[0, 0, 0][3] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            scalen2 = colbrd * scaleminor

            inds = indself - 1
            indf = indfor - 1
            indm = indminor - 1
            indsp = inds + 1
            indfp = indf + 1
            indmp = indm + 1
            jplp = jpl + 1
            jmn2p = jmn2 + 1

            for ig in range(ng15):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )
                n2m1 = ka_mn2[0, 0, 0][ig, jmn2, indm] + fmn2 * (
                    ka_mn2[0, 0, 0][ig, jmn2p, indm] - ka_mn2[0, 0, 0][ig, jmn2, indm]
                )
                n2m2 = ka_mn2[0, 0, 0][ig, jmn2, indmp] + fmn2 * (
                    ka_mn2[0, 0, 0][ig, jmn2p, indmp] - ka_mn2[0, 0, 0][ig, jmn2, indmp]
                )
                taun2 = scalen2 * (n2m1 + minorfrac * (n2m2 - n2m1))

                taug[0, 0, 0][ns15 + ig] = (
                    tau_major[0,0,0][ig]
                    + tauself
                    + taufor
                    + taun2
                )

                fracs[0, 0, 0][ns15 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            for ig2 in range(ng15):
                taug[0, 0, 0][ns15 + ig2] = 0.0
                fracs[0, 0, 0][ns15 + ig2] = 0.0




@stencil(
    backend=backend,
    rebuild=rebuild,
    externals={
        "nspa": nspa[15],
        "nspb": nspb[15],
        "ng16": ng16,
        "ns16": ns16,
        "oneminus": oneminus,
    },
)

def taugb16a(
    tau_major: Field[(DTYPE_FLT, (ng16,))],
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
    absa: Field[(DTYPE_FLT, (ng16, 585))],
    ind0: FIELD_INT,
    ind1: FIELD_INT,
    js: FIELD_INT,
    js1: FIELD_INT,
    id000: FIELD_INT,
    id010: FIELD_INT,
    id100: FIELD_INT,
    id110: FIELD_INT,
    id200: FIELD_INT,
    id210: FIELD_INT,
    id001: FIELD_INT,
    id011: FIELD_INT,
    id101: FIELD_INT,
    id111: FIELD_INT,
    id201: FIELD_INT,
    id211: FIELD_INT,
    # temporaries below here only necessary to work around a bug in gt4py
    # hopefully can be removed later
    speccomb: FIELD_FLT,
    speccomb1: FIELD_FLT,
    fac000: FIELD_FLT,
    fac100: FIELD_FLT,
    fac200: FIELD_FLT,
    fac010: FIELD_FLT,
    fac110: FIELD_FLT,
    fac210: FIELD_FLT,
    fac001: FIELD_FLT,
    fac101: FIELD_FLT,
    fac201: FIELD_FLT,
    fac011: FIELD_FLT,
    fac111: FIELD_FLT,
    fac211: FIELD_FLT,
    specparm: FIELD_FLT,
    specparm1: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng16, ns16, oneminus

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb = colamt[0, 0, 0][0] + rfrate[0, 0, 0][3, 0] * colamt[0, 0, 0][4]
            specparm = colamt[0, 0, 0][0] / speccomb
            specmult = 8.0 * min(specparm, oneminus)
            js = 1 + specmult
            fs = mod(specmult, 1.0)
            ind0 = ((jp - 1) * 5 + (jt - 1)) * nspa + js - 1

            speccomb1 = colamt[0, 0, 0][0] + rfrate[0, 0, 0][3, 1] * colamt[0, 0, 0][4]
            specparm1 = colamt[0, 0, 0][0] / speccomb1
            specmult1 = 8.0 * min(specparm1, oneminus)
            js1 = 1 + specmult1
            fs1 = mod(specmult1, 1.0)
            ind1 = (jp * 5 + (jt1 - 1)) * nspa + js1 - 1

            fac000,fac100,fac200,fac010,fac110,fac210,id000,id010,id100,id110,id200,id210 = set_ids(specparm, fs, fac00, fac10,ind0)
            fac001,fac101,fac201,fac011,fac111,fac211,id001,id011,id101,id111,id201,id211 = set_ids(specparm1,fs1,fac01, fac11,ind1)

            for ig in range(ng16):

                tau_major[0, 0, 0][ig] = (
                    speccomb
                    * (
                        fac000 * absa[0, 0, 0][ig, id000]
                        + fac010 * absa[0, 0, 0][ig, id010]
                        + fac100 * absa[0, 0, 0][ig, id100]
                        + fac110 * absa[0, 0, 0][ig, id110]
                        + fac200 * absa[0, 0, 0][ig, id200]
                        + fac210 * absa[0, 0, 0][ig, id210]
                    )
                    + speccomb1
                    * (
                        fac001 * absa[0, 0, 0][ig, id001]
                        + fac011 * absa[0, 0, 0][ig, id011]
                        + fac101 * absa[0, 0, 0][ig, id101]
                        + fac111 * absa[0, 0, 0][ig, id111]
                        + fac201 * absa[0, 0, 0][ig, id201]
                        + fac211 * absa[0, 0, 0][ig, id211]
                    )
                )



def taugb16b(
    tau_major: Field[(DTYPE_FLT, (ng16,))],
    laytrop: FIELD_BOOL,
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
    absb: Field[(DTYPE_FLT, (ng16, 235))],
    selfref: Field[(DTYPE_FLT, (ng16, 10))],
    forref: Field[(DTYPE_FLT, (ng16, 4))],
    fracrefa: Field[(DTYPE_FLT, (ng16, 9))],
    fracrefb: Field[(DTYPE_FLT, (ng16,))],
    chi_mls: Field[(DTYPE_FLT, (7, 59))],
    ind0: FIELD_INT,
    ind0p: FIELD_INT,
    ind1: FIELD_INT,
    ind1p: FIELD_INT,
    inds: FIELD_INT,
    indsp: FIELD_INT,
    indf: FIELD_INT,
    indfp: FIELD_INT,
    tauself: FIELD_FLT,
    taufor: FIELD_FLT,
    jpl: FIELD_INT,
    jplp: FIELD_INT,
    fpl: FIELD_FLT,
):
    from __externals__ import nspa, nspb, ng16, ns16, oneminus

    with computation(PARALLEL):
        with interval(...):
            refrat_planck_a = (
                chi_mls[0, 0, 0][0, 5] / chi_mls[0, 0, 0][5, 5]
            )  # P = 387. mb (Level 6)

    with computation(PARALLEL), interval(1, None):
        if laytrop:
            speccomb_planck = colamt[0, 0, 0][0] + refrat_planck_a * colamt[0, 0, 0][4]
            specparm_planck = colamt[0, 0, 0][0] / speccomb_planck
            specmult_planck = 8.0 * min(specparm_planck, oneminus)
            jpl = 1 + specmult_planck - 1
            fpl = mod(specmult_planck, 1.0)

            inds = indself - 1
            indf = indfor - 1
            indsp = inds + 1
            indfp = indf + 1
            jplp = jpl + 1

            for ig in range(ng16):
                tauself = selffac * (
                    selfref[0, 0, 0][ig, inds]
                    + selffrac
                    * (selfref[0, 0, 0][ig, indsp] - selfref[0, 0, 0][ig, inds])
                )
                taufor = forfac * (
                    forref[0, 0, 0][ig, indf]
                    + forfrac * (forref[0, 0, 0][ig, indfp] - forref[0, 0, 0][ig, indf])
                )

                taug[0, 0, 0][ns16 + ig] = (
                    tau_major[0,0,0][ig]
                    + tauself
                    + taufor
                )

                fracs[0, 0, 0][ns16 + ig] = fracrefa[0, 0, 0][ig, jpl] + fpl * (
                    fracrefa[0, 0, 0][ig, jplp] - fracrefa[0, 0, 0][ig, jpl]
                )

        else:
            ind0 = ((jp - 13) * 5 + (jt - 1)) * nspb
            ind1 = ((jp - 12) * 5 + (jt1 - 1)) * nspb

            ind0p = ind0 + 1
            ind1p = ind1 + 1

            for ig2 in range(ng16):
                taug[0, 0, 0][ns16 + ig2] = colamt[0, 0, 0][4] * (
                    fac00 * absb[0, 0, 0][ig2, ind0]
                    + fac10 * absb[0, 0, 0][ig2, ind0p]
                    + fac01 * absb[0, 0, 0][ig2, ind1]
                    + fac11 * absb[0, 0, 0][ig2, ind1p]
                )

                fracs[0, 0, 0][ns16 + ig2] = fracrefb[0, 0, 0][ig2]


rec_6 = 0.166667
tblint = ntbl
flxfac = wtdiff * fluxfac
lhlw0 = True



@stencil(
    backend,
    rebuild=rebuild,
    externals={
        "rec_6": rec_6,
        "bpade": bpade,
        "tblint": tblint,
        "eps": eps,
        "flxfac": flxfac,
        "heatfac": heatfac,
        "lhlw0": lhlw0,
    },
)
def rtrnmc_a(
    semiss: Field[gtscript.IJ, type_nbands],
    secdif: Field[gtscript.IJ, type_nbands],
    taucld: Field[type_nbands],
    fracs: Field[type_ngptlw],
    tautot: Field[type_ngptlw],
    cldfmc: Field[type_ngptlw],
    pklay: Field[type_nbands],
    pklev: Field[type_nbands],
    exp_tbl: Field[type_ntbmx],
    tau_tbl: Field[type_ntbmx],
    tfn_tbl: Field[type_ntbmx],
    NGB: Field[gtscript.IJ, (np.int32, (140,))],
    clrdrad: Field[type_nbands],
    totdrad: Field[type_nbands],
    gassrcu: Field[type_ngptlw],
    totsrcu: Field[type_ngptlw],
    trngas: Field[type_ngptlw],
    efclrfr: Field[type_ngptlw],
    totsrcd: Field[type_ngptlw],
    gassrcd: Field[type_ngptlw],
    tblind: Field[type_ngptlw],
    odepth: Field[type_ngptlw],
    odtot: Field[type_ngptlw],
    odcld: Field[type_ngptlw],
    atrtot: Field[type_ngptlw],
    atrgas: Field[type_ngptlw],
    reflct: Field[type_ngptlw],
    totfac: Field[type_ngptlw],
    gasfac: Field[type_ngptlw],
    plfrac: Field[type_ngptlw],
    blay: Field[type_ngptlw],
    bbdgas: Field[type_ngptlw],
    bbdtot: Field[type_ngptlw],
    bbugas: Field[type_ngptlw],
    bbutot: Field[type_ngptlw],
    dplnku: Field[type_ngptlw],
    dplnkd: Field[type_ngptlw],
    radtotd: Field[type_ngptlw],
    radclrd: Field[type_ngptlw],
    clfm: Field[type_ngptlw],
    trng: Field[type_ngptlw],
    itgas: Field[(np.int32, (ngptlw,))],
    ittot: Field[(np.int32, (ngptlw,))],
    ib: FIELD_2DINT,
):
    from __externals__ import rec_6, bpade, tblint, eps, flxfac, heatfac, lhlw0

    # Downward radiative transfer loop.
    # - Clear sky, gases contribution
    # - Total sky, gases+clouds contribution
    # - Cloudy layer
    # - Total sky radiance
    # - Clear sky radiance
    with computation(FORWARD), interval(-2, -1):
        for ig0 in range(ngptlw):
            ib = NGB[0, 0][ig0] - 1

            # clear sky, gases contribution
            odepth[0, 0, 0][ig0] = max(0.0, secdif[0, 0][ib] * tautot[0, 0, 1][ig0])
            if odepth[0, 0, 0][ig0] <= 0.06:
                atrgas[0, 0, 0][ig0] = (
                    odepth[0, 0, 0][ig0]
                    - 0.5 * odepth[0, 0, 0][ig0] * odepth[0, 0, 0][ig0]
                )
                trng[0, 0, 0][ig0] = 1.0 - atrgas[0, 0, 0][ig0]
                gasfac[0, 0, 0][ig0] = rec_6 * odepth[0, 0, 0][ig0]
            else:
                tblind[0, 0, 0][ig0] = odepth[0, 0, 0][ig0] / (
                    bpade + odepth[0, 0, 0][ig0]
                )
                # Currently itgas needs to be a storage, and can't be a local temporary.
                itgas[0, 0, 0][ig0] = tblint * tblind[0, 0, 0][ig0] + 0.5
                trng[0, 0, 0][ig0] = exp_tbl[0, 0, 0][itgas[0, 0, 0][ig0]]
                atrgas[0, 0, 0][ig0] = 1.0 - trng[0, 0, 0][ig0]
                gasfac[0, 0, 0][ig0] = tfn_tbl[0, 0, 0][itgas[0, 0, 0][ig0]]
                odepth[0, 0, 0][ig0] = tau_tbl[0, 0, 0][itgas[0, 0, 0][ig0]]

            plfrac[0, 0, 0][ig0] = fracs[0, 0, 1][ig0]
            blay[0, 0, 0][ig0] = pklay[0, 0, 1][ib]

            dplnku[0, 0, 0][ig0] = pklev[0, 0, 1][ib] - blay[0, 0, 0][ig0]
            dplnkd[0, 0, 0][ig0] = pklev[0, 0, 0][ib] - blay[0, 0, 0][ig0]
            bbdgas[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (
                blay[0, 0, 0][ig0] + dplnkd[0, 0, 0][ig0] * gasfac[0, 0, 0][ig0]
            )
            bbugas[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (
                blay[0, 0, 0][ig0] + dplnku[0, 0, 0][ig0] * gasfac[0, 0, 0][ig0]
            )
            gassrcd[0, 0, 0][ig0] = bbdgas[0, 0, 0][ig0] * atrgas[0, 0, 0][ig0]
            gassrcu[0, 0, 0][ig0] = bbugas[0, 0, 0][ig0] * atrgas[0, 0, 0][ig0]
            trngas[0, 0, 0][ig0] = trng[0, 0, 0][ig0]

            # total sky, gases+clouds contribution
            clfm[0, 0, 0][ig0] = cldfmc[0, 0, 0][ig0]
            if clfm[0, 0, 0][ig0] >= eps:
                # cloudy layer
                odcld[0, 0, 0][ig0] = secdif[0, 0][ib] * taucld[0, 0, 1][ib]
                efclrfr[0, 0, 0][ig0] = (
                    1.0 - (1.0 - exp(-odcld[0, 0, 0][ig0])) * clfm[0, 0, 0][ig0]
                )
                odtot[0, 0, 0][ig0] = odepth[0, 0, 0][ig0] + odcld[0, 0, 0][ig0]
                if odtot[0, 0, 0][ig0] < 0.06:
                    totfac[0, 0, 0][ig0] = rec_6 * odtot[0, 0, 0][ig0]
                    atrtot[0, 0, 0][ig0] = (
                        odtot[0, 0, 0][ig0]
                        - 0.5 * odtot[0, 0, 0][ig0] * odtot[0, 0, 0][ig0]
                    )
                else:
                    tblind[0, 0, 0][ig0] = odtot[0, 0, 0][ig0] / (
                        bpade + odtot[0, 0, 0][ig0]
                    )
                    ittot[0, 0, 0][ig0] = tblint * tblind[0, 0, 0][ig0] + 0.5
                    totfac[0, 0, 0][ig0] = tfn_tbl[0, 0, 0][ittot[0, 0, 0][ig0]]
                    atrtot[0, 0, 0][ig0] = 1.0 - exp_tbl[0, 0, 0][ittot[0, 0, 0][ig0]]

                bbdtot[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (
                    blay[0, 0, 0][ig0] + dplnkd[0, 0, 0][ig0] * totfac[0, 0, 0][ig0]
                )
                bbutot[0, 0, 0][ig0] = plfrac[0, 0, 0][ig0] * (
                    blay[0, 0, 0][ig0] + dplnku[0, 0, 0][ig0] * totfac[0, 0, 0][ig0]
                )
                totsrcd[0, 0, 0][ig0] = bbdtot[0, 0, 0][ig0] * atrtot[0, 0, 0][ig0]
                totsrcu[0, 0, 0][ig0] = bbutot[0, 0, 0][ig0] * atrtot[0, 0, 0][ig0]

                # total sky radiance
                radtotd[0, 0, 0][ig0] = (
                    radtotd[0, 0, 0][ig0] * trng[0, 0, 0][ig0] * efclrfr[0, 0, 0][ig0]
                    + gassrcd[0, 0, 0][ig0]
                    + clfm[0, 0, 0][ig0]
                    * (totsrcd[0, 0, 0][ig0] - gassrcd[0, 0, 0][ig0])
                )
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig0]

                # clear sky radiance
                radclrd[0, 0, 0][ig0] = (
                    radclrd[0, 0, 0][ig0] * trng[0, 0, 0][ig0] + gassrcd[0, 0, 0][ig0]
                )
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig0]
            else:
                # clear layer

                # total sky radiance
                radtotd[0, 0, 0][ig0] = (
                    radtotd[0, 0, 0][ig0] * trng[0, 0, 0][ig0] + gassrcd[0, 0, 0][ig0]
                )
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig0]

                # clear sky radiance
                radclrd[0, 0, 0][ig0] = (
                    radclrd[0, 0, 0][ig0] * trng[0, 0, 0][ig0] + gassrcd[0, 0, 0][ig0]
                )
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig0]

            reflct[0, 0, 0][ig0] = 1.0 - semiss[0, 0][ib]

    with computation(BACKWARD), interval(0, -2):
        for ig in range(ngptlw):
            ib = NGB[0, 0][ig] - 1

            # clear sky, gases contribution
            odepth[0, 0, 0][ig] = max(0.0, secdif[0, 0][ib] * tautot[0, 0, 1][ig])
            if odepth[0, 0, 0][ig] <= 0.06:
                atrgas[0, 0, 0][ig] = (
                    odepth[0, 0, 0][ig]
                    - 0.5 * odepth[0, 0, 0][ig] * odepth[0, 0, 0][ig]
                )
                trng[0, 0, 0][ig] = 1.0 - atrgas[0, 0, 0][ig]
                gasfac[0, 0, 0][ig] = rec_6 * odepth[0, 0, 0][ig]
            else:
                tblind[0, 0, 0][ig] = odepth[0, 0, 0][ig] / (
                    bpade + odepth[0, 0, 0][ig]
                )
                itgas[0, 0, 0][ig] = tblint * tblind[0, 0, 0][ig] + 0.5
                trng[0, 0, 0][ig] = exp_tbl[0, 0, 0][itgas[0, 0, 0][ig]]
                atrgas[0, 0, 0][ig] = 1.0 - trng[0, 0, 0][ig]
                gasfac[0, 0, 0][ig] = tfn_tbl[0, 0, 0][itgas[0, 0, 0][ig]]
                odepth[0, 0, 0][ig] = tau_tbl[0, 0, 0][itgas[0, 0, 0][ig]]

            plfrac[0, 0, 0][ig] = fracs[0, 0, 1][ig]
            blay[0, 0, 0][ig] = pklay[0, 0, 1][ib]

            dplnku[0, 0, 0][ig] = pklev[0, 0, 1][ib] - blay[0, 0, 0][ig]
            dplnkd[0, 0, 0][ig] = pklev[0, 0, 0][ib] - blay[0, 0, 0][ig]
            bbdgas[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (
                blay[0, 0, 0][ig] + dplnkd[0, 0, 0][ig] * gasfac[0, 0, 0][ig]
            )
            bbugas[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (
                blay[0, 0, 0][ig] + dplnku[0, 0, 0][ig] * gasfac[0, 0, 0][ig]
            )
            gassrcd[0, 0, 0][ig] = bbdgas[0, 0, 0][ig] * atrgas[0, 0, 0][ig]
            gassrcu[0, 0, 0][ig] = bbugas[0, 0, 0][ig] * atrgas[0, 0, 0][ig]
            trngas[0, 0, 0][ig] = trng[0, 0, 0][ig]

            # total sky, gases+clouds contribution
            clfm[0, 0, 0][ig] = cldfmc[0, 0, 1][ig]
            if clfm[0, 0, 0][ig] >= eps:
                # cloudy layer
                odcld[0, 0, 0][ig] = secdif[0, 0][ib] * taucld[0, 0, 1][ib]
                efclrfr[0, 0, 0][ig] = (
                    1.0 - (1.0 - exp(-odcld[0, 0, 0][ig])) * clfm[0, 0, 0][ig]
                )
                odtot[0, 0, 0][ig] = odepth[0, 0, 0][ig] + odcld[0, 0, 0][ig]
                if odtot[0, 0, 0][ig] < 0.06:
                    totfac[0, 0, 0][ig] = rec_6 * odtot[0, 0, 0][ig]
                    atrtot[0, 0, 0][ig] = (
                        odtot[0, 0, 0][ig]
                        - 0.5 * odtot[0, 0, 0][ig] * odtot[0, 0, 0][ig]
                    )
                else:
                    tblind[0, 0, 0][ig] = odtot[0, 0, 0][ig] / (
                        bpade + odtot[0, 0, 0][ig]
                    )
                    ittot[0, 0, 0][ig] = tblint * tblind[0, 0, 0][ig] + 0.5
                    totfac[0, 0, 0][ig] = tfn_tbl[0, 0, 0][ittot[0, 0, 0][ig]]
                    atrtot[0, 0, 0][ig] = 1.0 - exp_tbl[0, 0, 0][ittot[0, 0, 0][ig]]

                bbdtot[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (
                    blay[0, 0, 0][ig] + dplnkd[0, 0, 0][ig] * totfac[0, 0, 0][ig]
                )
                bbutot[0, 0, 0][ig] = plfrac[0, 0, 0][ig] * (
                    blay[0, 0, 0][ig] + dplnku[0, 0, 0][ig] * totfac[0, 0, 0][ig]
                )
                totsrcd[0, 0, 0][ig] = bbdtot[0, 0, 0][ig] * atrtot[0, 0, 0][ig]
                totsrcu[0, 0, 0][ig] = bbutot[0, 0, 0][ig] * atrtot[0, 0, 0][ig]

                # total sky radiance
                radtotd[0, 0, 0][ig] = (
                    radtotd[0, 0, 1][ig] * trng[0, 0, 0][ig] * efclrfr[0, 0, 0][ig]
                    + gassrcd[0, 0, 0][ig]
                    + clfm[0, 0, 0][ig] * (totsrcd[0, 0, 0][ig] - gassrcd[0, 0, 0][ig])
                )
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig]

                # clear sky radiance
                radclrd[0, 0, 0][ig] = (
                    radclrd[0, 0, 1][ig] * trng[0, 0, 0][ig] + gassrcd[0, 0, 0][ig]
                )
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig]
            else:
                # clear layer

                # total sky radiance
                radtotd[0, 0, 0][ig] = (
                    radtotd[0, 0, 1][ig] * trng[0, 0, 0][ig] + gassrcd[0, 0, 0][ig]
                )
                totdrad[0, 0, 0][ib] = totdrad[0, 0, 0][ib] + radtotd[0, 0, 0][ig]

                # clear sky radiance
                radclrd[0, 0, 0][ig] = (
                    radclrd[0, 0, 1][ig] * trng[0, 0, 0][ig] + gassrcd[0, 0, 0][ig]
                )
                clrdrad[0, 0, 0][ib] = clrdrad[0, 0, 0][ib] + radclrd[0, 0, 0][ig]

            reflct[0, 0, 0][ig] = 1.0 - semiss[0, 0][ib]

    

@stencil(
    backend,
    rebuild=rebuild,
    externals={
        "rec_6": rec_6,
        "bpade": bpade,
        "tblint": tblint,
        "eps": eps,
        "flxfac": flxfac,
        "heatfac": heatfac,
        "lhlw0": lhlw0,
    },
)
def rtrnmc_b(
    semiss: Field[gtscript.IJ, type_nbands],
    delp: FIELD_FLT,
    fracs: Field[type_ngptlw],
    cldfmc: Field[type_ngptlw],
    pklay: Field[type_nbands],
    NGB: Field[gtscript.IJ, (np.int32, (140,))],
    totuflux: FIELD_FLT,
    totdflux: FIELD_FLT,
    totuclfl: FIELD_FLT,
    totdclfl: FIELD_FLT,
    upfxc_t: FIELD_2D,
    upfx0_t: FIELD_2D,
    upfxc_s: FIELD_2D,
    upfx0_s: FIELD_2D,
    dnfxc_s: FIELD_2D,
    dnfx0_s: FIELD_2D,
    hlwc: FIELD_FLT,
    hlw0: FIELD_FLT,
    clrurad: Field[type_nbands],
    clrdrad: Field[type_nbands],
    toturad: Field[type_nbands],
    totdrad: Field[type_nbands],
    gassrcu: Field[type_ngptlw],
    totsrcu: Field[type_ngptlw],
    trngas: Field[type_ngptlw],
    efclrfr: Field[type_ngptlw],
    rfdelp: FIELD_FLT,
    fnet: FIELD_FLT,
    fnetc: FIELD_FLT,
    reflct: Field[type_ngptlw],
    radtotu: Field[type_ngptlw],
    radclru: Field[type_ngptlw],
    radtotd: Field[type_ngptlw],
    radclrd: Field[type_ngptlw],
    rad0: Field[type_ngptlw],
    clfm: Field[type_ngptlw],
    trng: Field[type_ngptlw],
    gasu: Field[type_ngptlw],
    ib: FIELD_2DINT,
):
    from __externals__ import rec_6, bpade, tblint, eps, flxfac, heatfac, lhlw0

# Compute spectral emissivity & reflectance, include the
    # contribution of spectrally varying longwave emissivity and
    # reflection from the surface to the upward radiative transfer.
    # note: spectral and Lambertian reflection are identical for the
    #       diffusivity angle flux integration used here.

    with computation(FORWARD), interval(0, 1):
        for ig2 in range(ngptlw):
            ib = NGB[0, 0][ig2] - 1
            rad0[0, 0, 0][ig2] = (
                semiss[0, 0][ib] * fracs[0, 0, 1][ig2] * pklay[0, 0, 0][ib]
            )

            # Compute total sky radiance
            radtotu[0, 0, 0][ig2] = (
                rad0[0, 0, 0][ig2] + reflct[0, 0, 0][ig2] * radtotd[0, 0, 0][ig2]
            )
            toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, 0][ig2]

            # Compute clear sky radiance
            radclru[0, 0, 0][ig2] = (
                rad0[0, 0, 0][ig2] + reflct[0, 0, 0][ig2] * radclrd[0, 0, 0][ig2]
            )
            clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, 0][ig2]

    # Upward radiative transfer loop
    # - Compute total sky radiance
    # - Compute clear sky radiance

    # toturad holds summed radiance for total sky stream
    # clrurad holds summed radiance for clear sky stream

    with computation(FORWARD), interval(0, 1):
        for ig3 in range(ngptlw):
            ib = NGB[0, 0][ig3] - 1
            clfm[0, 0, 0][ig3] = cldfmc[0, 0, 1][ig3]
            trng[0, 0, 0][ig3] = trngas[0, 0, 0][ig3]
            gasu[0, 0, 0][ig3] = gassrcu[0, 0, 0][ig3]

            if clfm[0, 0, 0][ig3] > eps:
                #  --- ...  cloudy layer

                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig3] = (
                    radtotu[0, 0, 0][ig3] * trng[0, 0, 0][ig3] * efclrfr[0, 0, 0][ig3]
                    + gasu[0, 0, 0][ig3]
                    + clfm[0, 0, 0][ig3] * (totsrcu[0, 0, 0][ig3] - gasu[0, 0, 0][ig3])
                )

                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig3] = (
                    radclru[0, 0, 0][ig3] * trng[0, 0, 0][ig3] + gasu[0, 0, 0][ig3]
                )

            else:
                #  --- ...  clear layer

                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig3] = (
                    radtotu[0, 0, 0][ig3] * trng[0, 0, 0][ig3] + gasu[0, 0, 0][ig3]
                )

                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig3] = (
                    radclru[0, 0, 0][ig3] * trng[0, 0, 0][ig3] + gasu[0, 0, 0][ig3]
                )

    with computation(FORWARD), interval(1, -1):
        for ig4 in range(ngptlw):
            ib = NGB[0, 0][ig4] - 1
            clfm[0, 0, 0][ig4] = cldfmc[0, 0, 1][ig4]
            trng[0, 0, 0][ig4] = trngas[0, 0, 0][ig4]
            gasu[0, 0, 0][ig4] = gassrcu[0, 0, 0][ig4]

            if clfm[0, 0, 0][ig4] > eps:
                #  --- ...  cloudy layer
                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig4] = (
                    radtotu[0, 0, -1][ig4] * trng[0, 0, 0][ig4] * efclrfr[0, 0, 0][ig4]
                    + gasu[0, 0, 0][ig4]
                    + clfm[0, 0, 0][ig4] * (totsrcu[0, 0, 0][ig4] - gasu[0, 0, 0][ig4])
                )
                toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, -1][ig4]
                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig4] = (
                    radclru[0, 0, -1][ig4] * trng[0, 0, 0][ig4] + gasu[0, 0, 0][ig4]
                )
                clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, -1][ig4]
            else:
                #  --- ...  clear layer
                #  --- ... total sky radiance
                radtotu[0, 0, 0][ig4] = (
                    radtotu[0, 0, -1][ig4] * trng[0, 0, 0][ig4] + gasu[0, 0, 0][ig4]
                )
                toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, -1][ig4]
                #  --- ... clear sky radiance
                radclru[0, 0, 0][ig4] = (
                    radclru[0, 0, -1][ig4] * trng[0, 0, 0][ig4] + gasu[0, 0, 0][ig4]
                )
                clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, -1][ig4]

    with computation(FORWARD), interval(-1, None):
        for ig5 in range(ngptlw):
            ib = NGB[0, 0][ig5] - 1

            if clfm[0, 0, 0][ig5] > eps:
                #  --- ...  cloudy layer
                #  --- ... total sky radiance
                toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, -1][ig5]
                #  --- ... clear sky radiance
                clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, -1][ig5]
            else:
                #  --- ...  clear layer
                #  --- ... total sky radiance
                toturad[0, 0, 0][ib] = toturad[0, 0, 0][ib] + radtotu[0, 0, -1][ig5]
                #  --- ... clear sky radiance
                clrurad[0, 0, 0][ib] = clrurad[0, 0, 0][ib] + radclru[0, 0, -1][ig5]

    # Process longwave output from band for total and clear streams.
    # Calculate upward, downward, and net flux.
    with computation(PARALLEL), interval(...):
        for nb in range(nbands):
            totuflux = totuflux + toturad[0, 0, 0][nb]
            totdflux = totdflux + totdrad[0, 0, 0][nb]
            totuclfl = totuclfl + clrurad[0, 0, 0][nb]
            totdclfl = totdclfl + clrdrad[0, 0, 0][nb]

        totuflux = totuflux * flxfac
        totdflux = totdflux * flxfac
        totuclfl = totuclfl * flxfac
        totdclfl = totdclfl * flxfac

    # calculate net fluxes and heating rates (fnet, htr)
    # also compute optional clear sky heating rates (fnetc, htrcl)
    with computation(FORWARD):
        with interval(0, 1):
            # Output surface fluxes
            upfxc_s = totuflux
            upfx0_s = totuclfl
            dnfxc_s = totdflux
            dnfx0_s = totdclfl

            fnet = totuflux - totdflux
            if lhlw0:
                fnetc = totuclfl - totdclfl
        with interval(-1, None):
            # Output TOA fluxes
            upfxc_t = totuflux
            upfx0_t = totuclfl

    with computation(PARALLEL), interval(1, None):
        fnet = totuflux - totdflux
        if lhlw0:
            fnetc = totuclfl - totdclfl

    with computation(PARALLEL), interval(1, None):
        rfdelp = heatfac / delp
        hlwc = (fnet[0, 0, -1] - fnet) * rfdelp
        if lhlw0:
            hlw0 = (fnetc[0, 0, -1] - fnetc) * rfdelp
# This file was automatically created by FeynRules 2.4.35
# Mathematica version: 10.1.0  for Mac OS X x86 (64-bit) (March 24, 2015)
# Date: Thu 11 Feb 2016 16:03:33



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# This is a default parameter object representing the renormalization scale (MU_R).
MU_R = Parameter(name = 'MU_R',
                 nature = 'external',
                 type = 'real',
                 value = 91.188,
                 texname = '\\text{\\mu_r}',
                 lhablock = 'LOOP',
                 lhacode = [1])

# User-defined parameters.
Lambda = Parameter(name = 'Lambda',
                   nature = 'external',
                   type = 'real',
                   value = 1000,
                   texname = '\\Lambda',
                   lhablock = 'DIM6',
                   lhacode = [ 1 ])

RCtphi = Parameter(name = 'RCtphi',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{RC}_{\\text{t$\\phi $}}',
                   lhablock = 'DIM6',
                   lhacode = [ 2 ])

ICtphi = Parameter(name = 'ICtphi',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{IC}_{\\text{t$\\phi $}}',
                   lhablock = 'DIM6',
                   lhacode = [ 3 ])

RCtG = Parameter(name = 'RCtG',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{RC}_{\\text{tG}}',
                 lhablock = 'DIM6',
                 lhacode = [ 4 ])

ICtG = Parameter(name = 'ICtG',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{IC}_{\\text{tG}}',
                 lhablock = 'DIM6',
                 lhacode = [ 5 ])

RCtW = Parameter(name = 'RCtW',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{RC}_{\\text{tW}}',
                 lhablock = 'DIM6',
                 lhacode = [ 6 ])

ICtW = Parameter(name = 'ICtW',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{IC}_{\\text{tW}}',
                 lhablock = 'DIM6',
                 lhacode = [ 7 ])

RCtB = Parameter(name = 'RCtB',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{RC}_{\\text{tB}}',
                 lhablock = 'DIM6',
                 lhacode = [ 8 ])

ICtB = Parameter(name = 'ICtB',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{IC}_{\\text{tB}}',
                 lhablock = 'DIM6',
                 lhacode = [ 9 ])

RCuphi = Parameter(name = 'RCuphi',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{RC}_{\\text{u$\\phi $}}',
                   lhablock = 'DIM6',
                   lhacode = [ 10 ])

ICuphi = Parameter(name = 'ICuphi',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{IC}_{\\text{u$\\phi $}}',
                   lhablock = 'DIM6',
                   lhacode = [ 11 ])

RCuG = Parameter(name = 'RCuG',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{RC}_{\\text{uG}}',
                 lhablock = 'DIM6',
                 lhacode = [ 12 ])

ICuG = Parameter(name = 'ICuG',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{IC}_{\\text{uG}}',
                 lhablock = 'DIM6',
                 lhacode = [ 13 ])

RCuW = Parameter(name = 'RCuW',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{RC}_{\\text{uW}}',
                 lhablock = 'DIM6',
                 lhacode = [ 14 ])

ICuW = Parameter(name = 'ICuW',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{IC}_{\\text{uW}}',
                 lhablock = 'DIM6',
                 lhacode = [ 15 ])

RCuB = Parameter(name = 'RCuB',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{RC}_{\\text{uB}}',
                 lhablock = 'DIM6',
                 lhacode = [ 16 ])

ICuB = Parameter(name = 'ICuB',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{IC}_{\\text{uB}}',
                 lhablock = 'DIM6',
                 lhacode = [ 17 ])

RC1utR = Parameter(name = 'RC1utR',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{RC},\\text{utR},1]',
                   lhablock = 'DIM6',
                   lhacode = [ 18 ])

IC1utR = Parameter(name = 'IC1utR',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{IC},\\text{utR},1]',
                   lhablock = 'DIM6',
                   lhacode = [ 19 ])

RC1utL = Parameter(name = 'RC1utL',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{RC},\\text{utL},1]',
                   lhablock = 'DIM6',
                   lhacode = [ 20 ])

IC1utL = Parameter(name = 'IC1utL',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{IC},\\text{utL},1]',
                   lhablock = 'DIM6',
                   lhacode = [ 21 ])

RC3utL = Parameter(name = 'RC3utL',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{RC},\\text{utL},3]',
                   lhablock = 'DIM6',
                   lhacode = [ 22 ])

IC3utL = Parameter(name = 'IC3utL',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{IC},\\text{utL},3]',
                   lhablock = 'DIM6',
                   lhacode = [ 23 ])

RCtcphi = Parameter(name = 'RCtcphi',
                    nature = 'external',
                    type = 'real',
                    value = 1,
                    texname = '\\text{RC}_{\\text{tc$\\phi $}}',
                    lhablock = 'DIM6',
                    lhacode = [ 24 ])

ICtcphi = Parameter(name = 'ICtcphi',
                    nature = 'external',
                    type = 'real',
                    value = 1,
                    texname = '\\text{IC}_{\\text{tc$\\phi $}}',
                    lhablock = 'DIM6',
                    lhacode = [ 25 ])

RCtcG = Parameter(name = 'RCtcG',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{RC}_{\\text{tcG}}',
                  lhablock = 'DIM6',
                  lhacode = [ 26 ])

ICtcG = Parameter(name = 'ICtcG',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{IC}_{\\text{tcG}}',
                  lhablock = 'DIM6',
                  lhacode = [ 27 ])

RCtcW = Parameter(name = 'RCtcW',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{RC}_{\\text{tcW}}',
                  lhablock = 'DIM6',
                  lhacode = [ 28 ])

ICtcW = Parameter(name = 'ICtcW',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{IC}_{\\text{tcW}}',
                  lhablock = 'DIM6',
                  lhacode = [ 29 ])

RCtcB = Parameter(name = 'RCtcB',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{RC}_{\\text{tcB}}',
                  lhablock = 'DIM6',
                  lhacode = [ 30 ])

ICtcB = Parameter(name = 'ICtcB',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{IC}_{\\text{tcB}}',
                  lhablock = 'DIM6',
                  lhacode = [ 31 ])

RCctphi = Parameter(name = 'RCctphi',
                    nature = 'external',
                    type = 'real',
                    value = 1,
                    texname = '\\text{RC}_{\\text{ct$\\phi $}}',
                    lhablock = 'DIM6',
                    lhacode = [ 32 ])

ICctphi = Parameter(name = 'ICctphi',
                    nature = 'external',
                    type = 'real',
                    value = 1,
                    texname = '\\text{IC}_{\\text{ct$\\phi $}}',
                    lhablock = 'DIM6',
                    lhacode = [ 33 ])

RCctG = Parameter(name = 'RCctG',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{RC}_{\\text{ctG}}',
                  lhablock = 'DIM6',
                  lhacode = [ 34 ])

ICctG = Parameter(name = 'ICctG',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{IC}_{\\text{ctG}}',
                  lhablock = 'DIM6',
                  lhacode = [ 35 ])

RCctW = Parameter(name = 'RCctW',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{RC}_{\\text{ctW}}',
                  lhablock = 'DIM6',
                  lhacode = [ 36 ])

ICctW = Parameter(name = 'ICctW',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{IC}_{\\text{ctW}}',
                  lhablock = 'DIM6',
                  lhacode = [ 37 ])

RCctB = Parameter(name = 'RCctB',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{RC}_{\\text{ctB}}',
                  lhablock = 'DIM6',
                  lhacode = [ 38 ])

ICctB = Parameter(name = 'ICctB',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{IC}_{\\text{ctB}}',
                  lhablock = 'DIM6',
                  lhacode = [ 39 ])

RC1ctR = Parameter(name = 'RC1ctR',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{RC},\\text{ctR},1]',
                   lhablock = 'DIM6',
                   lhacode = [ 40 ])

IC1ctR = Parameter(name = 'IC1ctR',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{IC},\\text{ctR},1]',
                   lhablock = 'DIM6',
                   lhacode = [ 41 ])

RC1ctL = Parameter(name = 'RC1ctL',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{RC},\\text{ctL},1]',
                   lhablock = 'DIM6',
                   lhacode = [ 42 ])

IC1ctL = Parameter(name = 'IC1ctL',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{IC},\\text{ctL},1]',
                   lhablock = 'DIM6',
                   lhacode = [ 43 ])

RC3ctL = Parameter(name = 'RC3ctL',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{RC},\\text{ctL},3]',
                   lhablock = 'DIM6',
                   lhacode = [ 44 ])

IC3ctL = Parameter(name = 'IC3ctL',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{Subsuperscript}[\\text{IC},\\text{ctL},3]',
                   lhablock = 'DIM6',
                   lhacode = [ 45 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.9,
                  texname = '\\text{aEWM1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.0000116637,
               texname = 'G_f',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1184,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

ymt = Parameter(name = 'ymt',
                nature = 'external',
                type = 'real',
                value = 172,
                texname = '\\text{ymt}',
                lhablock = 'YUKAWA',
                lhacode = [ 6 ])

ymtau = Parameter(name = 'ymtau',
                  nature = 'external',
                  type = 'real',
                  value = 1.777,
                  texname = '\\text{ymtau}',
                  lhablock = 'YUKAWA',
                  lhacode = [ 15 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 172,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.4952,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.085,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.50833649,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.00407,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])

aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = '1/aEWM1',
                texname = '\\alpha _{\\text{EW}}')

C1ctL = Parameter(name = 'C1ctL',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*IC1ctL + RC1ctL',
                  texname = '\\text{Subsuperscript}[C,\\text{ctL},1]')

C1ctR = Parameter(name = 'C1ctR',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*IC1ctR + RC1ctR',
                  texname = '\\text{Subsuperscript}[C,\\text{ctR},1]')

C1utL = Parameter(name = 'C1utL',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*IC1utL + RC1utL',
                  texname = '\\text{Subsuperscript}[C,\\text{utL},1]')

C1utR = Parameter(name = 'C1utR',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*IC1utR + RC1utR',
                  texname = '\\text{Subsuperscript}[C,\\text{utR},1]')

C3ctL = Parameter(name = 'C3ctL',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*IC3ctL + RC3ctL',
                  texname = '\\text{Subsuperscript}[C,\\text{ctL},3]')

C3utL = Parameter(name = 'C3utL',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*IC3utL + RC3utL',
                  texname = '\\text{Subsuperscript}[C,\\text{utL},3]')

CctB = Parameter(name = 'CctB',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*ICctB + RCctB',
                 texname = 'C_{\\text{ctB}}')

CctG = Parameter(name = 'CctG',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*ICctG + RCctG',
                 texname = 'C_{\\text{ctG}}')

Cctphi = Parameter(name = 'Cctphi',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICctphi + RCctphi',
                   texname = 'C_{\\text{ct$\\phi $}}')

CctW = Parameter(name = 'CctW',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*ICctW + RCctW',
                 texname = 'C_{\\text{ctW}}')

CtB = Parameter(name = 'CtB',
                nature = 'internal',
                type = 'complex',
                value = 'complex(0,1)*ICtB + RCtB',
                texname = 'C_{\\text{tB}}')

CtcB = Parameter(name = 'CtcB',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*ICtcB + RCtcB',
                 texname = 'C_{\\text{tcB}}')

CtcG = Parameter(name = 'CtcG',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*ICtcG + RCtcG',
                 texname = 'C_{\\text{tcG}}')

Ctcphi = Parameter(name = 'Ctcphi',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICtcphi + RCtcphi',
                   texname = 'C_{\\text{tc$\\phi $}}')

CtcW = Parameter(name = 'CtcW',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*ICtcW + RCtcW',
                 texname = 'C_{\\text{tcW}}')

CtG = Parameter(name = 'CtG',
                nature = 'internal',
                type = 'complex',
                value = 'complex(0,1)*ICtG + RCtG',
                texname = 'C_{\\text{tG}}')

Ctphi = Parameter(name = 'Ctphi',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICtphi + RCtphi',
                  texname = 'C_{\\text{t$\\phi $}}')

CtW = Parameter(name = 'CtW',
                nature = 'internal',
                type = 'complex',
                value = 'complex(0,1)*ICtW + RCtW',
                texname = 'C_{\\text{tW}}')

CuB = Parameter(name = 'CuB',
                nature = 'internal',
                type = 'complex',
                value = 'complex(0,1)*ICuB + RCuB',
                texname = 'C_{\\text{uB}}')

CuG = Parameter(name = 'CuG',
                nature = 'internal',
                type = 'complex',
                value = 'complex(0,1)*ICuG + RCuG',
                texname = 'C_{\\text{uG}}')

Cuphi = Parameter(name = 'Cuphi',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICuphi + RCuphi',
                  texname = 'C_{\\text{u$\\phi $}}')

CuW = Parameter(name = 'CuW',
                nature = 'internal',
                type = 'complex',
                value = 'complex(0,1)*ICuW + RCuW',
                texname = 'C_{\\text{uW}}')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

MW = Parameter(name = 'MW',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(MZ**2/2. + cmath.sqrt(MZ**4/4. - (aEW*cmath.pi*MZ**2)/(Gf*cmath.sqrt(2))))',
               texname = 'M_W')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)',
               texname = 'e')

sw2 = Parameter(name = 'sw2',
                nature = 'internal',
                type = 'real',
                value = '1 - MW**2/MZ**2',
                texname = '\\text{sw2}')

cw = Parameter(name = 'cw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 - sw2)',
               texname = 'c_w')

sw = Parameter(name = 'sw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(sw2)',
               texname = 's_w')

g1 = Parameter(name = 'g1',
               nature = 'internal',
               type = 'real',
               value = 'ee/cw',
               texname = 'g_1')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = 'ee/sw',
               texname = 'g_w')

vev = Parameter(name = 'vev',
                nature = 'internal',
                type = 'real',
                value = '(2*MW*sw)/ee',
                texname = '\\text{vev}')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = 'MH**2/(2.*vev**2)',
                texname = '\\text{lam}')

yt = Parameter(name = 'yt',
               nature = 'internal',
               type = 'real',
               value = '(ymt*cmath.sqrt(2))/vev',
               texname = '\\text{yt}')

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(lam*vev**2)',
                texname = '\\mu')

I2a33 = Parameter(name = 'I2a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yt',
                  texname = '\\text{I2a33}')

I3a33 = Parameter(name = 'I3a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yt',
                  texname = '\\text{I3a33}')


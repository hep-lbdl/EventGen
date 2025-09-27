# This file was automatically created by FeynRules 2.3.2
# Mathematica version: 10.0 for Linux x86 (64-bit) (September 9, 2014)
# Date: Tue 9 Jun 2015 02:27:01



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
RRd1x1 = Parameter(name = 'RRd1x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd1x1}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 1, 1 ])

RRd2x2 = Parameter(name = 'RRd2x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd2x2}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 2, 2 ])

RRd3x3 = Parameter(name = 'RRd3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.938737896,
                   texname = '\\text{RRd3x3}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 3, 3 ])

RRd3x6 = Parameter(name = 'RRd3x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.344631925,
                   texname = '\\text{RRd3x6}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 3, 6 ])

RRd4x4 = Parameter(name = 'RRd4x4',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd4x4}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 4, 4 ])

RRd5x5 = Parameter(name = 'RRd5x5',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd5x5}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 5, 5 ])

RRd6x3 = Parameter(name = 'RRd6x3',
                   nature = 'external',
                   type = 'real',
                   value = -0.344631925,
                   texname = '\\text{RRd6x3}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 6, 3 ])

RRd6x6 = Parameter(name = 'RRd6x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.938737896,
                   texname = '\\text{RRd6x6}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 6, 6 ])

alp = Parameter(name = 'alp',
                nature = 'external',
                type = 'real',
                value = -0.11382521,
                texname = '\\alpha',
                lhablock = 'FRALPHA',
                lhacode = [ 1 ])

RMUH = Parameter(name = 'RMUH',
                 nature = 'external',
                 type = 'real',
                 value = 357.680977,
                 texname = '\\text{RMUH}',
                 lhablock = 'HMIX',
                 lhacode = [ 1 ])

tb = Parameter(name = 'tb',
               nature = 'external',
               type = 'real',
               value = 9.74862403,
               texname = 't_b',
               lhablock = 'HMIX',
               lhacode = [ 2 ])

MA2 = Parameter(name = 'MA2',
                nature = 'external',
                type = 'real',
                value = 166439.065,
                texname = '\\text{Subsuperscript}[m,A,2]',
                lhablock = 'HMIX',
                lhacode = [ 4 ])

RmD21x1 = Parameter(name = 'RmD21x1',
                    nature = 'external',
                    type = 'real',
                    value = 273684.674,
                    texname = '\\text{RmD21x1}',
                    lhablock = 'MSD2',
                    lhacode = [ 1, 1 ])

RmD22x2 = Parameter(name = 'RmD22x2',
                    nature = 'external',
                    type = 'real',
                    value = 273684.674,
                    texname = '\\text{RmD22x2}',
                    lhablock = 'MSD2',
                    lhacode = [ 2, 2 ])

RmD23x3 = Parameter(name = 'RmD23x3',
                    nature = 'external',
                    type = 'real',
                    value = 270261.969,
                    texname = '\\text{RmD23x3}',
                    lhablock = 'MSD2',
                    lhacode = [ 3, 3 ])

RmE21x1 = Parameter(name = 'RmE21x1',
                    nature = 'external',
                    type = 'real',
                    value = 18630.6287,
                    texname = '\\text{RmE21x1}',
                    lhablock = 'MSE2',
                    lhacode = [ 1, 1 ])

RmE22x2 = Parameter(name = 'RmE22x2',
                    nature = 'external',
                    type = 'real',
                    value = 18630.6287,
                    texname = '\\text{RmE22x2}',
                    lhablock = 'MSE2',
                    lhacode = [ 2, 2 ])

RmE23x3 = Parameter(name = 'RmE23x3',
                    nature = 'external',
                    type = 'real',
                    value = 17967.6406,
                    texname = '\\text{RmE23x3}',
                    lhablock = 'MSE2',
                    lhacode = [ 3, 3 ])

RmL21x1 = Parameter(name = 'RmL21x1',
                    nature = 'external',
                    type = 'real',
                    value = 38155.67,
                    texname = '\\text{RmL21x1}',
                    lhablock = 'MSL2',
                    lhacode = [ 1, 1 ])

RmL22x2 = Parameter(name = 'RmL22x2',
                    nature = 'external',
                    type = 'real',
                    value = 38155.67,
                    texname = '\\text{RmL22x2}',
                    lhablock = 'MSL2',
                    lhacode = [ 2, 2 ])

RmL23x3 = Parameter(name = 'RmL23x3',
                    nature = 'external',
                    type = 'real',
                    value = 37828.6769,
                    texname = '\\text{RmL23x3}',
                    lhablock = 'MSL2',
                    lhacode = [ 3, 3 ])

RMx1 = Parameter(name = 'RMx1',
                 nature = 'external',
                 type = 'real',
                 value = 101.396534,
                 texname = '\\text{RMx1}',
                 lhablock = 'MSOFT',
                 lhacode = [ 1 ])

RMx2 = Parameter(name = 'RMx2',
                 nature = 'external',
                 type = 'real',
                 value = 191.504241,
                 texname = '\\text{RMx2}',
                 lhablock = 'MSOFT',
                 lhacode = [ 2 ])

RMx3 = Parameter(name = 'RMx3',
                 nature = 'external',
                 type = 'real',
                 value = 588.263031,
                 texname = '\\text{RMx3}',
                 lhablock = 'MSOFT',
                 lhacode = [ 3 ])

mHu2 = Parameter(name = 'mHu2',
                 nature = 'external',
                 type = 'real',
                 value = 32337.4943,
                 texname = '\\text{Subsuperscript}\\left[m,H_u,2\\right]',
                 lhablock = 'MSOFT',
                 lhacode = [ 21 ])

mHd2 = Parameter(name = 'mHd2',
                 nature = 'external',
                 type = 'real',
                 value = -128800.134,
                 texname = '\\text{Subsuperscript}\\left[m,H_d,2\\right]',
                 lhablock = 'MSOFT',
                 lhacode = [ 22 ])

RmQ21x1 = Parameter(name = 'RmQ21x1',
                    nature = 'external',
                    type = 'real',
                    value = 299836.701,
                    texname = '\\text{RmQ21x1}',
                    lhablock = 'MSQ2',
                    lhacode = [ 1, 1 ])

RmQ22x2 = Parameter(name = 'RmQ22x2',
                    nature = 'external',
                    type = 'real',
                    value = 299836.701,
                    texname = '\\text{RmQ22x2}',
                    lhablock = 'MSQ2',
                    lhacode = [ 2, 2 ])

RmQ23x3 = Parameter(name = 'RmQ23x3',
                    nature = 'external',
                    type = 'real',
                    value = 248765.367,
                    texname = '\\text{RmQ23x3}',
                    lhablock = 'MSQ2',
                    lhacode = [ 3, 3 ])

RmU21x1 = Parameter(name = 'RmU21x1',
                    nature = 'external',
                    type = 'real',
                    value = 280382.106,
                    texname = '\\text{RmU21x1}',
                    lhablock = 'MSU2',
                    lhacode = [ 1, 1 ])

RmU22x2 = Parameter(name = 'RmU22x2',
                    nature = 'external',
                    type = 'real',
                    value = 280382.106,
                    texname = '\\text{RmU22x2}',
                    lhablock = 'MSU2',
                    lhacode = [ 2, 2 ])

RmU23x3 = Parameter(name = 'RmU23x3',
                    nature = 'external',
                    type = 'real',
                    value = 179137.072,
                    texname = '\\text{RmU23x3}',
                    lhablock = 'MSU2',
                    lhacode = [ 3, 3 ])

RNN1x1 = Parameter(name = 'RNN1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.98636443,
                   texname = '\\text{RNN1x1}',
                   lhablock = 'NMIX',
                   lhacode = [ 1, 1 ])

RNN1x2 = Parameter(name = 'RNN1x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.0531103553,
                   texname = '\\text{RNN1x2}',
                   lhablock = 'NMIX',
                   lhacode = [ 1, 2 ])

RNN1x3 = Parameter(name = 'RNN1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.146433995,
                   texname = '\\text{RNN1x3}',
                   lhablock = 'NMIX',
                   lhacode = [ 1, 3 ])

RNN1x4 = Parameter(name = 'RNN1x4',
                   nature = 'external',
                   type = 'real',
                   value = -0.0531186117,
                   texname = '\\text{RNN1x4}',
                   lhablock = 'NMIX',
                   lhacode = [ 1, 4 ])

RNN2x1 = Parameter(name = 'RNN2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.0993505358,
                   texname = '\\text{RNN2x1}',
                   lhablock = 'NMIX',
                   lhacode = [ 2, 1 ])

RNN2x2 = Parameter(name = 'RNN2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.944949299,
                   texname = '\\text{RNN2x2}',
                   lhablock = 'NMIX',
                   lhacode = [ 2, 2 ])

RNN2x3 = Parameter(name = 'RNN2x3',
                   nature = 'external',
                   type = 'real',
                   value = -0.26984672,
                   texname = '\\text{RNN2x3}',
                   lhablock = 'NMIX',
                   lhacode = [ 2, 3 ])

RNN2x4 = Parameter(name = 'RNN2x4',
                   nature = 'external',
                   type = 'real',
                   value = 0.156150698,
                   texname = '\\text{RNN2x4}',
                   lhablock = 'NMIX',
                   lhacode = [ 2, 4 ])

RNN3x1 = Parameter(name = 'RNN3x1',
                   nature = 'external',
                   type = 'real',
                   value = -0.0603388002,
                   texname = '\\text{RNN3x1}',
                   lhablock = 'NMIX',
                   lhacode = [ 3, 1 ])

RNN3x2 = Parameter(name = 'RNN3x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.0877004854,
                   texname = '\\text{RNN3x2}',
                   lhablock = 'NMIX',
                   lhacode = [ 3, 2 ])

RNN3x3 = Parameter(name = 'RNN3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.695877493,
                   texname = '\\text{RNN3x3}',
                   lhablock = 'NMIX',
                   lhacode = [ 3, 3 ])

RNN3x4 = Parameter(name = 'RNN3x4',
                   nature = 'external',
                   type = 'real',
                   value = 0.710226984,
                   texname = '\\text{RNN3x4}',
                   lhablock = 'NMIX',
                   lhacode = [ 3, 4 ])

RNN4x1 = Parameter(name = 'RNN4x1',
                   nature = 'external',
                   type = 'real',
                   value = -0.116507132,
                   texname = '\\text{RNN4x1}',
                   lhablock = 'NMIX',
                   lhacode = [ 4, 1 ])

RNN4x2 = Parameter(name = 'RNN4x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.310739017,
                   texname = '\\text{RNN4x2}',
                   lhablock = 'NMIX',
                   lhacode = [ 4, 2 ])

RNN4x3 = Parameter(name = 'RNN4x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.64922596,
                   texname = '\\text{RNN4x3}',
                   lhablock = 'NMIX',
                   lhacode = [ 4, 3 ])

RNN4x4 = Parameter(name = 'RNN4x4',
                   nature = 'external',
                   type = 'real',
                   value = -0.684377823,
                   texname = '\\text{RNN4x4}',
                   lhablock = 'NMIX',
                   lhacode = [ 4, 4 ])

RLLE1x2x1 = Parameter(name = 'RLLE1x2x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLLE1x2x1}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 1, 2, 1 ])

RLLE1x2x2 = Parameter(name = 'RLLE1x2x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLLE1x2x2}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 1, 2, 2 ])

RLLE1x2x3 = Parameter(name = 'RLLE1x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLLE1x2x3}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 1, 2, 3 ])

RLLE1x3x1 = Parameter(name = 'RLLE1x3x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLLE1x3x1}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 1, 3, 1 ])

RLLE1x3x2 = Parameter(name = 'RLLE1x3x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLLE1x3x2}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 1, 3, 2 ])

RLLE1x3x3 = Parameter(name = 'RLLE1x3x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLLE1x3x3}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 1, 3, 3 ])

RLLE2x1x1 = Parameter(name = 'RLLE2x1x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLLE2x1x1}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 2, 1, 1 ])

RLLE2x1x2 = Parameter(name = 'RLLE2x1x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLLE2x1x2}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 2, 1, 2 ])

RLLE2x1x3 = Parameter(name = 'RLLE2x1x3',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLLE2x1x3}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 2, 1, 3 ])

RLLE2x3x1 = Parameter(name = 'RLLE2x3x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLLE2x3x1}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 2, 3, 1 ])

RLLE2x3x2 = Parameter(name = 'RLLE2x3x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLLE2x3x2}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 2, 3, 2 ])

RLLE2x3x3 = Parameter(name = 'RLLE2x3x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLLE2x3x3}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 2, 3, 3 ])

RLLE3x1x1 = Parameter(name = 'RLLE3x1x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLLE3x1x1}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 3, 1, 1 ])

RLLE3x1x2 = Parameter(name = 'RLLE3x1x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLLE3x1x2}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 3, 1, 2 ])

RLLE3x1x3 = Parameter(name = 'RLLE3x1x3',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLLE3x1x3}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 3, 1, 3 ])

RLLE3x2x1 = Parameter(name = 'RLLE3x2x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLLE3x2x1}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 3, 2, 1 ])

RLLE3x2x2 = Parameter(name = 'RLLE3x2x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLLE3x2x2}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 3, 2, 2 ])

RLLE3x2x3 = Parameter(name = 'RLLE3x2x3',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLLE3x2x3}',
                      lhablock = 'RVLAMLLE',
                      lhacode = [ 3, 2, 3 ])

RLQD1x1x1 = Parameter(name = 'RLQD1x1x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD1x1x1}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 1, 1, 1 ])

RLQD1x1x2 = Parameter(name = 'RLQD1x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD1x1x2}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 1, 1, 2 ])

RLQD1x1x3 = Parameter(name = 'RLQD1x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD1x1x3}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 1, 1, 3 ])

RLQD1x2x1 = Parameter(name = 'RLQD1x2x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD1x2x1}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 1, 2, 1 ])

RLQD1x2x2 = Parameter(name = 'RLQD1x2x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD1x2x2}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 1, 2, 2 ])

RLQD1x2x3 = Parameter(name = 'RLQD1x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD1x2x3}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 1, 2, 3 ])

RLQD1x3x1 = Parameter(name = 'RLQD1x3x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD1x3x1}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 1, 3, 1 ])

RLQD1x3x2 = Parameter(name = 'RLQD1x3x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD1x3x2}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 1, 3, 2 ])

RLQD1x3x3 = Parameter(name = 'RLQD1x3x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD1x3x3}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 1, 3, 3 ])

RLQD2x1x1 = Parameter(name = 'RLQD2x1x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD2x1x1}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 2, 1, 1 ])

RLQD2x1x2 = Parameter(name = 'RLQD2x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD2x1x2}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 2, 1, 2 ])

RLQD2x1x3 = Parameter(name = 'RLQD2x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD2x1x3}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 2, 1, 3 ])

RLQD2x2x1 = Parameter(name = 'RLQD2x2x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD2x2x1}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 2, 2, 1 ])

RLQD2x2x2 = Parameter(name = 'RLQD2x2x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD2x2x2}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 2, 2, 2 ])

RLQD2x2x3 = Parameter(name = 'RLQD2x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD2x2x3}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 2, 2, 3 ])

RLQD2x3x1 = Parameter(name = 'RLQD2x3x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD2x3x1}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 2, 3, 1 ])

RLQD2x3x2 = Parameter(name = 'RLQD2x3x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD2x3x2}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 2, 3, 2 ])

RLQD2x3x3 = Parameter(name = 'RLQD2x3x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD2x3x3}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 2, 3, 3 ])

RLQD3x1x1 = Parameter(name = 'RLQD3x1x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD3x1x1}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 3, 1, 1 ])

RLQD3x1x2 = Parameter(name = 'RLQD3x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD3x1x2}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 3, 1, 2 ])

RLQD3x1x3 = Parameter(name = 'RLQD3x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD3x1x3}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 3, 1, 3 ])

RLQD3x2x1 = Parameter(name = 'RLQD3x2x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD3x2x1}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 3, 2, 1 ])

RLQD3x2x2 = Parameter(name = 'RLQD3x2x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD3x2x2}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 3, 2, 2 ])

RLQD3x2x3 = Parameter(name = 'RLQD3x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD3x2x3}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 3, 2, 3 ])

RLQD3x3x1 = Parameter(name = 'RLQD3x3x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD3x3x1}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 3, 3, 1 ])

RLQD3x3x2 = Parameter(name = 'RLQD3x3x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD3x3x2}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 3, 3, 2 ])

RLQD3x3x3 = Parameter(name = 'RLQD3x3x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RLQD3x3x3}',
                      lhablock = 'RVLAMLQD',
                      lhacode = [ 3, 3, 3 ])

RLDD1x1x2 = Parameter(name = 'RLDD1x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLDD1x1x2}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 1, 1, 2 ])

RLDD1x1x3 = Parameter(name = 'RLDD1x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLDD1x1x3}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 1, 1, 3 ])

RLDD1x2x1 = Parameter(name = 'RLDD1x2x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLDD1x2x1}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 1, 2, 1 ])

RLDD1x2x3 = Parameter(name = 'RLDD1x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLDD1x2x3}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 1, 2, 3 ])

RLDD1x3x1 = Parameter(name = 'RLDD1x3x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLDD1x3x1}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 1, 3, 1 ])

RLDD1x3x2 = Parameter(name = 'RLDD1x3x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLDD1x3x2}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 1, 3, 2 ])

RLDD2x1x2 = Parameter(name = 'RLDD2x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLDD2x1x2}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 2, 1, 2 ])

RLDD2x1x3 = Parameter(name = 'RLDD2x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLDD2x1x3}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 2, 1, 3 ])

RLDD2x2x1 = Parameter(name = 'RLDD2x2x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLDD2x2x1}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 2, 2, 1 ])

RLDD2x2x3 = Parameter(name = 'RLDD2x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLDD2x2x3}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 2, 2, 3 ])

RLDD2x3x1 = Parameter(name = 'RLDD2x3x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLDD2x3x1}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 2, 3, 1 ])

RLDD2x3x2 = Parameter(name = 'RLDD2x3x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLDD2x3x2}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 2, 3, 2 ])

RLDD3x1x2 = Parameter(name = 'RLDD3x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLDD3x1x2}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 3, 1, 2 ])

RLDD3x1x3 = Parameter(name = 'RLDD3x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLDD3x1x3}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 3, 1, 3 ])

RLDD3x2x1 = Parameter(name = 'RLDD3x2x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLDD3x2x1}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 3, 2, 1 ])

RLDD3x2x3 = Parameter(name = 'RLDD3x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RLDD3x2x3}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 3, 2, 3 ])

RLDD3x3x1 = Parameter(name = 'RLDD3x3x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLDD3x3x1}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 3, 3, 1 ])

RLDD3x3x2 = Parameter(name = 'RLDD3x3x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RLDD3x3x2}',
                      lhablock = 'RVLAMUDD',
                      lhacode = [ 3, 3, 2 ])

RTLE1x2x1 = Parameter(name = 'RTLE1x2x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTLE1x2x1}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 1, 2, 1 ])

RTLE1x2x2 = Parameter(name = 'RTLE1x2x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTLE1x2x2}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 1, 2, 2 ])

RTLE1x2x3 = Parameter(name = 'RTLE1x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTLE1x2x3}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 1, 2, 3 ])

RTLE1x3x1 = Parameter(name = 'RTLE1x3x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTLE1x3x1}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 1, 3, 1 ])

RTLE1x3x2 = Parameter(name = 'RTLE1x3x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTLE1x3x2}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 1, 3, 2 ])

RTLE1x3x3 = Parameter(name = 'RTLE1x3x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTLE1x3x3}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 1, 3, 3 ])

RTLE2x1x1 = Parameter(name = 'RTLE2x1x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTLE2x1x1}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 2, 1, 1 ])

RTLE2x1x2 = Parameter(name = 'RTLE2x1x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTLE2x1x2}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 2, 1, 2 ])

RTLE2x1x3 = Parameter(name = 'RTLE2x1x3',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTLE2x1x3}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 2, 1, 3 ])

RTLE2x3x1 = Parameter(name = 'RTLE2x3x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTLE2x3x1}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 2, 3, 1 ])

RTLE2x3x2 = Parameter(name = 'RTLE2x3x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTLE2x3x2}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 2, 3, 2 ])

RTLE2x3x3 = Parameter(name = 'RTLE2x3x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTLE2x3x3}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 2, 3, 3 ])

RTLE3x1x1 = Parameter(name = 'RTLE3x1x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTLE3x1x1}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 3, 1, 1 ])

RTLE3x1x2 = Parameter(name = 'RTLE3x1x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTLE3x1x2}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 3, 1, 2 ])

RTLE3x1x3 = Parameter(name = 'RTLE3x1x3',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTLE3x1x3}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 3, 1, 3 ])

RTLE3x2x1 = Parameter(name = 'RTLE3x2x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTLE3x2x1}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 3, 2, 1 ])

RTLE3x2x2 = Parameter(name = 'RTLE3x2x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTLE3x2x2}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 3, 2, 2 ])

RTLE3x2x3 = Parameter(name = 'RTLE3x2x3',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTLE3x2x3}',
                      lhablock = 'RVTLLE',
                      lhacode = [ 3, 2, 3 ])

RTQD1x1x1 = Parameter(name = 'RTQD1x1x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD1x1x1}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 1, 1, 1 ])

RTQD1x1x2 = Parameter(name = 'RTQD1x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD1x1x2}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 1, 1, 2 ])

RTQD1x1x3 = Parameter(name = 'RTQD1x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD1x1x3}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 1, 1, 3 ])

RTQD1x2x1 = Parameter(name = 'RTQD1x2x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD1x2x1}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 1, 2, 1 ])

RTQD1x2x2 = Parameter(name = 'RTQD1x2x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD1x2x2}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 1, 2, 2 ])

RTQD1x2x3 = Parameter(name = 'RTQD1x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD1x2x3}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 1, 2, 3 ])

RTQD1x3x1 = Parameter(name = 'RTQD1x3x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD1x3x1}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 1, 3, 1 ])

RTQD1x3x2 = Parameter(name = 'RTQD1x3x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD1x3x2}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 1, 3, 2 ])

RTQD1x3x3 = Parameter(name = 'RTQD1x3x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD1x3x3}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 1, 3, 3 ])

RTQD2x1x1 = Parameter(name = 'RTQD2x1x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD2x1x1}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 2, 1, 1 ])

RTQD2x1x2 = Parameter(name = 'RTQD2x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD2x1x2}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 2, 1, 2 ])

RTQD2x1x3 = Parameter(name = 'RTQD2x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD2x1x3}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 2, 1, 3 ])

RTQD2x2x1 = Parameter(name = 'RTQD2x2x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD2x2x1}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 2, 2, 1 ])

RTQD2x2x2 = Parameter(name = 'RTQD2x2x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD2x2x2}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 2, 2, 2 ])

RTQD2x2x3 = Parameter(name = 'RTQD2x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD2x2x3}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 2, 2, 3 ])

RTQD2x3x1 = Parameter(name = 'RTQD2x3x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD2x3x1}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 2, 3, 1 ])

RTQD2x3x2 = Parameter(name = 'RTQD2x3x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD2x3x2}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 2, 3, 2 ])

RTQD2x3x3 = Parameter(name = 'RTQD2x3x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD2x3x3}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 2, 3, 3 ])

RTQD3x1x1 = Parameter(name = 'RTQD3x1x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD3x1x1}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 3, 1, 1 ])

RTQD3x1x2 = Parameter(name = 'RTQD3x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD3x1x2}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 3, 1, 2 ])

RTQD3x1x3 = Parameter(name = 'RTQD3x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD3x1x3}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 3, 1, 3 ])

RTQD3x2x1 = Parameter(name = 'RTQD3x2x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD3x2x1}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 3, 2, 1 ])

RTQD3x2x2 = Parameter(name = 'RTQD3x2x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD3x2x2}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 3, 2, 2 ])

RTQD3x2x3 = Parameter(name = 'RTQD3x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD3x2x3}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 3, 2, 3 ])

RTQD3x3x1 = Parameter(name = 'RTQD3x3x1',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD3x3x1}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 3, 3, 1 ])

RTQD3x3x2 = Parameter(name = 'RTQD3x3x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD3x3x2}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 3, 3, 2 ])

RTQD3x3x3 = Parameter(name = 'RTQD3x3x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.1,
                      texname = '\\text{RTQD3x3x3}',
                      lhablock = 'RVTLQD',
                      lhacode = [ 3, 3, 3 ])

RTDD1x1x2 = Parameter(name = 'RTDD1x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTDD1x1x2}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 1, 1, 2 ])

RTDD1x1x3 = Parameter(name = 'RTDD1x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTDD1x1x3}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 1, 1, 3 ])

RTDD1x2x1 = Parameter(name = 'RTDD1x2x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTDD1x2x1}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 1, 2, 1 ])

RTDD1x2x3 = Parameter(name = 'RTDD1x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTDD1x2x3}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 1, 2, 3 ])

RTDD1x3x1 = Parameter(name = 'RTDD1x3x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTDD1x3x1}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 1, 3, 1 ])

RTDD1x3x2 = Parameter(name = 'RTDD1x3x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTDD1x3x2}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 1, 3, 2 ])

RTDD2x1x2 = Parameter(name = 'RTDD2x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTDD2x1x2}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 2, 1, 2 ])

RTDD2x1x3 = Parameter(name = 'RTDD2x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTDD2x1x3}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 2, 1, 3 ])

RTDD2x2x1 = Parameter(name = 'RTDD2x2x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTDD2x2x1}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 2, 2, 1 ])

RTDD2x2x3 = Parameter(name = 'RTDD2x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTDD2x2x3}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 2, 2, 3 ])

RTDD2x3x1 = Parameter(name = 'RTDD2x3x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTDD2x3x1}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 2, 3, 1 ])

RTDD2x3x2 = Parameter(name = 'RTDD2x3x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTDD2x3x2}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 2, 3, 2 ])

RTDD3x1x2 = Parameter(name = 'RTDD3x1x2',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTDD3x1x2}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 3, 1, 2 ])

RTDD3x1x3 = Parameter(name = 'RTDD3x1x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTDD3x1x3}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 3, 1, 3 ])

RTDD3x2x1 = Parameter(name = 'RTDD3x2x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTDD3x2x1}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 3, 2, 1 ])

RTDD3x2x3 = Parameter(name = 'RTDD3x2x3',
                      nature = 'external',
                      type = 'real',
                      value = 0.2,
                      texname = '\\text{RTDD3x2x3}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 3, 2, 3 ])

RTDD3x3x1 = Parameter(name = 'RTDD3x3x1',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTDD3x3x1}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 3, 3, 1 ])

RTDD3x3x2 = Parameter(name = 'RTDD3x3x2',
                      nature = 'external',
                      type = 'real',
                      value = -0.2,
                      texname = '\\text{RTDD3x3x2}',
                      lhablock = 'RVTUDD',
                      lhacode = [ 3, 3, 2 ])

RRl1x1 = Parameter(name = 'RRl1x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl1x1}',
                   lhablock = 'SELMIX',
                   lhacode = [ 1, 1 ])

RRl2x2 = Parameter(name = 'RRl2x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl2x2}',
                   lhablock = 'SELMIX',
                   lhacode = [ 2, 2 ])

RRl3x3 = Parameter(name = 'RRl3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.28248719,
                   texname = '\\text{RRl3x3}',
                   lhablock = 'SELMIX',
                   lhacode = [ 3, 3 ])

RRl3x6 = Parameter(name = 'RRl3x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.959271071,
                   texname = '\\text{RRl3x6}',
                   lhablock = 'SELMIX',
                   lhacode = [ 3, 6 ])

RRl4x4 = Parameter(name = 'RRl4x4',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl4x4}',
                   lhablock = 'SELMIX',
                   lhacode = [ 4, 4 ])

RRl5x5 = Parameter(name = 'RRl5x5',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl5x5}',
                   lhablock = 'SELMIX',
                   lhacode = [ 5, 5 ])

RRl6x3 = Parameter(name = 'RRl6x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.959271071,
                   texname = '\\text{RRl6x3}',
                   lhablock = 'SELMIX',
                   lhacode = [ 6, 3 ])

RRl6x6 = Parameter(name = 'RRl6x6',
                   nature = 'external',
                   type = 'real',
                   value = -0.28248719,
                   texname = '\\text{RRl6x6}',
                   lhablock = 'SELMIX',
                   lhacode = [ 6, 6 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.934,
                  texname = '\\text{Subsuperscript}[\\alpha ,w,-1]',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.118,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

RRn1x1 = Parameter(name = 'RRn1x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRn1x1}',
                   lhablock = 'SNUMIX',
                   lhacode = [ 1, 1 ])

RRn2x2 = Parameter(name = 'RRn2x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRn2x2}',
                   lhablock = 'SNUMIX',
                   lhacode = [ 2, 2 ])

RRn3x3 = Parameter(name = 'RRn3x3',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRn3x3}',
                   lhablock = 'SNUMIX',
                   lhacode = [ 3, 3 ])

Rtd3x3 = Parameter(name = 'Rtd3x3',
                   nature = 'external',
                   type = 'real',
                   value = -110.693742,
                   texname = '\\text{Rtd3x3}',
                   lhablock = 'TD',
                   lhacode = [ 3, 3 ])

Rte3x3 = Parameter(name = 'Rte3x3',
                   nature = 'external',
                   type = 'real',
                   value = -25.4019727,
                   texname = '\\text{Rte3x3}',
                   lhablock = 'TE',
                   lhacode = [ 3, 3 ])

Rtu3x3 = Parameter(name = 'Rtu3x3',
                   nature = 'external',
                   type = 'real',
                   value = -444.752457,
                   texname = '\\text{Rtu3x3}',
                   lhablock = 'TU',
                   lhacode = [ 3, 3 ])

RUU1x1 = Parameter(name = 'RUU1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.916834859,
                   texname = '\\text{RUU1x1}',
                   lhablock = 'UMIX',
                   lhacode = [ 1, 1 ])

RUU1x2 = Parameter(name = 'RUU1x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.399266629,
                   texname = '\\text{RUU1x2}',
                   lhablock = 'UMIX',
                   lhacode = [ 1, 2 ])

RUU2x1 = Parameter(name = 'RUU2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.399266629,
                   texname = '\\text{RUU2x1}',
                   lhablock = 'UMIX',
                   lhacode = [ 2, 1 ])

RUU2x2 = Parameter(name = 'RUU2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.916834859,
                   texname = '\\text{RUU2x2}',
                   lhablock = 'UMIX',
                   lhacode = [ 2, 2 ])

RMNS1x1 = Parameter(name = 'RMNS1x1',
                    nature = 'external',
                    type = 'real',
                    value = 1.,
                    texname = '\\text{RMNS1x1}',
                    lhablock = 'UPMNS',
                    lhacode = [ 1, 1 ])

RMNS2x2 = Parameter(name = 'RMNS2x2',
                    nature = 'external',
                    type = 'real',
                    value = 1.,
                    texname = '\\text{RMNS2x2}',
                    lhablock = 'UPMNS',
                    lhacode = [ 2, 2 ])

RMNS3x3 = Parameter(name = 'RMNS3x3',
                    nature = 'external',
                    type = 'real',
                    value = 1.,
                    texname = '\\text{RMNS3x3}',
                    lhablock = 'UPMNS',
                    lhacode = [ 3, 3 ])

RRu1x1 = Parameter(name = 'RRu1x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu1x1}',
                   lhablock = 'USQMIX',
                   lhacode = [ 1, 1 ])

RRu2x2 = Parameter(name = 'RRu2x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu2x2}',
                   lhablock = 'USQMIX',
                   lhacode = [ 2, 2 ])

RRu3x3 = Parameter(name = 'RRu3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.55364496,
                   texname = '\\text{RRu3x3}',
                   lhablock = 'USQMIX',
                   lhacode = [ 3, 3 ])

RRu3x6 = Parameter(name = 'RRu3x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.83275282,
                   texname = '\\text{RRu3x6}',
                   lhablock = 'USQMIX',
                   lhacode = [ 3, 6 ])

RRu4x4 = Parameter(name = 'RRu4x4',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu4x4}',
                   lhablock = 'USQMIX',
                   lhacode = [ 4, 4 ])

RRu5x5 = Parameter(name = 'RRu5x5',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu5x5}',
                   lhablock = 'USQMIX',
                   lhacode = [ 5, 5 ])

RRu6x3 = Parameter(name = 'RRu6x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.83275282,
                   texname = '\\text{RRu6x3}',
                   lhablock = 'USQMIX',
                   lhacode = [ 6, 3 ])

RRu6x6 = Parameter(name = 'RRu6x6',
                   nature = 'external',
                   type = 'real',
                   value = -0.55364496,
                   texname = '\\text{RRu6x6}',
                   lhablock = 'USQMIX',
                   lhacode = [ 6, 6 ])

RCKM1x1 = Parameter(name = 'RCKM1x1',
                    nature = 'external',
                    type = 'real',
                    value = 0.9738404,
                    texname = '\\text{RCKM1x1}',
                    lhablock = 'VCKM',
                    lhacode = [ 1, 1 ])

RCKM1x2 = Parameter(name = 'RCKM1x2',
                    nature = 'external',
                    type = 'real',
                    value = 0.2271982,
                    texname = '\\text{RCKM1x2}',
                    lhablock = 'VCKM',
                    lhacode = [ 1, 2 ])

RCKM1x3 = Parameter(name = 'RCKM1x3',
                    nature = 'external',
                    type = 'real',
                    value = 0.002173989,
                    texname = '\\text{RCKM1x3}',
                    lhablock = 'VCKM',
                    lhacode = [ 1, 3 ])

RCKM2x1 = Parameter(name = 'RCKM2x1',
                    nature = 'external',
                    type = 'real',
                    value = -0.2270868,
                    texname = '\\text{RCKM2x1}',
                    lhablock = 'VCKM',
                    lhacode = [ 2, 1 ])

RCKM2x2 = Parameter(name = 'RCKM2x2',
                    nature = 'external',
                    type = 'real',
                    value = 0.9729587,
                    texname = '\\text{RCKM2x2}',
                    lhablock = 'VCKM',
                    lhacode = [ 2, 2 ])

RCKM2x3 = Parameter(name = 'RCKM2x3',
                    nature = 'external',
                    type = 'real',
                    value = 0.04222469,
                    texname = '\\text{RCKM2x3}',
                    lhablock = 'VCKM',
                    lhacode = [ 2, 3 ])

RCKM3x1 = Parameter(name = 'RCKM3x1',
                    nature = 'external',
                    type = 'real',
                    value = 0.007478279,
                    texname = '\\text{RCKM3x1}',
                    lhablock = 'VCKM',
                    lhacode = [ 3, 1 ])

RCKM3x2 = Parameter(name = 'RCKM3x2',
                    nature = 'external',
                    type = 'real',
                    value = -0.04161426,
                    texname = '\\text{RCKM3x2}',
                    lhablock = 'VCKM',
                    lhacode = [ 3, 2 ])

RCKM3x3 = Parameter(name = 'RCKM3x3',
                    nature = 'external',
                    type = 'real',
                    value = 0.9991002,
                    texname = '\\text{RCKM3x3}',
                    lhablock = 'VCKM',
                    lhacode = [ 3, 3 ])

RVV1x1 = Parameter(name = 'RVV1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.972557835,
                   texname = '\\text{RVV1x1}',
                   lhablock = 'VMIX',
                   lhacode = [ 1, 1 ])

RVV1x2 = Parameter(name = 'RVV1x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.232661249,
                   texname = '\\text{RVV1x2}',
                   lhablock = 'VMIX',
                   lhacode = [ 1, 2 ])

RVV2x1 = Parameter(name = 'RVV2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.232661249,
                   texname = '\\text{RVV2x1}',
                   lhablock = 'VMIX',
                   lhacode = [ 2, 1 ])

RVV2x2 = Parameter(name = 'RVV2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.972557835,
                   texname = '\\text{RVV2x2}',
                   lhablock = 'VMIX',
                   lhacode = [ 2, 2 ])

Ryd3x3 = Parameter(name = 'Ryd3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.138840206,
                   texname = '\\text{Ryd3x3}',
                   lhablock = 'YD',
                   lhacode = [ 3, 3 ])

Rye3x3 = Parameter(name = 'Rye3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.10089081,
                   texname = '\\text{Rye3x3}',
                   lhablock = 'YE',
                   lhacode = [ 3, 3 ])

Ryu3x3 = Parameter(name = 'Ryu3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.89284455,
                   texname = '\\text{Ryu3x3}',
                   lhablock = 'YU',
                   lhacode = [ 3, 3 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MW = Parameter(name = 'MW',
               nature = 'external',
               type = 'real',
               value = 79.8290131,
               texname = '\\text{MW}',
               lhablock = 'MASS',
               lhacode = [ 24 ])

Mneu1 = Parameter(name = 'Mneu1',
                  nature = 'external',
                  type = 'real',
                  value = 96.6880686,
                  texname = '\\text{Mneu1}',
                  lhablock = 'MASS',
                  lhacode = [ 1000022 ])

Mneu2 = Parameter(name = 'Mneu2',
                  nature = 'external',
                  type = 'real',
                  value = 181.088157,
                  texname = '\\text{Mneu2}',
                  lhablock = 'MASS',
                  lhacode = [ 1000023 ])

Mneu3 = Parameter(name = 'Mneu3',
                  nature = 'external',
                  type = 'real',
                  value = -363.756027,
                  texname = '\\text{Mneu3}',
                  lhablock = 'MASS',
                  lhacode = [ 1000025 ])

Mneu4 = Parameter(name = 'Mneu4',
                  nature = 'external',
                  type = 'real',
                  value = 381.729382,
                  texname = '\\text{Mneu4}',
                  lhablock = 'MASS',
                  lhacode = [ 1000035 ])

Mch1 = Parameter(name = 'Mch1',
                 nature = 'external',
                 type = 'real',
                 value = 181.696474,
                 texname = '\\text{Mch1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000024 ])

Mch2 = Parameter(name = 'Mch2',
                 nature = 'external',
                 type = 'real',
                 value = 379.93932,
                 texname = '\\text{Mch2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000037 ])

Mgo = Parameter(name = 'Mgo',
                nature = 'external',
                type = 'real',
                value = 607.713704,
                texname = '\\text{Mgo}',
                lhablock = 'MASS',
                lhacode = [ 1000021 ])

MH01 = Parameter(name = 'MH01',
                 nature = 'external',
                 type = 'real',
                 value = 110.899057,
                 texname = '\\text{MH01}',
                 lhablock = 'MASS',
                 lhacode = [ 25 ])

MH02 = Parameter(name = 'MH02',
                 nature = 'external',
                 type = 'real',
                 value = 399.960116,
                 texname = '\\text{MH02}',
                 lhablock = 'MASS',
                 lhacode = [ 35 ])

MA0 = Parameter(name = 'MA0',
                nature = 'external',
                type = 'real',
                value = 399.583917,
                texname = '\\text{MA0}',
                lhablock = 'MASS',
                lhacode = [ 36 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 407.879012,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 37 ])

Mta = Parameter(name = 'Mta',
                nature = 'external',
                type = 'real',
                value = 1.777,
                texname = '\\text{Mta}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 175.,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.88991651,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 5 ])

Msn1 = Parameter(name = 'Msn1',
                 nature = 'external',
                 type = 'real',
                 value = 185.258326,
                 texname = '\\text{Msn1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000012 ])

Msn2 = Parameter(name = 'Msn2',
                 nature = 'external',
                 type = 'real',
                 value = 185.258326,
                 texname = '\\text{Msn2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000014 ])

Msn3 = Parameter(name = 'Msn3',
                 nature = 'external',
                 type = 'real',
                 value = 184.708464,
                 texname = '\\text{Msn3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000016 ])

Msl1 = Parameter(name = 'Msl1',
                 nature = 'external',
                 type = 'real',
                 value = 202.91569,
                 texname = '\\text{Msl1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000011 ])

Msl2 = Parameter(name = 'Msl2',
                 nature = 'external',
                 type = 'real',
                 value = 202.91569,
                 texname = '\\text{Msl2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000013 ])

Msl3 = Parameter(name = 'Msl3',
                 nature = 'external',
                 type = 'real',
                 value = 134.490864,
                 texname = '\\text{Msl3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000015 ])

Msl4 = Parameter(name = 'Msl4',
                 nature = 'external',
                 type = 'real',
                 value = 144.102799,
                 texname = '\\text{Msl4}',
                 lhablock = 'MASS',
                 lhacode = [ 2000011 ])

Msl5 = Parameter(name = 'Msl5',
                 nature = 'external',
                 type = 'real',
                 value = 144.102799,
                 texname = '\\text{Msl5}',
                 lhablock = 'MASS',
                 lhacode = [ 2000013 ])

Msl6 = Parameter(name = 'Msl6',
                 nature = 'external',
                 type = 'real',
                 value = 206.867805,
                 texname = '\\text{Msl6}',
                 lhablock = 'MASS',
                 lhacode = [ 2000015 ])

Msu1 = Parameter(name = 'Msu1',
                 nature = 'external',
                 type = 'real',
                 value = 561.119014,
                 texname = '\\text{Msu1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000002 ])

Msu2 = Parameter(name = 'Msu2',
                 nature = 'external',
                 type = 'real',
                 value = 561.119014,
                 texname = '\\text{Msu2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000004 ])

Msu3 = Parameter(name = 'Msu3',
                 nature = 'external',
                 type = 'real',
                 value = 399.668493,
                 texname = '\\text{Msu3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000006 ])

Msu4 = Parameter(name = 'Msu4',
                 nature = 'external',
                 type = 'real',
                 value = 549.259265,
                 texname = '\\text{Msu4}',
                 lhablock = 'MASS',
                 lhacode = [ 2000002 ])

Msu5 = Parameter(name = 'Msu5',
                 nature = 'external',
                 type = 'real',
                 value = 549.259265,
                 texname = '\\text{Msu5}',
                 lhablock = 'MASS',
                 lhacode = [ 2000004 ])

Msu6 = Parameter(name = 'Msu6',
                 nature = 'external',
                 type = 'real',
                 value = 585.785818,
                 texname = '\\text{Msu6}',
                 lhablock = 'MASS',
                 lhacode = [ 2000006 ])

Msd1 = Parameter(name = 'Msd1',
                 nature = 'external',
                 type = 'real',
                 value = 568.441109,
                 texname = '\\text{Msd1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000001 ])

Msd2 = Parameter(name = 'Msd2',
                 nature = 'external',
                 type = 'real',
                 value = 568.441109,
                 texname = '\\text{Msd2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000003 ])

Msd3 = Parameter(name = 'Msd3',
                 nature = 'external',
                 type = 'real',
                 value = 513.065179,
                 texname = '\\text{Msd3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000005 ])

Msd4 = Parameter(name = 'Msd4',
                 nature = 'external',
                 type = 'real',
                 value = 545.228462,
                 texname = '\\text{Msd4}',
                 lhablock = 'MASS',
                 lhacode = [ 2000001 ])

Msd5 = Parameter(name = 'Msd5',
                 nature = 'external',
                 type = 'real',
                 value = 545.228462,
                 texname = '\\text{Msd5}',
                 lhablock = 'MASS',
                 lhacode = [ 2000003 ])

Msd6 = Parameter(name = 'Msd6',
                 nature = 'external',
                 type = 'real',
                 value = 543.726676,
                 texname = '\\text{Msd6}',
                 lhablock = 'MASS',
                 lhacode = [ 2000005 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.41143316,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.00282196,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

Wneu1 = Parameter(name = 'Wneu1',
                  nature = 'external',
                  type = 'real',
                  value = 0.0027770048,
                  texname = '\\text{Wneu1}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000022 ])

Wneu2 = Parameter(name = 'Wneu2',
                  nature = 'external',
                  type = 'real',
                  value = 0.0207770048,
                  texname = '\\text{Wneu2}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000023 ])

Wneu3 = Parameter(name = 'Wneu3',
                  nature = 'external',
                  type = 'real',
                  value = 1.91598495,
                  texname = '\\text{Wneu3}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000025 ])

Wneu4 = Parameter(name = 'Wneu4',
                  nature = 'external',
                  type = 'real',
                  value = 2.58585079,
                  texname = '\\text{Wneu4}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000035 ])

Wch1 = Parameter(name = 'Wch1',
                 nature = 'external',
                 type = 'real',
                 value = 0.0170414503,
                 texname = '\\text{Wch1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000024 ])

Wch2 = Parameter(name = 'Wch2',
                 nature = 'external',
                 type = 'real',
                 value = 2.4868951,
                 texname = '\\text{Wch2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000037 ])

Wgo = Parameter(name = 'Wgo',
                nature = 'external',
                type = 'real',
                value = 5.50675438,
                texname = '\\text{Wgo}',
                lhablock = 'DECAY',
                lhacode = [ 1000021 ])

WH01 = Parameter(name = 'WH01',
                 nature = 'external',
                 type = 'real',
                 value = 0.00198610799,
                 texname = '\\text{WH01}',
                 lhablock = 'DECAY',
                 lhacode = [ 25 ])

WH02 = Parameter(name = 'WH02',
                 nature = 'external',
                 type = 'real',
                 value = 0.574801389,
                 texname = '\\text{WH02}',
                 lhablock = 'DECAY',
                 lhacode = [ 35 ])

WA0 = Parameter(name = 'WA0',
                nature = 'external',
                type = 'real',
                value = 0.632178488,
                texname = '\\text{WA0}',
                lhablock = 'DECAY',
                lhacode = [ 36 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.546962813,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 37 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.56194983,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

Wsn1 = Parameter(name = 'Wsn1',
                 nature = 'external',
                 type = 'real',
                 value = 0.149881634,
                 texname = '\\text{Wsn1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000012 ])

Wsn2 = Parameter(name = 'Wsn2',
                 nature = 'external',
                 type = 'real',
                 value = 0.149881634,
                 texname = '\\text{Wsn2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000014 ])

Wsn3 = Parameter(name = 'Wsn3',
                 nature = 'external',
                 type = 'real',
                 value = 0.147518977,
                 texname = '\\text{Wsn3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000016 ])

Wsl1 = Parameter(name = 'Wsl1',
                 nature = 'external',
                 type = 'real',
                 value = 0.213682161,
                 texname = '\\text{Wsl1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000011 ])

Wsl2 = Parameter(name = 'Wsl2',
                 nature = 'external',
                 type = 'real',
                 value = 0.213682161,
                 texname = '\\text{Wsl2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000013 ])

Wsl3 = Parameter(name = 'Wsl3',
                 nature = 'external',
                 type = 'real',
                 value = 0.148327268,
                 texname = '\\text{Wsl3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000015 ])

Wsl4 = Parameter(name = 'Wsl4',
                 nature = 'external',
                 type = 'real',
                 value = 0.216121626,
                 texname = '\\text{Wsl4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000011 ])

Wsl5 = Parameter(name = 'Wsl5',
                 nature = 'external',
                 type = 'real',
                 value = 0.216121626,
                 texname = '\\text{Wsl5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000013 ])

Wsl6 = Parameter(name = 'Wsl6',
                 nature = 'external',
                 type = 'real',
                 value = 0.269906096,
                 texname = '\\text{Wsl6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000015 ])

Wsu1 = Parameter(name = 'Wsu1',
                 nature = 'external',
                 type = 'real',
                 value = 5.47719539,
                 texname = '\\text{Wsu1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000002 ])

Wsu2 = Parameter(name = 'Wsu2',
                 nature = 'external',
                 type = 'real',
                 value = 5.47719539,
                 texname = '\\text{Wsu2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000004 ])

Wsu3 = Parameter(name = 'Wsu3',
                 nature = 'external',
                 type = 'real',
                 value = 2.02159578,
                 texname = '\\text{Wsu3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000006 ])

Wsu4 = Parameter(name = 'Wsu4',
                 nature = 'external',
                 type = 'real',
                 value = 1.15297292,
                 texname = '\\text{Wsu4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000002 ])

Wsu5 = Parameter(name = 'Wsu5',
                 nature = 'external',
                 type = 'real',
                 value = 1.15297292,
                 texname = '\\text{Wsu5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000004 ])

Wsu6 = Parameter(name = 'Wsu6',
                 nature = 'external',
                 type = 'real',
                 value = 7.37313275,
                 texname = '\\text{Wsu6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000006 ])

Wsd1 = Parameter(name = 'Wsd1',
                 nature = 'external',
                 type = 'real',
                 value = 5.31278772,
                 texname = '\\text{Wsd1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000001 ])

Wsd2 = Parameter(name = 'Wsd2',
                 nature = 'external',
                 type = 'real',
                 value = 5.31278772,
                 texname = '\\text{Wsd2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000003 ])

Wsd3 = Parameter(name = 'Wsd3',
                 nature = 'external',
                 type = 'real',
                 value = 3.73627601,
                 texname = '\\text{Wsd3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000005 ])

Wsd4 = Parameter(name = 'Wsd4',
                 nature = 'external',
                 type = 'real',
                 value = 0.285812308,
                 texname = '\\text{Wsd4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000001 ])

Wsd5 = Parameter(name = 'Wsd5',
                 nature = 'external',
                 type = 'real',
                 value = 0.285812308,
                 texname = '\\text{Wsd5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000003 ])

Wsd6 = Parameter(name = 'Wsd6',
                 nature = 'external',
                 type = 'real',
                 value = 0.801566294,
                 texname = '\\text{Wsd6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000005 ])

beta = Parameter(name = 'beta',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.atan(tb)',
                 texname = '\\beta')

cw = Parameter(name = 'cw',
               nature = 'internal',
               type = 'real',
               value = 'MW/MZ',
               texname = 'c_w')

mD21x1 = Parameter(name = 'mD21x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmD21x1',
                   texname = '\\text{mD21x1}')

mD22x2 = Parameter(name = 'mD22x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmD22x2',
                   texname = '\\text{mD22x2}')

mD23x3 = Parameter(name = 'mD23x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmD23x3',
                   texname = '\\text{mD23x3}')

mE21x1 = Parameter(name = 'mE21x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmE21x1',
                   texname = '\\text{mE21x1}')

mE22x2 = Parameter(name = 'mE22x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmE22x2',
                   texname = '\\text{mE22x2}')

mE23x3 = Parameter(name = 'mE23x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmE23x3',
                   texname = '\\text{mE23x3}')

mL21x1 = Parameter(name = 'mL21x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmL21x1',
                   texname = '\\text{mL21x1}')

mL22x2 = Parameter(name = 'mL22x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmL22x2',
                   texname = '\\text{mL22x2}')

mL23x3 = Parameter(name = 'mL23x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmL23x3',
                   texname = '\\text{mL23x3}')

mQ21x1 = Parameter(name = 'mQ21x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmQ21x1',
                   texname = '\\text{mQ21x1}')

mQ22x2 = Parameter(name = 'mQ22x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmQ22x2',
                   texname = '\\text{mQ22x2}')

mQ23x3 = Parameter(name = 'mQ23x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmQ23x3',
                   texname = '\\text{mQ23x3}')

mU21x1 = Parameter(name = 'mU21x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmU21x1',
                   texname = '\\text{mU21x1}')

mU22x2 = Parameter(name = 'mU22x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmU22x2',
                   texname = '\\text{mU22x2}')

mU23x3 = Parameter(name = 'mU23x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmU23x3',
                   texname = '\\text{mU23x3}')

MUH = Parameter(name = 'MUH',
                nature = 'internal',
                type = 'complex',
                value = 'RMUH',
                texname = '\\mu')

Mx1 = Parameter(name = 'Mx1',
                nature = 'internal',
                type = 'complex',
                value = 'RMx1',
                texname = 'M_1')

Mx2 = Parameter(name = 'Mx2',
                nature = 'internal',
                type = 'complex',
                value = 'RMx2',
                texname = 'M_2')

Mx3 = Parameter(name = 'Mx3',
                nature = 'internal',
                type = 'complex',
                value = 'RMx3',
                texname = 'M_3')

NN1x1 = Parameter(name = 'NN1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN1x1',
                  texname = '\\text{NN1x1}')

NN1x2 = Parameter(name = 'NN1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN1x2',
                  texname = '\\text{NN1x2}')

NN1x3 = Parameter(name = 'NN1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN1x3',
                  texname = '\\text{NN1x3}')

NN1x4 = Parameter(name = 'NN1x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN1x4',
                  texname = '\\text{NN1x4}')

NN2x1 = Parameter(name = 'NN2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN2x1',
                  texname = '\\text{NN2x1}')

NN2x2 = Parameter(name = 'NN2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN2x2',
                  texname = '\\text{NN2x2}')

NN2x3 = Parameter(name = 'NN2x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN2x3',
                  texname = '\\text{NN2x3}')

NN2x4 = Parameter(name = 'NN2x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN2x4',
                  texname = '\\text{NN2x4}')

NN3x1 = Parameter(name = 'NN3x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN3x1',
                  texname = '\\text{NN3x1}')

NN3x2 = Parameter(name = 'NN3x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN3x2',
                  texname = '\\text{NN3x2}')

NN3x3 = Parameter(name = 'NN3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN3x3',
                  texname = '\\text{NN3x3}')

NN3x4 = Parameter(name = 'NN3x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN3x4',
                  texname = '\\text{NN3x4}')

NN4x1 = Parameter(name = 'NN4x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN4x1',
                  texname = '\\text{NN4x1}')

NN4x2 = Parameter(name = 'NN4x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN4x2',
                  texname = '\\text{NN4x2}')

NN4x3 = Parameter(name = 'NN4x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN4x3',
                  texname = '\\text{NN4x3}')

NN4x4 = Parameter(name = 'NN4x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN4x4',
                  texname = '\\text{NN4x4}')

Rd1x1 = Parameter(name = 'Rd1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd1x1',
                  texname = '\\text{Rd1x1}')

Rd2x2 = Parameter(name = 'Rd2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd2x2',
                  texname = '\\text{Rd2x2}')

Rd3x3 = Parameter(name = 'Rd3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd3x3',
                  texname = '\\text{Rd3x3}')

Rd3x6 = Parameter(name = 'Rd3x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd3x6',
                  texname = '\\text{Rd3x6}')

Rd4x4 = Parameter(name = 'Rd4x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd4x4',
                  texname = '\\text{Rd4x4}')

Rd5x5 = Parameter(name = 'Rd5x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd5x5',
                  texname = '\\text{Rd5x5}')

Rd6x3 = Parameter(name = 'Rd6x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd6x3',
                  texname = '\\text{Rd6x3}')

Rd6x6 = Parameter(name = 'Rd6x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd6x6',
                  texname = '\\text{Rd6x6}')

Rl1x1 = Parameter(name = 'Rl1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl1x1',
                  texname = '\\text{Rl1x1}')

Rl2x2 = Parameter(name = 'Rl2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl2x2',
                  texname = '\\text{Rl2x2}')

Rl3x3 = Parameter(name = 'Rl3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl3x3',
                  texname = '\\text{Rl3x3}')

Rl3x6 = Parameter(name = 'Rl3x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl3x6',
                  texname = '\\text{Rl3x6}')

Rl4x4 = Parameter(name = 'Rl4x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl4x4',
                  texname = '\\text{Rl4x4}')

Rl5x5 = Parameter(name = 'Rl5x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl5x5',
                  texname = '\\text{Rl5x5}')

Rl6x3 = Parameter(name = 'Rl6x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl6x3',
                  texname = '\\text{Rl6x3}')

Rl6x6 = Parameter(name = 'Rl6x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl6x6',
                  texname = '\\text{Rl6x6}')

Rn1x1 = Parameter(name = 'Rn1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn1x1',
                  texname = '\\text{Rn1x1}')

Rn2x2 = Parameter(name = 'Rn2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn2x2',
                  texname = '\\text{Rn2x2}')

Rn3x3 = Parameter(name = 'Rn3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn3x3',
                  texname = '\\text{Rn3x3}')

Ru1x1 = Parameter(name = 'Ru1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu1x1',
                  texname = '\\text{Ru1x1}')

Ru2x2 = Parameter(name = 'Ru2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu2x2',
                  texname = '\\text{Ru2x2}')

Ru3x3 = Parameter(name = 'Ru3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu3x3',
                  texname = '\\text{Ru3x3}')

Ru3x6 = Parameter(name = 'Ru3x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu3x6',
                  texname = '\\text{Ru3x6}')

Ru4x4 = Parameter(name = 'Ru4x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu4x4',
                  texname = '\\text{Ru4x4}')

Ru5x5 = Parameter(name = 'Ru5x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu5x5',
                  texname = '\\text{Ru5x5}')

Ru6x3 = Parameter(name = 'Ru6x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu6x3',
                  texname = '\\text{Ru6x3}')

Ru6x6 = Parameter(name = 'Ru6x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu6x6',
                  texname = '\\text{Ru6x6}')

UU1x1 = Parameter(name = 'UU1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RUU1x1',
                  texname = '\\text{UU1x1}')

UU1x2 = Parameter(name = 'UU1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RUU1x2',
                  texname = '\\text{UU1x2}')

UU2x1 = Parameter(name = 'UU2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RUU2x1',
                  texname = '\\text{UU2x1}')

UU2x2 = Parameter(name = 'UU2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RUU2x2',
                  texname = '\\text{UU2x2}')

VV1x1 = Parameter(name = 'VV1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RVV1x1',
                  texname = '\\text{VV1x1}')

VV1x2 = Parameter(name = 'VV1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RVV1x2',
                  texname = '\\text{VV1x2}')

VV2x1 = Parameter(name = 'VV2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RVV2x1',
                  texname = '\\text{VV2x1}')

VV2x2 = Parameter(name = 'VV2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RVV2x2',
                  texname = '\\text{VV2x2}')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(1/aEWM1)*cmath.sqrt(cmath.pi)',
               texname = 'e')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

LLLE1x2x1 = Parameter(name = 'LLLE1x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE1x2x1',
                      texname = '\\text{LLLE1x2x1}')

LLLE1x2x2 = Parameter(name = 'LLLE1x2x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE1x2x2',
                      texname = '\\text{LLLE1x2x2}')

LLLE1x2x3 = Parameter(name = 'LLLE1x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE1x2x3',
                      texname = '\\text{LLLE1x2x3}')

LLLE1x3x1 = Parameter(name = 'LLLE1x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE1x3x1',
                      texname = '\\text{LLLE1x3x1}')

LLLE1x3x2 = Parameter(name = 'LLLE1x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE1x3x2',
                      texname = '\\text{LLLE1x3x2}')

LLLE1x3x3 = Parameter(name = 'LLLE1x3x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE1x3x3',
                      texname = '\\text{LLLE1x3x3}')

LLLE2x1x1 = Parameter(name = 'LLLE2x1x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE2x1x1',
                      texname = '\\text{LLLE2x1x1}')

LLLE2x1x2 = Parameter(name = 'LLLE2x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE2x1x2',
                      texname = '\\text{LLLE2x1x2}')

LLLE2x1x3 = Parameter(name = 'LLLE2x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE2x1x3',
                      texname = '\\text{LLLE2x1x3}')

LLLE2x3x1 = Parameter(name = 'LLLE2x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE2x3x1',
                      texname = '\\text{LLLE2x3x1}')

LLLE2x3x2 = Parameter(name = 'LLLE2x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE2x3x2',
                      texname = '\\text{LLLE2x3x2}')

LLLE2x3x3 = Parameter(name = 'LLLE2x3x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE2x3x3',
                      texname = '\\text{LLLE2x3x3}')

LLLE3x1x1 = Parameter(name = 'LLLE3x1x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE3x1x1',
                      texname = '\\text{LLLE3x1x1}')

LLLE3x1x2 = Parameter(name = 'LLLE3x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE3x1x2',
                      texname = '\\text{LLLE3x1x2}')

LLLE3x1x3 = Parameter(name = 'LLLE3x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE3x1x3',
                      texname = '\\text{LLLE3x1x3}')

LLLE3x2x1 = Parameter(name = 'LLLE3x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE3x2x1',
                      texname = '\\text{LLLE3x2x1}')

LLLE3x2x2 = Parameter(name = 'LLLE3x2x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE3x2x2',
                      texname = '\\text{LLLE3x2x2}')

LLLE3x2x3 = Parameter(name = 'LLLE3x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLLE3x2x3',
                      texname = '\\text{LLLE3x2x3}')

LLQD1x1x1 = Parameter(name = 'LLQD1x1x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD1x1x1',
                      texname = '\\text{LLQD1x1x1}')

LLQD1x1x2 = Parameter(name = 'LLQD1x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD1x1x2',
                      texname = '\\text{LLQD1x1x2}')

LLQD1x1x3 = Parameter(name = 'LLQD1x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD1x1x3',
                      texname = '\\text{LLQD1x1x3}')

LLQD1x2x1 = Parameter(name = 'LLQD1x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD1x2x1',
                      texname = '\\text{LLQD1x2x1}')

LLQD1x2x2 = Parameter(name = 'LLQD1x2x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD1x2x2',
                      texname = '\\text{LLQD1x2x2}')

LLQD1x2x3 = Parameter(name = 'LLQD1x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD1x2x3',
                      texname = '\\text{LLQD1x2x3}')

LLQD1x3x1 = Parameter(name = 'LLQD1x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD1x3x1',
                      texname = '\\text{LLQD1x3x1}')

LLQD1x3x2 = Parameter(name = 'LLQD1x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD1x3x2',
                      texname = '\\text{LLQD1x3x2}')

LLQD1x3x3 = Parameter(name = 'LLQD1x3x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD1x3x3',
                      texname = '\\text{LLQD1x3x3}')

LLQD2x1x1 = Parameter(name = 'LLQD2x1x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD2x1x1',
                      texname = '\\text{LLQD2x1x1}')

LLQD2x1x2 = Parameter(name = 'LLQD2x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD2x1x2',
                      texname = '\\text{LLQD2x1x2}')

LLQD2x1x3 = Parameter(name = 'LLQD2x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD2x1x3',
                      texname = '\\text{LLQD2x1x3}')

LLQD2x2x1 = Parameter(name = 'LLQD2x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD2x2x1',
                      texname = '\\text{LLQD2x2x1}')

LLQD2x2x2 = Parameter(name = 'LLQD2x2x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD2x2x2',
                      texname = '\\text{LLQD2x2x2}')

LLQD2x2x3 = Parameter(name = 'LLQD2x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD2x2x3',
                      texname = '\\text{LLQD2x2x3}')

LLQD2x3x1 = Parameter(name = 'LLQD2x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD2x3x1',
                      texname = '\\text{LLQD2x3x1}')

LLQD2x3x2 = Parameter(name = 'LLQD2x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD2x3x2',
                      texname = '\\text{LLQD2x3x2}')

LLQD2x3x3 = Parameter(name = 'LLQD2x3x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD2x3x3',
                      texname = '\\text{LLQD2x3x3}')

LLQD3x1x1 = Parameter(name = 'LLQD3x1x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD3x1x1',
                      texname = '\\text{LLQD3x1x1}')

LLQD3x1x2 = Parameter(name = 'LLQD3x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD3x1x2',
                      texname = '\\text{LLQD3x1x2}')

LLQD3x1x3 = Parameter(name = 'LLQD3x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD3x1x3',
                      texname = '\\text{LLQD3x1x3}')

LLQD3x2x1 = Parameter(name = 'LLQD3x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD3x2x1',
                      texname = '\\text{LLQD3x2x1}')

LLQD3x2x2 = Parameter(name = 'LLQD3x2x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD3x2x2',
                      texname = '\\text{LLQD3x2x2}')

LLQD3x2x3 = Parameter(name = 'LLQD3x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD3x2x3',
                      texname = '\\text{LLQD3x2x3}')

LLQD3x3x1 = Parameter(name = 'LLQD3x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD3x3x1',
                      texname = '\\text{LLQD3x3x1}')

LLQD3x3x2 = Parameter(name = 'LLQD3x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD3x3x2',
                      texname = '\\text{LLQD3x3x2}')

LLQD3x3x3 = Parameter(name = 'LLQD3x3x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLQD3x3x3',
                      texname = '\\text{LLQD3x3x3}')

LUDD1x1x2 = Parameter(name = 'LUDD1x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD1x1x2',
                      texname = '\\text{LUDD1x1x2}')

LUDD1x1x3 = Parameter(name = 'LUDD1x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD1x1x3',
                      texname = '\\text{LUDD1x1x3}')

LUDD1x2x1 = Parameter(name = 'LUDD1x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD1x2x1',
                      texname = '\\text{LUDD1x2x1}')

LUDD1x2x3 = Parameter(name = 'LUDD1x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD1x2x3',
                      texname = '\\text{LUDD1x2x3}')

LUDD1x3x1 = Parameter(name = 'LUDD1x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD1x3x1',
                      texname = '\\text{LUDD1x3x1}')

LUDD1x3x2 = Parameter(name = 'LUDD1x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD1x3x2',
                      texname = '\\text{LUDD1x3x2}')

LUDD2x1x2 = Parameter(name = 'LUDD2x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD2x1x2',
                      texname = '\\text{LUDD2x1x2}')

LUDD2x1x3 = Parameter(name = 'LUDD2x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD2x1x3',
                      texname = '\\text{LUDD2x1x3}')

LUDD2x2x1 = Parameter(name = 'LUDD2x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD2x2x1',
                      texname = '\\text{LUDD2x2x1}')

LUDD2x2x3 = Parameter(name = 'LUDD2x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD2x2x3',
                      texname = '\\text{LUDD2x2x3}')

LUDD2x3x1 = Parameter(name = 'LUDD2x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD2x3x1',
                      texname = '\\text{LUDD2x3x1}')

LUDD2x3x2 = Parameter(name = 'LUDD2x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD2x3x2',
                      texname = '\\text{LUDD2x3x2}')

LUDD3x1x2 = Parameter(name = 'LUDD3x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD3x1x2',
                      texname = '\\text{LUDD3x1x2}')

LUDD3x1x3 = Parameter(name = 'LUDD3x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD3x1x3',
                      texname = '\\text{LUDD3x1x3}')

LUDD3x2x1 = Parameter(name = 'LUDD3x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD3x2x1',
                      texname = '\\text{LUDD3x2x1}')

LUDD3x2x3 = Parameter(name = 'LUDD3x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD3x2x3',
                      texname = '\\text{LUDD3x2x3}')

LUDD3x3x1 = Parameter(name = 'LUDD3x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD3x3x1',
                      texname = '\\text{LUDD3x3x1}')

LUDD3x3x2 = Parameter(name = 'LUDD3x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RLDD3x3x2',
                      texname = '\\text{LUDD3x3x2}')

td3x3 = Parameter(name = 'td3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rtd3x3',
                  texname = '\\text{td3x3}')

te3x3 = Parameter(name = 'te3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rte3x3',
                  texname = '\\text{te3x3}')

TLLE1x2x1 = Parameter(name = 'TLLE1x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE1x2x1',
                      texname = '\\text{TLLE1x2x1}')

TLLE1x2x2 = Parameter(name = 'TLLE1x2x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE1x2x2',
                      texname = '\\text{TLLE1x2x2}')

TLLE1x2x3 = Parameter(name = 'TLLE1x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE1x2x3',
                      texname = '\\text{TLLE1x2x3}')

TLLE1x3x1 = Parameter(name = 'TLLE1x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE1x3x1',
                      texname = '\\text{TLLE1x3x1}')

TLLE1x3x2 = Parameter(name = 'TLLE1x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE1x3x2',
                      texname = '\\text{TLLE1x3x2}')

TLLE1x3x3 = Parameter(name = 'TLLE1x3x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE1x3x3',
                      texname = '\\text{TLLE1x3x3}')

TLLE2x1x1 = Parameter(name = 'TLLE2x1x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE2x1x1',
                      texname = '\\text{TLLE2x1x1}')

TLLE2x1x2 = Parameter(name = 'TLLE2x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE2x1x2',
                      texname = '\\text{TLLE2x1x2}')

TLLE2x1x3 = Parameter(name = 'TLLE2x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE2x1x3',
                      texname = '\\text{TLLE2x1x3}')

TLLE2x3x1 = Parameter(name = 'TLLE2x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE2x3x1',
                      texname = '\\text{TLLE2x3x1}')

TLLE2x3x2 = Parameter(name = 'TLLE2x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE2x3x2',
                      texname = '\\text{TLLE2x3x2}')

TLLE2x3x3 = Parameter(name = 'TLLE2x3x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE2x3x3',
                      texname = '\\text{TLLE2x3x3}')

TLLE3x1x1 = Parameter(name = 'TLLE3x1x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE3x1x1',
                      texname = '\\text{TLLE3x1x1}')

TLLE3x1x2 = Parameter(name = 'TLLE3x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE3x1x2',
                      texname = '\\text{TLLE3x1x2}')

TLLE3x1x3 = Parameter(name = 'TLLE3x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE3x1x3',
                      texname = '\\text{TLLE3x1x3}')

TLLE3x2x1 = Parameter(name = 'TLLE3x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE3x2x1',
                      texname = '\\text{TLLE3x2x1}')

TLLE3x2x2 = Parameter(name = 'TLLE3x2x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE3x2x2',
                      texname = '\\text{TLLE3x2x2}')

TLLE3x2x3 = Parameter(name = 'TLLE3x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTLE3x2x3',
                      texname = '\\text{TLLE3x2x3}')

TLQD1x1x1 = Parameter(name = 'TLQD1x1x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD1x1x1',
                      texname = '\\text{TLQD1x1x1}')

TLQD1x1x2 = Parameter(name = 'TLQD1x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD1x1x2',
                      texname = '\\text{TLQD1x1x2}')

TLQD1x1x3 = Parameter(name = 'TLQD1x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD1x1x3',
                      texname = '\\text{TLQD1x1x3}')

TLQD1x2x1 = Parameter(name = 'TLQD1x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD1x2x1',
                      texname = '\\text{TLQD1x2x1}')

TLQD1x2x2 = Parameter(name = 'TLQD1x2x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD1x2x2',
                      texname = '\\text{TLQD1x2x2}')

TLQD1x2x3 = Parameter(name = 'TLQD1x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD1x2x3',
                      texname = '\\text{TLQD1x2x3}')

TLQD1x3x1 = Parameter(name = 'TLQD1x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD1x3x1',
                      texname = '\\text{TLQD1x3x1}')

TLQD1x3x2 = Parameter(name = 'TLQD1x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD1x3x2',
                      texname = '\\text{TLQD1x3x2}')

TLQD1x3x3 = Parameter(name = 'TLQD1x3x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD1x3x3',
                      texname = '\\text{TLQD1x3x3}')

TLQD2x1x1 = Parameter(name = 'TLQD2x1x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD2x1x1',
                      texname = '\\text{TLQD2x1x1}')

TLQD2x1x2 = Parameter(name = 'TLQD2x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD2x1x2',
                      texname = '\\text{TLQD2x1x2}')

TLQD2x1x3 = Parameter(name = 'TLQD2x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD2x1x3',
                      texname = '\\text{TLQD2x1x3}')

TLQD2x2x1 = Parameter(name = 'TLQD2x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD2x2x1',
                      texname = '\\text{TLQD2x2x1}')

TLQD2x2x2 = Parameter(name = 'TLQD2x2x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD2x2x2',
                      texname = '\\text{TLQD2x2x2}')

TLQD2x2x3 = Parameter(name = 'TLQD2x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD2x2x3',
                      texname = '\\text{TLQD2x2x3}')

TLQD2x3x1 = Parameter(name = 'TLQD2x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD2x3x1',
                      texname = '\\text{TLQD2x3x1}')

TLQD2x3x2 = Parameter(name = 'TLQD2x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD2x3x2',
                      texname = '\\text{TLQD2x3x2}')

TLQD2x3x3 = Parameter(name = 'TLQD2x3x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD2x3x3',
                      texname = '\\text{TLQD2x3x3}')

TLQD3x1x1 = Parameter(name = 'TLQD3x1x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD3x1x1',
                      texname = '\\text{TLQD3x1x1}')

TLQD3x1x2 = Parameter(name = 'TLQD3x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD3x1x2',
                      texname = '\\text{TLQD3x1x2}')

TLQD3x1x3 = Parameter(name = 'TLQD3x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD3x1x3',
                      texname = '\\text{TLQD3x1x3}')

TLQD3x2x1 = Parameter(name = 'TLQD3x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD3x2x1',
                      texname = '\\text{TLQD3x2x1}')

TLQD3x2x2 = Parameter(name = 'TLQD3x2x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD3x2x2',
                      texname = '\\text{TLQD3x2x2}')

TLQD3x2x3 = Parameter(name = 'TLQD3x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD3x2x3',
                      texname = '\\text{TLQD3x2x3}')

TLQD3x3x1 = Parameter(name = 'TLQD3x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD3x3x1',
                      texname = '\\text{TLQD3x3x1}')

TLQD3x3x2 = Parameter(name = 'TLQD3x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD3x3x2',
                      texname = '\\text{TLQD3x3x2}')

TLQD3x3x3 = Parameter(name = 'TLQD3x3x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTQD3x3x3',
                      texname = '\\text{TLQD3x3x3}')

tu3x3 = Parameter(name = 'tu3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rtu3x3',
                  texname = '\\text{tu3x3}')

TUDD1x1x2 = Parameter(name = 'TUDD1x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD1x1x2',
                      texname = '\\text{TUDD1x1x2}')

TUDD1x1x3 = Parameter(name = 'TUDD1x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD1x1x3',
                      texname = '\\text{TUDD1x1x3}')

TUDD1x2x1 = Parameter(name = 'TUDD1x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD1x2x1',
                      texname = '\\text{TUDD1x2x1}')

TUDD1x2x3 = Parameter(name = 'TUDD1x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD1x2x3',
                      texname = '\\text{TUDD1x2x3}')

TUDD1x3x1 = Parameter(name = 'TUDD1x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD1x3x1',
                      texname = '\\text{TUDD1x3x1}')

TUDD1x3x2 = Parameter(name = 'TUDD1x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD1x3x2',
                      texname = '\\text{TUDD1x3x2}')

TUDD2x1x2 = Parameter(name = 'TUDD2x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD2x1x2',
                      texname = '\\text{TUDD2x1x2}')

TUDD2x1x3 = Parameter(name = 'TUDD2x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD2x1x3',
                      texname = '\\text{TUDD2x1x3}')

TUDD2x2x1 = Parameter(name = 'TUDD2x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD2x2x1',
                      texname = '\\text{TUDD2x2x1}')

TUDD2x2x3 = Parameter(name = 'TUDD2x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD2x2x3',
                      texname = '\\text{TUDD2x2x3}')

TUDD2x3x1 = Parameter(name = 'TUDD2x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD2x3x1',
                      texname = '\\text{TUDD2x3x1}')

TUDD2x3x2 = Parameter(name = 'TUDD2x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD2x3x2',
                      texname = '\\text{TUDD2x3x2}')

TUDD3x1x2 = Parameter(name = 'TUDD3x1x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD3x1x2',
                      texname = '\\text{TUDD3x1x2}')

TUDD3x1x3 = Parameter(name = 'TUDD3x1x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD3x1x3',
                      texname = '\\text{TUDD3x1x3}')

TUDD3x2x1 = Parameter(name = 'TUDD3x2x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD3x2x1',
                      texname = '\\text{TUDD3x2x1}')

TUDD3x2x3 = Parameter(name = 'TUDD3x2x3',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD3x2x3',
                      texname = '\\text{TUDD3x2x3}')

TUDD3x3x1 = Parameter(name = 'TUDD3x3x1',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD3x3x1',
                      texname = '\\text{TUDD3x3x1}')

TUDD3x3x2 = Parameter(name = 'TUDD3x3x2',
                      nature = 'internal',
                      type = 'complex',
                      value = 'RTDD3x3x2',
                      texname = '\\text{TUDD3x3x2}')

yd3x3 = Parameter(name = 'yd3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ryd3x3',
                  texname = '\\text{yd3x3}')

ye3x3 = Parameter(name = 'ye3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rye3x3',
                  texname = '\\text{ye3x3}')

yu3x3 = Parameter(name = 'yu3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ryu3x3',
                  texname = '\\text{yu3x3}')

bb = Parameter(name = 'bb',
               nature = 'internal',
               type = 'complex',
               value = '((-mHd2 + mHu2 - MZ**2*cmath.cos(2*beta))*cmath.tan(2*beta))/2.',
               texname = 'b')

sw = Parameter(name = 'sw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 - cw**2)',
               texname = 's_w')

gp = Parameter(name = 'gp',
               nature = 'internal',
               type = 'real',
               value = 'ee/cw',
               texname = 'g\'')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = 'ee/sw',
               texname = 'g_w')

vev = Parameter(name = 'vev',
                nature = 'internal',
                type = 'real',
                value = '(2*cw*MZ*sw)/ee',
                texname = 'v')

vd = Parameter(name = 'vd',
               nature = 'internal',
               type = 'real',
               value = 'vev*cmath.cos(beta)',
               texname = 'v_d')

vu = Parameter(name = 'vu',
               nature = 'internal',
               type = 'real',
               value = 'vev*cmath.sin(beta)',
               texname = 'v_u')

I1a33 = Parameter(name = 'I1a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(yu3x3)',
                  texname = '\\text{I1a33}')

I10a113 = Parameter(name = 'I10a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x3x1)*complexconjugate(Rd3x6)',
                    texname = '\\text{I10a113}')

I10a115 = Parameter(name = 'I10a115',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x2x1)*complexconjugate(Rd5x5)',
                    texname = '\\text{I10a115}')

I10a116 = Parameter(name = 'I10a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x3x1)*complexconjugate(Rd6x6)',
                    texname = '\\text{I10a116}')

I10a123 = Parameter(name = 'I10a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x3x1)*complexconjugate(Rd3x6)',
                    texname = '\\text{I10a123}')

I10a125 = Parameter(name = 'I10a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x2x1)*complexconjugate(Rd5x5)',
                    texname = '\\text{I10a125}')

I10a126 = Parameter(name = 'I10a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x3x1)*complexconjugate(Rd6x6)',
                    texname = '\\text{I10a126}')

I10a133 = Parameter(name = 'I10a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x1)*complexconjugate(Rd3x6)',
                    texname = '\\text{I10a133}')

I10a135 = Parameter(name = 'I10a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x1)*complexconjugate(Rd5x5)',
                    texname = '\\text{I10a135}')

I10a136 = Parameter(name = 'I10a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x1)*complexconjugate(Rd6x6)',
                    texname = '\\text{I10a136}')

I10a213 = Parameter(name = 'I10a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x3x2)*complexconjugate(Rd3x6)',
                    texname = '\\text{I10a213}')

I10a214 = Parameter(name = 'I10a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x1x2)*complexconjugate(Rd4x4)',
                    texname = '\\text{I10a214}')

I10a216 = Parameter(name = 'I10a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x3x2)*complexconjugate(Rd6x6)',
                    texname = '\\text{I10a216}')

I10a223 = Parameter(name = 'I10a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x3x2)*complexconjugate(Rd3x6)',
                    texname = '\\text{I10a223}')

I10a224 = Parameter(name = 'I10a224',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x1x2)*complexconjugate(Rd4x4)',
                    texname = '\\text{I10a224}')

I10a226 = Parameter(name = 'I10a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x3x2)*complexconjugate(Rd6x6)',
                    texname = '\\text{I10a226}')

I10a233 = Parameter(name = 'I10a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x2)*complexconjugate(Rd3x6)',
                    texname = '\\text{I10a233}')

I10a234 = Parameter(name = 'I10a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x2)*complexconjugate(Rd4x4)',
                    texname = '\\text{I10a234}')

I10a236 = Parameter(name = 'I10a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x2)*complexconjugate(Rd6x6)',
                    texname = '\\text{I10a236}')

I10a314 = Parameter(name = 'I10a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x1x3)*complexconjugate(Rd4x4)',
                    texname = '\\text{I10a314}')

I10a315 = Parameter(name = 'I10a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x2x3)*complexconjugate(Rd5x5)',
                    texname = '\\text{I10a315}')

I10a324 = Parameter(name = 'I10a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x1x3)*complexconjugate(Rd4x4)',
                    texname = '\\text{I10a324}')

I10a325 = Parameter(name = 'I10a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x2x3)*complexconjugate(Rd5x5)',
                    texname = '\\text{I10a325}')

I10a334 = Parameter(name = 'I10a334',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x3)*complexconjugate(Rd4x4)',
                    texname = '\\text{I10a334}')

I10a335 = Parameter(name = 'I10a335',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x3)*complexconjugate(Rd5x5)',
                    texname = '\\text{I10a335}')

I100a343 = Parameter(name = 'I100a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x3x1)',
                     texname = '\\text{I100a343}')

I100a344 = Parameter(name = 'I100a344',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x3x1)',
                     texname = '\\text{I100a344}')

I100a345 = Parameter(name = 'I100a345',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x3x1)',
                     texname = '\\text{I100a345}')

I100a346 = Parameter(name = 'I100a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x3x1)',
                     texname = '\\text{I100a346}')

I100a353 = Parameter(name = 'I100a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x3x2)',
                     texname = '\\text{I100a353}')

I100a354 = Parameter(name = 'I100a354',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x3x2)',
                     texname = '\\text{I100a354}')

I100a355 = Parameter(name = 'I100a355',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x3x2)',
                     texname = '\\text{I100a355}')

I100a356 = Parameter(name = 'I100a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x3x2)',
                     texname = '\\text{I100a356}')

I100a433 = Parameter(name = 'I100a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x1x3)',
                     texname = '\\text{I100a433}')

I100a434 = Parameter(name = 'I100a434',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x1x3)',
                     texname = '\\text{I100a434}')

I100a435 = Parameter(name = 'I100a435',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x1x3)',
                     texname = '\\text{I100a435}')

I100a436 = Parameter(name = 'I100a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x1x3)',
                     texname = '\\text{I100a436}')

I100a453 = Parameter(name = 'I100a453',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x1x2)',
                     texname = '\\text{I100a453}')

I100a454 = Parameter(name = 'I100a454',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x1x2)',
                     texname = '\\text{I100a454}')

I100a455 = Parameter(name = 'I100a455',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x1x2)',
                     texname = '\\text{I100a455}')

I100a456 = Parameter(name = 'I100a456',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x1x2)',
                     texname = '\\text{I100a456}')

I100a463 = Parameter(name = 'I100a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x1x3)',
                     texname = '\\text{I100a463}')

I100a464 = Parameter(name = 'I100a464',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x1x3)',
                     texname = '\\text{I100a464}')

I100a465 = Parameter(name = 'I100a465',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x1x3)',
                     texname = '\\text{I100a465}')

I100a466 = Parameter(name = 'I100a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x1x3)',
                     texname = '\\text{I100a466}')

I100a533 = Parameter(name = 'I100a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x2x3)',
                     texname = '\\text{I100a533}')

I100a534 = Parameter(name = 'I100a534',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x2x3)',
                     texname = '\\text{I100a534}')

I100a535 = Parameter(name = 'I100a535',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x2x3)',
                     texname = '\\text{I100a535}')

I100a536 = Parameter(name = 'I100a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x2x3)',
                     texname = '\\text{I100a536}')

I100a543 = Parameter(name = 'I100a543',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x2x1)',
                     texname = '\\text{I100a543}')

I100a544 = Parameter(name = 'I100a544',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x2x1)',
                     texname = '\\text{I100a544}')

I100a545 = Parameter(name = 'I100a545',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x2x1)',
                     texname = '\\text{I100a545}')

I100a546 = Parameter(name = 'I100a546',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x2x1)',
                     texname = '\\text{I100a546}')

I100a563 = Parameter(name = 'I100a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x2x3)',
                     texname = '\\text{I100a563}')

I100a564 = Parameter(name = 'I100a564',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x2x3)',
                     texname = '\\text{I100a564}')

I100a565 = Parameter(name = 'I100a565',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x2x3)',
                     texname = '\\text{I100a565}')

I100a566 = Parameter(name = 'I100a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x2x3)',
                     texname = '\\text{I100a566}')

I100a643 = Parameter(name = 'I100a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x3x1)',
                     texname = '\\text{I100a643}')

I100a644 = Parameter(name = 'I100a644',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x3x1)',
                     texname = '\\text{I100a644}')

I100a645 = Parameter(name = 'I100a645',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x3x1)',
                     texname = '\\text{I100a645}')

I100a646 = Parameter(name = 'I100a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x3x1)',
                     texname = '\\text{I100a646}')

I100a653 = Parameter(name = 'I100a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x3x2)',
                     texname = '\\text{I100a653}')

I100a654 = Parameter(name = 'I100a654',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x3x2)',
                     texname = '\\text{I100a654}')

I100a655 = Parameter(name = 'I100a655',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x3x2)',
                     texname = '\\text{I100a655}')

I100a656 = Parameter(name = 'I100a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x3x2)',
                     texname = '\\text{I100a656}')

I101a343 = Parameter(name = 'I101a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x1x3)',
                     texname = '\\text{I101a343}')

I101a344 = Parameter(name = 'I101a344',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x1x3)',
                     texname = '\\text{I101a344}')

I101a345 = Parameter(name = 'I101a345',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x1x3)',
                     texname = '\\text{I101a345}')

I101a346 = Parameter(name = 'I101a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x1x3)',
                     texname = '\\text{I101a346}')

I101a353 = Parameter(name = 'I101a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x2x3)',
                     texname = '\\text{I101a353}')

I101a354 = Parameter(name = 'I101a354',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x2x3)',
                     texname = '\\text{I101a354}')

I101a355 = Parameter(name = 'I101a355',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x2x3)',
                     texname = '\\text{I101a355}')

I101a356 = Parameter(name = 'I101a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x2x3)',
                     texname = '\\text{I101a356}')

I101a433 = Parameter(name = 'I101a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x3x1)',
                     texname = '\\text{I101a433}')

I101a434 = Parameter(name = 'I101a434',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x3x1)',
                     texname = '\\text{I101a434}')

I101a435 = Parameter(name = 'I101a435',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x3x1)',
                     texname = '\\text{I101a435}')

I101a436 = Parameter(name = 'I101a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x3x1)',
                     texname = '\\text{I101a436}')

I101a453 = Parameter(name = 'I101a453',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x2x1)',
                     texname = '\\text{I101a453}')

I101a454 = Parameter(name = 'I101a454',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x2x1)',
                     texname = '\\text{I101a454}')

I101a455 = Parameter(name = 'I101a455',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x2x1)',
                     texname = '\\text{I101a455}')

I101a456 = Parameter(name = 'I101a456',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x2x1)',
                     texname = '\\text{I101a456}')

I101a463 = Parameter(name = 'I101a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x3x1)',
                     texname = '\\text{I101a463}')

I101a464 = Parameter(name = 'I101a464',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x3x1)',
                     texname = '\\text{I101a464}')

I101a465 = Parameter(name = 'I101a465',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x3x1)',
                     texname = '\\text{I101a465}')

I101a466 = Parameter(name = 'I101a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x3x1)',
                     texname = '\\text{I101a466}')

I101a533 = Parameter(name = 'I101a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x3x2)',
                     texname = '\\text{I101a533}')

I101a534 = Parameter(name = 'I101a534',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x3x2)',
                     texname = '\\text{I101a534}')

I101a535 = Parameter(name = 'I101a535',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x3x2)',
                     texname = '\\text{I101a535}')

I101a536 = Parameter(name = 'I101a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x3x2)',
                     texname = '\\text{I101a536}')

I101a543 = Parameter(name = 'I101a543',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x1x2)',
                     texname = '\\text{I101a543}')

I101a544 = Parameter(name = 'I101a544',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x1x2)',
                     texname = '\\text{I101a544}')

I101a545 = Parameter(name = 'I101a545',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x1x2)',
                     texname = '\\text{I101a545}')

I101a546 = Parameter(name = 'I101a546',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x1x2)',
                     texname = '\\text{I101a546}')

I101a563 = Parameter(name = 'I101a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x3x2)',
                     texname = '\\text{I101a563}')

I101a564 = Parameter(name = 'I101a564',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x3x2)',
                     texname = '\\text{I101a564}')

I101a565 = Parameter(name = 'I101a565',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x3x2)',
                     texname = '\\text{I101a565}')

I101a566 = Parameter(name = 'I101a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x3x2)',
                     texname = '\\text{I101a566}')

I101a643 = Parameter(name = 'I101a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x1x3)',
                     texname = '\\text{I101a643}')

I101a644 = Parameter(name = 'I101a644',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x1x3)',
                     texname = '\\text{I101a644}')

I101a645 = Parameter(name = 'I101a645',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x1x3)',
                     texname = '\\text{I101a645}')

I101a646 = Parameter(name = 'I101a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x1x3)',
                     texname = '\\text{I101a646}')

I101a653 = Parameter(name = 'I101a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru3x6)*complexconjugate(TUDD3x2x3)',
                     texname = '\\text{I101a653}')

I101a654 = Parameter(name = 'I101a654',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru4x4)*complexconjugate(TUDD1x2x3)',
                     texname = '\\text{I101a654}')

I101a655 = Parameter(name = 'I101a655',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru5x5)*complexconjugate(TUDD2x2x3)',
                     texname = '\\text{I101a655}')

I101a656 = Parameter(name = 'I101a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru6x6)*complexconjugate(TUDD3x2x3)',
                     texname = '\\text{I101a656}')

I102a433 = Parameter(name = 'I102a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru3x6)',
                     texname = '\\text{I102a433}')

I102a434 = Parameter(name = 'I102a434',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x3x1)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru4x4)',
                     texname = '\\text{I102a434}')

I102a435 = Parameter(name = 'I102a435',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x3x1)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru5x5)',
                     texname = '\\text{I102a435}')

I102a436 = Parameter(name = 'I102a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru6x6)',
                     texname = '\\text{I102a436}')

I102a463 = Parameter(name = 'I102a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru3x6)',
                     texname = '\\text{I102a463}')

I102a464 = Parameter(name = 'I102a464',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru4x4)',
                     texname = '\\text{I102a464}')

I102a465 = Parameter(name = 'I102a465',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru5x5)',
                     texname = '\\text{I102a465}')

I102a466 = Parameter(name = 'I102a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru6x6)',
                     texname = '\\text{I102a466}')

I102a533 = Parameter(name = 'I102a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)',
                     texname = '\\text{I102a533}')

I102a534 = Parameter(name = 'I102a534',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x3x2)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)',
                     texname = '\\text{I102a534}')

I102a535 = Parameter(name = 'I102a535',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x3x2)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)',
                     texname = '\\text{I102a535}')

I102a536 = Parameter(name = 'I102a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)',
                     texname = '\\text{I102a536}')

I102a563 = Parameter(name = 'I102a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru3x6)',
                     texname = '\\text{I102a563}')

I102a564 = Parameter(name = 'I102a564',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru4x4)',
                     texname = '\\text{I102a564}')

I102a565 = Parameter(name = 'I102a565',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru5x5)',
                     texname = '\\text{I102a565}')

I102a566 = Parameter(name = 'I102a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru6x6)',
                     texname = '\\text{I102a566}')

I103a343 = Parameter(name = 'I103a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru3x6)',
                     texname = '\\text{I103a343}')

I103a344 = Parameter(name = 'I103a344',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x3x1)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru4x4)',
                     texname = '\\text{I103a344}')

I103a345 = Parameter(name = 'I103a345',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x3x1)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru5x5)',
                     texname = '\\text{I103a345}')

I103a346 = Parameter(name = 'I103a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru6x6)',
                     texname = '\\text{I103a346}')

I103a353 = Parameter(name = 'I103a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)',
                     texname = '\\text{I103a353}')

I103a354 = Parameter(name = 'I103a354',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x3x2)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)',
                     texname = '\\text{I103a354}')

I103a355 = Parameter(name = 'I103a355',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x3x2)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)',
                     texname = '\\text{I103a355}')

I103a356 = Parameter(name = 'I103a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)',
                     texname = '\\text{I103a356}')

I103a643 = Parameter(name = 'I103a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru3x6)',
                     texname = '\\text{I103a643}')

I103a644 = Parameter(name = 'I103a644',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru4x4)',
                     texname = '\\text{I103a644}')

I103a645 = Parameter(name = 'I103a645',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru5x5)',
                     texname = '\\text{I103a645}')

I103a646 = Parameter(name = 'I103a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru6x6)',
                     texname = '\\text{I103a646}')

I103a653 = Parameter(name = 'I103a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru3x6)',
                     texname = '\\text{I103a653}')

I103a654 = Parameter(name = 'I103a654',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru4x4)',
                     texname = '\\text{I103a654}')

I103a655 = Parameter(name = 'I103a655',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru5x5)',
                     texname = '\\text{I103a655}')

I103a656 = Parameter(name = 'I103a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru6x6)',
                     texname = '\\text{I103a656}')

I104a433 = Parameter(name = 'I104a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru3x6)',
                     texname = '\\text{I104a433}')

I104a434 = Parameter(name = 'I104a434',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x1x3)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru4x4)',
                     texname = '\\text{I104a434}')

I104a435 = Parameter(name = 'I104a435',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x1x3)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru5x5)',
                     texname = '\\text{I104a435}')

I104a436 = Parameter(name = 'I104a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru6x6)',
                     texname = '\\text{I104a436}')

I104a463 = Parameter(name = 'I104a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru3x6)',
                     texname = '\\text{I104a463}')

I104a464 = Parameter(name = 'I104a464',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru4x4)',
                     texname = '\\text{I104a464}')

I104a465 = Parameter(name = 'I104a465',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru5x5)',
                     texname = '\\text{I104a465}')

I104a466 = Parameter(name = 'I104a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru6x6)',
                     texname = '\\text{I104a466}')

I104a533 = Parameter(name = 'I104a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)',
                     texname = '\\text{I104a533}')

I104a534 = Parameter(name = 'I104a534',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x2x3)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)',
                     texname = '\\text{I104a534}')

I104a535 = Parameter(name = 'I104a535',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x2x3)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)',
                     texname = '\\text{I104a535}')

I104a536 = Parameter(name = 'I104a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)',
                     texname = '\\text{I104a536}')

I104a563 = Parameter(name = 'I104a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru3x6)',
                     texname = '\\text{I104a563}')

I104a564 = Parameter(name = 'I104a564',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru4x4)',
                     texname = '\\text{I104a564}')

I104a565 = Parameter(name = 'I104a565',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru5x5)',
                     texname = '\\text{I104a565}')

I104a566 = Parameter(name = 'I104a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru6x6)',
                     texname = '\\text{I104a566}')

I105a343 = Parameter(name = 'I105a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru3x6)',
                     texname = '\\text{I105a343}')

I105a344 = Parameter(name = 'I105a344',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x1x3)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru4x4)',
                     texname = '\\text{I105a344}')

I105a345 = Parameter(name = 'I105a345',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x1x3)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru5x5)',
                     texname = '\\text{I105a345}')

I105a346 = Parameter(name = 'I105a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd3x3)*complexconjugate(Rd4x4)*complexconjugate(Ru6x6)',
                     texname = '\\text{I105a346}')

I105a353 = Parameter(name = 'I105a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru3x6)',
                     texname = '\\text{I105a353}')

I105a354 = Parameter(name = 'I105a354',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x2x3)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru4x4)',
                     texname = '\\text{I105a354}')

I105a355 = Parameter(name = 'I105a355',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x2x3)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru5x5)',
                     texname = '\\text{I105a355}')

I105a356 = Parameter(name = 'I105a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd3x3)*complexconjugate(Rd5x5)*complexconjugate(Ru6x6)',
                     texname = '\\text{I105a356}')

I105a643 = Parameter(name = 'I105a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru3x6)',
                     texname = '\\text{I105a643}')

I105a644 = Parameter(name = 'I105a644',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru4x4)',
                     texname = '\\text{I105a644}')

I105a645 = Parameter(name = 'I105a645',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru5x5)',
                     texname = '\\text{I105a645}')

I105a646 = Parameter(name = 'I105a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x3)*complexconjugate(Ru6x6)',
                     texname = '\\text{I105a646}')

I105a653 = Parameter(name = 'I105a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru3x6)',
                     texname = '\\text{I105a653}')

I105a654 = Parameter(name = 'I105a654',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD1x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru4x4)',
                     texname = '\\text{I105a654}')

I105a655 = Parameter(name = 'I105a655',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD2x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru5x5)',
                     texname = '\\text{I105a655}')

I105a656 = Parameter(name = 'I105a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yd3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x3)*complexconjugate(Ru6x6)',
                     texname = '\\text{I105a656}')

I106a343 = Parameter(name = 'I106a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru3x3)',
                     texname = '\\text{I106a343}')

I106a346 = Parameter(name = 'I106a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru6x3)',
                     texname = '\\text{I106a346}')

I106a353 = Parameter(name = 'I106a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru3x3)',
                     texname = '\\text{I106a353}')

I106a356 = Parameter(name = 'I106a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru6x3)',
                     texname = '\\text{I106a356}')

I106a433 = Parameter(name = 'I106a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru3x3)',
                     texname = '\\text{I106a433}')

I106a436 = Parameter(name = 'I106a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru6x3)',
                     texname = '\\text{I106a436}')

I106a453 = Parameter(name = 'I106a453',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x1)*complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru3x3)',
                     texname = '\\text{I106a453}')

I106a456 = Parameter(name = 'I106a456',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x1)*complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru6x3)',
                     texname = '\\text{I106a456}')

I106a463 = Parameter(name = 'I106a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru3x3)',
                     texname = '\\text{I106a463}')

I106a466 = Parameter(name = 'I106a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru6x3)',
                     texname = '\\text{I106a466}')

I106a533 = Parameter(name = 'I106a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru3x3)',
                     texname = '\\text{I106a533}')

I106a536 = Parameter(name = 'I106a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru6x3)',
                     texname = '\\text{I106a536}')

I106a543 = Parameter(name = 'I106a543',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x2)*complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru3x3)',
                     texname = '\\text{I106a543}')

I106a546 = Parameter(name = 'I106a546',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x2)*complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru6x3)',
                     texname = '\\text{I106a546}')

I106a563 = Parameter(name = 'I106a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru3x3)',
                     texname = '\\text{I106a563}')

I106a566 = Parameter(name = 'I106a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru6x3)',
                     texname = '\\text{I106a566}')

I106a643 = Parameter(name = 'I106a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru3x3)',
                     texname = '\\text{I106a643}')

I106a646 = Parameter(name = 'I106a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru6x3)',
                     texname = '\\text{I106a646}')

I106a653 = Parameter(name = 'I106a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru3x3)',
                     texname = '\\text{I106a653}')

I106a656 = Parameter(name = 'I106a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru6x3)',
                     texname = '\\text{I106a656}')

I107a343 = Parameter(name = 'I107a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru3x3)',
                     texname = '\\text{I107a343}')

I107a346 = Parameter(name = 'I107a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru6x3)',
                     texname = '\\text{I107a346}')

I107a353 = Parameter(name = 'I107a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru3x3)',
                     texname = '\\text{I107a353}')

I107a356 = Parameter(name = 'I107a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru6x3)',
                     texname = '\\text{I107a356}')

I107a433 = Parameter(name = 'I107a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru3x3)',
                     texname = '\\text{I107a433}')

I107a436 = Parameter(name = 'I107a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd3x6)*complexconjugate(Rd4x4)*complexconjugate(Ru6x3)',
                     texname = '\\text{I107a436}')

I107a453 = Parameter(name = 'I107a453',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x2)*complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru3x3)',
                     texname = '\\text{I107a453}')

I107a456 = Parameter(name = 'I107a456',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x2)*complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru6x3)',
                     texname = '\\text{I107a456}')

I107a463 = Parameter(name = 'I107a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru3x3)',
                     texname = '\\text{I107a463}')

I107a466 = Parameter(name = 'I107a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x1x3)*complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru6x3)',
                     texname = '\\text{I107a466}')

I107a533 = Parameter(name = 'I107a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru3x3)',
                     texname = '\\text{I107a533}')

I107a536 = Parameter(name = 'I107a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd3x6)*complexconjugate(Rd5x5)*complexconjugate(Ru6x3)',
                     texname = '\\text{I107a536}')

I107a543 = Parameter(name = 'I107a543',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x1)*complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru3x3)',
                     texname = '\\text{I107a543}')

I107a546 = Parameter(name = 'I107a546',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x1)*complexconjugate(Rd4x4)*complexconjugate(Rd5x5)*complexconjugate(Ru6x3)',
                     texname = '\\text{I107a546}')

I107a563 = Parameter(name = 'I107a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru3x3)',
                     texname = '\\text{I107a563}')

I107a566 = Parameter(name = 'I107a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x2x3)*complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru6x3)',
                     texname = '\\text{I107a566}')

I107a643 = Parameter(name = 'I107a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru3x3)',
                     texname = '\\text{I107a643}')

I107a646 = Parameter(name = 'I107a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x1)*complexconjugate(Rd4x4)*complexconjugate(Rd6x6)*complexconjugate(Ru6x3)',
                     texname = '\\text{I107a646}')

I107a653 = Parameter(name = 'I107a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru3x3)',
                     texname = '\\text{I107a653}')

I107a656 = Parameter(name = 'I107a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'yu3x3*complexconjugate(LUDD3x3x2)*complexconjugate(Rd5x5)*complexconjugate(Rd6x6)*complexconjugate(Ru6x3)',
                     texname = '\\text{I107a656}')

I108a33 = Parameter(name = 'I108a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(yu3x3)',
                    texname = '\\text{I108a33}')

I108a36 = Parameter(name = 'I108a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(yu3x3)',
                    texname = '\\text{I108a36}')

I109a33 = Parameter(name = 'I109a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*yu3x3',
                    texname = '\\text{I109a33}')

I109a36 = Parameter(name = 'I109a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*yu3x3',
                    texname = '\\text{I109a36}')

I11a113 = Parameter(name = 'I11a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x1x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I11a113}')

I11a115 = Parameter(name = 'I11a115',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x1x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I11a115}')

I11a116 = Parameter(name = 'I11a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x1x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I11a116}')

I11a123 = Parameter(name = 'I11a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x1x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I11a123}')

I11a125 = Parameter(name = 'I11a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x1x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I11a125}')

I11a126 = Parameter(name = 'I11a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x1x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I11a126}')

I11a133 = Parameter(name = 'I11a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I11a133}')

I11a135 = Parameter(name = 'I11a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I11a135}')

I11a136 = Parameter(name = 'I11a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I11a136}')

I11a213 = Parameter(name = 'I11a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x2x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I11a213}')

I11a214 = Parameter(name = 'I11a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x2x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I11a214}')

I11a216 = Parameter(name = 'I11a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x2x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I11a216}')

I11a223 = Parameter(name = 'I11a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x2x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I11a223}')

I11a224 = Parameter(name = 'I11a224',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x2x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I11a224}')

I11a226 = Parameter(name = 'I11a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x2x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I11a226}')

I11a233 = Parameter(name = 'I11a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I11a233}')

I11a234 = Parameter(name = 'I11a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I11a234}')

I11a236 = Parameter(name = 'I11a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I11a236}')

I11a314 = Parameter(name = 'I11a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x3x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I11a314}')

I11a315 = Parameter(name = 'I11a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x3x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I11a315}')

I11a324 = Parameter(name = 'I11a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x3x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I11a324}')

I11a325 = Parameter(name = 'I11a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x3x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I11a325}')

I11a334 = Parameter(name = 'I11a334',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I11a334}')

I11a335 = Parameter(name = 'I11a335',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I11a335}')

I110a11 = Parameter(name = 'I110a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru1x1',
                    texname = '\\text{I110a11}')

I110a22 = Parameter(name = 'I110a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru2x2',
                    texname = '\\text{I110a22}')

I110a33 = Parameter(name = 'I110a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3',
                    texname = '\\text{I110a33}')

I110a36 = Parameter(name = 'I110a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3',
                    texname = '\\text{I110a36}')

I111a33 = Parameter(name = 'I111a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(yd3x3)',
                    texname = '\\text{I111a33}')

I111a36 = Parameter(name = 'I111a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(yd3x3)',
                    texname = '\\text{I111a36}')

I112a33 = Parameter(name = 'I112a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*yu3x3',
                    texname = '\\text{I112a33}')

I112a36 = Parameter(name = 'I112a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*yu3x3',
                    texname = '\\text{I112a36}')

I113a111 = Parameter(name = 'I113a111',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru1x1*complexconjugate(LLQD1x1x1)',
                     texname = '\\text{I113a111}')

I113a112 = Parameter(name = 'I113a112',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru2x2*complexconjugate(LLQD1x2x1)',
                     texname = '\\text{I113a112}')

I113a113 = Parameter(name = 'I113a113',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru3x3*complexconjugate(LLQD1x3x1)',
                     texname = '\\text{I113a113}')

I113a116 = Parameter(name = 'I113a116',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru6x3*complexconjugate(LLQD1x3x1)',
                     texname = '\\text{I113a116}')

I113a121 = Parameter(name = 'I113a121',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru1x1*complexconjugate(LLQD1x1x2)',
                     texname = '\\text{I113a121}')

I113a122 = Parameter(name = 'I113a122',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru2x2*complexconjugate(LLQD1x2x2)',
                     texname = '\\text{I113a122}')

I113a123 = Parameter(name = 'I113a123',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru3x3*complexconjugate(LLQD1x3x2)',
                     texname = '\\text{I113a123}')

I113a126 = Parameter(name = 'I113a126',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru6x3*complexconjugate(LLQD1x3x2)',
                     texname = '\\text{I113a126}')

I113a131 = Parameter(name = 'I113a131',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru1x1*complexconjugate(LLQD1x1x3)',
                     texname = '\\text{I113a131}')

I113a132 = Parameter(name = 'I113a132',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru2x2*complexconjugate(LLQD1x2x3)',
                     texname = '\\text{I113a132}')

I113a133 = Parameter(name = 'I113a133',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru3x3*complexconjugate(LLQD1x3x3)',
                     texname = '\\text{I113a133}')

I113a136 = Parameter(name = 'I113a136',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru6x3*complexconjugate(LLQD1x3x3)',
                     texname = '\\text{I113a136}')

I113a211 = Parameter(name = 'I113a211',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru1x1*complexconjugate(LLQD2x1x1)',
                     texname = '\\text{I113a211}')

I113a212 = Parameter(name = 'I113a212',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru2x2*complexconjugate(LLQD2x2x1)',
                     texname = '\\text{I113a212}')

I113a213 = Parameter(name = 'I113a213',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru3x3*complexconjugate(LLQD2x3x1)',
                     texname = '\\text{I113a213}')

I113a216 = Parameter(name = 'I113a216',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru6x3*complexconjugate(LLQD2x3x1)',
                     texname = '\\text{I113a216}')

I113a221 = Parameter(name = 'I113a221',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru1x1*complexconjugate(LLQD2x1x2)',
                     texname = '\\text{I113a221}')

I113a222 = Parameter(name = 'I113a222',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru2x2*complexconjugate(LLQD2x2x2)',
                     texname = '\\text{I113a222}')

I113a223 = Parameter(name = 'I113a223',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru3x3*complexconjugate(LLQD2x3x2)',
                     texname = '\\text{I113a223}')

I113a226 = Parameter(name = 'I113a226',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru6x3*complexconjugate(LLQD2x3x2)',
                     texname = '\\text{I113a226}')

I113a231 = Parameter(name = 'I113a231',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru1x1*complexconjugate(LLQD2x1x3)',
                     texname = '\\text{I113a231}')

I113a232 = Parameter(name = 'I113a232',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru2x2*complexconjugate(LLQD2x2x3)',
                     texname = '\\text{I113a232}')

I113a233 = Parameter(name = 'I113a233',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru3x3*complexconjugate(LLQD2x3x3)',
                     texname = '\\text{I113a233}')

I113a236 = Parameter(name = 'I113a236',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru6x3*complexconjugate(LLQD2x3x3)',
                     texname = '\\text{I113a236}')

I113a311 = Parameter(name = 'I113a311',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru1x1*complexconjugate(LLQD3x1x1)',
                     texname = '\\text{I113a311}')

I113a312 = Parameter(name = 'I113a312',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru2x2*complexconjugate(LLQD3x2x1)',
                     texname = '\\text{I113a312}')

I113a313 = Parameter(name = 'I113a313',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru3x3*complexconjugate(LLQD3x3x1)',
                     texname = '\\text{I113a313}')

I113a316 = Parameter(name = 'I113a316',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru6x3*complexconjugate(LLQD3x3x1)',
                     texname = '\\text{I113a316}')

I113a321 = Parameter(name = 'I113a321',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru1x1*complexconjugate(LLQD3x1x2)',
                     texname = '\\text{I113a321}')

I113a322 = Parameter(name = 'I113a322',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru2x2*complexconjugate(LLQD3x2x2)',
                     texname = '\\text{I113a322}')

I113a323 = Parameter(name = 'I113a323',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru3x3*complexconjugate(LLQD3x3x2)',
                     texname = '\\text{I113a323}')

I113a326 = Parameter(name = 'I113a326',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru6x3*complexconjugate(LLQD3x3x2)',
                     texname = '\\text{I113a326}')

I113a331 = Parameter(name = 'I113a331',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru1x1*complexconjugate(LLQD3x1x3)',
                     texname = '\\text{I113a331}')

I113a332 = Parameter(name = 'I113a332',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru2x2*complexconjugate(LLQD3x2x3)',
                     texname = '\\text{I113a332}')

I113a333 = Parameter(name = 'I113a333',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru3x3*complexconjugate(LLQD3x3x3)',
                     texname = '\\text{I113a333}')

I113a336 = Parameter(name = 'I113a336',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Ru6x3*complexconjugate(LLQD3x3x3)',
                     texname = '\\text{I113a336}')

I114a123 = Parameter(name = 'I114a123',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x2*Ru3x6',
                     texname = '\\text{I114a123}')

I114a124 = Parameter(name = 'I114a124',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x1x2*Ru4x4',
                     texname = '\\text{I114a124}')

I114a125 = Parameter(name = 'I114a125',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x1x2*Ru5x5',
                     texname = '\\text{I114a125}')

I114a126 = Parameter(name = 'I114a126',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x2*Ru6x6',
                     texname = '\\text{I114a126}')

I114a133 = Parameter(name = 'I114a133',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Ru3x6',
                     texname = '\\text{I114a133}')

I114a134 = Parameter(name = 'I114a134',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x1x3*Ru4x4',
                     texname = '\\text{I114a134}')

I114a135 = Parameter(name = 'I114a135',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x1x3*Ru5x5',
                     texname = '\\text{I114a135}')

I114a136 = Parameter(name = 'I114a136',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Ru6x6',
                     texname = '\\text{I114a136}')

I114a213 = Parameter(name = 'I114a213',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x1*Ru3x6',
                     texname = '\\text{I114a213}')

I114a214 = Parameter(name = 'I114a214',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x2x1*Ru4x4',
                     texname = '\\text{I114a214}')

I114a215 = Parameter(name = 'I114a215',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x2x1*Ru5x5',
                     texname = '\\text{I114a215}')

I114a216 = Parameter(name = 'I114a216',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x1*Ru6x6',
                     texname = '\\text{I114a216}')

I114a233 = Parameter(name = 'I114a233',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Ru3x6',
                     texname = '\\text{I114a233}')

I114a234 = Parameter(name = 'I114a234',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x2x3*Ru4x4',
                     texname = '\\text{I114a234}')

I114a235 = Parameter(name = 'I114a235',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x2x3*Ru5x5',
                     texname = '\\text{I114a235}')

I114a236 = Parameter(name = 'I114a236',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Ru6x6',
                     texname = '\\text{I114a236}')

I114a313 = Parameter(name = 'I114a313',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Ru3x6',
                     texname = '\\text{I114a313}')

I114a314 = Parameter(name = 'I114a314',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x1*Ru4x4',
                     texname = '\\text{I114a314}')

I114a315 = Parameter(name = 'I114a315',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x1*Ru5x5',
                     texname = '\\text{I114a315}')

I114a316 = Parameter(name = 'I114a316',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Ru6x6',
                     texname = '\\text{I114a316}')

I114a323 = Parameter(name = 'I114a323',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Ru3x6',
                     texname = '\\text{I114a323}')

I114a324 = Parameter(name = 'I114a324',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x2*Ru4x4',
                     texname = '\\text{I114a324}')

I114a325 = Parameter(name = 'I114a325',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x2*Ru5x5',
                     texname = '\\text{I114a325}')

I114a326 = Parameter(name = 'I114a326',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Ru6x6',
                     texname = '\\text{I114a326}')

I115a123 = Parameter(name = 'I115a123',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x1*Ru3x6',
                     texname = '\\text{I115a123}')

I115a124 = Parameter(name = 'I115a124',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x2x1*Ru4x4',
                     texname = '\\text{I115a124}')

I115a125 = Parameter(name = 'I115a125',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x2x1*Ru5x5',
                     texname = '\\text{I115a125}')

I115a126 = Parameter(name = 'I115a126',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x1*Ru6x6',
                     texname = '\\text{I115a126}')

I115a133 = Parameter(name = 'I115a133',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Ru3x6',
                     texname = '\\text{I115a133}')

I115a134 = Parameter(name = 'I115a134',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x1*Ru4x4',
                     texname = '\\text{I115a134}')

I115a135 = Parameter(name = 'I115a135',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x1*Ru5x5',
                     texname = '\\text{I115a135}')

I115a136 = Parameter(name = 'I115a136',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Ru6x6',
                     texname = '\\text{I115a136}')

I115a213 = Parameter(name = 'I115a213',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x2*Ru3x6',
                     texname = '\\text{I115a213}')

I115a214 = Parameter(name = 'I115a214',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x1x2*Ru4x4',
                     texname = '\\text{I115a214}')

I115a215 = Parameter(name = 'I115a215',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x1x2*Ru5x5',
                     texname = '\\text{I115a215}')

I115a216 = Parameter(name = 'I115a216',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x2*Ru6x6',
                     texname = '\\text{I115a216}')

I115a233 = Parameter(name = 'I115a233',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Ru3x6',
                     texname = '\\text{I115a233}')

I115a234 = Parameter(name = 'I115a234',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x2*Ru4x4',
                     texname = '\\text{I115a234}')

I115a235 = Parameter(name = 'I115a235',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x2*Ru5x5',
                     texname = '\\text{I115a235}')

I115a236 = Parameter(name = 'I115a236',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Ru6x6',
                     texname = '\\text{I115a236}')

I115a313 = Parameter(name = 'I115a313',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Ru3x6',
                     texname = '\\text{I115a313}')

I115a314 = Parameter(name = 'I115a314',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x1x3*Ru4x4',
                     texname = '\\text{I115a314}')

I115a315 = Parameter(name = 'I115a315',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x1x3*Ru5x5',
                     texname = '\\text{I115a315}')

I115a316 = Parameter(name = 'I115a316',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Ru6x6',
                     texname = '\\text{I115a316}')

I115a323 = Parameter(name = 'I115a323',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Ru3x6',
                     texname = '\\text{I115a323}')

I115a324 = Parameter(name = 'I115a324',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x2x3*Ru4x4',
                     texname = '\\text{I115a324}')

I115a325 = Parameter(name = 'I115a325',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x2x3*Ru5x5',
                     texname = '\\text{I115a325}')

I115a326 = Parameter(name = 'I115a326',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Ru6x6',
                     texname = '\\text{I115a326}')

I116a11 = Parameter(name = 'I116a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru1x1*complexconjugate(Rd1x1)',
                    texname = '\\text{I116a11}')

I116a22 = Parameter(name = 'I116a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru2x2*complexconjugate(Rd2x2)',
                    texname = '\\text{I116a22}')

I116a33 = Parameter(name = 'I116a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I116a33}')

I116a36 = Parameter(name = 'I116a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I116a36}')

I116a63 = Parameter(name = 'I116a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I116a63}')

I116a66 = Parameter(name = 'I116a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I116a66}')

I117a33 = Parameter(name = 'I117a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                    texname = '\\text{I117a33}')

I117a36 = Parameter(name = 'I117a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                    texname = '\\text{I117a36}')

I117a63 = Parameter(name = 'I117a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                    texname = '\\text{I117a63}')

I117a66 = Parameter(name = 'I117a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                    texname = '\\text{I117a66}')

I118a33 = Parameter(name = 'I118a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Rd3x6)*complexconjugate(td3x3)',
                    texname = '\\text{I118a33}')

I118a36 = Parameter(name = 'I118a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Rd3x6)*complexconjugate(td3x3)',
                    texname = '\\text{I118a36}')

I118a63 = Parameter(name = 'I118a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)',
                    texname = '\\text{I118a63}')

I118a66 = Parameter(name = 'I118a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)',
                    texname = '\\text{I118a66}')

I119a33 = Parameter(name = 'I119a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*tu3x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I119a33}')

I119a36 = Parameter(name = 'I119a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*tu3x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I119a36}')

I119a63 = Parameter(name = 'I119a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*tu3x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I119a63}')

I119a66 = Parameter(name = 'I119a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*tu3x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I119a66}')

I12a33 = Parameter(name = 'I12a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(yd3x3)',
                   texname = '\\text{I12a33}')

I12a36 = Parameter(name = 'I12a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(yd3x3)',
                   texname = '\\text{I12a36}')

I120a33 = Parameter(name = 'I120a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I120a33}')

I120a36 = Parameter(name = 'I120a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I120a36}')

I120a63 = Parameter(name = 'I120a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*yd3x3*complexconjugate(Rd6x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I120a63}')

I120a66 = Parameter(name = 'I120a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*yd3x3*complexconjugate(Rd6x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I120a66}')

I121a33 = Parameter(name = 'I121a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)',
                    texname = '\\text{I121a33}')

I121a36 = Parameter(name = 'I121a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)',
                    texname = '\\text{I121a36}')

I121a63 = Parameter(name = 'I121a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*yu3x3*complexconjugate(Rd6x3)*complexconjugate(yu3x3)',
                    texname = '\\text{I121a63}')

I121a66 = Parameter(name = 'I121a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*yu3x3*complexconjugate(Rd6x3)*complexconjugate(yu3x3)',
                    texname = '\\text{I121a66}')

I122a33 = Parameter(name = 'I122a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*yu3x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I122a33}')

I122a36 = Parameter(name = 'I122a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*yu3x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I122a36}')

I122a63 = Parameter(name = 'I122a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*yu3x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I122a63}')

I122a66 = Parameter(name = 'I122a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*yu3x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I122a66}')

I123a33 = Parameter(name = 'I123a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*yu3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                    texname = '\\text{I123a33}')

I123a36 = Parameter(name = 'I123a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*yu3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                    texname = '\\text{I123a36}')

I123a63 = Parameter(name = 'I123a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                    texname = '\\text{I123a63}')

I123a66 = Parameter(name = 'I123a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                    texname = '\\text{I123a66}')

I124a11 = Parameter(name = 'I124a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru1x1*complexconjugate(Ru1x1)',
                    texname = '\\text{I124a11}')

I124a22 = Parameter(name = 'I124a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru2x2*complexconjugate(Ru2x2)',
                    texname = '\\text{I124a22}')

I124a33 = Parameter(name = 'I124a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I124a33}')

I124a36 = Parameter(name = 'I124a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I124a36}')

I124a63 = Parameter(name = 'I124a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I124a63}')

I124a66 = Parameter(name = 'I124a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I124a66}')

I125a33 = Parameter(name = 'I125a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*complexconjugate(Ru3x6)',
                    texname = '\\text{I125a33}')

I125a36 = Parameter(name = 'I125a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*complexconjugate(Ru3x6)',
                    texname = '\\text{I125a36}')

I125a44 = Parameter(name = 'I125a44',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru4x4*complexconjugate(Ru4x4)',
                    texname = '\\text{I125a44}')

I125a55 = Parameter(name = 'I125a55',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru5x5*complexconjugate(Ru5x5)',
                    texname = '\\text{I125a55}')

I125a63 = Parameter(name = 'I125a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*complexconjugate(Ru6x6)',
                    texname = '\\text{I125a63}')

I125a66 = Parameter(name = 'I125a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*complexconjugate(Ru6x6)',
                    texname = '\\text{I125a66}')

I126a33 = Parameter(name = 'I126a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I126a33}')

I126a36 = Parameter(name = 'I126a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I126a36}')

I126a63 = Parameter(name = 'I126a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I126a63}')

I126a66 = Parameter(name = 'I126a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I126a66}')

I127a33 = Parameter(name = 'I127a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Ru3x6)*complexconjugate(tu3x3)',
                    texname = '\\text{I127a33}')

I127a36 = Parameter(name = 'I127a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Ru3x6)*complexconjugate(tu3x3)',
                    texname = '\\text{I127a36}')

I127a63 = Parameter(name = 'I127a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Ru6x6)*complexconjugate(tu3x3)',
                    texname = '\\text{I127a63}')

I127a66 = Parameter(name = 'I127a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Ru6x6)*complexconjugate(tu3x3)',
                    texname = '\\text{I127a66}')

I128a33 = Parameter(name = 'I128a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*tu3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I128a33}')

I128a36 = Parameter(name = 'I128a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*tu3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I128a36}')

I128a63 = Parameter(name = 'I128a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*tu3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I128a63}')

I128a66 = Parameter(name = 'I128a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*tu3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I128a66}')

I129a33 = Parameter(name = 'I129a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*yu3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I129a33}')

I129a36 = Parameter(name = 'I129a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*yu3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I129a36}')

I129a63 = Parameter(name = 'I129a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*yu3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I129a63}')

I129a66 = Parameter(name = 'I129a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*yu3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I129a66}')

I13a33 = Parameter(name = 'I13a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3',
                   texname = '\\text{I13a33}')

I13a36 = Parameter(name = 'I13a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3',
                   texname = '\\text{I13a36}')

I130a33 = Parameter(name = 'I130a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)',
                    texname = '\\text{I130a33}')

I130a36 = Parameter(name = 'I130a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)',
                    texname = '\\text{I130a36}')

I130a63 = Parameter(name = 'I130a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*yu3x3*complexconjugate(Ru6x3)*complexconjugate(yu3x3)',
                    texname = '\\text{I130a63}')

I130a66 = Parameter(name = 'I130a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*yu3x3*complexconjugate(Ru6x3)*complexconjugate(yu3x3)',
                    texname = '\\text{I130a66}')

I131a33 = Parameter(name = 'I131a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*yu3x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I131a33}')

I131a36 = Parameter(name = 'I131a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*yu3x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I131a36}')

I131a63 = Parameter(name = 'I131a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I131a63}')

I131a66 = Parameter(name = 'I131a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I131a66}')

I132a311 = Parameter(name = 'I132a311',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru1x1*complexconjugate(Rd3x6)*complexconjugate(TLQD1x1x3)',
                     texname = '\\text{I132a311}')

I132a312 = Parameter(name = 'I132a312',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru2x2*complexconjugate(Rd3x6)*complexconjugate(TLQD1x2x3)',
                     texname = '\\text{I132a312}')

I132a313 = Parameter(name = 'I132a313',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru3x3*complexconjugate(Rd3x6)*complexconjugate(TLQD1x3x3)',
                     texname = '\\text{I132a313}')

I132a316 = Parameter(name = 'I132a316',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru6x3*complexconjugate(Rd3x6)*complexconjugate(TLQD1x3x3)',
                     texname = '\\text{I132a316}')

I132a321 = Parameter(name = 'I132a321',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru1x1*complexconjugate(Rd3x6)*complexconjugate(TLQD2x1x3)',
                     texname = '\\text{I132a321}')

I132a322 = Parameter(name = 'I132a322',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru2x2*complexconjugate(Rd3x6)*complexconjugate(TLQD2x2x3)',
                     texname = '\\text{I132a322}')

I132a323 = Parameter(name = 'I132a323',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru3x3*complexconjugate(Rd3x6)*complexconjugate(TLQD2x3x3)',
                     texname = '\\text{I132a323}')

I132a326 = Parameter(name = 'I132a326',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru6x3*complexconjugate(Rd3x6)*complexconjugate(TLQD2x3x3)',
                     texname = '\\text{I132a326}')

I132a331 = Parameter(name = 'I132a331',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru1x1*complexconjugate(Rd3x6)*complexconjugate(TLQD3x1x3)',
                     texname = '\\text{I132a331}')

I132a332 = Parameter(name = 'I132a332',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru2x2*complexconjugate(Rd3x6)*complexconjugate(TLQD3x2x3)',
                     texname = '\\text{I132a332}')

I132a333 = Parameter(name = 'I132a333',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru3x3*complexconjugate(Rd3x6)*complexconjugate(TLQD3x3x3)',
                     texname = '\\text{I132a333}')

I132a336 = Parameter(name = 'I132a336',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru6x3*complexconjugate(Rd3x6)*complexconjugate(TLQD3x3x3)',
                     texname = '\\text{I132a336}')

I132a361 = Parameter(name = 'I132a361',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru1x1*complexconjugate(Rd3x6)*complexconjugate(TLQD3x1x3)',
                     texname = '\\text{I132a361}')

I132a362 = Parameter(name = 'I132a362',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru2x2*complexconjugate(Rd3x6)*complexconjugate(TLQD3x2x3)',
                     texname = '\\text{I132a362}')

I132a363 = Parameter(name = 'I132a363',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru3x3*complexconjugate(Rd3x6)*complexconjugate(TLQD3x3x3)',
                     texname = '\\text{I132a363}')

I132a366 = Parameter(name = 'I132a366',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru6x3*complexconjugate(Rd3x6)*complexconjugate(TLQD3x3x3)',
                     texname = '\\text{I132a366}')

I132a411 = Parameter(name = 'I132a411',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru1x1*complexconjugate(Rd4x4)*complexconjugate(TLQD1x1x1)',
                     texname = '\\text{I132a411}')

I132a412 = Parameter(name = 'I132a412',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru2x2*complexconjugate(Rd4x4)*complexconjugate(TLQD1x2x1)',
                     texname = '\\text{I132a412}')

I132a413 = Parameter(name = 'I132a413',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru3x3*complexconjugate(Rd4x4)*complexconjugate(TLQD1x3x1)',
                     texname = '\\text{I132a413}')

I132a416 = Parameter(name = 'I132a416',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru6x3*complexconjugate(Rd4x4)*complexconjugate(TLQD1x3x1)',
                     texname = '\\text{I132a416}')

I132a421 = Parameter(name = 'I132a421',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru1x1*complexconjugate(Rd4x4)*complexconjugate(TLQD2x1x1)',
                     texname = '\\text{I132a421}')

I132a422 = Parameter(name = 'I132a422',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru2x2*complexconjugate(Rd4x4)*complexconjugate(TLQD2x2x1)',
                     texname = '\\text{I132a422}')

I132a423 = Parameter(name = 'I132a423',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru3x3*complexconjugate(Rd4x4)*complexconjugate(TLQD2x3x1)',
                     texname = '\\text{I132a423}')

I132a426 = Parameter(name = 'I132a426',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru6x3*complexconjugate(Rd4x4)*complexconjugate(TLQD2x3x1)',
                     texname = '\\text{I132a426}')

I132a431 = Parameter(name = 'I132a431',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru1x1*complexconjugate(Rd4x4)*complexconjugate(TLQD3x1x1)',
                     texname = '\\text{I132a431}')

I132a432 = Parameter(name = 'I132a432',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru2x2*complexconjugate(Rd4x4)*complexconjugate(TLQD3x2x1)',
                     texname = '\\text{I132a432}')

I132a433 = Parameter(name = 'I132a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru3x3*complexconjugate(Rd4x4)*complexconjugate(TLQD3x3x1)',
                     texname = '\\text{I132a433}')

I132a436 = Parameter(name = 'I132a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru6x3*complexconjugate(Rd4x4)*complexconjugate(TLQD3x3x1)',
                     texname = '\\text{I132a436}')

I132a461 = Parameter(name = 'I132a461',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru1x1*complexconjugate(Rd4x4)*complexconjugate(TLQD3x1x1)',
                     texname = '\\text{I132a461}')

I132a462 = Parameter(name = 'I132a462',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru2x2*complexconjugate(Rd4x4)*complexconjugate(TLQD3x2x1)',
                     texname = '\\text{I132a462}')

I132a463 = Parameter(name = 'I132a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru3x3*complexconjugate(Rd4x4)*complexconjugate(TLQD3x3x1)',
                     texname = '\\text{I132a463}')

I132a466 = Parameter(name = 'I132a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru6x3*complexconjugate(Rd4x4)*complexconjugate(TLQD3x3x1)',
                     texname = '\\text{I132a466}')

I132a511 = Parameter(name = 'I132a511',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru1x1*complexconjugate(Rd5x5)*complexconjugate(TLQD1x1x2)',
                     texname = '\\text{I132a511}')

I132a512 = Parameter(name = 'I132a512',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru2x2*complexconjugate(Rd5x5)*complexconjugate(TLQD1x2x2)',
                     texname = '\\text{I132a512}')

I132a513 = Parameter(name = 'I132a513',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru3x3*complexconjugate(Rd5x5)*complexconjugate(TLQD1x3x2)',
                     texname = '\\text{I132a513}')

I132a516 = Parameter(name = 'I132a516',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru6x3*complexconjugate(Rd5x5)*complexconjugate(TLQD1x3x2)',
                     texname = '\\text{I132a516}')

I132a521 = Parameter(name = 'I132a521',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru1x1*complexconjugate(Rd5x5)*complexconjugate(TLQD2x1x2)',
                     texname = '\\text{I132a521}')

I132a522 = Parameter(name = 'I132a522',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru2x2*complexconjugate(Rd5x5)*complexconjugate(TLQD2x2x2)',
                     texname = '\\text{I132a522}')

I132a523 = Parameter(name = 'I132a523',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru3x3*complexconjugate(Rd5x5)*complexconjugate(TLQD2x3x2)',
                     texname = '\\text{I132a523}')

I132a526 = Parameter(name = 'I132a526',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru6x3*complexconjugate(Rd5x5)*complexconjugate(TLQD2x3x2)',
                     texname = '\\text{I132a526}')

I132a531 = Parameter(name = 'I132a531',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru1x1*complexconjugate(Rd5x5)*complexconjugate(TLQD3x1x2)',
                     texname = '\\text{I132a531}')

I132a532 = Parameter(name = 'I132a532',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru2x2*complexconjugate(Rd5x5)*complexconjugate(TLQD3x2x2)',
                     texname = '\\text{I132a532}')

I132a533 = Parameter(name = 'I132a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru3x3*complexconjugate(Rd5x5)*complexconjugate(TLQD3x3x2)',
                     texname = '\\text{I132a533}')

I132a536 = Parameter(name = 'I132a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru6x3*complexconjugate(Rd5x5)*complexconjugate(TLQD3x3x2)',
                     texname = '\\text{I132a536}')

I132a561 = Parameter(name = 'I132a561',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru1x1*complexconjugate(Rd5x5)*complexconjugate(TLQD3x1x2)',
                     texname = '\\text{I132a561}')

I132a562 = Parameter(name = 'I132a562',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru2x2*complexconjugate(Rd5x5)*complexconjugate(TLQD3x2x2)',
                     texname = '\\text{I132a562}')

I132a563 = Parameter(name = 'I132a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru3x3*complexconjugate(Rd5x5)*complexconjugate(TLQD3x3x2)',
                     texname = '\\text{I132a563}')

I132a566 = Parameter(name = 'I132a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru6x3*complexconjugate(Rd5x5)*complexconjugate(TLQD3x3x2)',
                     texname = '\\text{I132a566}')

I132a611 = Parameter(name = 'I132a611',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru1x1*complexconjugate(Rd6x6)*complexconjugate(TLQD1x1x3)',
                     texname = '\\text{I132a611}')

I132a612 = Parameter(name = 'I132a612',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru2x2*complexconjugate(Rd6x6)*complexconjugate(TLQD1x2x3)',
                     texname = '\\text{I132a612}')

I132a613 = Parameter(name = 'I132a613',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(TLQD1x3x3)',
                     texname = '\\text{I132a613}')

I132a616 = Parameter(name = 'I132a616',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru6x3*complexconjugate(Rd6x6)*complexconjugate(TLQD1x3x3)',
                     texname = '\\text{I132a616}')

I132a621 = Parameter(name = 'I132a621',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru1x1*complexconjugate(Rd6x6)*complexconjugate(TLQD2x1x3)',
                     texname = '\\text{I132a621}')

I132a622 = Parameter(name = 'I132a622',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru2x2*complexconjugate(Rd6x6)*complexconjugate(TLQD2x2x3)',
                     texname = '\\text{I132a622}')

I132a623 = Parameter(name = 'I132a623',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(TLQD2x3x3)',
                     texname = '\\text{I132a623}')

I132a626 = Parameter(name = 'I132a626',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru6x3*complexconjugate(Rd6x6)*complexconjugate(TLQD2x3x3)',
                     texname = '\\text{I132a626}')

I132a631 = Parameter(name = 'I132a631',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru1x1*complexconjugate(Rd6x6)*complexconjugate(TLQD3x1x3)',
                     texname = '\\text{I132a631}')

I132a632 = Parameter(name = 'I132a632',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru2x2*complexconjugate(Rd6x6)*complexconjugate(TLQD3x2x3)',
                     texname = '\\text{I132a632}')

I132a633 = Parameter(name = 'I132a633',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(TLQD3x3x3)',
                     texname = '\\text{I132a633}')

I132a636 = Parameter(name = 'I132a636',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru6x3*complexconjugate(Rd6x6)*complexconjugate(TLQD3x3x3)',
                     texname = '\\text{I132a636}')

I132a661 = Parameter(name = 'I132a661',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru1x1*complexconjugate(Rd6x6)*complexconjugate(TLQD3x1x3)',
                     texname = '\\text{I132a661}')

I132a662 = Parameter(name = 'I132a662',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru2x2*complexconjugate(Rd6x6)*complexconjugate(TLQD3x2x3)',
                     texname = '\\text{I132a662}')

I132a663 = Parameter(name = 'I132a663',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(TLQD3x3x3)',
                     texname = '\\text{I132a663}')

I132a666 = Parameter(name = 'I132a666',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru6x3*complexconjugate(Rd6x6)*complexconjugate(TLQD3x3x3)',
                     texname = '\\text{I132a666}')

I133a311 = Parameter(name = 'I133a311',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru1x1*yd3x3*complexconjugate(LLQD1x1x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a311}')

I133a312 = Parameter(name = 'I133a312',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru2x2*yd3x3*complexconjugate(LLQD1x2x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a312}')

I133a313 = Parameter(name = 'I133a313',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru3x3*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a313}')

I133a316 = Parameter(name = 'I133a316',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru6x3*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a316}')

I133a321 = Parameter(name = 'I133a321',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru1x1*yd3x3*complexconjugate(LLQD2x1x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a321}')

I133a322 = Parameter(name = 'I133a322',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru2x2*yd3x3*complexconjugate(LLQD2x2x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a322}')

I133a323 = Parameter(name = 'I133a323',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru3x3*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a323}')

I133a326 = Parameter(name = 'I133a326',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru6x3*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a326}')

I133a331 = Parameter(name = 'I133a331',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru1x1*yd3x3*complexconjugate(LLQD3x1x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a331}')

I133a332 = Parameter(name = 'I133a332',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru2x2*yd3x3*complexconjugate(LLQD3x2x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a332}')

I133a333 = Parameter(name = 'I133a333',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a333}')

I133a336 = Parameter(name = 'I133a336',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru6x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a336}')

I133a361 = Parameter(name = 'I133a361',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru1x1*yd3x3*complexconjugate(LLQD3x1x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a361}')

I133a362 = Parameter(name = 'I133a362',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru2x2*yd3x3*complexconjugate(LLQD3x2x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a362}')

I133a363 = Parameter(name = 'I133a363',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a363}')

I133a366 = Parameter(name = 'I133a366',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru6x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x3)',
                     texname = '\\text{I133a366}')

I133a611 = Parameter(name = 'I133a611',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru1x1*yd3x3*complexconjugate(LLQD1x1x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a611}')

I133a612 = Parameter(name = 'I133a612',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru2x2*yd3x3*complexconjugate(LLQD1x2x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a612}')

I133a613 = Parameter(name = 'I133a613',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru3x3*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a613}')

I133a616 = Parameter(name = 'I133a616',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru6x3*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a616}')

I133a621 = Parameter(name = 'I133a621',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru1x1*yd3x3*complexconjugate(LLQD2x1x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a621}')

I133a622 = Parameter(name = 'I133a622',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru2x2*yd3x3*complexconjugate(LLQD2x2x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a622}')

I133a623 = Parameter(name = 'I133a623',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru3x3*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a623}')

I133a626 = Parameter(name = 'I133a626',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru6x3*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a626}')

I133a631 = Parameter(name = 'I133a631',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru1x1*yd3x3*complexconjugate(LLQD3x1x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a631}')

I133a632 = Parameter(name = 'I133a632',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru2x2*yd3x3*complexconjugate(LLQD3x2x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a632}')

I133a633 = Parameter(name = 'I133a633',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a633}')

I133a636 = Parameter(name = 'I133a636',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru6x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a636}')

I133a661 = Parameter(name = 'I133a661',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru1x1*yd3x3*complexconjugate(LLQD3x1x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a661}')

I133a662 = Parameter(name = 'I133a662',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru2x2*yd3x3*complexconjugate(LLQD3x2x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a662}')

I133a663 = Parameter(name = 'I133a663',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a663}')

I133a666 = Parameter(name = 'I133a666',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru6x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x3)',
                     texname = '\\text{I133a666}')

I134a331 = Parameter(name = 'I134a331',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru1x1*ye3x3*complexconjugate(LLQD3x1x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I134a331}')

I134a332 = Parameter(name = 'I134a332',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru2x2*ye3x3*complexconjugate(LLQD3x2x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I134a332}')

I134a333 = Parameter(name = 'I134a333',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru3x3*ye3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I134a333}')

I134a336 = Parameter(name = 'I134a336',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru6x3*ye3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I134a336}')

I134a361 = Parameter(name = 'I134a361',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru1x1*ye3x3*complexconjugate(LLQD3x1x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I134a361}')

I134a362 = Parameter(name = 'I134a362',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru2x2*ye3x3*complexconjugate(LLQD3x2x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I134a362}')

I134a363 = Parameter(name = 'I134a363',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru3x3*ye3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I134a363}')

I134a366 = Parameter(name = 'I134a366',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru6x3*ye3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I134a366}')

I134a431 = Parameter(name = 'I134a431',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru1x1*ye3x3*complexconjugate(LLQD3x1x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I134a431}')

I134a432 = Parameter(name = 'I134a432',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru2x2*ye3x3*complexconjugate(LLQD3x2x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I134a432}')

I134a433 = Parameter(name = 'I134a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru3x3*ye3x3*complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I134a433}')

I134a436 = Parameter(name = 'I134a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru6x3*ye3x3*complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I134a436}')

I134a461 = Parameter(name = 'I134a461',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru1x1*ye3x3*complexconjugate(LLQD3x1x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I134a461}')

I134a462 = Parameter(name = 'I134a462',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru2x2*ye3x3*complexconjugate(LLQD3x2x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I134a462}')

I134a463 = Parameter(name = 'I134a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru3x3*ye3x3*complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I134a463}')

I134a466 = Parameter(name = 'I134a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru6x3*ye3x3*complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I134a466}')

I134a531 = Parameter(name = 'I134a531',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru1x1*ye3x3*complexconjugate(LLQD3x1x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I134a531}')

I134a532 = Parameter(name = 'I134a532',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru2x2*ye3x3*complexconjugate(LLQD3x2x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I134a532}')

I134a533 = Parameter(name = 'I134a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru3x3*ye3x3*complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I134a533}')

I134a536 = Parameter(name = 'I134a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru6x3*ye3x3*complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I134a536}')

I134a561 = Parameter(name = 'I134a561',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru1x1*ye3x3*complexconjugate(LLQD3x1x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I134a561}')

I134a562 = Parameter(name = 'I134a562',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru2x2*ye3x3*complexconjugate(LLQD3x2x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I134a562}')

I134a563 = Parameter(name = 'I134a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru3x3*ye3x3*complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I134a563}')

I134a566 = Parameter(name = 'I134a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru6x3*ye3x3*complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I134a566}')

I134a631 = Parameter(name = 'I134a631',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru1x1*ye3x3*complexconjugate(LLQD3x1x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I134a631}')

I134a632 = Parameter(name = 'I134a632',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru2x2*ye3x3*complexconjugate(LLQD3x2x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I134a632}')

I134a633 = Parameter(name = 'I134a633',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru3x3*ye3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I134a633}')

I134a636 = Parameter(name = 'I134a636',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x6*Ru6x3*ye3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I134a636}')

I134a661 = Parameter(name = 'I134a661',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru1x1*ye3x3*complexconjugate(LLQD3x1x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I134a661}')

I134a662 = Parameter(name = 'I134a662',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru2x2*ye3x3*complexconjugate(LLQD3x2x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I134a662}')

I134a663 = Parameter(name = 'I134a663',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru3x3*ye3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I134a663}')

I134a666 = Parameter(name = 'I134a666',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x6*Ru6x3*ye3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I134a666}')

I135a313 = Parameter(name = 'I135a313',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru3x6*yu3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I135a313}')

I135a316 = Parameter(name = 'I135a316',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru6x6*yu3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I135a316}')

I135a323 = Parameter(name = 'I135a323',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru3x6*yu3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I135a323}')

I135a326 = Parameter(name = 'I135a326',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru6x6*yu3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I135a326}')

I135a333 = Parameter(name = 'I135a333',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru3x6*yu3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I135a333}')

I135a336 = Parameter(name = 'I135a336',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru6x6*yu3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I135a336}')

I135a363 = Parameter(name = 'I135a363',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru3x6*yu3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I135a363}')

I135a366 = Parameter(name = 'I135a366',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru6x6*yu3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                     texname = '\\text{I135a366}')

I135a413 = Parameter(name = 'I135a413',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru3x6*yu3x3*complexconjugate(LLQD1x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I135a413}')

I135a416 = Parameter(name = 'I135a416',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru6x6*yu3x3*complexconjugate(LLQD1x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I135a416}')

I135a423 = Parameter(name = 'I135a423',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru3x6*yu3x3*complexconjugate(LLQD2x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I135a423}')

I135a426 = Parameter(name = 'I135a426',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru6x6*yu3x3*complexconjugate(LLQD2x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I135a426}')

I135a433 = Parameter(name = 'I135a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru3x6*yu3x3*complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I135a433}')

I135a436 = Parameter(name = 'I135a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru6x6*yu3x3*complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I135a436}')

I135a463 = Parameter(name = 'I135a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru3x6*yu3x3*complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I135a463}')

I135a466 = Parameter(name = 'I135a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru6x6*yu3x3*complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                     texname = '\\text{I135a466}')

I135a513 = Parameter(name = 'I135a513',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru3x6*yu3x3*complexconjugate(LLQD1x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I135a513}')

I135a516 = Parameter(name = 'I135a516',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru6x6*yu3x3*complexconjugate(LLQD1x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I135a516}')

I135a523 = Parameter(name = 'I135a523',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru3x6*yu3x3*complexconjugate(LLQD2x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I135a523}')

I135a526 = Parameter(name = 'I135a526',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru6x6*yu3x3*complexconjugate(LLQD2x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I135a526}')

I135a533 = Parameter(name = 'I135a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru3x6*yu3x3*complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I135a533}')

I135a536 = Parameter(name = 'I135a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru6x6*yu3x3*complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I135a536}')

I135a563 = Parameter(name = 'I135a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru3x6*yu3x3*complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I135a563}')

I135a566 = Parameter(name = 'I135a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru6x6*yu3x3*complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                     texname = '\\text{I135a566}')

I135a613 = Parameter(name = 'I135a613',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru3x6*yu3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I135a613}')

I135a616 = Parameter(name = 'I135a616',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl1x1*Ru6x6*yu3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I135a616}')

I135a623 = Parameter(name = 'I135a623',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru3x6*yu3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I135a623}')

I135a626 = Parameter(name = 'I135a626',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl2x2*Ru6x6*yu3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I135a626}')

I135a633 = Parameter(name = 'I135a633',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru3x6*yu3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I135a633}')

I135a636 = Parameter(name = 'I135a636',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl3x3*Ru6x6*yu3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I135a636}')

I135a663 = Parameter(name = 'I135a663',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru3x6*yu3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I135a663}')

I135a666 = Parameter(name = 'I135a666',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rl6x3*Ru6x6*yu3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                     texname = '\\text{I135a666}')

I136a343 = Parameter(name = 'I136a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd3x6*Rd4x4*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a343}')

I136a346 = Parameter(name = 'I136a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd3x6*Rd4x4*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a346}')

I136a353 = Parameter(name = 'I136a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd3x6*Rd5x5*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a353}')

I136a356 = Parameter(name = 'I136a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd3x6*Rd5x5*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a356}')

I136a433 = Parameter(name = 'I136a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd3x6*Rd4x4*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a433}')

I136a436 = Parameter(name = 'I136a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd3x6*Rd4x4*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a436}')

I136a453 = Parameter(name = 'I136a453',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x1*Rd4x4*Rd5x5*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a453}')

I136a456 = Parameter(name = 'I136a456',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x1*Rd4x4*Rd5x5*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a456}')

I136a463 = Parameter(name = 'I136a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd4x4*Rd6x6*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a463}')

I136a466 = Parameter(name = 'I136a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd4x4*Rd6x6*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a466}')

I136a533 = Parameter(name = 'I136a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd3x6*Rd5x5*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a533}')

I136a536 = Parameter(name = 'I136a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd3x6*Rd5x5*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a536}')

I136a543 = Parameter(name = 'I136a543',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x2*Rd4x4*Rd5x5*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a543}')

I136a546 = Parameter(name = 'I136a546',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x2*Rd4x4*Rd5x5*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a546}')

I136a563 = Parameter(name = 'I136a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd5x5*Rd6x6*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a563}')

I136a566 = Parameter(name = 'I136a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd5x5*Rd6x6*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a566}')

I136a643 = Parameter(name = 'I136a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd4x4*Rd6x6*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a643}')

I136a646 = Parameter(name = 'I136a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd4x4*Rd6x6*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a646}')

I136a653 = Parameter(name = 'I136a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd5x5*Rd6x6*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a653}')

I136a656 = Parameter(name = 'I136a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd5x5*Rd6x6*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I136a656}')

I137a343 = Parameter(name = 'I137a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd3x6*Rd4x4*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a343}')

I137a346 = Parameter(name = 'I137a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd3x6*Rd4x4*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a346}')

I137a353 = Parameter(name = 'I137a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd3x6*Rd5x5*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a353}')

I137a356 = Parameter(name = 'I137a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd3x6*Rd5x5*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a356}')

I137a433 = Parameter(name = 'I137a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd3x6*Rd4x4*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a433}')

I137a436 = Parameter(name = 'I137a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd3x6*Rd4x4*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a436}')

I137a453 = Parameter(name = 'I137a453',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x2*Rd4x4*Rd5x5*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a453}')

I137a456 = Parameter(name = 'I137a456',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x2*Rd4x4*Rd5x5*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a456}')

I137a463 = Parameter(name = 'I137a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd4x4*Rd6x6*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a463}')

I137a466 = Parameter(name = 'I137a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd4x4*Rd6x6*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a466}')

I137a533 = Parameter(name = 'I137a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd3x6*Rd5x5*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a533}')

I137a536 = Parameter(name = 'I137a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd3x6*Rd5x5*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a536}')

I137a543 = Parameter(name = 'I137a543',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x1*Rd4x4*Rd5x5*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a543}')

I137a546 = Parameter(name = 'I137a546',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x1*Rd4x4*Rd5x5*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a546}')

I137a563 = Parameter(name = 'I137a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd5x5*Rd6x6*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a563}')

I137a566 = Parameter(name = 'I137a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd5x5*Rd6x6*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a566}')

I137a643 = Parameter(name = 'I137a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd4x4*Rd6x6*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a643}')

I137a646 = Parameter(name = 'I137a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd4x4*Rd6x6*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a646}')

I137a653 = Parameter(name = 'I137a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd5x5*Rd6x6*Ru3x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a653}')

I137a656 = Parameter(name = 'I137a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd5x5*Rd6x6*Ru6x3*complexconjugate(yu3x3)',
                     texname = '\\text{I137a656}')

I138a433 = Parameter(name = 'I138a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd3x3*Rd4x4*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I138a433}')

I138a434 = Parameter(name = 'I138a434',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x1*Rd3x3*Rd4x4*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I138a434}')

I138a435 = Parameter(name = 'I138a435',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x1*Rd3x3*Rd4x4*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I138a435}')

I138a436 = Parameter(name = 'I138a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd3x3*Rd4x4*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I138a436}')

I138a463 = Parameter(name = 'I138a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd4x4*Rd6x3*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I138a463}')

I138a464 = Parameter(name = 'I138a464',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x1*Rd4x4*Rd6x3*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I138a464}')

I138a465 = Parameter(name = 'I138a465',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x1*Rd4x4*Rd6x3*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I138a465}')

I138a466 = Parameter(name = 'I138a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd4x4*Rd6x3*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I138a466}')

I138a533 = Parameter(name = 'I138a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd3x3*Rd5x5*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I138a533}')

I138a534 = Parameter(name = 'I138a534',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x2*Rd3x3*Rd5x5*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I138a534}')

I138a535 = Parameter(name = 'I138a535',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x2*Rd3x3*Rd5x5*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I138a535}')

I138a536 = Parameter(name = 'I138a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd3x3*Rd5x5*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I138a536}')

I138a563 = Parameter(name = 'I138a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd5x5*Rd6x3*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I138a563}')

I138a564 = Parameter(name = 'I138a564',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x2*Rd5x5*Rd6x3*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I138a564}')

I138a565 = Parameter(name = 'I138a565',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x2*Rd5x5*Rd6x3*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I138a565}')

I138a566 = Parameter(name = 'I138a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd5x5*Rd6x3*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I138a566}')

I139a433 = Parameter(name = 'I139a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd3x3*Rd4x4*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I139a433}')

I139a434 = Parameter(name = 'I139a434',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x1x3*Rd3x3*Rd4x4*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I139a434}')

I139a435 = Parameter(name = 'I139a435',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x1x3*Rd3x3*Rd4x4*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I139a435}')

I139a436 = Parameter(name = 'I139a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd3x3*Rd4x4*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I139a436}')

I139a463 = Parameter(name = 'I139a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd4x4*Rd6x3*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I139a463}')

I139a464 = Parameter(name = 'I139a464',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x1x3*Rd4x4*Rd6x3*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I139a464}')

I139a465 = Parameter(name = 'I139a465',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x1x3*Rd4x4*Rd6x3*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I139a465}')

I139a466 = Parameter(name = 'I139a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd4x4*Rd6x3*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I139a466}')

I139a533 = Parameter(name = 'I139a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd3x3*Rd5x5*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I139a533}')

I139a534 = Parameter(name = 'I139a534',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x2x3*Rd3x3*Rd5x5*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I139a534}')

I139a535 = Parameter(name = 'I139a535',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x2x3*Rd3x3*Rd5x5*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I139a535}')

I139a536 = Parameter(name = 'I139a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd3x3*Rd5x5*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I139a536}')

I139a563 = Parameter(name = 'I139a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd5x5*Rd6x3*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I139a563}')

I139a564 = Parameter(name = 'I139a564',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x2x3*Rd5x5*Rd6x3*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I139a564}')

I139a565 = Parameter(name = 'I139a565',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x2x3*Rd5x5*Rd6x3*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I139a565}')

I139a566 = Parameter(name = 'I139a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd5x5*Rd6x3*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I139a566}')

I14a11 = Parameter(name = 'I14a11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x1',
                   texname = '\\text{I14a11}')

I14a22 = Parameter(name = 'I14a22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x2',
                   texname = '\\text{I14a22}')

I14a33 = Parameter(name = 'I14a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3',
                   texname = '\\text{I14a33}')

I14a36 = Parameter(name = 'I14a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3',
                   texname = '\\text{I14a36}')

I140a343 = Parameter(name = 'I140a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd3x3*Rd4x4*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I140a343}')

I140a344 = Parameter(name = 'I140a344',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x1*Rd3x3*Rd4x4*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I140a344}')

I140a345 = Parameter(name = 'I140a345',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x1*Rd3x3*Rd4x4*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I140a345}')

I140a346 = Parameter(name = 'I140a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd3x3*Rd4x4*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I140a346}')

I140a353 = Parameter(name = 'I140a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd3x3*Rd5x5*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I140a353}')

I140a354 = Parameter(name = 'I140a354',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x2*Rd3x3*Rd5x5*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I140a354}')

I140a355 = Parameter(name = 'I140a355',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x2*Rd3x3*Rd5x5*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I140a355}')

I140a356 = Parameter(name = 'I140a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd3x3*Rd5x5*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I140a356}')

I140a643 = Parameter(name = 'I140a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd4x4*Rd6x3*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I140a643}')

I140a644 = Parameter(name = 'I140a644',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x1*Rd4x4*Rd6x3*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I140a644}')

I140a645 = Parameter(name = 'I140a645',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x1*Rd4x4*Rd6x3*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I140a645}')

I140a646 = Parameter(name = 'I140a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x1*Rd4x4*Rd6x3*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I140a646}')

I140a653 = Parameter(name = 'I140a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd5x5*Rd6x3*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I140a653}')

I140a654 = Parameter(name = 'I140a654',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x3x2*Rd5x5*Rd6x3*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I140a654}')

I140a655 = Parameter(name = 'I140a655',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x3x2*Rd5x5*Rd6x3*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I140a655}')

I140a656 = Parameter(name = 'I140a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x3x2*Rd5x5*Rd6x3*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I140a656}')

I141a343 = Parameter(name = 'I141a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd3x3*Rd4x4*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I141a343}')

I141a344 = Parameter(name = 'I141a344',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x1x3*Rd3x3*Rd4x4*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I141a344}')

I141a345 = Parameter(name = 'I141a345',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x1x3*Rd3x3*Rd4x4*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I141a345}')

I141a346 = Parameter(name = 'I141a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd3x3*Rd4x4*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I141a346}')

I141a353 = Parameter(name = 'I141a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd3x3*Rd5x5*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I141a353}')

I141a354 = Parameter(name = 'I141a354',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x2x3*Rd3x3*Rd5x5*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I141a354}')

I141a355 = Parameter(name = 'I141a355',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x2x3*Rd3x3*Rd5x5*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I141a355}')

I141a356 = Parameter(name = 'I141a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd3x3*Rd5x5*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I141a356}')

I141a643 = Parameter(name = 'I141a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd4x4*Rd6x3*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I141a643}')

I141a644 = Parameter(name = 'I141a644',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x1x3*Rd4x4*Rd6x3*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I141a644}')

I141a645 = Parameter(name = 'I141a645',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x1x3*Rd4x4*Rd6x3*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I141a645}')

I141a646 = Parameter(name = 'I141a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x1x3*Rd4x4*Rd6x3*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I141a646}')

I141a653 = Parameter(name = 'I141a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd5x5*Rd6x3*Ru3x6*complexconjugate(yd3x3)',
                     texname = '\\text{I141a653}')

I141a654 = Parameter(name = 'I141a654',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD1x2x3*Rd5x5*Rd6x3*Ru4x4*complexconjugate(yd3x3)',
                     texname = '\\text{I141a654}')

I141a655 = Parameter(name = 'I141a655',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD2x2x3*Rd5x5*Rd6x3*Ru5x5*complexconjugate(yd3x3)',
                     texname = '\\text{I141a655}')

I141a656 = Parameter(name = 'I141a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'LUDD3x2x3*Rd5x5*Rd6x3*Ru6x6*complexconjugate(yd3x3)',
                     texname = '\\text{I141a656}')

I142a343 = Parameter(name = 'I142a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru3x6*TUDD3x3x1',
                     texname = '\\text{I142a343}')

I142a344 = Parameter(name = 'I142a344',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru4x4*TUDD1x3x1',
                     texname = '\\text{I142a344}')

I142a345 = Parameter(name = 'I142a345',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru5x5*TUDD2x3x1',
                     texname = '\\text{I142a345}')

I142a346 = Parameter(name = 'I142a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru6x6*TUDD3x3x1',
                     texname = '\\text{I142a346}')

I142a353 = Parameter(name = 'I142a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru3x6*TUDD3x3x2',
                     texname = '\\text{I142a353}')

I142a354 = Parameter(name = 'I142a354',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru4x4*TUDD1x3x2',
                     texname = '\\text{I142a354}')

I142a355 = Parameter(name = 'I142a355',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru5x5*TUDD2x3x2',
                     texname = '\\text{I142a355}')

I142a356 = Parameter(name = 'I142a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru6x6*TUDD3x3x2',
                     texname = '\\text{I142a356}')

I142a433 = Parameter(name = 'I142a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru3x6*TUDD3x1x3',
                     texname = '\\text{I142a433}')

I142a434 = Parameter(name = 'I142a434',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru4x4*TUDD1x1x3',
                     texname = '\\text{I142a434}')

I142a435 = Parameter(name = 'I142a435',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru5x5*TUDD2x1x3',
                     texname = '\\text{I142a435}')

I142a436 = Parameter(name = 'I142a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru6x6*TUDD3x1x3',
                     texname = '\\text{I142a436}')

I142a453 = Parameter(name = 'I142a453',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru3x6*TUDD3x1x2',
                     texname = '\\text{I142a453}')

I142a454 = Parameter(name = 'I142a454',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru4x4*TUDD1x1x2',
                     texname = '\\text{I142a454}')

I142a455 = Parameter(name = 'I142a455',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru5x5*TUDD2x1x2',
                     texname = '\\text{I142a455}')

I142a456 = Parameter(name = 'I142a456',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru6x6*TUDD3x1x2',
                     texname = '\\text{I142a456}')

I142a463 = Parameter(name = 'I142a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru3x6*TUDD3x1x3',
                     texname = '\\text{I142a463}')

I142a464 = Parameter(name = 'I142a464',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru4x4*TUDD1x1x3',
                     texname = '\\text{I142a464}')

I142a465 = Parameter(name = 'I142a465',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru5x5*TUDD2x1x3',
                     texname = '\\text{I142a465}')

I142a466 = Parameter(name = 'I142a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru6x6*TUDD3x1x3',
                     texname = '\\text{I142a466}')

I142a533 = Parameter(name = 'I142a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru3x6*TUDD3x2x3',
                     texname = '\\text{I142a533}')

I142a534 = Parameter(name = 'I142a534',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru4x4*TUDD1x2x3',
                     texname = '\\text{I142a534}')

I142a535 = Parameter(name = 'I142a535',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru5x5*TUDD2x2x3',
                     texname = '\\text{I142a535}')

I142a536 = Parameter(name = 'I142a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru6x6*TUDD3x2x3',
                     texname = '\\text{I142a536}')

I142a543 = Parameter(name = 'I142a543',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru3x6*TUDD3x2x1',
                     texname = '\\text{I142a543}')

I142a544 = Parameter(name = 'I142a544',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru4x4*TUDD1x2x1',
                     texname = '\\text{I142a544}')

I142a545 = Parameter(name = 'I142a545',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru5x5*TUDD2x2x1',
                     texname = '\\text{I142a545}')

I142a546 = Parameter(name = 'I142a546',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru6x6*TUDD3x2x1',
                     texname = '\\text{I142a546}')

I142a563 = Parameter(name = 'I142a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru3x6*TUDD3x2x3',
                     texname = '\\text{I142a563}')

I142a564 = Parameter(name = 'I142a564',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru4x4*TUDD1x2x3',
                     texname = '\\text{I142a564}')

I142a565 = Parameter(name = 'I142a565',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru5x5*TUDD2x2x3',
                     texname = '\\text{I142a565}')

I142a566 = Parameter(name = 'I142a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru6x6*TUDD3x2x3',
                     texname = '\\text{I142a566}')

I142a643 = Parameter(name = 'I142a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru3x6*TUDD3x3x1',
                     texname = '\\text{I142a643}')

I142a644 = Parameter(name = 'I142a644',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru4x4*TUDD1x3x1',
                     texname = '\\text{I142a644}')

I142a645 = Parameter(name = 'I142a645',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru5x5*TUDD2x3x1',
                     texname = '\\text{I142a645}')

I142a646 = Parameter(name = 'I142a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru6x6*TUDD3x3x1',
                     texname = '\\text{I142a646}')

I142a653 = Parameter(name = 'I142a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru3x6*TUDD3x3x2',
                     texname = '\\text{I142a653}')

I142a654 = Parameter(name = 'I142a654',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru4x4*TUDD1x3x2',
                     texname = '\\text{I142a654}')

I142a655 = Parameter(name = 'I142a655',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru5x5*TUDD2x3x2',
                     texname = '\\text{I142a655}')

I142a656 = Parameter(name = 'I142a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru6x6*TUDD3x3x2',
                     texname = '\\text{I142a656}')

I143a343 = Parameter(name = 'I143a343',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru3x6*TUDD3x1x3',
                     texname = '\\text{I143a343}')

I143a344 = Parameter(name = 'I143a344',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru4x4*TUDD1x1x3',
                     texname = '\\text{I143a344}')

I143a345 = Parameter(name = 'I143a345',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru5x5*TUDD2x1x3',
                     texname = '\\text{I143a345}')

I143a346 = Parameter(name = 'I143a346',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru6x6*TUDD3x1x3',
                     texname = '\\text{I143a346}')

I143a353 = Parameter(name = 'I143a353',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru3x6*TUDD3x2x3',
                     texname = '\\text{I143a353}')

I143a354 = Parameter(name = 'I143a354',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru4x4*TUDD1x2x3',
                     texname = '\\text{I143a354}')

I143a355 = Parameter(name = 'I143a355',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru5x5*TUDD2x2x3',
                     texname = '\\text{I143a355}')

I143a356 = Parameter(name = 'I143a356',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru6x6*TUDD3x2x3',
                     texname = '\\text{I143a356}')

I143a433 = Parameter(name = 'I143a433',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru3x6*TUDD3x3x1',
                     texname = '\\text{I143a433}')

I143a434 = Parameter(name = 'I143a434',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru4x4*TUDD1x3x1',
                     texname = '\\text{I143a434}')

I143a435 = Parameter(name = 'I143a435',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru5x5*TUDD2x3x1',
                     texname = '\\text{I143a435}')

I143a436 = Parameter(name = 'I143a436',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd4x4*Ru6x6*TUDD3x3x1',
                     texname = '\\text{I143a436}')

I143a453 = Parameter(name = 'I143a453',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru3x6*TUDD3x2x1',
                     texname = '\\text{I143a453}')

I143a454 = Parameter(name = 'I143a454',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru4x4*TUDD1x2x1',
                     texname = '\\text{I143a454}')

I143a455 = Parameter(name = 'I143a455',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru5x5*TUDD2x2x1',
                     texname = '\\text{I143a455}')

I143a456 = Parameter(name = 'I143a456',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru6x6*TUDD3x2x1',
                     texname = '\\text{I143a456}')

I143a463 = Parameter(name = 'I143a463',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru3x6*TUDD3x3x1',
                     texname = '\\text{I143a463}')

I143a464 = Parameter(name = 'I143a464',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru4x4*TUDD1x3x1',
                     texname = '\\text{I143a464}')

I143a465 = Parameter(name = 'I143a465',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru5x5*TUDD2x3x1',
                     texname = '\\text{I143a465}')

I143a466 = Parameter(name = 'I143a466',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru6x6*TUDD3x3x1',
                     texname = '\\text{I143a466}')

I143a533 = Parameter(name = 'I143a533',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru3x6*TUDD3x3x2',
                     texname = '\\text{I143a533}')

I143a534 = Parameter(name = 'I143a534',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru4x4*TUDD1x3x2',
                     texname = '\\text{I143a534}')

I143a535 = Parameter(name = 'I143a535',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru5x5*TUDD2x3x2',
                     texname = '\\text{I143a535}')

I143a536 = Parameter(name = 'I143a536',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd3x6*Rd5x5*Ru6x6*TUDD3x3x2',
                     texname = '\\text{I143a536}')

I143a543 = Parameter(name = 'I143a543',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru3x6*TUDD3x1x2',
                     texname = '\\text{I143a543}')

I143a544 = Parameter(name = 'I143a544',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru4x4*TUDD1x1x2',
                     texname = '\\text{I143a544}')

I143a545 = Parameter(name = 'I143a545',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru5x5*TUDD2x1x2',
                     texname = '\\text{I143a545}')

I143a546 = Parameter(name = 'I143a546',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd5x5*Ru6x6*TUDD3x1x2',
                     texname = '\\text{I143a546}')

I143a563 = Parameter(name = 'I143a563',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru3x6*TUDD3x3x2',
                     texname = '\\text{I143a563}')

I143a564 = Parameter(name = 'I143a564',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru4x4*TUDD1x3x2',
                     texname = '\\text{I143a564}')

I143a565 = Parameter(name = 'I143a565',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru5x5*TUDD2x3x2',
                     texname = '\\text{I143a565}')

I143a566 = Parameter(name = 'I143a566',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru6x6*TUDD3x3x2',
                     texname = '\\text{I143a566}')

I143a643 = Parameter(name = 'I143a643',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru3x6*TUDD3x1x3',
                     texname = '\\text{I143a643}')

I143a644 = Parameter(name = 'I143a644',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru4x4*TUDD1x1x3',
                     texname = '\\text{I143a644}')

I143a645 = Parameter(name = 'I143a645',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru5x5*TUDD2x1x3',
                     texname = '\\text{I143a645}')

I143a646 = Parameter(name = 'I143a646',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd4x4*Rd6x6*Ru6x6*TUDD3x1x3',
                     texname = '\\text{I143a646}')

I143a653 = Parameter(name = 'I143a653',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru3x6*TUDD3x2x3',
                     texname = '\\text{I143a653}')

I143a654 = Parameter(name = 'I143a654',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru4x4*TUDD1x2x3',
                     texname = '\\text{I143a654}')

I143a655 = Parameter(name = 'I143a655',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru5x5*TUDD2x2x3',
                     texname = '\\text{I143a655}')

I143a656 = Parameter(name = 'I143a656',
                     nature = 'internal',
                     type = 'complex',
                     value = 'Rd5x5*Rd6x6*Ru6x6*TUDD3x2x3',
                     texname = '\\text{I143a656}')

I144a11 = Parameter(name = 'I144a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rd1x1)',
                    texname = '\\text{I144a11}')

I144a22 = Parameter(name = 'I144a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rd2x2)',
                    texname = '\\text{I144a22}')

I144a33 = Parameter(name = 'I144a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rd3x3)',
                    texname = '\\text{I144a33}')

I144a36 = Parameter(name = 'I144a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rd6x3)',
                    texname = '\\text{I144a36}')

I145a33 = Parameter(name = 'I145a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                    texname = '\\text{I145a33}')

I145a36 = Parameter(name = 'I145a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                    texname = '\\text{I145a36}')

I146a33 = Parameter(name = 'I146a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'yu3x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I146a33}')

I146a36 = Parameter(name = 'I146a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'yu3x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I146a36}')

I147a11 = Parameter(name = 'I147a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rl1x1)',
                    texname = '\\text{I147a11}')

I147a22 = Parameter(name = 'I147a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rl2x2)',
                    texname = '\\text{I147a22}')

I147a33 = Parameter(name = 'I147a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rl3x3)',
                    texname = '\\text{I147a33}')

I147a36 = Parameter(name = 'I147a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rl6x3)',
                    texname = '\\text{I147a36}')

I148a33 = Parameter(name = 'I148a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                    texname = '\\text{I148a33}')

I148a36 = Parameter(name = 'I148a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                    texname = '\\text{I148a36}')

I149a11 = Parameter(name = 'I149a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rn1x1)',
                    texname = '\\text{I149a11}')

I149a22 = Parameter(name = 'I149a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rn2x2)',
                    texname = '\\text{I149a22}')

I149a33 = Parameter(name = 'I149a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Rn3x3)',
                    texname = '\\text{I149a33}')

I15a33 = Parameter(name = 'I15a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(yu3x3)',
                   texname = '\\text{I15a33}')

I15a36 = Parameter(name = 'I15a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(yu3x3)',
                   texname = '\\text{I15a36}')

I150a33 = Parameter(name = 'I150a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'ye3x3*complexconjugate(Rn3x3)',
                    texname = '\\text{I150a33}')

I151a11 = Parameter(name = 'I151a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Ru1x1)',
                    texname = '\\text{I151a11}')

I151a22 = Parameter(name = 'I151a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Ru2x2)',
                    texname = '\\text{I151a22}')

I151a33 = Parameter(name = 'I151a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Ru3x3)',
                    texname = '\\text{I151a33}')

I151a36 = Parameter(name = 'I151a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Ru6x3)',
                    texname = '\\text{I151a36}')

I152a33 = Parameter(name = 'I152a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I152a33}')

I152a36 = Parameter(name = 'I152a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I152a36}')

I153a33 = Parameter(name = 'I153a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'yd3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I153a33}')

I153a36 = Parameter(name = 'I153a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'yd3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I153a36}')

I154a11 = Parameter(name = 'I154a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru1x1*complexconjugate(Rd1x1)',
                    texname = '\\text{I154a11}')

I154a22 = Parameter(name = 'I154a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru2x2*complexconjugate(Rd2x2)',
                    texname = '\\text{I154a22}')

I154a33 = Parameter(name = 'I154a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I154a33}')

I154a36 = Parameter(name = 'I154a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I154a36}')

I154a63 = Parameter(name = 'I154a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I154a63}')

I154a66 = Parameter(name = 'I154a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I154a66}')

I155a11 = Parameter(name = 'I155a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(Rl1x1)',
                    texname = '\\text{I155a11}')

I155a22 = Parameter(name = 'I155a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(Rl2x2)',
                    texname = '\\text{I155a22}')

I155a33 = Parameter(name = 'I155a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(Rl3x3)',
                    texname = '\\text{I155a33}')

I155a36 = Parameter(name = 'I155a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(Rl6x3)',
                    texname = '\\text{I155a36}')

I156a11 = Parameter(name = 'I156a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(Ru1x1)',
                    texname = '\\text{I156a11}')

I156a22 = Parameter(name = 'I156a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(Ru2x2)',
                    texname = '\\text{I156a22}')

I156a33 = Parameter(name = 'I156a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I156a33}')

I156a36 = Parameter(name = 'I156a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I156a36}')

I156a63 = Parameter(name = 'I156a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I156a63}')

I156a66 = Parameter(name = 'I156a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I156a66}')

I157a11 = Parameter(name = 'I157a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(Rn1x1)',
                    texname = '\\text{I157a11}')

I157a22 = Parameter(name = 'I157a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(Rn2x2)',
                    texname = '\\text{I157a22}')

I157a33 = Parameter(name = 'I157a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(Rn3x3)',
                    texname = '\\text{I157a33}')

I157a36 = Parameter(name = 'I157a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(Rn3x3)',
                    texname = '\\text{I157a36}')

I158a11 = Parameter(name = 'I158a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(Rd1x1)',
                    texname = '\\text{I158a11}')

I158a22 = Parameter(name = 'I158a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(Rd2x2)',
                    texname = '\\text{I158a22}')

I158a33 = Parameter(name = 'I158a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I158a33}')

I158a36 = Parameter(name = 'I158a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(Rd3x3)',
                    texname = '\\text{I158a36}')

I158a63 = Parameter(name = 'I158a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I158a63}')

I158a66 = Parameter(name = 'I158a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(Rd6x3)',
                    texname = '\\text{I158a66}')

I159a11 = Parameter(name = 'I159a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(Rl1x1)',
                    texname = '\\text{I159a11}')

I159a22 = Parameter(name = 'I159a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(Rl2x2)',
                    texname = '\\text{I159a22}')

I159a33 = Parameter(name = 'I159a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(Rl3x3)',
                    texname = '\\text{I159a33}')

I159a36 = Parameter(name = 'I159a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(Rl3x3)',
                    texname = '\\text{I159a36}')

I159a63 = Parameter(name = 'I159a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(Rl6x3)',
                    texname = '\\text{I159a63}')

I159a66 = Parameter(name = 'I159a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(Rl6x3)',
                    texname = '\\text{I159a66}')

I16a33 = Parameter(name = 'I16a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3',
                   texname = '\\text{I16a33}')

I16a36 = Parameter(name = 'I16a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3',
                   texname = '\\text{I16a36}')

I160a11 = Parameter(name = 'I160a11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru1x1*complexconjugate(Ru1x1)',
                    texname = '\\text{I160a11}')

I160a22 = Parameter(name = 'I160a22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru2x2*complexconjugate(Ru2x2)',
                    texname = '\\text{I160a22}')

I160a33 = Parameter(name = 'I160a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I160a33}')

I160a36 = Parameter(name = 'I160a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I160a36}')

I160a63 = Parameter(name = 'I160a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I160a63}')

I160a66 = Parameter(name = 'I160a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I160a66}')

I161a33 = Parameter(name = 'I161a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'ye3x3',
                    texname = '\\text{I161a33}')

I162a33 = Parameter(name = 'I162a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*complexconjugate(Rd3x6)',
                    texname = '\\text{I162a33}')

I162a36 = Parameter(name = 'I162a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*complexconjugate(Rd3x6)',
                    texname = '\\text{I162a36}')

I162a44 = Parameter(name = 'I162a44',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*complexconjugate(Rd4x4)',
                    texname = '\\text{I162a44}')

I162a55 = Parameter(name = 'I162a55',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*complexconjugate(Rd5x5)',
                    texname = '\\text{I162a55}')

I162a63 = Parameter(name = 'I162a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*complexconjugate(Rd6x6)',
                    texname = '\\text{I162a63}')

I162a66 = Parameter(name = 'I162a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*complexconjugate(Rd6x6)',
                    texname = '\\text{I162a66}')

I163a33 = Parameter(name = 'I163a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*complexconjugate(Rl3x6)',
                    texname = '\\text{I163a33}')

I163a36 = Parameter(name = 'I163a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*complexconjugate(Rl3x6)',
                    texname = '\\text{I163a36}')

I163a44 = Parameter(name = 'I163a44',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x4*complexconjugate(Rl4x4)',
                    texname = '\\text{I163a44}')

I163a55 = Parameter(name = 'I163a55',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl5x5*complexconjugate(Rl5x5)',
                    texname = '\\text{I163a55}')

I163a63 = Parameter(name = 'I163a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*complexconjugate(Rl6x6)',
                    texname = '\\text{I163a63}')

I163a66 = Parameter(name = 'I163a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*complexconjugate(Rl6x6)',
                    texname = '\\text{I163a66}')

I164a33 = Parameter(name = 'I164a33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*complexconjugate(Ru3x6)',
                    texname = '\\text{I164a33}')

I164a36 = Parameter(name = 'I164a36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*complexconjugate(Ru3x6)',
                    texname = '\\text{I164a36}')

I164a44 = Parameter(name = 'I164a44',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru4x4*complexconjugate(Ru4x4)',
                    texname = '\\text{I164a44}')

I164a55 = Parameter(name = 'I164a55',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru5x5*complexconjugate(Ru5x5)',
                    texname = '\\text{I164a55}')

I164a63 = Parameter(name = 'I164a63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*complexconjugate(Ru6x6)',
                    texname = '\\text{I164a63}')

I164a66 = Parameter(name = 'I164a66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*complexconjugate(Ru6x6)',
                    texname = '\\text{I164a66}')

I17a111 = Parameter(name = 'I17a111',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(LLQD1x1x1)',
                    texname = '\\text{I17a111}')

I17a112 = Parameter(name = 'I17a112',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(LLQD1x2x1)',
                    texname = '\\text{I17a112}')

I17a113 = Parameter(name = 'I17a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(LLQD1x3x1)',
                    texname = '\\text{I17a113}')

I17a116 = Parameter(name = 'I17a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(LLQD1x3x1)',
                    texname = '\\text{I17a116}')

I17a121 = Parameter(name = 'I17a121',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(LLQD1x1x2)',
                    texname = '\\text{I17a121}')

I17a122 = Parameter(name = 'I17a122',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(LLQD1x2x2)',
                    texname = '\\text{I17a122}')

I17a123 = Parameter(name = 'I17a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(LLQD1x3x2)',
                    texname = '\\text{I17a123}')

I17a126 = Parameter(name = 'I17a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(LLQD1x3x2)',
                    texname = '\\text{I17a126}')

I17a131 = Parameter(name = 'I17a131',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(LLQD1x1x3)',
                    texname = '\\text{I17a131}')

I17a132 = Parameter(name = 'I17a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(LLQD1x2x3)',
                    texname = '\\text{I17a132}')

I17a133 = Parameter(name = 'I17a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(LLQD1x3x3)',
                    texname = '\\text{I17a133}')

I17a136 = Parameter(name = 'I17a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(LLQD1x3x3)',
                    texname = '\\text{I17a136}')

I17a211 = Parameter(name = 'I17a211',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(LLQD2x1x1)',
                    texname = '\\text{I17a211}')

I17a212 = Parameter(name = 'I17a212',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(LLQD2x2x1)',
                    texname = '\\text{I17a212}')

I17a213 = Parameter(name = 'I17a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(LLQD2x3x1)',
                    texname = '\\text{I17a213}')

I17a216 = Parameter(name = 'I17a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(LLQD2x3x1)',
                    texname = '\\text{I17a216}')

I17a221 = Parameter(name = 'I17a221',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(LLQD2x1x2)',
                    texname = '\\text{I17a221}')

I17a222 = Parameter(name = 'I17a222',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(LLQD2x2x2)',
                    texname = '\\text{I17a222}')

I17a223 = Parameter(name = 'I17a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(LLQD2x3x2)',
                    texname = '\\text{I17a223}')

I17a226 = Parameter(name = 'I17a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(LLQD2x3x2)',
                    texname = '\\text{I17a226}')

I17a231 = Parameter(name = 'I17a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(LLQD2x1x3)',
                    texname = '\\text{I17a231}')

I17a232 = Parameter(name = 'I17a232',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(LLQD2x2x3)',
                    texname = '\\text{I17a232}')

I17a233 = Parameter(name = 'I17a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(LLQD2x3x3)',
                    texname = '\\text{I17a233}')

I17a236 = Parameter(name = 'I17a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(LLQD2x3x3)',
                    texname = '\\text{I17a236}')

I17a311 = Parameter(name = 'I17a311',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(LLQD3x1x1)',
                    texname = '\\text{I17a311}')

I17a312 = Parameter(name = 'I17a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(LLQD3x2x1)',
                    texname = '\\text{I17a312}')

I17a313 = Parameter(name = 'I17a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(LLQD3x3x1)',
                    texname = '\\text{I17a313}')

I17a316 = Parameter(name = 'I17a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(LLQD3x3x1)',
                    texname = '\\text{I17a316}')

I17a321 = Parameter(name = 'I17a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(LLQD3x1x2)',
                    texname = '\\text{I17a321}')

I17a322 = Parameter(name = 'I17a322',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(LLQD3x2x2)',
                    texname = '\\text{I17a322}')

I17a323 = Parameter(name = 'I17a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(LLQD3x3x2)',
                    texname = '\\text{I17a323}')

I17a326 = Parameter(name = 'I17a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(LLQD3x3x2)',
                    texname = '\\text{I17a326}')

I17a331 = Parameter(name = 'I17a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*complexconjugate(LLQD3x1x3)',
                    texname = '\\text{I17a331}')

I17a332 = Parameter(name = 'I17a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*complexconjugate(LLQD3x2x3)',
                    texname = '\\text{I17a332}')

I17a333 = Parameter(name = 'I17a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*complexconjugate(LLQD3x3x3)',
                    texname = '\\text{I17a333}')

I17a336 = Parameter(name = 'I17a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*complexconjugate(LLQD3x3x3)',
                    texname = '\\text{I17a336}')

I18a113 = Parameter(name = 'I18a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*Rd3x6',
                    texname = '\\text{I18a113}')

I18a114 = Parameter(name = 'I18a114',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x1*Rd4x4',
                    texname = '\\text{I18a114}')

I18a115 = Parameter(name = 'I18a115',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x2*Rd5x5',
                    texname = '\\text{I18a115}')

I18a116 = Parameter(name = 'I18a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*Rd6x6',
                    texname = '\\text{I18a116}')

I18a123 = Parameter(name = 'I18a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*Rd3x6',
                    texname = '\\text{I18a123}')

I18a124 = Parameter(name = 'I18a124',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x1*Rd4x4',
                    texname = '\\text{I18a124}')

I18a125 = Parameter(name = 'I18a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x2*Rd5x5',
                    texname = '\\text{I18a125}')

I18a126 = Parameter(name = 'I18a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*Rd6x6',
                    texname = '\\text{I18a126}')

I18a133 = Parameter(name = 'I18a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd3x6',
                    texname = '\\text{I18a133}')

I18a134 = Parameter(name = 'I18a134',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x1*Rd4x4',
                    texname = '\\text{I18a134}')

I18a135 = Parameter(name = 'I18a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x2*Rd5x5',
                    texname = '\\text{I18a135}')

I18a136 = Parameter(name = 'I18a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd6x6',
                    texname = '\\text{I18a136}')

I18a213 = Parameter(name = 'I18a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*Rd3x6',
                    texname = '\\text{I18a213}')

I18a214 = Parameter(name = 'I18a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x1*Rd4x4',
                    texname = '\\text{I18a214}')

I18a215 = Parameter(name = 'I18a215',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x2*Rd5x5',
                    texname = '\\text{I18a215}')

I18a216 = Parameter(name = 'I18a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*Rd6x6',
                    texname = '\\text{I18a216}')

I18a223 = Parameter(name = 'I18a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*Rd3x6',
                    texname = '\\text{I18a223}')

I18a224 = Parameter(name = 'I18a224',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x1*Rd4x4',
                    texname = '\\text{I18a224}')

I18a225 = Parameter(name = 'I18a225',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x2*Rd5x5',
                    texname = '\\text{I18a225}')

I18a226 = Parameter(name = 'I18a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*Rd6x6',
                    texname = '\\text{I18a226}')

I18a233 = Parameter(name = 'I18a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd3x6',
                    texname = '\\text{I18a233}')

I18a234 = Parameter(name = 'I18a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x1*Rd4x4',
                    texname = '\\text{I18a234}')

I18a235 = Parameter(name = 'I18a235',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x2*Rd5x5',
                    texname = '\\text{I18a235}')

I18a236 = Parameter(name = 'I18a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd6x6',
                    texname = '\\text{I18a236}')

I18a313 = Parameter(name = 'I18a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd3x6',
                    texname = '\\text{I18a313}')

I18a314 = Parameter(name = 'I18a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x1*Rd4x4',
                    texname = '\\text{I18a314}')

I18a315 = Parameter(name = 'I18a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x2*Rd5x5',
                    texname = '\\text{I18a315}')

I18a316 = Parameter(name = 'I18a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd6x6',
                    texname = '\\text{I18a316}')

I18a323 = Parameter(name = 'I18a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd3x6',
                    texname = '\\text{I18a323}')

I18a324 = Parameter(name = 'I18a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x1*Rd4x4',
                    texname = '\\text{I18a324}')

I18a325 = Parameter(name = 'I18a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x2*Rd5x5',
                    texname = '\\text{I18a325}')

I18a326 = Parameter(name = 'I18a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd6x6',
                    texname = '\\text{I18a326}')

I18a333 = Parameter(name = 'I18a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6',
                    texname = '\\text{I18a333}')

I18a334 = Parameter(name = 'I18a334',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4',
                    texname = '\\text{I18a334}')

I18a335 = Parameter(name = 'I18a335',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5',
                    texname = '\\text{I18a335}')

I18a336 = Parameter(name = 'I18a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6',
                    texname = '\\text{I18a336}')

I19a113 = Parameter(name = 'I19a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*Rd3x6',
                    texname = '\\text{I19a113}')

I19a114 = Parameter(name = 'I19a114',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x1*Rd4x4',
                    texname = '\\text{I19a114}')

I19a115 = Parameter(name = 'I19a115',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x2*Rd5x5',
                    texname = '\\text{I19a115}')

I19a116 = Parameter(name = 'I19a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*Rd6x6',
                    texname = '\\text{I19a116}')

I19a123 = Parameter(name = 'I19a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*Rd3x6',
                    texname = '\\text{I19a123}')

I19a124 = Parameter(name = 'I19a124',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x1*Rd4x4',
                    texname = '\\text{I19a124}')

I19a125 = Parameter(name = 'I19a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x2*Rd5x5',
                    texname = '\\text{I19a125}')

I19a126 = Parameter(name = 'I19a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*Rd6x6',
                    texname = '\\text{I19a126}')

I19a133 = Parameter(name = 'I19a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd3x6',
                    texname = '\\text{I19a133}')

I19a134 = Parameter(name = 'I19a134',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x1*Rd4x4',
                    texname = '\\text{I19a134}')

I19a135 = Parameter(name = 'I19a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x2*Rd5x5',
                    texname = '\\text{I19a135}')

I19a136 = Parameter(name = 'I19a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd6x6',
                    texname = '\\text{I19a136}')

I19a213 = Parameter(name = 'I19a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*Rd3x6',
                    texname = '\\text{I19a213}')

I19a214 = Parameter(name = 'I19a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x1*Rd4x4',
                    texname = '\\text{I19a214}')

I19a215 = Parameter(name = 'I19a215',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x2*Rd5x5',
                    texname = '\\text{I19a215}')

I19a216 = Parameter(name = 'I19a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*Rd6x6',
                    texname = '\\text{I19a216}')

I19a223 = Parameter(name = 'I19a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*Rd3x6',
                    texname = '\\text{I19a223}')

I19a224 = Parameter(name = 'I19a224',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x1*Rd4x4',
                    texname = '\\text{I19a224}')

I19a225 = Parameter(name = 'I19a225',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x2*Rd5x5',
                    texname = '\\text{I19a225}')

I19a226 = Parameter(name = 'I19a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*Rd6x6',
                    texname = '\\text{I19a226}')

I19a233 = Parameter(name = 'I19a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd3x6',
                    texname = '\\text{I19a233}')

I19a234 = Parameter(name = 'I19a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x1*Rd4x4',
                    texname = '\\text{I19a234}')

I19a235 = Parameter(name = 'I19a235',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x2*Rd5x5',
                    texname = '\\text{I19a235}')

I19a236 = Parameter(name = 'I19a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd6x6',
                    texname = '\\text{I19a236}')

I19a313 = Parameter(name = 'I19a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd3x6',
                    texname = '\\text{I19a313}')

I19a314 = Parameter(name = 'I19a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x1*Rd4x4',
                    texname = '\\text{I19a314}')

I19a315 = Parameter(name = 'I19a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x2*Rd5x5',
                    texname = '\\text{I19a315}')

I19a316 = Parameter(name = 'I19a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd6x6',
                    texname = '\\text{I19a316}')

I19a323 = Parameter(name = 'I19a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd3x6',
                    texname = '\\text{I19a323}')

I19a324 = Parameter(name = 'I19a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x1*Rd4x4',
                    texname = '\\text{I19a324}')

I19a325 = Parameter(name = 'I19a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x2*Rd5x5',
                    texname = '\\text{I19a325}')

I19a326 = Parameter(name = 'I19a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd6x6',
                    texname = '\\text{I19a326}')

I19a333 = Parameter(name = 'I19a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6',
                    texname = '\\text{I19a333}')

I19a334 = Parameter(name = 'I19a334',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4',
                    texname = '\\text{I19a334}')

I19a335 = Parameter(name = 'I19a335',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5',
                    texname = '\\text{I19a335}')

I19a336 = Parameter(name = 'I19a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6',
                    texname = '\\text{I19a336}')

I2a33 = Parameter(name = 'I2a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd3x3',
                  texname = '\\text{I2a33}')

I20a113 = Parameter(name = 'I20a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x3x1*Rd3x6',
                    texname = '\\text{I20a113}')

I20a115 = Parameter(name = 'I20a115',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x2x1*Rd5x5',
                    texname = '\\text{I20a115}')

I20a116 = Parameter(name = 'I20a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x3x1*Rd6x6',
                    texname = '\\text{I20a116}')

I20a123 = Parameter(name = 'I20a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x3x1*Rd3x6',
                    texname = '\\text{I20a123}')

I20a125 = Parameter(name = 'I20a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x2x1*Rd5x5',
                    texname = '\\text{I20a125}')

I20a126 = Parameter(name = 'I20a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x3x1*Rd6x6',
                    texname = '\\text{I20a126}')

I20a133 = Parameter(name = 'I20a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x3x1*Rd3x6',
                    texname = '\\text{I20a133}')

I20a135 = Parameter(name = 'I20a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x2x1*Rd5x5',
                    texname = '\\text{I20a135}')

I20a136 = Parameter(name = 'I20a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x3x1*Rd6x6',
                    texname = '\\text{I20a136}')

I20a213 = Parameter(name = 'I20a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x3x2*Rd3x6',
                    texname = '\\text{I20a213}')

I20a214 = Parameter(name = 'I20a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x1x2*Rd4x4',
                    texname = '\\text{I20a214}')

I20a216 = Parameter(name = 'I20a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x3x2*Rd6x6',
                    texname = '\\text{I20a216}')

I20a223 = Parameter(name = 'I20a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x3x2*Rd3x6',
                    texname = '\\text{I20a223}')

I20a224 = Parameter(name = 'I20a224',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x1x2*Rd4x4',
                    texname = '\\text{I20a224}')

I20a226 = Parameter(name = 'I20a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x3x2*Rd6x6',
                    texname = '\\text{I20a226}')

I20a233 = Parameter(name = 'I20a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x3x2*Rd3x6',
                    texname = '\\text{I20a233}')

I20a234 = Parameter(name = 'I20a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x1x2*Rd4x4',
                    texname = '\\text{I20a234}')

I20a236 = Parameter(name = 'I20a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x3x2*Rd6x6',
                    texname = '\\text{I20a236}')

I20a314 = Parameter(name = 'I20a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x1x3*Rd4x4',
                    texname = '\\text{I20a314}')

I20a315 = Parameter(name = 'I20a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x2x3*Rd5x5',
                    texname = '\\text{I20a315}')

I20a324 = Parameter(name = 'I20a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x1x3*Rd4x4',
                    texname = '\\text{I20a324}')

I20a325 = Parameter(name = 'I20a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x2x3*Rd5x5',
                    texname = '\\text{I20a325}')

I20a334 = Parameter(name = 'I20a334',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x1x3*Rd4x4',
                    texname = '\\text{I20a334}')

I20a335 = Parameter(name = 'I20a335',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x2x3*Rd5x5',
                    texname = '\\text{I20a335}')

I21a113 = Parameter(name = 'I21a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x1x3*Rd3x6',
                    texname = '\\text{I21a113}')

I21a115 = Parameter(name = 'I21a115',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x1x2*Rd5x5',
                    texname = '\\text{I21a115}')

I21a116 = Parameter(name = 'I21a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x1x3*Rd6x6',
                    texname = '\\text{I21a116}')

I21a123 = Parameter(name = 'I21a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x1x3*Rd3x6',
                    texname = '\\text{I21a123}')

I21a125 = Parameter(name = 'I21a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x1x2*Rd5x5',
                    texname = '\\text{I21a125}')

I21a126 = Parameter(name = 'I21a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x1x3*Rd6x6',
                    texname = '\\text{I21a126}')

I21a133 = Parameter(name = 'I21a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x1x3*Rd3x6',
                    texname = '\\text{I21a133}')

I21a135 = Parameter(name = 'I21a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x1x2*Rd5x5',
                    texname = '\\text{I21a135}')

I21a136 = Parameter(name = 'I21a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x1x3*Rd6x6',
                    texname = '\\text{I21a136}')

I21a213 = Parameter(name = 'I21a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x2x3*Rd3x6',
                    texname = '\\text{I21a213}')

I21a214 = Parameter(name = 'I21a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x2x1*Rd4x4',
                    texname = '\\text{I21a214}')

I21a216 = Parameter(name = 'I21a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x2x3*Rd6x6',
                    texname = '\\text{I21a216}')

I21a223 = Parameter(name = 'I21a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x2x3*Rd3x6',
                    texname = '\\text{I21a223}')

I21a224 = Parameter(name = 'I21a224',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x2x1*Rd4x4',
                    texname = '\\text{I21a224}')

I21a226 = Parameter(name = 'I21a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x2x3*Rd6x6',
                    texname = '\\text{I21a226}')

I21a233 = Parameter(name = 'I21a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x2x3*Rd3x6',
                    texname = '\\text{I21a233}')

I21a234 = Parameter(name = 'I21a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x2x1*Rd4x4',
                    texname = '\\text{I21a234}')

I21a236 = Parameter(name = 'I21a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x2x3*Rd6x6',
                    texname = '\\text{I21a236}')

I21a314 = Parameter(name = 'I21a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x3x1*Rd4x4',
                    texname = '\\text{I21a314}')

I21a315 = Parameter(name = 'I21a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD1x3x2*Rd5x5',
                    texname = '\\text{I21a315}')

I21a324 = Parameter(name = 'I21a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x3x1*Rd4x4',
                    texname = '\\text{I21a324}')

I21a325 = Parameter(name = 'I21a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD2x3x2*Rd5x5',
                    texname = '\\text{I21a325}')

I21a334 = Parameter(name = 'I21a334',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x3x1*Rd4x4',
                    texname = '\\text{I21a334}')

I21a335 = Parameter(name = 'I21a335',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LUDD3x3x2*Rd5x5',
                    texname = '\\text{I21a335}')

I22a11 = Parameter(name = 'I22a11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x1*complexconjugate(Rd1x1)',
                   texname = '\\text{I22a11}')

I22a22 = Parameter(name = 'I22a22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x2*complexconjugate(Rd2x2)',
                   texname = '\\text{I22a22}')

I22a33 = Parameter(name = 'I22a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I22a33}')

I22a36 = Parameter(name = 'I22a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I22a36}')

I22a63 = Parameter(name = 'I22a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I22a63}')

I22a66 = Parameter(name = 'I22a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I22a66}')

I23a33 = Parameter(name = 'I23a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*complexconjugate(Rd3x6)',
                   texname = '\\text{I23a33}')

I23a36 = Parameter(name = 'I23a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*complexconjugate(Rd3x6)',
                   texname = '\\text{I23a36}')

I23a44 = Parameter(name = 'I23a44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd4x4*complexconjugate(Rd4x4)',
                   texname = '\\text{I23a44}')

I23a55 = Parameter(name = 'I23a55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd5x5*complexconjugate(Rd5x5)',
                   texname = '\\text{I23a55}')

I23a63 = Parameter(name = 'I23a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*complexconjugate(Rd6x6)',
                   texname = '\\text{I23a63}')

I23a66 = Parameter(name = 'I23a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*complexconjugate(Rd6x6)',
                   texname = '\\text{I23a66}')

I24a33 = Parameter(name = 'I24a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd3x6)*complexconjugate(td3x3)',
                   texname = '\\text{I24a33}')

I24a36 = Parameter(name = 'I24a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd3x6)*complexconjugate(td3x3)',
                   texname = '\\text{I24a36}')

I24a63 = Parameter(name = 'I24a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)',
                   texname = '\\text{I24a63}')

I24a66 = Parameter(name = 'I24a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)',
                   texname = '\\text{I24a66}')

I25a33 = Parameter(name = 'I25a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I25a33}')

I25a36 = Parameter(name = 'I25a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I25a36}')

I25a63 = Parameter(name = 'I25a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I25a63}')

I25a66 = Parameter(name = 'I25a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I25a66}')

I26a33 = Parameter(name = 'I26a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*td3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I26a33}')

I26a36 = Parameter(name = 'I26a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*td3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I26a36}')

I26a63 = Parameter(name = 'I26a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*td3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I26a63}')

I26a66 = Parameter(name = 'I26a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*td3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I26a66}')

I27a33 = Parameter(name = 'I27a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I27a33}')

I27a36 = Parameter(name = 'I27a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I27a36}')

I27a63 = Parameter(name = 'I27a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yd3x3*complexconjugate(Rd6x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I27a63}')

I27a66 = Parameter(name = 'I27a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yd3x3*complexconjugate(Rd6x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I27a66}')

I28a33 = Parameter(name = 'I28a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I28a33}')

I28a36 = Parameter(name = 'I28a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I28a36}')

I28a63 = Parameter(name = 'I28a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I28a63}')

I28a66 = Parameter(name = 'I28a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I28a66}')

I29a33 = Parameter(name = 'I29a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I29a33}')

I29a36 = Parameter(name = 'I29a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I29a36}')

I29a63 = Parameter(name = 'I29a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I29a63}')

I29a66 = Parameter(name = 'I29a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I29a66}')

I3a33 = Parameter(name = 'I3a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                  texname = '\\text{I3a33}')

I3a36 = Parameter(name = 'I3a36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                  texname = '\\text{I3a36}')

I30a33 = Parameter(name = 'I30a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(yd3x3)',
                   texname = '\\text{I30a33}')

I31a33 = Parameter(name = 'I31a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yu3x3',
                   texname = '\\text{I31a33}')

I32a33 = Parameter(name = 'I32a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(ye3x3)',
                   texname = '\\text{I32a33}')

I33a33 = Parameter(name = 'I33a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I33a33}')

I33a36 = Parameter(name = 'I33a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I33a36}')

I34a33 = Parameter(name = 'I34a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I34a33}')

I34a36 = Parameter(name = 'I34a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I34a36}')

I35a112 = Parameter(name = 'I35a112',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x1*complexconjugate(Rl2x2)',
                    texname = '\\text{I35a112}')

I35a113 = Parameter(name = 'I35a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x1*complexconjugate(Rl3x3)',
                    texname = '\\text{I35a113}')

I35a116 = Parameter(name = 'I35a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x1*complexconjugate(Rl6x3)',
                    texname = '\\text{I35a116}')

I35a121 = Parameter(name = 'I35a121',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x1*complexconjugate(Rl1x1)',
                    texname = '\\text{I35a121}')

I35a123 = Parameter(name = 'I35a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x1*complexconjugate(Rl3x3)',
                    texname = '\\text{I35a123}')

I35a126 = Parameter(name = 'I35a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x1*complexconjugate(Rl6x3)',
                    texname = '\\text{I35a126}')

I35a131 = Parameter(name = 'I35a131',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x1*complexconjugate(Rl1x1)',
                    texname = '\\text{I35a131}')

I35a132 = Parameter(name = 'I35a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x1*complexconjugate(Rl2x2)',
                    texname = '\\text{I35a132}')

I35a212 = Parameter(name = 'I35a212',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x2*complexconjugate(Rl2x2)',
                    texname = '\\text{I35a212}')

I35a213 = Parameter(name = 'I35a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x2*complexconjugate(Rl3x3)',
                    texname = '\\text{I35a213}')

I35a216 = Parameter(name = 'I35a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x2*complexconjugate(Rl6x3)',
                    texname = '\\text{I35a216}')

I35a221 = Parameter(name = 'I35a221',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x2*complexconjugate(Rl1x1)',
                    texname = '\\text{I35a221}')

I35a223 = Parameter(name = 'I35a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x2*complexconjugate(Rl3x3)',
                    texname = '\\text{I35a223}')

I35a226 = Parameter(name = 'I35a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x2*complexconjugate(Rl6x3)',
                    texname = '\\text{I35a226}')

I35a231 = Parameter(name = 'I35a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x2*complexconjugate(Rl1x1)',
                    texname = '\\text{I35a231}')

I35a232 = Parameter(name = 'I35a232',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x2*complexconjugate(Rl2x2)',
                    texname = '\\text{I35a232}')

I35a312 = Parameter(name = 'I35a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x3*complexconjugate(Rl2x2)',
                    texname = '\\text{I35a312}')

I35a313 = Parameter(name = 'I35a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*complexconjugate(Rl3x3)',
                    texname = '\\text{I35a313}')

I35a316 = Parameter(name = 'I35a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*complexconjugate(Rl6x3)',
                    texname = '\\text{I35a316}')

I35a321 = Parameter(name = 'I35a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x3*complexconjugate(Rl1x1)',
                    texname = '\\text{I35a321}')

I35a323 = Parameter(name = 'I35a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*complexconjugate(Rl3x3)',
                    texname = '\\text{I35a323}')

I35a326 = Parameter(name = 'I35a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*complexconjugate(Rl6x3)',
                    texname = '\\text{I35a326}')

I35a331 = Parameter(name = 'I35a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x3*complexconjugate(Rl1x1)',
                    texname = '\\text{I35a331}')

I35a332 = Parameter(name = 'I35a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x3*complexconjugate(Rl2x2)',
                    texname = '\\text{I35a332}')

I36a111 = Parameter(name = 'I36a111',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x1*complexconjugate(Rl1x1)',
                    texname = '\\text{I36a111}')

I36a112 = Parameter(name = 'I36a112',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x1*complexconjugate(Rl2x2)',
                    texname = '\\text{I36a112}')

I36a113 = Parameter(name = 'I36a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x1*complexconjugate(Rl3x3)',
                    texname = '\\text{I36a113}')

I36a116 = Parameter(name = 'I36a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x1*complexconjugate(Rl6x3)',
                    texname = '\\text{I36a116}')

I36a121 = Parameter(name = 'I36a121',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x1*complexconjugate(Rl1x1)',
                    texname = '\\text{I36a121}')

I36a122 = Parameter(name = 'I36a122',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x1*complexconjugate(Rl2x2)',
                    texname = '\\text{I36a122}')

I36a123 = Parameter(name = 'I36a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x1*complexconjugate(Rl3x3)',
                    texname = '\\text{I36a123}')

I36a126 = Parameter(name = 'I36a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x1*complexconjugate(Rl6x3)',
                    texname = '\\text{I36a126}')

I36a131 = Parameter(name = 'I36a131',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x1*complexconjugate(Rl1x1)',
                    texname = '\\text{I36a131}')

I36a132 = Parameter(name = 'I36a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x1*complexconjugate(Rl2x2)',
                    texname = '\\text{I36a132}')

I36a133 = Parameter(name = 'I36a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*complexconjugate(Rl3x3)',
                    texname = '\\text{I36a133}')

I36a136 = Parameter(name = 'I36a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*complexconjugate(Rl6x3)',
                    texname = '\\text{I36a136}')

I36a211 = Parameter(name = 'I36a211',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x2*complexconjugate(Rl1x1)',
                    texname = '\\text{I36a211}')

I36a212 = Parameter(name = 'I36a212',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x2*complexconjugate(Rl2x2)',
                    texname = '\\text{I36a212}')

I36a213 = Parameter(name = 'I36a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x2*complexconjugate(Rl3x3)',
                    texname = '\\text{I36a213}')

I36a216 = Parameter(name = 'I36a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x2*complexconjugate(Rl6x3)',
                    texname = '\\text{I36a216}')

I36a221 = Parameter(name = 'I36a221',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x2*complexconjugate(Rl1x1)',
                    texname = '\\text{I36a221}')

I36a222 = Parameter(name = 'I36a222',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x2*complexconjugate(Rl2x2)',
                    texname = '\\text{I36a222}')

I36a223 = Parameter(name = 'I36a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x2*complexconjugate(Rl3x3)',
                    texname = '\\text{I36a223}')

I36a226 = Parameter(name = 'I36a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x2*complexconjugate(Rl6x3)',
                    texname = '\\text{I36a226}')

I36a231 = Parameter(name = 'I36a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x2*complexconjugate(Rl1x1)',
                    texname = '\\text{I36a231}')

I36a232 = Parameter(name = 'I36a232',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x2*complexconjugate(Rl2x2)',
                    texname = '\\text{I36a232}')

I36a233 = Parameter(name = 'I36a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*complexconjugate(Rl3x3)',
                    texname = '\\text{I36a233}')

I36a236 = Parameter(name = 'I36a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*complexconjugate(Rl6x3)',
                    texname = '\\text{I36a236}')

I36a311 = Parameter(name = 'I36a311',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*complexconjugate(Rl1x1)',
                    texname = '\\text{I36a311}')

I36a312 = Parameter(name = 'I36a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*complexconjugate(Rl2x2)',
                    texname = '\\text{I36a312}')

I36a313 = Parameter(name = 'I36a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*complexconjugate(Rl3x3)',
                    texname = '\\text{I36a313}')

I36a316 = Parameter(name = 'I36a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*complexconjugate(Rl6x3)',
                    texname = '\\text{I36a316}')

I36a321 = Parameter(name = 'I36a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*complexconjugate(Rl1x1)',
                    texname = '\\text{I36a321}')

I36a322 = Parameter(name = 'I36a322',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*complexconjugate(Rl2x2)',
                    texname = '\\text{I36a322}')

I36a323 = Parameter(name = 'I36a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*complexconjugate(Rl3x3)',
                    texname = '\\text{I36a323}')

I36a326 = Parameter(name = 'I36a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*complexconjugate(Rl6x3)',
                    texname = '\\text{I36a326}')

I36a331 = Parameter(name = 'I36a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*complexconjugate(Rl1x1)',
                    texname = '\\text{I36a331}')

I36a332 = Parameter(name = 'I36a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*complexconjugate(Rl2x2)',
                    texname = '\\text{I36a332}')

I36a333 = Parameter(name = 'I36a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*complexconjugate(Rl3x3)',
                    texname = '\\text{I36a333}')

I36a336 = Parameter(name = 'I36a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*complexconjugate(Rl6x3)',
                    texname = '\\text{I36a336}')

I37a123 = Parameter(name = 'I37a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE2x1x3)*complexconjugate(Rl3x6)',
                    texname = '\\text{I37a123}')

I37a124 = Parameter(name = 'I37a124',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE2x1x1)*complexconjugate(Rl4x4)',
                    texname = '\\text{I37a124}')

I37a125 = Parameter(name = 'I37a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE2x1x2)*complexconjugate(Rl5x5)',
                    texname = '\\text{I37a125}')

I37a126 = Parameter(name = 'I37a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE2x1x3)*complexconjugate(Rl6x6)',
                    texname = '\\text{I37a126}')

I37a133 = Parameter(name = 'I37a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE3x1x3)*complexconjugate(Rl3x6)',
                    texname = '\\text{I37a133}')

I37a134 = Parameter(name = 'I37a134',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE3x1x1)*complexconjugate(Rl4x4)',
                    texname = '\\text{I37a134}')

I37a135 = Parameter(name = 'I37a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE3x1x2)*complexconjugate(Rl5x5)',
                    texname = '\\text{I37a135}')

I37a136 = Parameter(name = 'I37a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE3x1x3)*complexconjugate(Rl6x6)',
                    texname = '\\text{I37a136}')

I37a213 = Parameter(name = 'I37a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE1x2x3)*complexconjugate(Rl3x6)',
                    texname = '\\text{I37a213}')

I37a214 = Parameter(name = 'I37a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE1x2x1)*complexconjugate(Rl4x4)',
                    texname = '\\text{I37a214}')

I37a215 = Parameter(name = 'I37a215',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE1x2x2)*complexconjugate(Rl5x5)',
                    texname = '\\text{I37a215}')

I37a216 = Parameter(name = 'I37a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE1x2x3)*complexconjugate(Rl6x6)',
                    texname = '\\text{I37a216}')

I37a233 = Parameter(name = 'I37a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE3x2x3)*complexconjugate(Rl3x6)',
                    texname = '\\text{I37a233}')

I37a234 = Parameter(name = 'I37a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE3x2x1)*complexconjugate(Rl4x4)',
                    texname = '\\text{I37a234}')

I37a235 = Parameter(name = 'I37a235',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE3x2x2)*complexconjugate(Rl5x5)',
                    texname = '\\text{I37a235}')

I37a236 = Parameter(name = 'I37a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE3x2x3)*complexconjugate(Rl6x6)',
                    texname = '\\text{I37a236}')

I37a313 = Parameter(name = 'I37a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE1x3x3)*complexconjugate(Rl3x6)',
                    texname = '\\text{I37a313}')

I37a314 = Parameter(name = 'I37a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE1x3x1)*complexconjugate(Rl4x4)',
                    texname = '\\text{I37a314}')

I37a315 = Parameter(name = 'I37a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE1x3x2)*complexconjugate(Rl5x5)',
                    texname = '\\text{I37a315}')

I37a316 = Parameter(name = 'I37a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE1x3x3)*complexconjugate(Rl6x6)',
                    texname = '\\text{I37a316}')

I37a323 = Parameter(name = 'I37a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE2x3x3)*complexconjugate(Rl3x6)',
                    texname = '\\text{I37a323}')

I37a324 = Parameter(name = 'I37a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE2x3x1)*complexconjugate(Rl4x4)',
                    texname = '\\text{I37a324}')

I37a325 = Parameter(name = 'I37a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE2x3x2)*complexconjugate(Rl5x5)',
                    texname = '\\text{I37a325}')

I37a326 = Parameter(name = 'I37a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LLLE2x3x3)*complexconjugate(Rl6x6)',
                    texname = '\\text{I37a326}')

I38a11 = Parameter(name = 'I38a11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x1*complexconjugate(Rl1x1)',
                   texname = '\\text{I38a11}')

I38a22 = Parameter(name = 'I38a22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x2*complexconjugate(Rl2x2)',
                   texname = '\\text{I38a22}')

I38a33 = Parameter(name = 'I38a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I38a33}')

I38a36 = Parameter(name = 'I38a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I38a36}')

I38a63 = Parameter(name = 'I38a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I38a63}')

I38a66 = Parameter(name = 'I38a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I38a66}')

I39a33 = Parameter(name = 'I39a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*complexconjugate(Rl3x6)',
                   texname = '\\text{I39a33}')

I39a36 = Parameter(name = 'I39a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*complexconjugate(Rl3x6)',
                   texname = '\\text{I39a36}')

I39a44 = Parameter(name = 'I39a44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x4*complexconjugate(Rl4x4)',
                   texname = '\\text{I39a44}')

I39a55 = Parameter(name = 'I39a55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl5x5*complexconjugate(Rl5x5)',
                   texname = '\\text{I39a55}')

I39a63 = Parameter(name = 'I39a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*complexconjugate(Rl6x6)',
                   texname = '\\text{I39a63}')

I39a66 = Parameter(name = 'I39a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*complexconjugate(Rl6x6)',
                   texname = '\\text{I39a66}')

I4a33 = Parameter(name = 'I4a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd3x3*complexconjugate(Rd3x3)',
                  texname = '\\text{I4a33}')

I4a36 = Parameter(name = 'I4a36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd3x3*complexconjugate(Rd6x3)',
                  texname = '\\text{I4a36}')

I40a33 = Parameter(name = 'I40a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(ye3x3)',
                   texname = '\\text{I40a33}')

I40a36 = Parameter(name = 'I40a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(ye3x3)',
                   texname = '\\text{I40a36}')

I41a33 = Parameter(name = 'I41a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3',
                   texname = '\\text{I41a33}')

I41a36 = Parameter(name = 'I41a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3',
                   texname = '\\text{I41a36}')

I42a11 = Parameter(name = 'I42a11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x1',
                   texname = '\\text{I42a11}')

I42a22 = Parameter(name = 'I42a22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x2',
                   texname = '\\text{I42a22}')

I42a33 = Parameter(name = 'I42a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3',
                   texname = '\\text{I42a33}')

I42a36 = Parameter(name = 'I42a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3',
                   texname = '\\text{I42a36}')

I43a33 = Parameter(name = 'I43a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3',
                   texname = '\\text{I43a33}')

I43a36 = Parameter(name = 'I43a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3',
                   texname = '\\text{I43a36}')

I44a112 = Parameter(name = 'I44a112',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLLE1x2x1)',
                    texname = '\\text{I44a112}')

I44a113 = Parameter(name = 'I44a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLLE1x3x1)',
                    texname = '\\text{I44a113}')

I44a116 = Parameter(name = 'I44a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLLE1x3x1)',
                    texname = '\\text{I44a116}')

I44a122 = Parameter(name = 'I44a122',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLLE1x2x2)',
                    texname = '\\text{I44a122}')

I44a123 = Parameter(name = 'I44a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLLE1x3x2)',
                    texname = '\\text{I44a123}')

I44a126 = Parameter(name = 'I44a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLLE1x3x2)',
                    texname = '\\text{I44a126}')

I44a132 = Parameter(name = 'I44a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLLE1x2x3)',
                    texname = '\\text{I44a132}')

I44a133 = Parameter(name = 'I44a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLLE1x3x3)',
                    texname = '\\text{I44a133}')

I44a136 = Parameter(name = 'I44a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLLE1x3x3)',
                    texname = '\\text{I44a136}')

I44a211 = Parameter(name = 'I44a211',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLLE2x1x1)',
                    texname = '\\text{I44a211}')

I44a213 = Parameter(name = 'I44a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLLE2x3x1)',
                    texname = '\\text{I44a213}')

I44a216 = Parameter(name = 'I44a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLLE2x3x1)',
                    texname = '\\text{I44a216}')

I44a221 = Parameter(name = 'I44a221',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLLE2x1x2)',
                    texname = '\\text{I44a221}')

I44a223 = Parameter(name = 'I44a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLLE2x3x2)',
                    texname = '\\text{I44a223}')

I44a226 = Parameter(name = 'I44a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLLE2x3x2)',
                    texname = '\\text{I44a226}')

I44a231 = Parameter(name = 'I44a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLLE2x1x3)',
                    texname = '\\text{I44a231}')

I44a233 = Parameter(name = 'I44a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLLE2x3x3)',
                    texname = '\\text{I44a233}')

I44a236 = Parameter(name = 'I44a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLLE2x3x3)',
                    texname = '\\text{I44a236}')

I44a311 = Parameter(name = 'I44a311',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLLE3x1x1)',
                    texname = '\\text{I44a311}')

I44a312 = Parameter(name = 'I44a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLLE3x2x1)',
                    texname = '\\text{I44a312}')

I44a321 = Parameter(name = 'I44a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLLE3x1x2)',
                    texname = '\\text{I44a321}')

I44a322 = Parameter(name = 'I44a322',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLLE3x2x2)',
                    texname = '\\text{I44a322}')

I44a331 = Parameter(name = 'I44a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLLE3x1x3)',
                    texname = '\\text{I44a331}')

I44a332 = Parameter(name = 'I44a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLLE3x2x3)',
                    texname = '\\text{I44a332}')

I45a111 = Parameter(name = 'I45a111',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLQD1x1x1)',
                    texname = '\\text{I45a111}')

I45a112 = Parameter(name = 'I45a112',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLQD2x1x1)',
                    texname = '\\text{I45a112}')

I45a113 = Parameter(name = 'I45a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLQD3x1x1)',
                    texname = '\\text{I45a113}')

I45a116 = Parameter(name = 'I45a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLQD3x1x1)',
                    texname = '\\text{I45a116}')

I45a121 = Parameter(name = 'I45a121',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLQD1x1x2)',
                    texname = '\\text{I45a121}')

I45a122 = Parameter(name = 'I45a122',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLQD2x1x2)',
                    texname = '\\text{I45a122}')

I45a123 = Parameter(name = 'I45a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLQD3x1x2)',
                    texname = '\\text{I45a123}')

I45a126 = Parameter(name = 'I45a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLQD3x1x2)',
                    texname = '\\text{I45a126}')

I45a131 = Parameter(name = 'I45a131',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLQD1x1x3)',
                    texname = '\\text{I45a131}')

I45a132 = Parameter(name = 'I45a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLQD2x1x3)',
                    texname = '\\text{I45a132}')

I45a133 = Parameter(name = 'I45a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLQD3x1x3)',
                    texname = '\\text{I45a133}')

I45a136 = Parameter(name = 'I45a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLQD3x1x3)',
                    texname = '\\text{I45a136}')

I45a211 = Parameter(name = 'I45a211',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLQD1x2x1)',
                    texname = '\\text{I45a211}')

I45a212 = Parameter(name = 'I45a212',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLQD2x2x1)',
                    texname = '\\text{I45a212}')

I45a213 = Parameter(name = 'I45a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLQD3x2x1)',
                    texname = '\\text{I45a213}')

I45a216 = Parameter(name = 'I45a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLQD3x2x1)',
                    texname = '\\text{I45a216}')

I45a221 = Parameter(name = 'I45a221',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLQD1x2x2)',
                    texname = '\\text{I45a221}')

I45a222 = Parameter(name = 'I45a222',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLQD2x2x2)',
                    texname = '\\text{I45a222}')

I45a223 = Parameter(name = 'I45a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLQD3x2x2)',
                    texname = '\\text{I45a223}')

I45a226 = Parameter(name = 'I45a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLQD3x2x2)',
                    texname = '\\text{I45a226}')

I45a231 = Parameter(name = 'I45a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLQD1x2x3)',
                    texname = '\\text{I45a231}')

I45a232 = Parameter(name = 'I45a232',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLQD2x2x3)',
                    texname = '\\text{I45a232}')

I45a233 = Parameter(name = 'I45a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLQD3x2x3)',
                    texname = '\\text{I45a233}')

I45a236 = Parameter(name = 'I45a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLQD3x2x3)',
                    texname = '\\text{I45a236}')

I45a311 = Parameter(name = 'I45a311',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLQD1x3x1)',
                    texname = '\\text{I45a311}')

I45a312 = Parameter(name = 'I45a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLQD2x3x1)',
                    texname = '\\text{I45a312}')

I45a313 = Parameter(name = 'I45a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLQD3x3x1)',
                    texname = '\\text{I45a313}')

I45a316 = Parameter(name = 'I45a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLQD3x3x1)',
                    texname = '\\text{I45a316}')

I45a321 = Parameter(name = 'I45a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLQD1x3x2)',
                    texname = '\\text{I45a321}')

I45a322 = Parameter(name = 'I45a322',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLQD2x3x2)',
                    texname = '\\text{I45a322}')

I45a323 = Parameter(name = 'I45a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLQD3x3x2)',
                    texname = '\\text{I45a323}')

I45a326 = Parameter(name = 'I45a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLQD3x3x2)',
                    texname = '\\text{I45a326}')

I45a331 = Parameter(name = 'I45a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*complexconjugate(LLQD1x3x3)',
                    texname = '\\text{I45a331}')

I45a332 = Parameter(name = 'I45a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*complexconjugate(LLQD2x3x3)',
                    texname = '\\text{I45a332}')

I45a333 = Parameter(name = 'I45a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*complexconjugate(LLQD3x3x3)',
                    texname = '\\text{I45a333}')

I45a336 = Parameter(name = 'I45a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*complexconjugate(LLQD3x3x3)',
                    texname = '\\text{I45a336}')

I46a123 = Parameter(name = 'I46a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x3*Rl3x6',
                    texname = '\\text{I46a123}')

I46a124 = Parameter(name = 'I46a124',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x1*Rl4x4',
                    texname = '\\text{I46a124}')

I46a125 = Parameter(name = 'I46a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x2*Rl5x5',
                    texname = '\\text{I46a125}')

I46a126 = Parameter(name = 'I46a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x3*Rl6x6',
                    texname = '\\text{I46a126}')

I46a133 = Parameter(name = 'I46a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x3*Rl3x6',
                    texname = '\\text{I46a133}')

I46a134 = Parameter(name = 'I46a134',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x1*Rl4x4',
                    texname = '\\text{I46a134}')

I46a135 = Parameter(name = 'I46a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x2*Rl5x5',
                    texname = '\\text{I46a135}')

I46a136 = Parameter(name = 'I46a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x3*Rl6x6',
                    texname = '\\text{I46a136}')

I46a213 = Parameter(name = 'I46a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x3*Rl3x6',
                    texname = '\\text{I46a213}')

I46a214 = Parameter(name = 'I46a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x1*Rl4x4',
                    texname = '\\text{I46a214}')

I46a215 = Parameter(name = 'I46a215',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x2*Rl5x5',
                    texname = '\\text{I46a215}')

I46a216 = Parameter(name = 'I46a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x3*Rl6x6',
                    texname = '\\text{I46a216}')

I46a233 = Parameter(name = 'I46a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x3*Rl3x6',
                    texname = '\\text{I46a233}')

I46a234 = Parameter(name = 'I46a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x1*Rl4x4',
                    texname = '\\text{I46a234}')

I46a235 = Parameter(name = 'I46a235',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x2*Rl5x5',
                    texname = '\\text{I46a235}')

I46a236 = Parameter(name = 'I46a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x3*Rl6x6',
                    texname = '\\text{I46a236}')

I46a313 = Parameter(name = 'I46a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*Rl3x6',
                    texname = '\\text{I46a313}')

I46a314 = Parameter(name = 'I46a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x1*Rl4x4',
                    texname = '\\text{I46a314}')

I46a315 = Parameter(name = 'I46a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x2*Rl5x5',
                    texname = '\\text{I46a315}')

I46a316 = Parameter(name = 'I46a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*Rl6x6',
                    texname = '\\text{I46a316}')

I46a323 = Parameter(name = 'I46a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*Rl3x6',
                    texname = '\\text{I46a323}')

I46a324 = Parameter(name = 'I46a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x1*Rl4x4',
                    texname = '\\text{I46a324}')

I46a325 = Parameter(name = 'I46a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x2*Rl5x5',
                    texname = '\\text{I46a325}')

I46a326 = Parameter(name = 'I46a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*Rl6x6',
                    texname = '\\text{I46a326}')

I47a11 = Parameter(name = 'I47a11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x1*complexconjugate(Rl1x1)',
                   texname = '\\text{I47a11}')

I47a22 = Parameter(name = 'I47a22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x2*complexconjugate(Rl2x2)',
                   texname = '\\text{I47a22}')

I47a33 = Parameter(name = 'I47a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I47a33}')

I47a36 = Parameter(name = 'I47a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I47a36}')

I47a63 = Parameter(name = 'I47a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I47a63}')

I47a66 = Parameter(name = 'I47a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I47a66}')

I48a33 = Parameter(name = 'I48a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*complexconjugate(Rl3x6)',
                   texname = '\\text{I48a33}')

I48a36 = Parameter(name = 'I48a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*complexconjugate(Rl3x6)',
                   texname = '\\text{I48a36}')

I48a44 = Parameter(name = 'I48a44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x4*complexconjugate(Rl4x4)',
                   texname = '\\text{I48a44}')

I48a55 = Parameter(name = 'I48a55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl5x5*complexconjugate(Rl5x5)',
                   texname = '\\text{I48a55}')

I48a63 = Parameter(name = 'I48a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*complexconjugate(Rl6x6)',
                   texname = '\\text{I48a63}')

I48a66 = Parameter(name = 'I48a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*complexconjugate(Rl6x6)',
                   texname = '\\text{I48a66}')

I49a33 = Parameter(name = 'I49a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl3x6)*complexconjugate(te3x3)',
                   texname = '\\text{I49a33}')

I49a36 = Parameter(name = 'I49a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl3x6)*complexconjugate(te3x3)',
                   texname = '\\text{I49a36}')

I49a63 = Parameter(name = 'I49a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)',
                   texname = '\\text{I49a63}')

I49a66 = Parameter(name = 'I49a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)',
                   texname = '\\text{I49a66}')

I5a111 = Parameter(name = 'I5a111',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x1x1*complexconjugate(Rd1x1)',
                   texname = '\\text{I5a111}')

I5a112 = Parameter(name = 'I5a112',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x2x1*complexconjugate(Rd2x2)',
                   texname = '\\text{I5a112}')

I5a113 = Parameter(name = 'I5a113',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x3x1*complexconjugate(Rd3x3)',
                   texname = '\\text{I5a113}')

I5a116 = Parameter(name = 'I5a116',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x3x1*complexconjugate(Rd6x3)',
                   texname = '\\text{I5a116}')

I5a121 = Parameter(name = 'I5a121',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x1x1*complexconjugate(Rd1x1)',
                   texname = '\\text{I5a121}')

I5a122 = Parameter(name = 'I5a122',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x2x1*complexconjugate(Rd2x2)',
                   texname = '\\text{I5a122}')

I5a123 = Parameter(name = 'I5a123',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x3x1*complexconjugate(Rd3x3)',
                   texname = '\\text{I5a123}')

I5a126 = Parameter(name = 'I5a126',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x3x1*complexconjugate(Rd6x3)',
                   texname = '\\text{I5a126}')

I5a131 = Parameter(name = 'I5a131',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x1x1*complexconjugate(Rd1x1)',
                   texname = '\\text{I5a131}')

I5a132 = Parameter(name = 'I5a132',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x2x1*complexconjugate(Rd2x2)',
                   texname = '\\text{I5a132}')

I5a133 = Parameter(name = 'I5a133',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x3x1*complexconjugate(Rd3x3)',
                   texname = '\\text{I5a133}')

I5a136 = Parameter(name = 'I5a136',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x3x1*complexconjugate(Rd6x3)',
                   texname = '\\text{I5a136}')

I5a211 = Parameter(name = 'I5a211',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x1x2*complexconjugate(Rd1x1)',
                   texname = '\\text{I5a211}')

I5a212 = Parameter(name = 'I5a212',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x2x2*complexconjugate(Rd2x2)',
                   texname = '\\text{I5a212}')

I5a213 = Parameter(name = 'I5a213',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x3x2*complexconjugate(Rd3x3)',
                   texname = '\\text{I5a213}')

I5a216 = Parameter(name = 'I5a216',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x3x2*complexconjugate(Rd6x3)',
                   texname = '\\text{I5a216}')

I5a221 = Parameter(name = 'I5a221',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x1x2*complexconjugate(Rd1x1)',
                   texname = '\\text{I5a221}')

I5a222 = Parameter(name = 'I5a222',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x2x2*complexconjugate(Rd2x2)',
                   texname = '\\text{I5a222}')

I5a223 = Parameter(name = 'I5a223',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x3x2*complexconjugate(Rd3x3)',
                   texname = '\\text{I5a223}')

I5a226 = Parameter(name = 'I5a226',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x3x2*complexconjugate(Rd6x3)',
                   texname = '\\text{I5a226}')

I5a231 = Parameter(name = 'I5a231',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x1x2*complexconjugate(Rd1x1)',
                   texname = '\\text{I5a231}')

I5a232 = Parameter(name = 'I5a232',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x2x2*complexconjugate(Rd2x2)',
                   texname = '\\text{I5a232}')

I5a233 = Parameter(name = 'I5a233',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x3x2*complexconjugate(Rd3x3)',
                   texname = '\\text{I5a233}')

I5a236 = Parameter(name = 'I5a236',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x3x2*complexconjugate(Rd6x3)',
                   texname = '\\text{I5a236}')

I5a311 = Parameter(name = 'I5a311',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x1x3*complexconjugate(Rd1x1)',
                   texname = '\\text{I5a311}')

I5a312 = Parameter(name = 'I5a312',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x2x3*complexconjugate(Rd2x2)',
                   texname = '\\text{I5a312}')

I5a313 = Parameter(name = 'I5a313',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I5a313}')

I5a316 = Parameter(name = 'I5a316',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD1x3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I5a316}')

I5a321 = Parameter(name = 'I5a321',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x1x3*complexconjugate(Rd1x1)',
                   texname = '\\text{I5a321}')

I5a322 = Parameter(name = 'I5a322',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x2x3*complexconjugate(Rd2x2)',
                   texname = '\\text{I5a322}')

I5a323 = Parameter(name = 'I5a323',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I5a323}')

I5a326 = Parameter(name = 'I5a326',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD2x3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I5a326}')

I5a331 = Parameter(name = 'I5a331',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x1x3*complexconjugate(Rd1x1)',
                   texname = '\\text{I5a331}')

I5a332 = Parameter(name = 'I5a332',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x2x3*complexconjugate(Rd2x2)',
                   texname = '\\text{I5a332}')

I5a333 = Parameter(name = 'I5a333',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I5a333}')

I5a336 = Parameter(name = 'I5a336',
                   nature = 'internal',
                   type = 'complex',
                   value = 'LLQD3x3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I5a336}')

I50a33 = Parameter(name = 'I50a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I50a33}')

I50a36 = Parameter(name = 'I50a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I50a36}')

I50a63 = Parameter(name = 'I50a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I50a63}')

I50a66 = Parameter(name = 'I50a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I50a66}')

I51a33 = Parameter(name = 'I51a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*te3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I51a33}')

I51a36 = Parameter(name = 'I51a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*te3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I51a36}')

I51a63 = Parameter(name = 'I51a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*te3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I51a63}')

I51a66 = Parameter(name = 'I51a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*te3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I51a66}')

I52a33 = Parameter(name = 'I52a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I52a33}')

I52a36 = Parameter(name = 'I52a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I52a36}')

I52a63 = Parameter(name = 'I52a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*ye3x3*complexconjugate(Rl6x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I52a63}')

I52a66 = Parameter(name = 'I52a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*ye3x3*complexconjugate(Rl6x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I52a66}')

I53a33 = Parameter(name = 'I53a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I53a33}')

I53a36 = Parameter(name = 'I53a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I53a36}')

I53a63 = Parameter(name = 'I53a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I53a63}')

I53a66 = Parameter(name = 'I53a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I53a66}')

I54a33 = Parameter(name = 'I54a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3*complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I54a33}')

I54a36 = Parameter(name = 'I54a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3*complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I54a36}')

I54a63 = Parameter(name = 'I54a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I54a63}')

I54a66 = Parameter(name = 'I54a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I54a66}')

I55a112 = Parameter(name = 'I55a112',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x1*complexconjugate(Rn2x2)',
                    texname = '\\text{I55a112}')

I55a113 = Parameter(name = 'I55a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x1*complexconjugate(Rn3x3)',
                    texname = '\\text{I55a113}')

I55a121 = Parameter(name = 'I55a121',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x1*complexconjugate(Rn1x1)',
                    texname = '\\text{I55a121}')

I55a123 = Parameter(name = 'I55a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x1*complexconjugate(Rn3x3)',
                    texname = '\\text{I55a123}')

I55a131 = Parameter(name = 'I55a131',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x1*complexconjugate(Rn1x1)',
                    texname = '\\text{I55a131}')

I55a132 = Parameter(name = 'I55a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x1*complexconjugate(Rn2x2)',
                    texname = '\\text{I55a132}')

I55a212 = Parameter(name = 'I55a212',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x2*complexconjugate(Rn2x2)',
                    texname = '\\text{I55a212}')

I55a213 = Parameter(name = 'I55a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x2*complexconjugate(Rn3x3)',
                    texname = '\\text{I55a213}')

I55a221 = Parameter(name = 'I55a221',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x2*complexconjugate(Rn1x1)',
                    texname = '\\text{I55a221}')

I55a223 = Parameter(name = 'I55a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x2*complexconjugate(Rn3x3)',
                    texname = '\\text{I55a223}')

I55a231 = Parameter(name = 'I55a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x2*complexconjugate(Rn1x1)',
                    texname = '\\text{I55a231}')

I55a232 = Parameter(name = 'I55a232',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x2*complexconjugate(Rn2x2)',
                    texname = '\\text{I55a232}')

I55a312 = Parameter(name = 'I55a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x3*complexconjugate(Rn2x2)',
                    texname = '\\text{I55a312}')

I55a313 = Parameter(name = 'I55a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x3*complexconjugate(Rn3x3)',
                    texname = '\\text{I55a313}')

I55a321 = Parameter(name = 'I55a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x3*complexconjugate(Rn1x1)',
                    texname = '\\text{I55a321}')

I55a323 = Parameter(name = 'I55a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x3*complexconjugate(Rn3x3)',
                    texname = '\\text{I55a323}')

I55a331 = Parameter(name = 'I55a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*complexconjugate(Rn1x1)',
                    texname = '\\text{I55a331}')

I55a332 = Parameter(name = 'I55a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*complexconjugate(Rn2x2)',
                    texname = '\\text{I55a332}')

I56a111 = Parameter(name = 'I56a111',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x1*complexconjugate(Rn1x1)',
                    texname = '\\text{I56a111}')

I56a112 = Parameter(name = 'I56a112',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x1*complexconjugate(Rn2x2)',
                    texname = '\\text{I56a112}')

I56a113 = Parameter(name = 'I56a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x1*complexconjugate(Rn3x3)',
                    texname = '\\text{I56a113}')

I56a121 = Parameter(name = 'I56a121',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x1*complexconjugate(Rn1x1)',
                    texname = '\\text{I56a121}')

I56a122 = Parameter(name = 'I56a122',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x1*complexconjugate(Rn2x2)',
                    texname = '\\text{I56a122}')

I56a123 = Parameter(name = 'I56a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x1*complexconjugate(Rn3x3)',
                    texname = '\\text{I56a123}')

I56a131 = Parameter(name = 'I56a131',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x1*complexconjugate(Rn1x1)',
                    texname = '\\text{I56a131}')

I56a132 = Parameter(name = 'I56a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x1*complexconjugate(Rn2x2)',
                    texname = '\\text{I56a132}')

I56a133 = Parameter(name = 'I56a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*complexconjugate(Rn3x3)',
                    texname = '\\text{I56a133}')

I56a211 = Parameter(name = 'I56a211',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x2*complexconjugate(Rn1x1)',
                    texname = '\\text{I56a211}')

I56a212 = Parameter(name = 'I56a212',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x2*complexconjugate(Rn2x2)',
                    texname = '\\text{I56a212}')

I56a213 = Parameter(name = 'I56a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x2*complexconjugate(Rn3x3)',
                    texname = '\\text{I56a213}')

I56a221 = Parameter(name = 'I56a221',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x2*complexconjugate(Rn1x1)',
                    texname = '\\text{I56a221}')

I56a222 = Parameter(name = 'I56a222',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x2*complexconjugate(Rn2x2)',
                    texname = '\\text{I56a222}')

I56a223 = Parameter(name = 'I56a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x2*complexconjugate(Rn3x3)',
                    texname = '\\text{I56a223}')

I56a231 = Parameter(name = 'I56a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x2*complexconjugate(Rn1x1)',
                    texname = '\\text{I56a231}')

I56a232 = Parameter(name = 'I56a232',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x2*complexconjugate(Rn2x2)',
                    texname = '\\text{I56a232}')

I56a233 = Parameter(name = 'I56a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*complexconjugate(Rn3x3)',
                    texname = '\\text{I56a233}')

I56a311 = Parameter(name = 'I56a311',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*complexconjugate(Rn1x1)',
                    texname = '\\text{I56a311}')

I56a312 = Parameter(name = 'I56a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*complexconjugate(Rn2x2)',
                    texname = '\\text{I56a312}')

I56a313 = Parameter(name = 'I56a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*complexconjugate(Rn3x3)',
                    texname = '\\text{I56a313}')

I56a321 = Parameter(name = 'I56a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*complexconjugate(Rn1x1)',
                    texname = '\\text{I56a321}')

I56a322 = Parameter(name = 'I56a322',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*complexconjugate(Rn2x2)',
                    texname = '\\text{I56a322}')

I56a323 = Parameter(name = 'I56a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*complexconjugate(Rn3x3)',
                    texname = '\\text{I56a323}')

I56a331 = Parameter(name = 'I56a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*complexconjugate(Rn1x1)',
                    texname = '\\text{I56a331}')

I56a332 = Parameter(name = 'I56a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*complexconjugate(Rn2x2)',
                    texname = '\\text{I56a332}')

I56a333 = Parameter(name = 'I56a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*complexconjugate(Rn3x3)',
                    texname = '\\text{I56a333}')

I57a11 = Parameter(name = 'I57a11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x1*complexconjugate(Rn1x1)',
                   texname = '\\text{I57a11}')

I57a22 = Parameter(name = 'I57a22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x2*complexconjugate(Rn2x2)',
                   texname = '\\text{I57a22}')

I57a33 = Parameter(name = 'I57a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I57a33}')

I57a36 = Parameter(name = 'I57a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I57a36}')

I58a33 = Parameter(name = 'I58a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*te3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I58a33}')

I58a36 = Parameter(name = 'I58a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*te3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I58a36}')

I59a33 = Parameter(name = 'I59a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I59a33}')

I59a36 = Parameter(name = 'I59a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I59a36}')

I6a113 = Parameter(name = 'I6a113',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x1x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I6a113}')

I6a114 = Parameter(name = 'I6a114',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x1x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I6a114}')

I6a115 = Parameter(name = 'I6a115',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x1x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I6a115}')

I6a116 = Parameter(name = 'I6a116',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x1x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I6a116}')

I6a123 = Parameter(name = 'I6a123',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x1x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I6a123}')

I6a124 = Parameter(name = 'I6a124',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x1x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I6a124}')

I6a125 = Parameter(name = 'I6a125',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x1x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I6a125}')

I6a126 = Parameter(name = 'I6a126',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x1x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I6a126}')

I6a133 = Parameter(name = 'I6a133',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x1x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I6a133}')

I6a134 = Parameter(name = 'I6a134',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x1x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I6a134}')

I6a135 = Parameter(name = 'I6a135',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x1x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I6a135}')

I6a136 = Parameter(name = 'I6a136',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x1x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I6a136}')

I6a213 = Parameter(name = 'I6a213',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x2x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I6a213}')

I6a214 = Parameter(name = 'I6a214',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x2x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I6a214}')

I6a215 = Parameter(name = 'I6a215',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x2x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I6a215}')

I6a216 = Parameter(name = 'I6a216',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x2x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I6a216}')

I6a223 = Parameter(name = 'I6a223',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x2x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I6a223}')

I6a224 = Parameter(name = 'I6a224',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x2x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I6a224}')

I6a225 = Parameter(name = 'I6a225',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x2x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I6a225}')

I6a226 = Parameter(name = 'I6a226',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x2x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I6a226}')

I6a233 = Parameter(name = 'I6a233',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x2x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I6a233}')

I6a234 = Parameter(name = 'I6a234',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x2x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I6a234}')

I6a235 = Parameter(name = 'I6a235',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x2x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I6a235}')

I6a236 = Parameter(name = 'I6a236',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x2x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I6a236}')

I6a313 = Parameter(name = 'I6a313',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x3x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I6a313}')

I6a314 = Parameter(name = 'I6a314',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x3x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I6a314}')

I6a315 = Parameter(name = 'I6a315',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x3x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I6a315}')

I6a316 = Parameter(name = 'I6a316',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x3x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I6a316}')

I6a323 = Parameter(name = 'I6a323',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x3x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I6a323}')

I6a324 = Parameter(name = 'I6a324',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x3x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I6a324}')

I6a325 = Parameter(name = 'I6a325',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x3x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I6a325}')

I6a326 = Parameter(name = 'I6a326',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x3x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I6a326}')

I6a333 = Parameter(name = 'I6a333',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I6a333}')

I6a334 = Parameter(name = 'I6a334',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I6a334}')

I6a335 = Parameter(name = 'I6a335',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I6a335}')

I6a336 = Parameter(name = 'I6a336',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I6a336}')

I60a33 = Parameter(name = 'I60a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I60a33}')

I60a36 = Parameter(name = 'I60a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I60a36}')

I61a113 = Parameter(name = 'I61a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*Rd3x3*complexconjugate(Rd1x1)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a113}')

I61a116 = Parameter(name = 'I61a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*Rd6x3*complexconjugate(Rd1x1)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a116}')

I61a123 = Parameter(name = 'I61a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*Rd3x3*complexconjugate(Rd2x2)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a123}')

I61a126 = Parameter(name = 'I61a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*Rd6x3*complexconjugate(Rd2x2)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a126}')

I61a133 = Parameter(name = 'I61a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd3x3*complexconjugate(Rd3x3)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a133}')

I61a136 = Parameter(name = 'I61a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd6x3*complexconjugate(Rd3x3)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a136}')

I61a163 = Parameter(name = 'I61a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd3x3*complexconjugate(Rd6x3)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a163}')

I61a166 = Parameter(name = 'I61a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd6x3*complexconjugate(Rd6x3)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a166}')

I61a213 = Parameter(name = 'I61a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*Rd3x3*complexconjugate(Rd1x1)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a213}')

I61a216 = Parameter(name = 'I61a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*Rd6x3*complexconjugate(Rd1x1)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a216}')

I61a223 = Parameter(name = 'I61a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*Rd3x3*complexconjugate(Rd2x2)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a223}')

I61a226 = Parameter(name = 'I61a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*Rd6x3*complexconjugate(Rd2x2)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a226}')

I61a233 = Parameter(name = 'I61a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd3x3*complexconjugate(Rd3x3)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a233}')

I61a236 = Parameter(name = 'I61a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd6x3*complexconjugate(Rd3x3)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a236}')

I61a263 = Parameter(name = 'I61a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd3x3*complexconjugate(Rd6x3)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a263}')

I61a266 = Parameter(name = 'I61a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd6x3*complexconjugate(Rd6x3)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a266}')

I61a313 = Parameter(name = 'I61a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd3x3*complexconjugate(Rd1x1)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a313}')

I61a316 = Parameter(name = 'I61a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd6x3*complexconjugate(Rd1x1)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a316}')

I61a323 = Parameter(name = 'I61a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd3x3*complexconjugate(Rd2x2)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a323}')

I61a326 = Parameter(name = 'I61a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd6x3*complexconjugate(Rd2x2)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a326}')

I61a333 = Parameter(name = 'I61a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x3*complexconjugate(Rd3x3)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a333}')

I61a336 = Parameter(name = 'I61a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x3*complexconjugate(Rd3x3)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a336}')

I61a363 = Parameter(name = 'I61a363',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x3*complexconjugate(Rd6x3)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a363}')

I61a366 = Parameter(name = 'I61a366',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x3*complexconjugate(Rd6x3)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I61a366}')

I62a133 = Parameter(name = 'I62a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd3x6*complexconjugate(Rd3x6)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a133}')

I62a134 = Parameter(name = 'I62a134',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x1*Rd4x4*complexconjugate(Rd3x6)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a134}')

I62a135 = Parameter(name = 'I62a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x2*Rd5x5*complexconjugate(Rd3x6)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a135}')

I62a136 = Parameter(name = 'I62a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd6x6*complexconjugate(Rd3x6)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a136}')

I62a163 = Parameter(name = 'I62a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd3x6*complexconjugate(Rd6x6)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a163}')

I62a164 = Parameter(name = 'I62a164',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x1*Rd4x4*complexconjugate(Rd6x6)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a164}')

I62a165 = Parameter(name = 'I62a165',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x2*Rd5x5*complexconjugate(Rd6x6)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a165}')

I62a166 = Parameter(name = 'I62a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd6x6*complexconjugate(Rd6x6)*complexconjugate(Rn1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a166}')

I62a233 = Parameter(name = 'I62a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd3x6*complexconjugate(Rd3x6)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a233}')

I62a234 = Parameter(name = 'I62a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x1*Rd4x4*complexconjugate(Rd3x6)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a234}')

I62a235 = Parameter(name = 'I62a235',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x2*Rd5x5*complexconjugate(Rd3x6)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a235}')

I62a236 = Parameter(name = 'I62a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd6x6*complexconjugate(Rd3x6)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a236}')

I62a263 = Parameter(name = 'I62a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd3x6*complexconjugate(Rd6x6)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a263}')

I62a264 = Parameter(name = 'I62a264',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x1*Rd4x4*complexconjugate(Rd6x6)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a264}')

I62a265 = Parameter(name = 'I62a265',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x2*Rd5x5*complexconjugate(Rd6x6)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a265}')

I62a266 = Parameter(name = 'I62a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd6x6*complexconjugate(Rd6x6)*complexconjugate(Rn2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a266}')

I62a333 = Parameter(name = 'I62a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6*complexconjugate(Rd3x6)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a333}')

I62a334 = Parameter(name = 'I62a334',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4*complexconjugate(Rd3x6)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a334}')

I62a335 = Parameter(name = 'I62a335',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5*complexconjugate(Rd3x6)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a335}')

I62a336 = Parameter(name = 'I62a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6*complexconjugate(Rd3x6)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a336}')

I62a363 = Parameter(name = 'I62a363',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6*complexconjugate(Rd6x6)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a363}')

I62a364 = Parameter(name = 'I62a364',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4*complexconjugate(Rd6x6)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a364}')

I62a365 = Parameter(name = 'I62a365',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5*complexconjugate(Rd6x6)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a365}')

I62a366 = Parameter(name = 'I62a366',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6*complexconjugate(Rd6x6)*complexconjugate(Rn3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I62a366}')

I63a113 = Parameter(name = 'I63a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD1x1x3*complexconjugate(Rd1x1)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a113}')

I63a114 = Parameter(name = 'I63a114',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD1x1x1*complexconjugate(Rd1x1)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a114}')

I63a115 = Parameter(name = 'I63a115',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD1x1x2*complexconjugate(Rd1x1)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a115}')

I63a116 = Parameter(name = 'I63a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD1x1x3*complexconjugate(Rd1x1)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a116}')

I63a123 = Parameter(name = 'I63a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD1x2x3*complexconjugate(Rd2x2)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a123}')

I63a124 = Parameter(name = 'I63a124',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD1x2x1*complexconjugate(Rd2x2)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a124}')

I63a125 = Parameter(name = 'I63a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD1x2x2*complexconjugate(Rd2x2)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a125}')

I63a126 = Parameter(name = 'I63a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD1x2x3*complexconjugate(Rd2x2)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a126}')

I63a133 = Parameter(name = 'I63a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD1x3x3*complexconjugate(Rd3x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a133}')

I63a134 = Parameter(name = 'I63a134',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD1x3x1*complexconjugate(Rd3x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a134}')

I63a135 = Parameter(name = 'I63a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD1x3x2*complexconjugate(Rd3x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a135}')

I63a136 = Parameter(name = 'I63a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD1x3x3*complexconjugate(Rd3x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a136}')

I63a163 = Parameter(name = 'I63a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD1x3x3*complexconjugate(Rd6x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a163}')

I63a164 = Parameter(name = 'I63a164',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD1x3x1*complexconjugate(Rd6x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a164}')

I63a165 = Parameter(name = 'I63a165',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD1x3x2*complexconjugate(Rd6x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a165}')

I63a166 = Parameter(name = 'I63a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD1x3x3*complexconjugate(Rd6x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I63a166}')

I63a213 = Parameter(name = 'I63a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD2x1x3*complexconjugate(Rd1x1)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a213}')

I63a214 = Parameter(name = 'I63a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD2x1x1*complexconjugate(Rd1x1)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a214}')

I63a215 = Parameter(name = 'I63a215',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD2x1x2*complexconjugate(Rd1x1)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a215}')

I63a216 = Parameter(name = 'I63a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD2x1x3*complexconjugate(Rd1x1)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a216}')

I63a223 = Parameter(name = 'I63a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD2x2x3*complexconjugate(Rd2x2)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a223}')

I63a224 = Parameter(name = 'I63a224',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD2x2x1*complexconjugate(Rd2x2)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a224}')

I63a225 = Parameter(name = 'I63a225',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD2x2x2*complexconjugate(Rd2x2)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a225}')

I63a226 = Parameter(name = 'I63a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD2x2x3*complexconjugate(Rd2x2)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a226}')

I63a233 = Parameter(name = 'I63a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD2x3x3*complexconjugate(Rd3x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a233}')

I63a234 = Parameter(name = 'I63a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD2x3x1*complexconjugate(Rd3x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a234}')

I63a235 = Parameter(name = 'I63a235',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD2x3x2*complexconjugate(Rd3x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a235}')

I63a236 = Parameter(name = 'I63a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD2x3x3*complexconjugate(Rd3x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a236}')

I63a263 = Parameter(name = 'I63a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD2x3x3*complexconjugate(Rd6x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a263}')

I63a264 = Parameter(name = 'I63a264',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD2x3x1*complexconjugate(Rd6x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a264}')

I63a265 = Parameter(name = 'I63a265',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD2x3x2*complexconjugate(Rd6x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a265}')

I63a266 = Parameter(name = 'I63a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD2x3x3*complexconjugate(Rd6x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I63a266}')

I63a313 = Parameter(name = 'I63a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x1x3*complexconjugate(Rd1x1)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a313}')

I63a314 = Parameter(name = 'I63a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x1x1*complexconjugate(Rd1x1)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a314}')

I63a315 = Parameter(name = 'I63a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x1x2*complexconjugate(Rd1x1)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a315}')

I63a316 = Parameter(name = 'I63a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x1x3*complexconjugate(Rd1x1)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a316}')

I63a323 = Parameter(name = 'I63a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x2x3*complexconjugate(Rd2x2)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a323}')

I63a324 = Parameter(name = 'I63a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x2x1*complexconjugate(Rd2x2)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a324}')

I63a325 = Parameter(name = 'I63a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x2x2*complexconjugate(Rd2x2)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a325}')

I63a326 = Parameter(name = 'I63a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x2x3*complexconjugate(Rd2x2)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a326}')

I63a333 = Parameter(name = 'I63a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x3x3*complexconjugate(Rd3x3)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a333}')

I63a334 = Parameter(name = 'I63a334',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x3x1*complexconjugate(Rd3x3)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a334}')

I63a335 = Parameter(name = 'I63a335',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x3x2*complexconjugate(Rd3x3)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a335}')

I63a336 = Parameter(name = 'I63a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x3x3*complexconjugate(Rd3x3)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a336}')

I63a363 = Parameter(name = 'I63a363',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x3x3*complexconjugate(Rd6x3)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a363}')

I63a364 = Parameter(name = 'I63a364',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x3x1*complexconjugate(Rd6x3)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a364}')

I63a365 = Parameter(name = 'I63a365',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x3x2*complexconjugate(Rd6x3)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a365}')

I63a366 = Parameter(name = 'I63a366',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x3x3*complexconjugate(Rd6x3)*complexconjugate(Rn3x3)',
                    texname = '\\text{I63a366}')

I64a123 = Parameter(name = 'I64a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x3*Rl3x3*complexconjugate(Rl2x2)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a123}')

I64a126 = Parameter(name = 'I64a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x2x3*Rl6x3*complexconjugate(Rl2x2)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a126}')

I64a133 = Parameter(name = 'I64a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*Rl3x3*complexconjugate(Rl3x3)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a133}')

I64a136 = Parameter(name = 'I64a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*Rl6x3*complexconjugate(Rl3x3)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a136}')

I64a163 = Parameter(name = 'I64a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*Rl3x3*complexconjugate(Rl6x3)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a163}')

I64a166 = Parameter(name = 'I64a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*Rl6x3*complexconjugate(Rl6x3)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a166}')

I64a213 = Parameter(name = 'I64a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x3*Rl3x3*complexconjugate(Rl1x1)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a213}')

I64a216 = Parameter(name = 'I64a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x1x3*Rl6x3*complexconjugate(Rl1x1)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a216}')

I64a233 = Parameter(name = 'I64a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*Rl3x3*complexconjugate(Rl3x3)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a233}')

I64a236 = Parameter(name = 'I64a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*Rl6x3*complexconjugate(Rl3x3)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a236}')

I64a263 = Parameter(name = 'I64a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*Rl3x3*complexconjugate(Rl6x3)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a263}')

I64a266 = Parameter(name = 'I64a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*Rl6x3*complexconjugate(Rl6x3)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a266}')

I64a313 = Parameter(name = 'I64a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x3*Rl3x3*complexconjugate(Rl1x1)*complexconjugate(Rn3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a313}')

I64a316 = Parameter(name = 'I64a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x1x3*Rl6x3*complexconjugate(Rl1x1)*complexconjugate(Rn3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a316}')

I64a323 = Parameter(name = 'I64a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x3*Rl3x3*complexconjugate(Rl2x2)*complexconjugate(Rn3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a323}')

I64a326 = Parameter(name = 'I64a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE3x2x3*Rl6x3*complexconjugate(Rl2x2)*complexconjugate(Rn3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I64a326}')

I65a133 = Parameter(name = 'I65a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*Rl3x6*complexconjugate(Rl3x6)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a133}')

I65a134 = Parameter(name = 'I65a134',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x1*Rl4x4*complexconjugate(Rl3x6)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a134}')

I65a135 = Parameter(name = 'I65a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x2*Rl5x5*complexconjugate(Rl3x6)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a135}')

I65a136 = Parameter(name = 'I65a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*Rl6x6*complexconjugate(Rl3x6)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a136}')

I65a163 = Parameter(name = 'I65a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*Rl3x6*complexconjugate(Rl6x6)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a163}')

I65a164 = Parameter(name = 'I65a164',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x1*Rl4x4*complexconjugate(Rl6x6)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a164}')

I65a165 = Parameter(name = 'I65a165',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x2*Rl5x5*complexconjugate(Rl6x6)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a165}')

I65a166 = Parameter(name = 'I65a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE1x3x3*Rl6x6*complexconjugate(Rl6x6)*complexconjugate(Rn1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a166}')

I65a233 = Parameter(name = 'I65a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*Rl3x6*complexconjugate(Rl3x6)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a233}')

I65a234 = Parameter(name = 'I65a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x1*Rl4x4*complexconjugate(Rl3x6)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a234}')

I65a235 = Parameter(name = 'I65a235',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x2*Rl5x5*complexconjugate(Rl3x6)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a235}')

I65a236 = Parameter(name = 'I65a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*Rl6x6*complexconjugate(Rl3x6)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a236}')

I65a263 = Parameter(name = 'I65a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*Rl3x6*complexconjugate(Rl6x6)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a263}')

I65a264 = Parameter(name = 'I65a264',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x1*Rl4x4*complexconjugate(Rl6x6)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a264}')

I65a265 = Parameter(name = 'I65a265',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x2*Rl5x5*complexconjugate(Rl6x6)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a265}')

I65a266 = Parameter(name = 'I65a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLLE2x3x3*Rl6x6*complexconjugate(Rl6x6)*complexconjugate(Rn2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I65a266}')

I66a123 = Parameter(name = 'I66a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*TLLE1x2x3*complexconjugate(Rl2x2)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a123}')

I66a124 = Parameter(name = 'I66a124',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x4*TLLE1x2x1*complexconjugate(Rl2x2)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a124}')

I66a125 = Parameter(name = 'I66a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl5x5*TLLE1x2x2*complexconjugate(Rl2x2)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a125}')

I66a126 = Parameter(name = 'I66a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*TLLE1x2x3*complexconjugate(Rl2x2)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a126}')

I66a133 = Parameter(name = 'I66a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*TLLE1x3x3*complexconjugate(Rl3x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a133}')

I66a134 = Parameter(name = 'I66a134',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x4*TLLE1x3x1*complexconjugate(Rl3x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a134}')

I66a135 = Parameter(name = 'I66a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl5x5*TLLE1x3x2*complexconjugate(Rl3x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a135}')

I66a136 = Parameter(name = 'I66a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*TLLE1x3x3*complexconjugate(Rl3x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a136}')

I66a163 = Parameter(name = 'I66a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*TLLE1x3x3*complexconjugate(Rl6x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a163}')

I66a164 = Parameter(name = 'I66a164',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x4*TLLE1x3x1*complexconjugate(Rl6x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a164}')

I66a165 = Parameter(name = 'I66a165',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl5x5*TLLE1x3x2*complexconjugate(Rl6x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a165}')

I66a166 = Parameter(name = 'I66a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*TLLE1x3x3*complexconjugate(Rl6x3)*complexconjugate(Rn1x1)',
                    texname = '\\text{I66a166}')

I66a213 = Parameter(name = 'I66a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*TLLE2x1x3*complexconjugate(Rl1x1)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a213}')

I66a214 = Parameter(name = 'I66a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x4*TLLE2x1x1*complexconjugate(Rl1x1)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a214}')

I66a215 = Parameter(name = 'I66a215',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl5x5*TLLE2x1x2*complexconjugate(Rl1x1)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a215}')

I66a216 = Parameter(name = 'I66a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*TLLE2x1x3*complexconjugate(Rl1x1)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a216}')

I66a233 = Parameter(name = 'I66a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*TLLE2x3x3*complexconjugate(Rl3x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a233}')

I66a234 = Parameter(name = 'I66a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x4*TLLE2x3x1*complexconjugate(Rl3x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a234}')

I66a235 = Parameter(name = 'I66a235',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl5x5*TLLE2x3x2*complexconjugate(Rl3x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a235}')

I66a236 = Parameter(name = 'I66a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*TLLE2x3x3*complexconjugate(Rl3x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a236}')

I66a263 = Parameter(name = 'I66a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*TLLE2x3x3*complexconjugate(Rl6x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a263}')

I66a264 = Parameter(name = 'I66a264',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x4*TLLE2x3x1*complexconjugate(Rl6x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a264}')

I66a265 = Parameter(name = 'I66a265',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl5x5*TLLE2x3x2*complexconjugate(Rl6x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a265}')

I66a266 = Parameter(name = 'I66a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*TLLE2x3x3*complexconjugate(Rl6x3)*complexconjugate(Rn2x2)',
                    texname = '\\text{I66a266}')

I66a313 = Parameter(name = 'I66a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*TLLE3x1x3*complexconjugate(Rl1x1)*complexconjugate(Rn3x3)',
                    texname = '\\text{I66a313}')

I66a314 = Parameter(name = 'I66a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x4*TLLE3x1x1*complexconjugate(Rl1x1)*complexconjugate(Rn3x3)',
                    texname = '\\text{I66a314}')

I66a315 = Parameter(name = 'I66a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl5x5*TLLE3x1x2*complexconjugate(Rl1x1)*complexconjugate(Rn3x3)',
                    texname = '\\text{I66a315}')

I66a316 = Parameter(name = 'I66a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*TLLE3x1x3*complexconjugate(Rl1x1)*complexconjugate(Rn3x3)',
                    texname = '\\text{I66a316}')

I66a323 = Parameter(name = 'I66a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*TLLE3x2x3*complexconjugate(Rl2x2)*complexconjugate(Rn3x3)',
                    texname = '\\text{I66a323}')

I66a324 = Parameter(name = 'I66a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x4*TLLE3x2x1*complexconjugate(Rl2x2)*complexconjugate(Rn3x3)',
                    texname = '\\text{I66a324}')

I66a325 = Parameter(name = 'I66a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl5x5*TLLE3x2x2*complexconjugate(Rl2x2)*complexconjugate(Rn3x3)',
                    texname = '\\text{I66a325}')

I66a326 = Parameter(name = 'I66a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*TLLE3x2x3*complexconjugate(Rl2x2)*complexconjugate(Rn3x3)',
                    texname = '\\text{I66a326}')

I67a11 = Parameter(name = 'I67a11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x1',
                   texname = '\\text{I67a11}')

I67a22 = Parameter(name = 'I67a22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn2x2',
                   texname = '\\text{I67a22}')

I67a33 = Parameter(name = 'I67a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3',
                   texname = '\\text{I67a33}')

I68a33 = Parameter(name = 'I68a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(ye3x3)',
                   texname = '\\text{I68a33}')

I69a11 = Parameter(name = 'I69a11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x1*complexconjugate(Rl1x1)',
                   texname = '\\text{I69a11}')

I69a22 = Parameter(name = 'I69a22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn2x2*complexconjugate(Rl2x2)',
                   texname = '\\text{I69a22}')

I69a33 = Parameter(name = 'I69a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I69a33}')

I69a36 = Parameter(name = 'I69a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I69a36}')

I7a113 = Parameter(name = 'I7a113',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x1x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I7a113}')

I7a114 = Parameter(name = 'I7a114',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x1x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I7a114}')

I7a115 = Parameter(name = 'I7a115',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x1x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I7a115}')

I7a116 = Parameter(name = 'I7a116',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x1x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I7a116}')

I7a123 = Parameter(name = 'I7a123',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x2x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I7a123}')

I7a124 = Parameter(name = 'I7a124',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x2x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I7a124}')

I7a125 = Parameter(name = 'I7a125',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x2x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I7a125}')

I7a126 = Parameter(name = 'I7a126',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x2x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I7a126}')

I7a133 = Parameter(name = 'I7a133',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x3x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I7a133}')

I7a134 = Parameter(name = 'I7a134',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x3x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I7a134}')

I7a135 = Parameter(name = 'I7a135',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x3x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I7a135}')

I7a136 = Parameter(name = 'I7a136',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD1x3x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I7a136}')

I7a213 = Parameter(name = 'I7a213',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x1x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I7a213}')

I7a214 = Parameter(name = 'I7a214',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x1x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I7a214}')

I7a215 = Parameter(name = 'I7a215',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x1x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I7a215}')

I7a216 = Parameter(name = 'I7a216',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x1x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I7a216}')

I7a223 = Parameter(name = 'I7a223',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x2x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I7a223}')

I7a224 = Parameter(name = 'I7a224',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x2x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I7a224}')

I7a225 = Parameter(name = 'I7a225',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x2x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I7a225}')

I7a226 = Parameter(name = 'I7a226',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x2x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I7a226}')

I7a233 = Parameter(name = 'I7a233',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x3x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I7a233}')

I7a234 = Parameter(name = 'I7a234',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x3x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I7a234}')

I7a235 = Parameter(name = 'I7a235',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x3x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I7a235}')

I7a236 = Parameter(name = 'I7a236',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD2x3x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I7a236}')

I7a313 = Parameter(name = 'I7a313',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x1x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I7a313}')

I7a314 = Parameter(name = 'I7a314',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x1x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I7a314}')

I7a315 = Parameter(name = 'I7a315',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x1x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I7a315}')

I7a316 = Parameter(name = 'I7a316',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x1x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I7a316}')

I7a323 = Parameter(name = 'I7a323',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x2x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I7a323}')

I7a324 = Parameter(name = 'I7a324',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x2x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I7a324}')

I7a325 = Parameter(name = 'I7a325',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x2x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I7a325}')

I7a326 = Parameter(name = 'I7a326',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x2x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I7a326}')

I7a333 = Parameter(name = 'I7a333',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                   texname = '\\text{I7a333}')

I7a334 = Parameter(name = 'I7a334',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                   texname = '\\text{I7a334}')

I7a335 = Parameter(name = 'I7a335',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                   texname = '\\text{I7a335}')

I7a336 = Parameter(name = 'I7a336',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                   texname = '\\text{I7a336}')

I70a33 = Parameter(name = 'I70a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I70a33}')

I70a36 = Parameter(name = 'I70a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I70a36}')

I71a33 = Parameter(name = 'I71a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl3x6)*complexconjugate(te3x3)',
                   texname = '\\text{I71a33}')

I71a36 = Parameter(name = 'I71a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)',
                   texname = '\\text{I71a36}')

I72a33 = Parameter(name = 'I72a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I72a33}')

I72a36 = Parameter(name = 'I72a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*ye3x3*complexconjugate(Rl6x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I72a36}')

I73a111 = Parameter(name = 'I73a111',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLQD1x1x1)',
                    texname = '\\text{I73a111}')

I73a112 = Parameter(name = 'I73a112',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLQD2x1x1)',
                    texname = '\\text{I73a112}')

I73a113 = Parameter(name = 'I73a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLQD3x1x1)',
                    texname = '\\text{I73a113}')

I73a121 = Parameter(name = 'I73a121',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLQD1x1x2)',
                    texname = '\\text{I73a121}')

I73a122 = Parameter(name = 'I73a122',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLQD2x1x2)',
                    texname = '\\text{I73a122}')

I73a123 = Parameter(name = 'I73a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLQD3x1x2)',
                    texname = '\\text{I73a123}')

I73a131 = Parameter(name = 'I73a131',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLQD1x1x3)',
                    texname = '\\text{I73a131}')

I73a132 = Parameter(name = 'I73a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLQD2x1x3)',
                    texname = '\\text{I73a132}')

I73a133 = Parameter(name = 'I73a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLQD3x1x3)',
                    texname = '\\text{I73a133}')

I73a211 = Parameter(name = 'I73a211',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLQD1x2x1)',
                    texname = '\\text{I73a211}')

I73a212 = Parameter(name = 'I73a212',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLQD2x2x1)',
                    texname = '\\text{I73a212}')

I73a213 = Parameter(name = 'I73a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLQD3x2x1)',
                    texname = '\\text{I73a213}')

I73a221 = Parameter(name = 'I73a221',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLQD1x2x2)',
                    texname = '\\text{I73a221}')

I73a222 = Parameter(name = 'I73a222',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLQD2x2x2)',
                    texname = '\\text{I73a222}')

I73a223 = Parameter(name = 'I73a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLQD3x2x2)',
                    texname = '\\text{I73a223}')

I73a231 = Parameter(name = 'I73a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLQD1x2x3)',
                    texname = '\\text{I73a231}')

I73a232 = Parameter(name = 'I73a232',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLQD2x2x3)',
                    texname = '\\text{I73a232}')

I73a233 = Parameter(name = 'I73a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLQD3x2x3)',
                    texname = '\\text{I73a233}')

I73a311 = Parameter(name = 'I73a311',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLQD1x3x1)',
                    texname = '\\text{I73a311}')

I73a312 = Parameter(name = 'I73a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLQD2x3x1)',
                    texname = '\\text{I73a312}')

I73a313 = Parameter(name = 'I73a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLQD3x3x1)',
                    texname = '\\text{I73a313}')

I73a321 = Parameter(name = 'I73a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLQD1x3x2)',
                    texname = '\\text{I73a321}')

I73a322 = Parameter(name = 'I73a322',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLQD2x3x2)',
                    texname = '\\text{I73a322}')

I73a323 = Parameter(name = 'I73a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLQD3x3x2)',
                    texname = '\\text{I73a323}')

I73a331 = Parameter(name = 'I73a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLQD1x3x3)',
                    texname = '\\text{I73a331}')

I73a332 = Parameter(name = 'I73a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLQD2x3x3)',
                    texname = '\\text{I73a332}')

I73a333 = Parameter(name = 'I73a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLQD3x3x3)',
                    texname = '\\text{I73a333}')

I74a112 = Parameter(name = 'I74a112',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLLE2x1x1)',
                    texname = '\\text{I74a112}')

I74a113 = Parameter(name = 'I74a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLLE3x1x1)',
                    texname = '\\text{I74a113}')

I74a122 = Parameter(name = 'I74a122',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLLE2x1x2)',
                    texname = '\\text{I74a122}')

I74a123 = Parameter(name = 'I74a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLLE3x1x2)',
                    texname = '\\text{I74a123}')

I74a132 = Parameter(name = 'I74a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLLE2x1x3)',
                    texname = '\\text{I74a132}')

I74a133 = Parameter(name = 'I74a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLLE3x1x3)',
                    texname = '\\text{I74a133}')

I74a211 = Parameter(name = 'I74a211',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLLE1x2x1)',
                    texname = '\\text{I74a211}')

I74a213 = Parameter(name = 'I74a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLLE3x2x1)',
                    texname = '\\text{I74a213}')

I74a221 = Parameter(name = 'I74a221',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLLE1x2x2)',
                    texname = '\\text{I74a221}')

I74a223 = Parameter(name = 'I74a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLLE3x2x2)',
                    texname = '\\text{I74a223}')

I74a231 = Parameter(name = 'I74a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLLE1x2x3)',
                    texname = '\\text{I74a231}')

I74a233 = Parameter(name = 'I74a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn3x3*complexconjugate(LLLE3x2x3)',
                    texname = '\\text{I74a233}')

I74a311 = Parameter(name = 'I74a311',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLLE1x3x1)',
                    texname = '\\text{I74a311}')

I74a312 = Parameter(name = 'I74a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLLE2x3x1)',
                    texname = '\\text{I74a312}')

I74a321 = Parameter(name = 'I74a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLLE1x3x2)',
                    texname = '\\text{I74a321}')

I74a322 = Parameter(name = 'I74a322',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLLE2x3x2)',
                    texname = '\\text{I74a322}')

I74a331 = Parameter(name = 'I74a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn1x1*complexconjugate(LLLE1x3x3)',
                    texname = '\\text{I74a331}')

I74a332 = Parameter(name = 'I74a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rn2x2*complexconjugate(LLLE2x3x3)',
                    texname = '\\text{I74a332}')

I75a131 = Parameter(name = 'I75a131',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn1x1*complexconjugate(Rd3x6)*complexconjugate(TLQD1x1x3)',
                    texname = '\\text{I75a131}')

I75a132 = Parameter(name = 'I75a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn1x1*complexconjugate(Rd3x6)*complexconjugate(TLQD1x2x3)',
                    texname = '\\text{I75a132}')

I75a133 = Parameter(name = 'I75a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn1x1*complexconjugate(Rd3x6)*complexconjugate(TLQD1x3x3)',
                    texname = '\\text{I75a133}')

I75a136 = Parameter(name = 'I75a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn1x1*complexconjugate(Rd3x6)*complexconjugate(TLQD1x3x3)',
                    texname = '\\text{I75a136}')

I75a141 = Parameter(name = 'I75a141',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn1x1*complexconjugate(Rd4x4)*complexconjugate(TLQD1x1x1)',
                    texname = '\\text{I75a141}')

I75a142 = Parameter(name = 'I75a142',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn1x1*complexconjugate(Rd4x4)*complexconjugate(TLQD1x2x1)',
                    texname = '\\text{I75a142}')

I75a143 = Parameter(name = 'I75a143',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn1x1*complexconjugate(Rd4x4)*complexconjugate(TLQD1x3x1)',
                    texname = '\\text{I75a143}')

I75a146 = Parameter(name = 'I75a146',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn1x1*complexconjugate(Rd4x4)*complexconjugate(TLQD1x3x1)',
                    texname = '\\text{I75a146}')

I75a151 = Parameter(name = 'I75a151',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn1x1*complexconjugate(Rd5x5)*complexconjugate(TLQD1x1x2)',
                    texname = '\\text{I75a151}')

I75a152 = Parameter(name = 'I75a152',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn1x1*complexconjugate(Rd5x5)*complexconjugate(TLQD1x2x2)',
                    texname = '\\text{I75a152}')

I75a153 = Parameter(name = 'I75a153',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn1x1*complexconjugate(Rd5x5)*complexconjugate(TLQD1x3x2)',
                    texname = '\\text{I75a153}')

I75a156 = Parameter(name = 'I75a156',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn1x1*complexconjugate(Rd5x5)*complexconjugate(TLQD1x3x2)',
                    texname = '\\text{I75a156}')

I75a161 = Parameter(name = 'I75a161',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn1x1*complexconjugate(Rd6x6)*complexconjugate(TLQD1x1x3)',
                    texname = '\\text{I75a161}')

I75a162 = Parameter(name = 'I75a162',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn1x1*complexconjugate(Rd6x6)*complexconjugate(TLQD1x2x3)',
                    texname = '\\text{I75a162}')

I75a163 = Parameter(name = 'I75a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn1x1*complexconjugate(Rd6x6)*complexconjugate(TLQD1x3x3)',
                    texname = '\\text{I75a163}')

I75a166 = Parameter(name = 'I75a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn1x1*complexconjugate(Rd6x6)*complexconjugate(TLQD1x3x3)',
                    texname = '\\text{I75a166}')

I75a231 = Parameter(name = 'I75a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn2x2*complexconjugate(Rd3x6)*complexconjugate(TLQD2x1x3)',
                    texname = '\\text{I75a231}')

I75a232 = Parameter(name = 'I75a232',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn2x2*complexconjugate(Rd3x6)*complexconjugate(TLQD2x2x3)',
                    texname = '\\text{I75a232}')

I75a233 = Parameter(name = 'I75a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn2x2*complexconjugate(Rd3x6)*complexconjugate(TLQD2x3x3)',
                    texname = '\\text{I75a233}')

I75a236 = Parameter(name = 'I75a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn2x2*complexconjugate(Rd3x6)*complexconjugate(TLQD2x3x3)',
                    texname = '\\text{I75a236}')

I75a241 = Parameter(name = 'I75a241',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn2x2*complexconjugate(Rd4x4)*complexconjugate(TLQD2x1x1)',
                    texname = '\\text{I75a241}')

I75a242 = Parameter(name = 'I75a242',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn2x2*complexconjugate(Rd4x4)*complexconjugate(TLQD2x2x1)',
                    texname = '\\text{I75a242}')

I75a243 = Parameter(name = 'I75a243',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn2x2*complexconjugate(Rd4x4)*complexconjugate(TLQD2x3x1)',
                    texname = '\\text{I75a243}')

I75a246 = Parameter(name = 'I75a246',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn2x2*complexconjugate(Rd4x4)*complexconjugate(TLQD2x3x1)',
                    texname = '\\text{I75a246}')

I75a251 = Parameter(name = 'I75a251',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn2x2*complexconjugate(Rd5x5)*complexconjugate(TLQD2x1x2)',
                    texname = '\\text{I75a251}')

I75a252 = Parameter(name = 'I75a252',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn2x2*complexconjugate(Rd5x5)*complexconjugate(TLQD2x2x2)',
                    texname = '\\text{I75a252}')

I75a253 = Parameter(name = 'I75a253',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn2x2*complexconjugate(Rd5x5)*complexconjugate(TLQD2x3x2)',
                    texname = '\\text{I75a253}')

I75a256 = Parameter(name = 'I75a256',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn2x2*complexconjugate(Rd5x5)*complexconjugate(TLQD2x3x2)',
                    texname = '\\text{I75a256}')

I75a261 = Parameter(name = 'I75a261',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn2x2*complexconjugate(Rd6x6)*complexconjugate(TLQD2x1x3)',
                    texname = '\\text{I75a261}')

I75a262 = Parameter(name = 'I75a262',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn2x2*complexconjugate(Rd6x6)*complexconjugate(TLQD2x2x3)',
                    texname = '\\text{I75a262}')

I75a263 = Parameter(name = 'I75a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn2x2*complexconjugate(Rd6x6)*complexconjugate(TLQD2x3x3)',
                    texname = '\\text{I75a263}')

I75a266 = Parameter(name = 'I75a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn2x2*complexconjugate(Rd6x6)*complexconjugate(TLQD2x3x3)',
                    texname = '\\text{I75a266}')

I75a331 = Parameter(name = 'I75a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn3x3*complexconjugate(Rd3x6)*complexconjugate(TLQD3x1x3)',
                    texname = '\\text{I75a331}')

I75a332 = Parameter(name = 'I75a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn3x3*complexconjugate(Rd3x6)*complexconjugate(TLQD3x2x3)',
                    texname = '\\text{I75a332}')

I75a333 = Parameter(name = 'I75a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn3x3*complexconjugate(Rd3x6)*complexconjugate(TLQD3x3x3)',
                    texname = '\\text{I75a333}')

I75a336 = Parameter(name = 'I75a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn3x3*complexconjugate(Rd3x6)*complexconjugate(TLQD3x3x3)',
                    texname = '\\text{I75a336}')

I75a341 = Parameter(name = 'I75a341',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn3x3*complexconjugate(Rd4x4)*complexconjugate(TLQD3x1x1)',
                    texname = '\\text{I75a341}')

I75a342 = Parameter(name = 'I75a342',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn3x3*complexconjugate(Rd4x4)*complexconjugate(TLQD3x2x1)',
                    texname = '\\text{I75a342}')

I75a343 = Parameter(name = 'I75a343',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn3x3*complexconjugate(Rd4x4)*complexconjugate(TLQD3x3x1)',
                    texname = '\\text{I75a343}')

I75a346 = Parameter(name = 'I75a346',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn3x3*complexconjugate(Rd4x4)*complexconjugate(TLQD3x3x1)',
                    texname = '\\text{I75a346}')

I75a351 = Parameter(name = 'I75a351',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn3x3*complexconjugate(Rd5x5)*complexconjugate(TLQD3x1x2)',
                    texname = '\\text{I75a351}')

I75a352 = Parameter(name = 'I75a352',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn3x3*complexconjugate(Rd5x5)*complexconjugate(TLQD3x2x2)',
                    texname = '\\text{I75a352}')

I75a353 = Parameter(name = 'I75a353',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn3x3*complexconjugate(Rd5x5)*complexconjugate(TLQD3x3x2)',
                    texname = '\\text{I75a353}')

I75a356 = Parameter(name = 'I75a356',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn3x3*complexconjugate(Rd5x5)*complexconjugate(TLQD3x3x2)',
                    texname = '\\text{I75a356}')

I75a361 = Parameter(name = 'I75a361',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn3x3*complexconjugate(Rd6x6)*complexconjugate(TLQD3x1x3)',
                    texname = '\\text{I75a361}')

I75a362 = Parameter(name = 'I75a362',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn3x3*complexconjugate(Rd6x6)*complexconjugate(TLQD3x2x3)',
                    texname = '\\text{I75a362}')

I75a363 = Parameter(name = 'I75a363',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn3x3*complexconjugate(Rd6x6)*complexconjugate(TLQD3x3x3)',
                    texname = '\\text{I75a363}')

I75a366 = Parameter(name = 'I75a366',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn3x3*complexconjugate(Rd6x6)*complexconjugate(TLQD3x3x3)',
                    texname = '\\text{I75a366}')

I76a131 = Parameter(name = 'I76a131',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn1x1*yd3x3*complexconjugate(LLQD1x1x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a131}')

I76a132 = Parameter(name = 'I76a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn1x1*yd3x3*complexconjugate(LLQD1x2x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a132}')

I76a133 = Parameter(name = 'I76a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn1x1*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a133}')

I76a136 = Parameter(name = 'I76a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn1x1*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a136}')

I76a161 = Parameter(name = 'I76a161',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn1x1*yd3x3*complexconjugate(LLQD1x1x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a161}')

I76a162 = Parameter(name = 'I76a162',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn1x1*yd3x3*complexconjugate(LLQD1x2x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a162}')

I76a163 = Parameter(name = 'I76a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn1x1*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a163}')

I76a166 = Parameter(name = 'I76a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn1x1*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a166}')

I76a231 = Parameter(name = 'I76a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn2x2*yd3x3*complexconjugate(LLQD2x1x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a231}')

I76a232 = Parameter(name = 'I76a232',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn2x2*yd3x3*complexconjugate(LLQD2x2x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a232}')

I76a233 = Parameter(name = 'I76a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn2x2*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a233}')

I76a236 = Parameter(name = 'I76a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn2x2*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a236}')

I76a261 = Parameter(name = 'I76a261',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn2x2*yd3x3*complexconjugate(LLQD2x1x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a261}')

I76a262 = Parameter(name = 'I76a262',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn2x2*yd3x3*complexconjugate(LLQD2x2x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a262}')

I76a263 = Parameter(name = 'I76a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn2x2*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a263}')

I76a266 = Parameter(name = 'I76a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn2x2*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a266}')

I76a331 = Parameter(name = 'I76a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn3x3*yd3x3*complexconjugate(LLQD3x1x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a331}')

I76a332 = Parameter(name = 'I76a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn3x3*yd3x3*complexconjugate(LLQD3x2x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a332}')

I76a333 = Parameter(name = 'I76a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a333}')

I76a336 = Parameter(name = 'I76a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x3)',
                    texname = '\\text{I76a336}')

I76a361 = Parameter(name = 'I76a361',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x1*Rn3x3*yd3x3*complexconjugate(LLQD3x1x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a361}')

I76a362 = Parameter(name = 'I76a362',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x2*Rn3x3*yd3x3*complexconjugate(LLQD3x2x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a362}')

I76a363 = Parameter(name = 'I76a363',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x3*Rn3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a363}')

I76a366 = Parameter(name = 'I76a366',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x3*Rn3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x3)',
                    texname = '\\text{I76a366}')

I77a133 = Parameter(name = 'I77a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn1x1*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I77a133}')

I77a136 = Parameter(name = 'I77a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn1x1*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I77a136}')

I77a143 = Parameter(name = 'I77a143',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn1x1*yd3x3*complexconjugate(LLQD1x3x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I77a143}')

I77a146 = Parameter(name = 'I77a146',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn1x1*yd3x3*complexconjugate(LLQD1x3x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I77a146}')

I77a153 = Parameter(name = 'I77a153',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn1x1*yd3x3*complexconjugate(LLQD1x3x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I77a153}')

I77a156 = Parameter(name = 'I77a156',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn1x1*yd3x3*complexconjugate(LLQD1x3x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I77a156}')

I77a163 = Parameter(name = 'I77a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn1x1*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I77a163}')

I77a166 = Parameter(name = 'I77a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn1x1*yd3x3*complexconjugate(LLQD1x3x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I77a166}')

I77a233 = Parameter(name = 'I77a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn2x2*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I77a233}')

I77a236 = Parameter(name = 'I77a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn2x2*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I77a236}')

I77a243 = Parameter(name = 'I77a243',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn2x2*yd3x3*complexconjugate(LLQD2x3x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I77a243}')

I77a246 = Parameter(name = 'I77a246',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn2x2*yd3x3*complexconjugate(LLQD2x3x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I77a246}')

I77a253 = Parameter(name = 'I77a253',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn2x2*yd3x3*complexconjugate(LLQD2x3x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I77a253}')

I77a256 = Parameter(name = 'I77a256',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn2x2*yd3x3*complexconjugate(LLQD2x3x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I77a256}')

I77a263 = Parameter(name = 'I77a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn2x2*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I77a263}')

I77a266 = Parameter(name = 'I77a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn2x2*yd3x3*complexconjugate(LLQD2x3x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I77a266}')

I77a333 = Parameter(name = 'I77a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I77a333}')

I77a336 = Parameter(name = 'I77a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd3x6)',
                    texname = '\\text{I77a336}')

I77a343 = Parameter(name = 'I77a343',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn3x3*yd3x3*complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I77a343}')

I77a346 = Parameter(name = 'I77a346',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn3x3*yd3x3*complexconjugate(LLQD3x3x1)*complexconjugate(Rd4x4)',
                    texname = '\\text{I77a346}')

I77a353 = Parameter(name = 'I77a353',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn3x3*yd3x3*complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I77a353}')

I77a356 = Parameter(name = 'I77a356',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn3x3*yd3x3*complexconjugate(LLQD3x3x2)*complexconjugate(Rd5x5)',
                    texname = '\\text{I77a356}')

I77a363 = Parameter(name = 'I77a363',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*Rn3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I77a363}')

I77a366 = Parameter(name = 'I77a366',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*Rn3x3*yd3x3*complexconjugate(LLQD3x3x3)*complexconjugate(Rd6x6)',
                    texname = '\\text{I77a366}')

I78a132 = Parameter(name = 'I78a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn1x1*complexconjugate(Rl3x6)*complexconjugate(TLLE1x2x3)',
                    texname = '\\text{I78a132}')

I78a133 = Parameter(name = 'I78a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn1x1*complexconjugate(Rl3x6)*complexconjugate(TLLE1x3x3)',
                    texname = '\\text{I78a133}')

I78a136 = Parameter(name = 'I78a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn1x1*complexconjugate(Rl3x6)*complexconjugate(TLLE1x3x3)',
                    texname = '\\text{I78a136}')

I78a142 = Parameter(name = 'I78a142',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn1x1*complexconjugate(Rl4x4)*complexconjugate(TLLE1x2x1)',
                    texname = '\\text{I78a142}')

I78a143 = Parameter(name = 'I78a143',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn1x1*complexconjugate(Rl4x4)*complexconjugate(TLLE1x3x1)',
                    texname = '\\text{I78a143}')

I78a146 = Parameter(name = 'I78a146',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn1x1*complexconjugate(Rl4x4)*complexconjugate(TLLE1x3x1)',
                    texname = '\\text{I78a146}')

I78a152 = Parameter(name = 'I78a152',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn1x1*complexconjugate(Rl5x5)*complexconjugate(TLLE1x2x2)',
                    texname = '\\text{I78a152}')

I78a153 = Parameter(name = 'I78a153',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn1x1*complexconjugate(Rl5x5)*complexconjugate(TLLE1x3x2)',
                    texname = '\\text{I78a153}')

I78a156 = Parameter(name = 'I78a156',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn1x1*complexconjugate(Rl5x5)*complexconjugate(TLLE1x3x2)',
                    texname = '\\text{I78a156}')

I78a162 = Parameter(name = 'I78a162',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn1x1*complexconjugate(Rl6x6)*complexconjugate(TLLE1x2x3)',
                    texname = '\\text{I78a162}')

I78a163 = Parameter(name = 'I78a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn1x1*complexconjugate(Rl6x6)*complexconjugate(TLLE1x3x3)',
                    texname = '\\text{I78a163}')

I78a166 = Parameter(name = 'I78a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn1x1*complexconjugate(Rl6x6)*complexconjugate(TLLE1x3x3)',
                    texname = '\\text{I78a166}')

I78a231 = Parameter(name = 'I78a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn2x2*complexconjugate(Rl3x6)*complexconjugate(TLLE2x1x3)',
                    texname = '\\text{I78a231}')

I78a233 = Parameter(name = 'I78a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn2x2*complexconjugate(Rl3x6)*complexconjugate(TLLE2x3x3)',
                    texname = '\\text{I78a233}')

I78a236 = Parameter(name = 'I78a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn2x2*complexconjugate(Rl3x6)*complexconjugate(TLLE2x3x3)',
                    texname = '\\text{I78a236}')

I78a241 = Parameter(name = 'I78a241',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn2x2*complexconjugate(Rl4x4)*complexconjugate(TLLE2x1x1)',
                    texname = '\\text{I78a241}')

I78a243 = Parameter(name = 'I78a243',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn2x2*complexconjugate(Rl4x4)*complexconjugate(TLLE2x3x1)',
                    texname = '\\text{I78a243}')

I78a246 = Parameter(name = 'I78a246',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn2x2*complexconjugate(Rl4x4)*complexconjugate(TLLE2x3x1)',
                    texname = '\\text{I78a246}')

I78a251 = Parameter(name = 'I78a251',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn2x2*complexconjugate(Rl5x5)*complexconjugate(TLLE2x1x2)',
                    texname = '\\text{I78a251}')

I78a253 = Parameter(name = 'I78a253',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn2x2*complexconjugate(Rl5x5)*complexconjugate(TLLE2x3x2)',
                    texname = '\\text{I78a253}')

I78a256 = Parameter(name = 'I78a256',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn2x2*complexconjugate(Rl5x5)*complexconjugate(TLLE2x3x2)',
                    texname = '\\text{I78a256}')

I78a261 = Parameter(name = 'I78a261',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn2x2*complexconjugate(Rl6x6)*complexconjugate(TLLE2x1x3)',
                    texname = '\\text{I78a261}')

I78a263 = Parameter(name = 'I78a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn2x2*complexconjugate(Rl6x6)*complexconjugate(TLLE2x3x3)',
                    texname = '\\text{I78a263}')

I78a266 = Parameter(name = 'I78a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn2x2*complexconjugate(Rl6x6)*complexconjugate(TLLE2x3x3)',
                    texname = '\\text{I78a266}')

I78a331 = Parameter(name = 'I78a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn3x3*complexconjugate(Rl3x6)*complexconjugate(TLLE3x1x3)',
                    texname = '\\text{I78a331}')

I78a332 = Parameter(name = 'I78a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn3x3*complexconjugate(Rl3x6)*complexconjugate(TLLE3x2x3)',
                    texname = '\\text{I78a332}')

I78a341 = Parameter(name = 'I78a341',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn3x3*complexconjugate(Rl4x4)*complexconjugate(TLLE3x1x1)',
                    texname = '\\text{I78a341}')

I78a342 = Parameter(name = 'I78a342',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn3x3*complexconjugate(Rl4x4)*complexconjugate(TLLE3x2x1)',
                    texname = '\\text{I78a342}')

I78a351 = Parameter(name = 'I78a351',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn3x3*complexconjugate(Rl5x5)*complexconjugate(TLLE3x1x2)',
                    texname = '\\text{I78a351}')

I78a352 = Parameter(name = 'I78a352',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn3x3*complexconjugate(Rl5x5)*complexconjugate(TLLE3x2x2)',
                    texname = '\\text{I78a352}')

I78a361 = Parameter(name = 'I78a361',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn3x3*complexconjugate(Rl6x6)*complexconjugate(TLLE3x1x3)',
                    texname = '\\text{I78a361}')

I78a362 = Parameter(name = 'I78a362',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn3x3*complexconjugate(Rl6x6)*complexconjugate(TLLE3x2x3)',
                    texname = '\\text{I78a362}')

I79a132 = Parameter(name = 'I79a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn1x1*ye3x3*complexconjugate(LLLE1x2x3)*complexconjugate(Rl3x3)',
                    texname = '\\text{I79a132}')

I79a133 = Parameter(name = 'I79a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn1x1*ye3x3*complexconjugate(LLLE1x3x3)*complexconjugate(Rl3x3)',
                    texname = '\\text{I79a133}')

I79a136 = Parameter(name = 'I79a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn1x1*ye3x3*complexconjugate(LLLE1x3x3)*complexconjugate(Rl3x3)',
                    texname = '\\text{I79a136}')

I79a162 = Parameter(name = 'I79a162',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn1x1*ye3x3*complexconjugate(LLLE1x2x3)*complexconjugate(Rl6x3)',
                    texname = '\\text{I79a162}')

I79a163 = Parameter(name = 'I79a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn1x1*ye3x3*complexconjugate(LLLE1x3x3)*complexconjugate(Rl6x3)',
                    texname = '\\text{I79a163}')

I79a166 = Parameter(name = 'I79a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn1x1*ye3x3*complexconjugate(LLLE1x3x3)*complexconjugate(Rl6x3)',
                    texname = '\\text{I79a166}')

I79a231 = Parameter(name = 'I79a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn2x2*ye3x3*complexconjugate(LLLE2x1x3)*complexconjugate(Rl3x3)',
                    texname = '\\text{I79a231}')

I79a233 = Parameter(name = 'I79a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn2x2*ye3x3*complexconjugate(LLLE2x3x3)*complexconjugate(Rl3x3)',
                    texname = '\\text{I79a233}')

I79a236 = Parameter(name = 'I79a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn2x2*ye3x3*complexconjugate(LLLE2x3x3)*complexconjugate(Rl3x3)',
                    texname = '\\text{I79a236}')

I79a261 = Parameter(name = 'I79a261',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn2x2*ye3x3*complexconjugate(LLLE2x1x3)*complexconjugate(Rl6x3)',
                    texname = '\\text{I79a261}')

I79a263 = Parameter(name = 'I79a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x3*Rn2x2*ye3x3*complexconjugate(LLLE2x3x3)*complexconjugate(Rl6x3)',
                    texname = '\\text{I79a263}')

I79a266 = Parameter(name = 'I79a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x3*Rn2x2*ye3x3*complexconjugate(LLLE2x3x3)*complexconjugate(Rl6x3)',
                    texname = '\\text{I79a266}')

I79a331 = Parameter(name = 'I79a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn3x3*ye3x3*complexconjugate(LLLE3x1x3)*complexconjugate(Rl3x3)',
                    texname = '\\text{I79a331}')

I79a332 = Parameter(name = 'I79a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn3x3*ye3x3*complexconjugate(LLLE3x2x3)*complexconjugate(Rl3x3)',
                    texname = '\\text{I79a332}')

I79a361 = Parameter(name = 'I79a361',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x1*Rn3x3*ye3x3*complexconjugate(LLLE3x1x3)*complexconjugate(Rl6x3)',
                    texname = '\\text{I79a361}')

I79a362 = Parameter(name = 'I79a362',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x2*Rn3x3*ye3x3*complexconjugate(LLLE3x2x3)*complexconjugate(Rl6x3)',
                    texname = '\\text{I79a362}')

I8a11 = Parameter(name = 'I8a11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd1x1*complexconjugate(Rd1x1)',
                  texname = '\\text{I8a11}')

I8a22 = Parameter(name = 'I8a22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd2x2*complexconjugate(Rd2x2)',
                  texname = '\\text{I8a22}')

I8a33 = Parameter(name = 'I8a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x3*complexconjugate(Rd3x3)',
                  texname = '\\text{I8a33}')

I8a36 = Parameter(name = 'I8a36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x3*complexconjugate(Rd3x3)',
                  texname = '\\text{I8a36}')

I8a63 = Parameter(name = 'I8a63',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x3*complexconjugate(Rd6x3)',
                  texname = '\\text{I8a63}')

I8a66 = Parameter(name = 'I8a66',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x3*complexconjugate(Rd6x3)',
                  texname = '\\text{I8a66}')

I80a133 = Parameter(name = 'I80a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*Rn1x1*ye3x3*complexconjugate(LLLE1x3x3)*complexconjugate(Rl3x6)',
                    texname = '\\text{I80a133}')

I80a136 = Parameter(name = 'I80a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*Rn1x1*ye3x3*complexconjugate(LLLE1x3x3)*complexconjugate(Rl3x6)',
                    texname = '\\text{I80a136}')

I80a143 = Parameter(name = 'I80a143',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*Rn1x1*ye3x3*complexconjugate(LLLE1x3x1)*complexconjugate(Rl4x4)',
                    texname = '\\text{I80a143}')

I80a146 = Parameter(name = 'I80a146',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*Rn1x1*ye3x3*complexconjugate(LLLE1x3x1)*complexconjugate(Rl4x4)',
                    texname = '\\text{I80a146}')

I80a153 = Parameter(name = 'I80a153',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*Rn1x1*ye3x3*complexconjugate(LLLE1x3x2)*complexconjugate(Rl5x5)',
                    texname = '\\text{I80a153}')

I80a156 = Parameter(name = 'I80a156',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*Rn1x1*ye3x3*complexconjugate(LLLE1x3x2)*complexconjugate(Rl5x5)',
                    texname = '\\text{I80a156}')

I80a163 = Parameter(name = 'I80a163',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*Rn1x1*ye3x3*complexconjugate(LLLE1x3x3)*complexconjugate(Rl6x6)',
                    texname = '\\text{I80a163}')

I80a166 = Parameter(name = 'I80a166',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*Rn1x1*ye3x3*complexconjugate(LLLE1x3x3)*complexconjugate(Rl6x6)',
                    texname = '\\text{I80a166}')

I80a233 = Parameter(name = 'I80a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*Rn2x2*ye3x3*complexconjugate(LLLE2x3x3)*complexconjugate(Rl3x6)',
                    texname = '\\text{I80a233}')

I80a236 = Parameter(name = 'I80a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*Rn2x2*ye3x3*complexconjugate(LLLE2x3x3)*complexconjugate(Rl3x6)',
                    texname = '\\text{I80a236}')

I80a243 = Parameter(name = 'I80a243',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*Rn2x2*ye3x3*complexconjugate(LLLE2x3x1)*complexconjugate(Rl4x4)',
                    texname = '\\text{I80a243}')

I80a246 = Parameter(name = 'I80a246',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*Rn2x2*ye3x3*complexconjugate(LLLE2x3x1)*complexconjugate(Rl4x4)',
                    texname = '\\text{I80a246}')

I80a253 = Parameter(name = 'I80a253',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*Rn2x2*ye3x3*complexconjugate(LLLE2x3x2)*complexconjugate(Rl5x5)',
                    texname = '\\text{I80a253}')

I80a256 = Parameter(name = 'I80a256',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*Rn2x2*ye3x3*complexconjugate(LLLE2x3x2)*complexconjugate(Rl5x5)',
                    texname = '\\text{I80a256}')

I80a263 = Parameter(name = 'I80a263',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*Rn2x2*ye3x3*complexconjugate(LLLE2x3x3)*complexconjugate(Rl6x6)',
                    texname = '\\text{I80a263}')

I80a266 = Parameter(name = 'I80a266',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*Rn2x2*ye3x3*complexconjugate(LLLE2x3x3)*complexconjugate(Rl6x6)',
                    texname = '\\text{I80a266}')

I81a33 = Parameter(name = 'I81a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I81a33}')

I81a36 = Parameter(name = 'I81a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I81a36}')

I82a33 = Parameter(name = 'I82a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yu3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I82a33}')

I82a36 = Parameter(name = 'I82a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yu3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I82a36}')

I83a111 = Parameter(name = 'I83a111',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x1*complexconjugate(Ru1x1)',
                    texname = '\\text{I83a111}')

I83a112 = Parameter(name = 'I83a112',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x1*complexconjugate(Ru2x2)',
                    texname = '\\text{I83a112}')

I83a113 = Parameter(name = 'I83a113',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x1*complexconjugate(Ru3x3)',
                    texname = '\\text{I83a113}')

I83a116 = Parameter(name = 'I83a116',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x1*complexconjugate(Ru6x3)',
                    texname = '\\text{I83a116}')

I83a121 = Parameter(name = 'I83a121',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x1*complexconjugate(Ru1x1)',
                    texname = '\\text{I83a121}')

I83a122 = Parameter(name = 'I83a122',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x1*complexconjugate(Ru2x2)',
                    texname = '\\text{I83a122}')

I83a123 = Parameter(name = 'I83a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x1*complexconjugate(Ru3x3)',
                    texname = '\\text{I83a123}')

I83a126 = Parameter(name = 'I83a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x1*complexconjugate(Ru6x3)',
                    texname = '\\text{I83a126}')

I83a131 = Parameter(name = 'I83a131',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x1*complexconjugate(Ru1x1)',
                    texname = '\\text{I83a131}')

I83a132 = Parameter(name = 'I83a132',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x1*complexconjugate(Ru2x2)',
                    texname = '\\text{I83a132}')

I83a133 = Parameter(name = 'I83a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*complexconjugate(Ru3x3)',
                    texname = '\\text{I83a133}')

I83a136 = Parameter(name = 'I83a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*complexconjugate(Ru6x3)',
                    texname = '\\text{I83a136}')

I83a211 = Parameter(name = 'I83a211',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x2*complexconjugate(Ru1x1)',
                    texname = '\\text{I83a211}')

I83a212 = Parameter(name = 'I83a212',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x2*complexconjugate(Ru2x2)',
                    texname = '\\text{I83a212}')

I83a213 = Parameter(name = 'I83a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x2*complexconjugate(Ru3x3)',
                    texname = '\\text{I83a213}')

I83a216 = Parameter(name = 'I83a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x2*complexconjugate(Ru6x3)',
                    texname = '\\text{I83a216}')

I83a221 = Parameter(name = 'I83a221',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x2*complexconjugate(Ru1x1)',
                    texname = '\\text{I83a221}')

I83a222 = Parameter(name = 'I83a222',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x2*complexconjugate(Ru2x2)',
                    texname = '\\text{I83a222}')

I83a223 = Parameter(name = 'I83a223',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x2*complexconjugate(Ru3x3)',
                    texname = '\\text{I83a223}')

I83a226 = Parameter(name = 'I83a226',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x2*complexconjugate(Ru6x3)',
                    texname = '\\text{I83a226}')

I83a231 = Parameter(name = 'I83a231',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x2*complexconjugate(Ru1x1)',
                    texname = '\\text{I83a231}')

I83a232 = Parameter(name = 'I83a232',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x2*complexconjugate(Ru2x2)',
                    texname = '\\text{I83a232}')

I83a233 = Parameter(name = 'I83a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*complexconjugate(Ru3x3)',
                    texname = '\\text{I83a233}')

I83a236 = Parameter(name = 'I83a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*complexconjugate(Ru6x3)',
                    texname = '\\text{I83a236}')

I83a311 = Parameter(name = 'I83a311',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*complexconjugate(Ru1x1)',
                    texname = '\\text{I83a311}')

I83a312 = Parameter(name = 'I83a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*complexconjugate(Ru2x2)',
                    texname = '\\text{I83a312}')

I83a313 = Parameter(name = 'I83a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I83a313}')

I83a316 = Parameter(name = 'I83a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I83a316}')

I83a321 = Parameter(name = 'I83a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*complexconjugate(Ru1x1)',
                    texname = '\\text{I83a321}')

I83a322 = Parameter(name = 'I83a322',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*complexconjugate(Ru2x2)',
                    texname = '\\text{I83a322}')

I83a323 = Parameter(name = 'I83a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I83a323}')

I83a326 = Parameter(name = 'I83a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I83a326}')

I83a331 = Parameter(name = 'I83a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*complexconjugate(Ru1x1)',
                    texname = '\\text{I83a331}')

I83a332 = Parameter(name = 'I83a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*complexconjugate(Ru2x2)',
                    texname = '\\text{I83a332}')

I83a333 = Parameter(name = 'I83a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*complexconjugate(Ru3x3)',
                    texname = '\\text{I83a333}')

I83a336 = Parameter(name = 'I83a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*complexconjugate(Ru6x3)',
                    texname = '\\text{I83a336}')

I84a11 = Parameter(name = 'I84a11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x1*complexconjugate(Ru1x1)',
                   texname = '\\text{I84a11}')

I84a22 = Parameter(name = 'I84a22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x2*complexconjugate(Ru2x2)',
                   texname = '\\text{I84a22}')

I84a33 = Parameter(name = 'I84a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I84a33}')

I84a36 = Parameter(name = 'I84a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I84a36}')

I84a63 = Parameter(name = 'I84a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I84a63}')

I84a66 = Parameter(name = 'I84a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I84a66}')

I85a33 = Parameter(name = 'I85a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*complexconjugate(Ru3x6)',
                   texname = '\\text{I85a33}')

I85a36 = Parameter(name = 'I85a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*complexconjugate(Ru3x6)',
                   texname = '\\text{I85a36}')

I85a44 = Parameter(name = 'I85a44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru4x4*complexconjugate(Ru4x4)',
                   texname = '\\text{I85a44}')

I85a55 = Parameter(name = 'I85a55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru5x5*complexconjugate(Ru5x5)',
                   texname = '\\text{I85a55}')

I85a63 = Parameter(name = 'I85a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*complexconjugate(Ru6x6)',
                   texname = '\\text{I85a63}')

I85a66 = Parameter(name = 'I85a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*complexconjugate(Ru6x6)',
                   texname = '\\text{I85a66}')

I86a11 = Parameter(name = 'I86a11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x1*complexconjugate(Ru1x1)',
                   texname = '\\text{I86a11}')

I86a22 = Parameter(name = 'I86a22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x2*complexconjugate(Ru2x2)',
                   texname = '\\text{I86a22}')

I86a33 = Parameter(name = 'I86a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I86a33}')

I86a36 = Parameter(name = 'I86a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I86a36}')

I86a63 = Parameter(name = 'I86a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I86a63}')

I86a66 = Parameter(name = 'I86a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I86a66}')

I87a33 = Parameter(name = 'I87a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Ru3x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I87a33}')

I87a36 = Parameter(name = 'I87a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Ru6x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I87a36}')

I87a63 = Parameter(name = 'I87a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Ru3x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I87a63}')

I87a66 = Parameter(name = 'I87a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Ru6x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I87a66}')

I88a33 = Parameter(name = 'I88a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I88a33}')

I88a36 = Parameter(name = 'I88a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I88a36}')

I88a63 = Parameter(name = 'I88a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I88a63}')

I88a66 = Parameter(name = 'I88a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I88a66}')

I89a33 = Parameter(name = 'I89a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*td3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I89a33}')

I89a36 = Parameter(name = 'I89a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*td3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I89a36}')

I89a63 = Parameter(name = 'I89a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*td3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I89a63}')

I89a66 = Parameter(name = 'I89a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*td3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I89a66}')

I9a33 = Parameter(name = 'I9a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x6*complexconjugate(Rd3x6)',
                  texname = '\\text{I9a33}')

I9a36 = Parameter(name = 'I9a36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x6*complexconjugate(Rd3x6)',
                  texname = '\\text{I9a36}')

I9a44 = Parameter(name = 'I9a44',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd4x4*complexconjugate(Rd4x4)',
                  texname = '\\text{I9a44}')

I9a55 = Parameter(name = 'I9a55',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd5x5*complexconjugate(Rd5x5)',
                  texname = '\\text{I9a55}')

I9a63 = Parameter(name = 'I9a63',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x6*complexconjugate(Rd6x6)',
                  texname = '\\text{I9a63}')

I9a66 = Parameter(name = 'I9a66',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x6*complexconjugate(Rd6x6)',
                  texname = '\\text{I9a66}')

I90a33 = Parameter(name = 'I90a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I90a33}')

I90a36 = Parameter(name = 'I90a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I90a36}')

I90a63 = Parameter(name = 'I90a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I90a63}')

I90a66 = Parameter(name = 'I90a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I90a66}')

I91a33 = Parameter(name = 'I91a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yd3x3*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I91a33}')

I91a36 = Parameter(name = 'I91a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yd3x3*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I91a36}')

I91a63 = Parameter(name = 'I91a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yd3x3*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I91a63}')

I91a66 = Parameter(name = 'I91a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yd3x3*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I91a66}')

I92a33 = Parameter(name = 'I92a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I92a33}')

I92a36 = Parameter(name = 'I92a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I92a36}')

I92a63 = Parameter(name = 'I92a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I92a63}')

I92a66 = Parameter(name = 'I92a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I92a66}')

I93a33 = Parameter(name = 'I93a33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I93a33}')

I93a36 = Parameter(name = 'I93a36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yu3x3*complexconjugate(Ru6x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I93a36}')

I93a63 = Parameter(name = 'I93a63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I93a63}')

I93a66 = Parameter(name = 'I93a66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yu3x3*complexconjugate(Ru6x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I93a66}')

I94a123 = Parameter(name = 'I94a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x2)*complexconjugate(Ru3x6)',
                    texname = '\\text{I94a123}')

I94a124 = Parameter(name = 'I94a124',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x1x2)*complexconjugate(Ru4x4)',
                    texname = '\\text{I94a124}')

I94a125 = Parameter(name = 'I94a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x1x2)*complexconjugate(Ru5x5)',
                    texname = '\\text{I94a125}')

I94a126 = Parameter(name = 'I94a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x2)*complexconjugate(Ru6x6)',
                    texname = '\\text{I94a126}')

I94a133 = Parameter(name = 'I94a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x3)*complexconjugate(Ru3x6)',
                    texname = '\\text{I94a133}')

I94a134 = Parameter(name = 'I94a134',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x1x3)*complexconjugate(Ru4x4)',
                    texname = '\\text{I94a134}')

I94a135 = Parameter(name = 'I94a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x1x3)*complexconjugate(Ru5x5)',
                    texname = '\\text{I94a135}')

I94a136 = Parameter(name = 'I94a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x3)*complexconjugate(Ru6x6)',
                    texname = '\\text{I94a136}')

I94a213 = Parameter(name = 'I94a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x1)*complexconjugate(Ru3x6)',
                    texname = '\\text{I94a213}')

I94a214 = Parameter(name = 'I94a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x2x1)*complexconjugate(Ru4x4)',
                    texname = '\\text{I94a214}')

I94a215 = Parameter(name = 'I94a215',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x2x1)*complexconjugate(Ru5x5)',
                    texname = '\\text{I94a215}')

I94a216 = Parameter(name = 'I94a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x1)*complexconjugate(Ru6x6)',
                    texname = '\\text{I94a216}')

I94a233 = Parameter(name = 'I94a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x3)*complexconjugate(Ru3x6)',
                    texname = '\\text{I94a233}')

I94a234 = Parameter(name = 'I94a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x2x3)*complexconjugate(Ru4x4)',
                    texname = '\\text{I94a234}')

I94a235 = Parameter(name = 'I94a235',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x2x3)*complexconjugate(Ru5x5)',
                    texname = '\\text{I94a235}')

I94a236 = Parameter(name = 'I94a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x3)*complexconjugate(Ru6x6)',
                    texname = '\\text{I94a236}')

I94a313 = Parameter(name = 'I94a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x1)*complexconjugate(Ru3x6)',
                    texname = '\\text{I94a313}')

I94a314 = Parameter(name = 'I94a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x3x1)*complexconjugate(Ru4x4)',
                    texname = '\\text{I94a314}')

I94a315 = Parameter(name = 'I94a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x3x1)*complexconjugate(Ru5x5)',
                    texname = '\\text{I94a315}')

I94a316 = Parameter(name = 'I94a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x1)*complexconjugate(Ru6x6)',
                    texname = '\\text{I94a316}')

I94a323 = Parameter(name = 'I94a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x2)*complexconjugate(Ru3x6)',
                    texname = '\\text{I94a323}')

I94a324 = Parameter(name = 'I94a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x3x2)*complexconjugate(Ru4x4)',
                    texname = '\\text{I94a324}')

I94a325 = Parameter(name = 'I94a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x3x2)*complexconjugate(Ru5x5)',
                    texname = '\\text{I94a325}')

I94a326 = Parameter(name = 'I94a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x2)*complexconjugate(Ru6x6)',
                    texname = '\\text{I94a326}')

I95a123 = Parameter(name = 'I95a123',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x1)*complexconjugate(Ru3x6)',
                    texname = '\\text{I95a123}')

I95a124 = Parameter(name = 'I95a124',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x2x1)*complexconjugate(Ru4x4)',
                    texname = '\\text{I95a124}')

I95a125 = Parameter(name = 'I95a125',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x2x1)*complexconjugate(Ru5x5)',
                    texname = '\\text{I95a125}')

I95a126 = Parameter(name = 'I95a126',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x1)*complexconjugate(Ru6x6)',
                    texname = '\\text{I95a126}')

I95a133 = Parameter(name = 'I95a133',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x1)*complexconjugate(Ru3x6)',
                    texname = '\\text{I95a133}')

I95a134 = Parameter(name = 'I95a134',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x3x1)*complexconjugate(Ru4x4)',
                    texname = '\\text{I95a134}')

I95a135 = Parameter(name = 'I95a135',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x3x1)*complexconjugate(Ru5x5)',
                    texname = '\\text{I95a135}')

I95a136 = Parameter(name = 'I95a136',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x1)*complexconjugate(Ru6x6)',
                    texname = '\\text{I95a136}')

I95a213 = Parameter(name = 'I95a213',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x2)*complexconjugate(Ru3x6)',
                    texname = '\\text{I95a213}')

I95a214 = Parameter(name = 'I95a214',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x1x2)*complexconjugate(Ru4x4)',
                    texname = '\\text{I95a214}')

I95a215 = Parameter(name = 'I95a215',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x1x2)*complexconjugate(Ru5x5)',
                    texname = '\\text{I95a215}')

I95a216 = Parameter(name = 'I95a216',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x2)*complexconjugate(Ru6x6)',
                    texname = '\\text{I95a216}')

I95a233 = Parameter(name = 'I95a233',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x2)*complexconjugate(Ru3x6)',
                    texname = '\\text{I95a233}')

I95a234 = Parameter(name = 'I95a234',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x3x2)*complexconjugate(Ru4x4)',
                    texname = '\\text{I95a234}')

I95a235 = Parameter(name = 'I95a235',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x3x2)*complexconjugate(Ru5x5)',
                    texname = '\\text{I95a235}')

I95a236 = Parameter(name = 'I95a236',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x3x2)*complexconjugate(Ru6x6)',
                    texname = '\\text{I95a236}')

I95a313 = Parameter(name = 'I95a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x3)*complexconjugate(Ru3x6)',
                    texname = '\\text{I95a313}')

I95a314 = Parameter(name = 'I95a314',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x1x3)*complexconjugate(Ru4x4)',
                    texname = '\\text{I95a314}')

I95a315 = Parameter(name = 'I95a315',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x1x3)*complexconjugate(Ru5x5)',
                    texname = '\\text{I95a315}')

I95a316 = Parameter(name = 'I95a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x1x3)*complexconjugate(Ru6x6)',
                    texname = '\\text{I95a316}')

I95a323 = Parameter(name = 'I95a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x3)*complexconjugate(Ru3x6)',
                    texname = '\\text{I95a323}')

I95a324 = Parameter(name = 'I95a324',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD1x2x3)*complexconjugate(Ru4x4)',
                    texname = '\\text{I95a324}')

I95a325 = Parameter(name = 'I95a325',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD2x2x3)*complexconjugate(Ru5x5)',
                    texname = '\\text{I95a325}')

I95a326 = Parameter(name = 'I95a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'complexconjugate(LUDD3x2x3)*complexconjugate(Ru6x6)',
                    texname = '\\text{I95a326}')

I96a311 = Parameter(name = 'I96a311',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*Rd3x3*complexconjugate(Rl1x1)*complexconjugate(Ru1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a311}')

I96a312 = Parameter(name = 'I96a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*Rd3x3*complexconjugate(Rl1x1)*complexconjugate(Ru2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a312}')

I96a313 = Parameter(name = 'I96a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd3x3*complexconjugate(Rl1x1)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a313}')

I96a316 = Parameter(name = 'I96a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd3x3*complexconjugate(Rl1x1)*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a316}')

I96a321 = Parameter(name = 'I96a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*Rd3x3*complexconjugate(Rl2x2)*complexconjugate(Ru1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a321}')

I96a322 = Parameter(name = 'I96a322',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*Rd3x3*complexconjugate(Rl2x2)*complexconjugate(Ru2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a322}')

I96a323 = Parameter(name = 'I96a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd3x3*complexconjugate(Rl2x2)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a323}')

I96a326 = Parameter(name = 'I96a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd3x3*complexconjugate(Rl2x2)*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a326}')

I96a331 = Parameter(name = 'I96a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd3x3*complexconjugate(Rl3x3)*complexconjugate(Ru1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a331}')

I96a332 = Parameter(name = 'I96a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd3x3*complexconjugate(Rl3x3)*complexconjugate(Ru2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a332}')

I96a333 = Parameter(name = 'I96a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x3*complexconjugate(Rl3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a333}')

I96a336 = Parameter(name = 'I96a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x3*complexconjugate(Rl3x3)*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a336}')

I96a361 = Parameter(name = 'I96a361',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd3x3*complexconjugate(Rl6x3)*complexconjugate(Ru1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a361}')

I96a362 = Parameter(name = 'I96a362',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd3x3*complexconjugate(Rl6x3)*complexconjugate(Ru2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a362}')

I96a363 = Parameter(name = 'I96a363',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x3*complexconjugate(Rl6x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a363}')

I96a366 = Parameter(name = 'I96a366',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x3*complexconjugate(Rl6x3)*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a366}')

I96a611 = Parameter(name = 'I96a611',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x1x3*Rd6x3*complexconjugate(Rl1x1)*complexconjugate(Ru1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a611}')

I96a612 = Parameter(name = 'I96a612',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x2x3*Rd6x3*complexconjugate(Rl1x1)*complexconjugate(Ru2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a612}')

I96a613 = Parameter(name = 'I96a613',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd6x3*complexconjugate(Rl1x1)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a613}')

I96a616 = Parameter(name = 'I96a616',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd6x3*complexconjugate(Rl1x1)*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a616}')

I96a621 = Parameter(name = 'I96a621',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x1x3*Rd6x3*complexconjugate(Rl2x2)*complexconjugate(Ru1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a621}')

I96a622 = Parameter(name = 'I96a622',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x2x3*Rd6x3*complexconjugate(Rl2x2)*complexconjugate(Ru2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a622}')

I96a623 = Parameter(name = 'I96a623',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd6x3*complexconjugate(Rl2x2)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a623}')

I96a626 = Parameter(name = 'I96a626',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd6x3*complexconjugate(Rl2x2)*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a626}')

I96a631 = Parameter(name = 'I96a631',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd6x3*complexconjugate(Rl3x3)*complexconjugate(Ru1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a631}')

I96a632 = Parameter(name = 'I96a632',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd6x3*complexconjugate(Rl3x3)*complexconjugate(Ru2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a632}')

I96a633 = Parameter(name = 'I96a633',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x3*complexconjugate(Rl3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a633}')

I96a636 = Parameter(name = 'I96a636',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x3*complexconjugate(Rl3x3)*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a636}')

I96a661 = Parameter(name = 'I96a661',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd6x3*complexconjugate(Rl6x3)*complexconjugate(Ru1x1)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a661}')

I96a662 = Parameter(name = 'I96a662',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd6x3*complexconjugate(Rl6x3)*complexconjugate(Ru2x2)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a662}')

I96a663 = Parameter(name = 'I96a663',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x3*complexconjugate(Rl6x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a663}')

I96a666 = Parameter(name = 'I96a666',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x3*complexconjugate(Rl6x3)*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                    texname = '\\text{I96a666}')

I97a331 = Parameter(name = 'I97a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd3x6*complexconjugate(Rl3x6)*complexconjugate(Ru1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a331}')

I97a332 = Parameter(name = 'I97a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd3x6*complexconjugate(Rl3x6)*complexconjugate(Ru2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a332}')

I97a333 = Parameter(name = 'I97a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6*complexconjugate(Rl3x6)*complexconjugate(Ru3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a333}')

I97a336 = Parameter(name = 'I97a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6*complexconjugate(Rl3x6)*complexconjugate(Ru6x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a336}')

I97a361 = Parameter(name = 'I97a361',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd3x6*complexconjugate(Rl6x6)*complexconjugate(Ru1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a361}')

I97a362 = Parameter(name = 'I97a362',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd3x6*complexconjugate(Rl6x6)*complexconjugate(Ru2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a362}')

I97a363 = Parameter(name = 'I97a363',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6*complexconjugate(Rl6x6)*complexconjugate(Ru3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a363}')

I97a366 = Parameter(name = 'I97a366',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6*complexconjugate(Rl6x6)*complexconjugate(Ru6x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a366}')

I97a431 = Parameter(name = 'I97a431',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x1*Rd4x4*complexconjugate(Rl3x6)*complexconjugate(Ru1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a431}')

I97a432 = Parameter(name = 'I97a432',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x1*Rd4x4*complexconjugate(Rl3x6)*complexconjugate(Ru2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a432}')

I97a433 = Parameter(name = 'I97a433',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4*complexconjugate(Rl3x6)*complexconjugate(Ru3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a433}')

I97a436 = Parameter(name = 'I97a436',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4*complexconjugate(Rl3x6)*complexconjugate(Ru6x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a436}')

I97a461 = Parameter(name = 'I97a461',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x1*Rd4x4*complexconjugate(Rl6x6)*complexconjugate(Ru1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a461}')

I97a462 = Parameter(name = 'I97a462',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x1*Rd4x4*complexconjugate(Rl6x6)*complexconjugate(Ru2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a462}')

I97a463 = Parameter(name = 'I97a463',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4*complexconjugate(Rl6x6)*complexconjugate(Ru3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a463}')

I97a466 = Parameter(name = 'I97a466',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4*complexconjugate(Rl6x6)*complexconjugate(Ru6x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a466}')

I97a531 = Parameter(name = 'I97a531',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x2*Rd5x5*complexconjugate(Rl3x6)*complexconjugate(Ru1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a531}')

I97a532 = Parameter(name = 'I97a532',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x2*Rd5x5*complexconjugate(Rl3x6)*complexconjugate(Ru2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a532}')

I97a533 = Parameter(name = 'I97a533',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5*complexconjugate(Rl3x6)*complexconjugate(Ru3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a533}')

I97a536 = Parameter(name = 'I97a536',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5*complexconjugate(Rl3x6)*complexconjugate(Ru6x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a536}')

I97a561 = Parameter(name = 'I97a561',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x2*Rd5x5*complexconjugate(Rl6x6)*complexconjugate(Ru1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a561}')

I97a562 = Parameter(name = 'I97a562',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x2*Rd5x5*complexconjugate(Rl6x6)*complexconjugate(Ru2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a562}')

I97a563 = Parameter(name = 'I97a563',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5*complexconjugate(Rl6x6)*complexconjugate(Ru3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a563}')

I97a566 = Parameter(name = 'I97a566',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5*complexconjugate(Rl6x6)*complexconjugate(Ru6x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a566}')

I97a631 = Parameter(name = 'I97a631',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd6x6*complexconjugate(Rl3x6)*complexconjugate(Ru1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a631}')

I97a632 = Parameter(name = 'I97a632',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd6x6*complexconjugate(Rl3x6)*complexconjugate(Ru2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a632}')

I97a633 = Parameter(name = 'I97a633',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6*complexconjugate(Rl3x6)*complexconjugate(Ru3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a633}')

I97a636 = Parameter(name = 'I97a636',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6*complexconjugate(Rl3x6)*complexconjugate(Ru6x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a636}')

I97a661 = Parameter(name = 'I97a661',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x1x3*Rd6x6*complexconjugate(Rl6x6)*complexconjugate(Ru1x1)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a661}')

I97a662 = Parameter(name = 'I97a662',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x2x3*Rd6x6*complexconjugate(Rl6x6)*complexconjugate(Ru2x2)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a662}')

I97a663 = Parameter(name = 'I97a663',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6*complexconjugate(Rl6x6)*complexconjugate(Ru3x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a663}')

I97a666 = Parameter(name = 'I97a666',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6*complexconjugate(Rl6x6)*complexconjugate(Ru6x3)*complexconjugate(ye3x3)',
                    texname = '\\text{I97a666}')

I98a313 = Parameter(name = 'I98a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd3x6*complexconjugate(Rl1x1)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a313}')

I98a316 = Parameter(name = 'I98a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd3x6*complexconjugate(Rl1x1)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a316}')

I98a323 = Parameter(name = 'I98a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd3x6*complexconjugate(Rl2x2)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a323}')

I98a326 = Parameter(name = 'I98a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd3x6*complexconjugate(Rl2x2)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a326}')

I98a333 = Parameter(name = 'I98a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6*complexconjugate(Rl3x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a333}')

I98a336 = Parameter(name = 'I98a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6*complexconjugate(Rl3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a336}')

I98a363 = Parameter(name = 'I98a363',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6*complexconjugate(Rl6x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a363}')

I98a366 = Parameter(name = 'I98a366',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd3x6*complexconjugate(Rl6x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a366}')

I98a413 = Parameter(name = 'I98a413',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x1*Rd4x4*complexconjugate(Rl1x1)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a413}')

I98a416 = Parameter(name = 'I98a416',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x1*Rd4x4*complexconjugate(Rl1x1)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a416}')

I98a423 = Parameter(name = 'I98a423',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x1*Rd4x4*complexconjugate(Rl2x2)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a423}')

I98a426 = Parameter(name = 'I98a426',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x1*Rd4x4*complexconjugate(Rl2x2)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a426}')

I98a433 = Parameter(name = 'I98a433',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4*complexconjugate(Rl3x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a433}')

I98a436 = Parameter(name = 'I98a436',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4*complexconjugate(Rl3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a436}')

I98a463 = Parameter(name = 'I98a463',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4*complexconjugate(Rl6x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a463}')

I98a466 = Parameter(name = 'I98a466',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x1*Rd4x4*complexconjugate(Rl6x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a466}')

I98a513 = Parameter(name = 'I98a513',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x2*Rd5x5*complexconjugate(Rl1x1)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a513}')

I98a516 = Parameter(name = 'I98a516',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x2*Rd5x5*complexconjugate(Rl1x1)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a516}')

I98a523 = Parameter(name = 'I98a523',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x2*Rd5x5*complexconjugate(Rl2x2)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a523}')

I98a526 = Parameter(name = 'I98a526',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x2*Rd5x5*complexconjugate(Rl2x2)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a526}')

I98a533 = Parameter(name = 'I98a533',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5*complexconjugate(Rl3x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a533}')

I98a536 = Parameter(name = 'I98a536',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5*complexconjugate(Rl3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a536}')

I98a563 = Parameter(name = 'I98a563',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5*complexconjugate(Rl6x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a563}')

I98a566 = Parameter(name = 'I98a566',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x2*Rd5x5*complexconjugate(Rl6x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a566}')

I98a613 = Parameter(name = 'I98a613',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd6x6*complexconjugate(Rl1x1)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a613}')

I98a616 = Parameter(name = 'I98a616',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD1x3x3*Rd6x6*complexconjugate(Rl1x1)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a616}')

I98a623 = Parameter(name = 'I98a623',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd6x6*complexconjugate(Rl2x2)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a623}')

I98a626 = Parameter(name = 'I98a626',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD2x3x3*Rd6x6*complexconjugate(Rl2x2)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a626}')

I98a633 = Parameter(name = 'I98a633',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6*complexconjugate(Rl3x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a633}')

I98a636 = Parameter(name = 'I98a636',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6*complexconjugate(Rl3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a636}')

I98a663 = Parameter(name = 'I98a663',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6*complexconjugate(Rl6x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a663}')

I98a666 = Parameter(name = 'I98a666',
                    nature = 'internal',
                    type = 'complex',
                    value = 'LLQD3x3x3*Rd6x6*complexconjugate(Rl6x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                    texname = '\\text{I98a666}')

I99a311 = Parameter(name = 'I99a311',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD1x1x3*complexconjugate(Rl1x1)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a311}')

I99a312 = Parameter(name = 'I99a312',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD1x2x3*complexconjugate(Rl1x1)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a312}')

I99a313 = Parameter(name = 'I99a313',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD1x3x3*complexconjugate(Rl1x1)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a313}')

I99a316 = Parameter(name = 'I99a316',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD1x3x3*complexconjugate(Rl1x1)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a316}')

I99a321 = Parameter(name = 'I99a321',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD2x1x3*complexconjugate(Rl2x2)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a321}')

I99a322 = Parameter(name = 'I99a322',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD2x2x3*complexconjugate(Rl2x2)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a322}')

I99a323 = Parameter(name = 'I99a323',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD2x3x3*complexconjugate(Rl2x2)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a323}')

I99a326 = Parameter(name = 'I99a326',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD2x3x3*complexconjugate(Rl2x2)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a326}')

I99a331 = Parameter(name = 'I99a331',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x1x3*complexconjugate(Rl3x3)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a331}')

I99a332 = Parameter(name = 'I99a332',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x2x3*complexconjugate(Rl3x3)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a332}')

I99a333 = Parameter(name = 'I99a333',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x3x3*complexconjugate(Rl3x3)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a333}')

I99a336 = Parameter(name = 'I99a336',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x3x3*complexconjugate(Rl3x3)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a336}')

I99a361 = Parameter(name = 'I99a361',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x1x3*complexconjugate(Rl6x3)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a361}')

I99a362 = Parameter(name = 'I99a362',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x2x3*complexconjugate(Rl6x3)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a362}')

I99a363 = Parameter(name = 'I99a363',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x3x3*complexconjugate(Rl6x3)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a363}')

I99a366 = Parameter(name = 'I99a366',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*TLQD3x3x3*complexconjugate(Rl6x3)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a366}')

I99a411 = Parameter(name = 'I99a411',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD1x1x1*complexconjugate(Rl1x1)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a411}')

I99a412 = Parameter(name = 'I99a412',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD1x2x1*complexconjugate(Rl1x1)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a412}')

I99a413 = Parameter(name = 'I99a413',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD1x3x1*complexconjugate(Rl1x1)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a413}')

I99a416 = Parameter(name = 'I99a416',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD1x3x1*complexconjugate(Rl1x1)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a416}')

I99a421 = Parameter(name = 'I99a421',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD2x1x1*complexconjugate(Rl2x2)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a421}')

I99a422 = Parameter(name = 'I99a422',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD2x2x1*complexconjugate(Rl2x2)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a422}')

I99a423 = Parameter(name = 'I99a423',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD2x3x1*complexconjugate(Rl2x2)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a423}')

I99a426 = Parameter(name = 'I99a426',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD2x3x1*complexconjugate(Rl2x2)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a426}')

I99a431 = Parameter(name = 'I99a431',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x1x1*complexconjugate(Rl3x3)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a431}')

I99a432 = Parameter(name = 'I99a432',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x2x1*complexconjugate(Rl3x3)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a432}')

I99a433 = Parameter(name = 'I99a433',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x3x1*complexconjugate(Rl3x3)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a433}')

I99a436 = Parameter(name = 'I99a436',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x3x1*complexconjugate(Rl3x3)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a436}')

I99a461 = Parameter(name = 'I99a461',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x1x1*complexconjugate(Rl6x3)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a461}')

I99a462 = Parameter(name = 'I99a462',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x2x1*complexconjugate(Rl6x3)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a462}')

I99a463 = Parameter(name = 'I99a463',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x3x1*complexconjugate(Rl6x3)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a463}')

I99a466 = Parameter(name = 'I99a466',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*TLQD3x3x1*complexconjugate(Rl6x3)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a466}')

I99a511 = Parameter(name = 'I99a511',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD1x1x2*complexconjugate(Rl1x1)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a511}')

I99a512 = Parameter(name = 'I99a512',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD1x2x2*complexconjugate(Rl1x1)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a512}')

I99a513 = Parameter(name = 'I99a513',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD1x3x2*complexconjugate(Rl1x1)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a513}')

I99a516 = Parameter(name = 'I99a516',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD1x3x2*complexconjugate(Rl1x1)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a516}')

I99a521 = Parameter(name = 'I99a521',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD2x1x2*complexconjugate(Rl2x2)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a521}')

I99a522 = Parameter(name = 'I99a522',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD2x2x2*complexconjugate(Rl2x2)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a522}')

I99a523 = Parameter(name = 'I99a523',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD2x3x2*complexconjugate(Rl2x2)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a523}')

I99a526 = Parameter(name = 'I99a526',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD2x3x2*complexconjugate(Rl2x2)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a526}')

I99a531 = Parameter(name = 'I99a531',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x1x2*complexconjugate(Rl3x3)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a531}')

I99a532 = Parameter(name = 'I99a532',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x2x2*complexconjugate(Rl3x3)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a532}')

I99a533 = Parameter(name = 'I99a533',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x3x2*complexconjugate(Rl3x3)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a533}')

I99a536 = Parameter(name = 'I99a536',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x3x2*complexconjugate(Rl3x3)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a536}')

I99a561 = Parameter(name = 'I99a561',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x1x2*complexconjugate(Rl6x3)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a561}')

I99a562 = Parameter(name = 'I99a562',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x2x2*complexconjugate(Rl6x3)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a562}')

I99a563 = Parameter(name = 'I99a563',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x3x2*complexconjugate(Rl6x3)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a563}')

I99a566 = Parameter(name = 'I99a566',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*TLQD3x3x2*complexconjugate(Rl6x3)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a566}')

I99a611 = Parameter(name = 'I99a611',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD1x1x3*complexconjugate(Rl1x1)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a611}')

I99a612 = Parameter(name = 'I99a612',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD1x2x3*complexconjugate(Rl1x1)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a612}')

I99a613 = Parameter(name = 'I99a613',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD1x3x3*complexconjugate(Rl1x1)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a613}')

I99a616 = Parameter(name = 'I99a616',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD1x3x3*complexconjugate(Rl1x1)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a616}')

I99a621 = Parameter(name = 'I99a621',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD2x1x3*complexconjugate(Rl2x2)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a621}')

I99a622 = Parameter(name = 'I99a622',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD2x2x3*complexconjugate(Rl2x2)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a622}')

I99a623 = Parameter(name = 'I99a623',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD2x3x3*complexconjugate(Rl2x2)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a623}')

I99a626 = Parameter(name = 'I99a626',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD2x3x3*complexconjugate(Rl2x2)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a626}')

I99a631 = Parameter(name = 'I99a631',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x1x3*complexconjugate(Rl3x3)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a631}')

I99a632 = Parameter(name = 'I99a632',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x2x3*complexconjugate(Rl3x3)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a632}')

I99a633 = Parameter(name = 'I99a633',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x3x3*complexconjugate(Rl3x3)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a633}')

I99a636 = Parameter(name = 'I99a636',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x3x3*complexconjugate(Rl3x3)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a636}')

I99a661 = Parameter(name = 'I99a661',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x1x3*complexconjugate(Rl6x3)*complexconjugate(Ru1x1)',
                    texname = '\\text{I99a661}')

I99a662 = Parameter(name = 'I99a662',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x2x3*complexconjugate(Rl6x3)*complexconjugate(Ru2x2)',
                    texname = '\\text{I99a662}')

I99a663 = Parameter(name = 'I99a663',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x3x3*complexconjugate(Rl6x3)*complexconjugate(Ru3x3)',
                    texname = '\\text{I99a663}')

I99a666 = Parameter(name = 'I99a666',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*TLQD3x3x3*complexconjugate(Rl6x3)*complexconjugate(Ru6x3)',
                    texname = '\\text{I99a666}')


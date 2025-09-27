# This file was automatically created by FeynRules 2.4.35
# Mathematica version: 10.1.0  for Mac OS X x86 (64-bit) (March 24, 2015)
# Date: Thu 11 Feb 2016 16:03:33


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot
try:
   import form_factors as ForFac 
except ImportError:
   pass


FF1 = Lorentz(name = 'FF1',
              spins = [ 2, 2 ],
              structure = 'P(-1,1)*Gamma(-1,2,1)')

FF2 = Lorentz(name = 'FF2',
              spins = [ 2, 2 ],
              structure = 'ProjM(2,1)')

FF3 = Lorentz(name = 'FF3',
              spins = [ 2, 2 ],
              structure = 'P(-1,1)**2*ProjM(2,1)')

FF4 = Lorentz(name = 'FF4',
              spins = [ 2, 2 ],
              structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1)')

FF5 = Lorentz(name = 'FF5',
              spins = [ 2, 2 ],
              structure = 'ProjP(2,1)')

FF6 = Lorentz(name = 'FF6',
              spins = [ 2, 2 ],
              structure = 'P(-1,1)**2*ProjP(2,1)')

FF7 = Lorentz(name = 'FF7',
              spins = [ 2, 2 ],
              structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

VV1 = Lorentz(name = 'VV1',
              spins = [ 3, 3 ],
              structure = 'P(1,2)*P(2,2)')

VV2 = Lorentz(name = 'VV2',
              spins = [ 3, 3 ],
              structure = 'Metric(1,2)')

VV3 = Lorentz(name = 'VV3',
              spins = [ 3, 3 ],
              structure = 'P(-1,2)**2*Metric(1,2)')

UUS1 = Lorentz(name = 'UUS1',
               spins = [ -1, -1, 1 ],
               structure = '1')

UUV1 = Lorentz(name = 'UUV1',
               spins = [ -1, -1, 3 ],
               structure = 'P(3,2) + P(3,3)')

SSS1 = Lorentz(name = 'SSS1',
               spins = [ 1, 1, 1 ],
               structure = '1')

FFS1 = Lorentz(name = 'FFS1',
               spins = [ 2, 2, 1 ],
               structure = 'Gamma5(2,1)')

FFS2 = Lorentz(name = 'FFS2',
               spins = [ 2, 2, 1 ],
               structure = 'Identity(2,1)')

FFS3 = Lorentz(name = 'FFS3',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1)')

FFS4 = Lorentz(name = 'FFS4',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,1)**2*ProjM(2,1)')

FFS5 = Lorentz(name = 'FFS5',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS6 = Lorentz(name = 'FFS6',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS7 = Lorentz(name = 'FFS7',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) - 2*P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS8 = Lorentz(name = 'FFS8',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) - (P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1))/2.')

FFS9 = Lorentz(name = 'FFS9',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + (P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1))/2.')

FFS10 = Lorentz(name = 'FFS10',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS11 = Lorentz(name = 'FFS11',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + (3*P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1))/2.')

FFS12 = Lorentz(name = 'FFS12',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + 3*P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS13 = Lorentz(name = 'FFS13',
                spins = [ 2, 2, 1 ],
                structure = 'P(-2,3)*P(-1,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*ProjM(-3,1) + 3*P(-1,1)**2*ProjM(2,1) + 4*P(-1,1)*P(-1,3)*ProjM(2,1) + 3*P(-1,3)**2*ProjM(2,1)')

FFS14 = Lorentz(name = 'FFS14',
                spins = [ 2, 2, 1 ],
                structure = 'P(-2,3)*P(-1,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*ProjM(-3,1) + (9*P(-1,1)**2*ProjM(2,1))/2. + 7*P(-1,1)*P(-1,3)*ProjM(2,1) + (9*P(-1,3)**2*ProjM(2,1))/2.')

FFS15 = Lorentz(name = 'FFS15',
                spins = [ 2, 2, 1 ],
                structure = 'ProjM(2,1) - ProjP(2,1)')

FFS16 = Lorentz(name = 'FFS16',
                spins = [ 2, 2, 1 ],
                structure = 'Gamma5(2,1) - ProjM(2,1)/2. + ProjP(2,1)/2.')

FFS17 = Lorentz(name = 'FFS17',
                spins = [ 2, 2, 1 ],
                structure = 'Identity(2,1) + ProjM(2,1)/2. + ProjP(2,1)/2.')

FFS18 = Lorentz(name = 'FFS18',
                spins = [ 2, 2, 1 ],
                structure = 'ProjP(2,1)')

FFS19 = Lorentz(name = 'FFS19',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)**2*ProjP(2,1)')

FFS20 = Lorentz(name = 'FFS20',
                spins = [ 2, 2, 1 ],
                structure = 'ProjM(2,1) + ProjP(2,1)')

FFS21 = Lorentz(name = 'FFS21',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFS22 = Lorentz(name = 'FFS22',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFS23 = Lorentz(name = 'FFS23',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1) - 2*P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFS24 = Lorentz(name = 'FFS24',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1) - (P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1))/2.')

FFS25 = Lorentz(name = 'FFS25',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1) + (P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1))/2.')

FFS26 = Lorentz(name = 'FFS26',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFS27 = Lorentz(name = 'FFS27',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1) + (3*P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1))/2.')

FFS28 = Lorentz(name = 'FFS28',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1) + 3*P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFS29 = Lorentz(name = 'FFS29',
                spins = [ 2, 2, 1 ],
                structure = 'P(-2,3)*P(-1,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3)*ProjP(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*ProjP(-3,1) + 3*P(-1,1)**2*ProjP(2,1) + 4*P(-1,1)*P(-1,3)*ProjP(2,1) + 3*P(-1,3)**2*ProjP(2,1)')

FFS30 = Lorentz(name = 'FFS30',
                spins = [ 2, 2, 1 ],
                structure = 'P(-2,3)*P(-1,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3)*ProjP(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*ProjP(-3,1) + (9*P(-1,1)**2*ProjP(2,1))/2. + 7*P(-1,1)*P(-1,3)*ProjP(2,1) + (9*P(-1,3)**2*ProjP(2,1))/2.')

FFV1 = Lorentz(name = 'FFV1',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,1)')

FFV2 = Lorentz(name = 'FFV2',
               spins = [ 2, 2, 3 ],
               structure = 'P(3,1)*ProjM(2,1)')

FFV3 = Lorentz(name = 'FFV3',
               spins = [ 2, 2, 3 ],
               structure = 'P(3,2)*ProjM(2,1)')

FFV4 = Lorentz(name = 'FFV4',
               spins = [ 2, 2, 3 ],
               structure = 'P(3,1)*ProjM(2,1) + P(3,2)*ProjM(2,1)')

FFV5 = Lorentz(name = 'FFV5',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFV6 = Lorentz(name = 'FFV6',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)')

FFV7 = Lorentz(name = 'FFV7',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)')

FFV8 = Lorentz(name = 'FFV8',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV9 = Lorentz(name = 'FFV9',
               spins = [ 2, 2, 3 ],
               structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV10 = Lorentz(name = 'FFV10',
                spins = [ 2, 2, 3 ],
                structure = '(13*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV11 = Lorentz(name = 'FFV11',
                spins = [ 2, 2, 3 ],
                structure = '(13*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (8*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + (4*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/11.')

FFV12 = Lorentz(name = 'FFV12',
                spins = [ 2, 2, 3 ],
                structure = '(11*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/13. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/13. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + (5*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/13.')

FFV13 = Lorentz(name = 'FFV13',
                spins = [ 2, 2, 3 ],
                structure = '(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/2. + (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/2. + P(3,1)*ProjM(2,1) + P(3,2)*ProjM(2,1)')

FFV14 = Lorentz(name = 'FFV14',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/3. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/3.')

FFV15 = Lorentz(name = 'FFV15',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + P(3,1)*ProjM(2,1) + P(3,2)*ProjM(2,1)')

FFV16 = Lorentz(name = 'FFV16',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV17 = Lorentz(name = 'FFV17',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV18 = Lorentz(name = 'FFV18',
                spins = [ 2, 2, 3 ],
                structure = '2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV19 = Lorentz(name = 'FFV19',
                spins = [ 2, 2, 3 ],
                structure = 'P(3,1)*ProjP(2,1)')

FFV20 = Lorentz(name = 'FFV20',
                spins = [ 2, 2, 3 ],
                structure = 'P(3,2)*ProjP(2,1)')

FFV21 = Lorentz(name = 'FFV21',
                spins = [ 2, 2, 3 ],
                structure = 'P(3,1)*ProjP(2,1) + P(3,2)*ProjP(2,1)')

FFV22 = Lorentz(name = 'FFV22',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFV23 = Lorentz(name = 'FFV23',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) - 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFV24 = Lorentz(name = 'FFV24',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,1) - Gamma(3,2,-1)*ProjM(-1,1) - Gamma(3,2,-1)*ProjP(-1,1)')

FFV25 = Lorentz(name = 'FFV25',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,1) - (4*Gamma(3,2,-1)*ProjM(-1,1))/13. - (4*Gamma(3,2,-1)*ProjP(-1,1))/13.')

FFV26 = Lorentz(name = 'FFV26',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,1) + (8*Gamma(3,2,-1)*ProjM(-1,1))/13. + (8*Gamma(3,2,-1)*ProjP(-1,1))/13.')

FFV27 = Lorentz(name = 'FFV27',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + Gamma(3,2,-1)*ProjP(-1,1)')

FFV28 = Lorentz(name = 'FFV28',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFV29 = Lorentz(name = 'FFV29',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,1) + 2*Gamma(3,2,-1)*ProjM(-1,1) + 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFV30 = Lorentz(name = 'FFV30',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + 4*Gamma(3,2,-1)*ProjP(-1,1)')

FFV31 = Lorentz(name = 'FFV31',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFV32 = Lorentz(name = 'FFV32',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFV33 = Lorentz(name = 'FFV33',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV34 = Lorentz(name = 'FFV34',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV35 = Lorentz(name = 'FFV35',
                spins = [ 2, 2, 3 ],
                structure = '(13*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV36 = Lorentz(name = 'FFV36',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + P(3,1)*ProjM(2,1) + P(3,2)*ProjM(2,1) - P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) - P(3,1)*ProjP(2,1) - P(3,2)*ProjP(2,1)')

FFV37 = Lorentz(name = 'FFV37',
                spins = [ 2, 2, 3 ],
                structure = '(13*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (8*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) + (4*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/11.')

FFV38 = Lorentz(name = 'FFV38',
                spins = [ 2, 2, 3 ],
                structure = '(11*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/13. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/13. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) + (5*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/13.')

FFV39 = Lorentz(name = 'FFV39',
                spins = [ 2, 2, 3 ],
                structure = '(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/2. + (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/2. + P(3,1)*ProjP(2,1) + P(3,2)*ProjP(2,1)')

FFV40 = Lorentz(name = 'FFV40',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/3. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) + (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/3.')

FFV41 = Lorentz(name = 'FFV41',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) + P(3,1)*ProjP(2,1) + P(3,2)*ProjP(2,1)')

FFV42 = Lorentz(name = 'FFV42',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + P(3,1)*ProjM(2,1) + P(3,2)*ProjM(2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) + P(3,1)*ProjP(2,1) + P(3,2)*ProjP(2,1)')

FFV43 = Lorentz(name = 'FFV43',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV44 = Lorentz(name = 'FFV44',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV45 = Lorentz(name = 'FFV45',
                spins = [ 2, 2, 3 ],
                structure = '2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

VSS1 = Lorentz(name = 'VSS1',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) - P(1,3)')

VVS1 = Lorentz(name = 'VVS1',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,2)*P(2,1)')

VVS2 = Lorentz(name = 'VVS2',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1)')

VVS3 = Lorentz(name = 'VVS3',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,3)*P(2,1)')

VVS4 = Lorentz(name = 'VVS4',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,2)*P(2,3)')

VVS5 = Lorentz(name = 'VVS5',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1)')

VVS6 = Lorentz(name = 'VVS6',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2)')

VVS7 = Lorentz(name = 'VVS7',
               spins = [ 3, 3, 1 ],
               structure = 'Metric(1,2)')

VVS8 = Lorentz(name = 'VVS8',
               spins = [ 3, 3, 1 ],
               structure = 'P(-1,1)*P(-1,2)*Metric(1,2)')

VVS9 = Lorentz(name = 'VVS9',
               spins = [ 3, 3, 1 ],
               structure = 'P(-1,2)**2*Metric(1,2)')

VVS10 = Lorentz(name = 'VVS10',
                spins = [ 3, 3, 1 ],
                structure = 'P(-1,1)*P(-1,3)*Metric(1,2)')

VVS11 = Lorentz(name = 'VVS11',
                spins = [ 3, 3, 1 ],
                structure = 'P(-1,2)*P(-1,3)*Metric(1,2)')

VVV1 = Lorentz(name = 'VVV1',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,1))')

VVV2 = Lorentz(name = 'VVV2',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,2))')

VVV3 = Lorentz(name = 'VVV3',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2)')

VVV4 = Lorentz(name = 'VVV4',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,2)*Metric(1,2)')

VVV5 = Lorentz(name = 'VVV5',
               spins = [ 3, 3, 3 ],
               structure = 'P(2,1)*Metric(1,3)')

VVV6 = Lorentz(name = 'VVV6',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,2)*Metric(2,3)')

VVV7 = Lorentz(name = 'VVV7',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

SSSS1 = Lorentz(name = 'SSSS1',
                spins = [ 1, 1, 1, 1 ],
                structure = '1')

FFSS1 = Lorentz(name = 'FFSS1',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjM(2,1)')

FFSS2 = Lorentz(name = 'FFSS2',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS3 = Lorentz(name = 'FFSS3',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS4 = Lorentz(name = 'FFSS4',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS5 = Lorentz(name = 'FFSS5',
                spins = [ 2, 2, 1, 1 ],
                structure = 'ProjP(2,1)')

FFSS6 = Lorentz(name = 'FFSS6',
                spins = [ 2, 2, 1, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFFF1 = Lorentz(name = 'FFFF1',
                spins = [ 2, 2, 2, 2 ],
                structure = 'ProjM(2,1)*ProjM(4,3)')

FFFF2 = Lorentz(name = 'FFFF2',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Gamma(-2,-4,-3)*Gamma(-2,2,-6)*Gamma(-1,-6,-5)*Gamma(-1,4,-4)*ProjM(-5,1)*ProjM(-3,3)')

FFFF3 = Lorentz(name = 'FFFF3',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Gamma(-2,-4,-3)*Gamma(-2,4,-6)*Gamma(-1,-6,-5)*Gamma(-1,2,-4)*ProjM(-5,3)*ProjM(-3,1)')

FFFF4 = Lorentz(name = 'FFFF4',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Gamma(-2,-6,-5)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*Gamma(-1,4,-6)*ProjM(-5,3)*ProjM(-3,1)')

FFFF5 = Lorentz(name = 'FFFF5',
                spins = [ 2, 2, 2, 2 ],
                structure = 'ProjM(4,3)*ProjP(2,1)')

FFFF6 = Lorentz(name = 'FFFF6',
                spins = [ 2, 2, 2, 2 ],
                structure = 'ProjM(2,1)*ProjP(4,3)')

FFFF7 = Lorentz(name = 'FFFF7',
                spins = [ 2, 2, 2, 2 ],
                structure = 'ProjP(2,1)*ProjP(4,3)')

FFFF8 = Lorentz(name = 'FFFF8',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Gamma(-2,-4,-3)*Gamma(-2,4,-6)*Gamma(-1,-6,-5)*Gamma(-1,2,-4)*ProjM(-5,3)*ProjP(-3,1)')

FFFF9 = Lorentz(name = 'FFFF9',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Gamma(-2,-4,-3)*Gamma(-2,2,-6)*Gamma(-1,-6,-5)*Gamma(-1,4,-4)*ProjM(-5,1)*ProjP(-3,3)')

FFFF10 = Lorentz(name = 'FFFF10',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-4,-3)*Gamma(-2,2,-6)*Gamma(-1,-6,-5)*Gamma(-1,4,-4)*ProjM(-3,3)*ProjP(-5,1)')

FFFF11 = Lorentz(name = 'FFFF11',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-6,-5)*Gamma(-2,-4,-3)*Gamma(-1,2,-6)*Gamma(-1,4,-4)*ProjM(-3,3)*ProjP(-5,1)')

FFFF12 = Lorentz(name = 'FFFF12',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-4,-3)*Gamma(-2,2,-6)*Gamma(-1,-6,-5)*Gamma(-1,4,-4)*ProjP(-5,1)*ProjP(-3,3)')

FFFF13 = Lorentz(name = 'FFFF13',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-4,-3)*Gamma(-2,4,-6)*Gamma(-1,-6,-5)*Gamma(-1,2,-4)*ProjM(-3,1)*ProjP(-5,3)')

FFFF14 = Lorentz(name = 'FFFF14',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-6,-5)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*Gamma(-1,4,-6)*ProjM(-3,1)*ProjP(-5,3)')

FFFF15 = Lorentz(name = 'FFFF15',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-4,-3)*Gamma(-2,4,-6)*Gamma(-1,-6,-5)*Gamma(-1,2,-4)*ProjP(-5,3)*ProjP(-3,1)')

FFFF16 = Lorentz(name = 'FFFF16',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-6,-5)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*Gamma(-1,4,-6)*ProjP(-5,3)*ProjP(-3,1)')

FFVS1 = Lorentz(name = 'FFVS1',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(3,1)*ProjM(2,1)')

FFVS2 = Lorentz(name = 'FFVS2',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(3,2)*ProjM(2,1)')

FFVS3 = Lorentz(name = 'FFVS3',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(3,3)*ProjM(2,1)')

FFVS4 = Lorentz(name = 'FFVS4',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(3,4)*ProjM(2,1)')

FFVS5 = Lorentz(name = 'FFVS5',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(3,1)*ProjM(2,1) + P(3,2)*ProjM(2,1) + P(3,4)*ProjM(2,1)')

FFVS6 = Lorentz(name = 'FFVS6',
                spins = [ 2, 2, 3, 1 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFVS7 = Lorentz(name = 'FFVS7',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)')

FFVS8 = Lorentz(name = 'FFVS8',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)')

FFVS9 = Lorentz(name = 'FFVS9',
                spins = [ 2, 2, 3, 1 ],
                structure = 'P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)')

FFVS10 = Lorentz(name = 'FFVS10',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS11 = Lorentz(name = 'FFVS11',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS12 = Lorentz(name = 'FFVS12',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS13 = Lorentz(name = 'FFVS13',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(13*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS14 = Lorentz(name = 'FFVS14',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '-(P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)) + P(3,3)*ProjM(2,1)')

FFVS15 = Lorentz(name = 'FFVS15',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(13*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + (4*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/11.')

FFVS16 = Lorentz(name = 'FFVS16',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(13*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (8*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + (4*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/11.')

FFVS17 = Lorentz(name = 'FFVS17',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/2. + (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/2. + P(3,1)*ProjM(2,1) + P(3,2)*ProjM(2,1) + P(3,4)*ProjM(2,1)')

FFVS18 = Lorentz(name = 'FFVS18',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/3. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/3.')

FFVS19 = Lorentz(name = 'FFVS19',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + P(3,1)*ProjM(2,1) + P(3,2)*ProjM(2,1) + P(3,4)*ProjM(2,1)')

FFVS20 = Lorentz(name = 'FFVS20',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS21 = Lorentz(name = 'FFVS21',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS22 = Lorentz(name = 'FFVS22',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS23 = Lorentz(name = 'FFVS23',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(11*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/13. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/13. + (11*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/13. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + (5*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/13. + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS24 = Lorentz(name = 'FFVS24',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/3. + P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/3. + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS25 = Lorentz(name = 'FFVS25',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,1)*ProjP(2,1)')

FFVS26 = Lorentz(name = 'FFVS26',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,2)*ProjP(2,1)')

FFVS27 = Lorentz(name = 'FFVS27',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,3)*ProjP(2,1)')

FFVS28 = Lorentz(name = 'FFVS28',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,4)*ProjP(2,1)')

FFVS29 = Lorentz(name = 'FFVS29',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,1)*ProjP(2,1) + P(3,2)*ProjP(2,1) + P(3,4)*ProjP(2,1)')

FFVS30 = Lorentz(name = 'FFVS30',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFVS31 = Lorentz(name = 'FFVS31',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFVS32 = Lorentz(name = 'FFVS32',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFVS33 = Lorentz(name = 'FFVS33',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFVS34 = Lorentz(name = 'FFVS34',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS35 = Lorentz(name = 'FFVS35',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS36 = Lorentz(name = 'FFVS36',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS37 = Lorentz(name = 'FFVS37',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(13*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS38 = Lorentz(name = 'FFVS38',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '-(P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)) + P(3,3)*ProjP(2,1)')

FFVS39 = Lorentz(name = 'FFVS39',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(13*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (8*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) + (4*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/11.')

FFVS40 = Lorentz(name = 'FFVS40',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/2. + (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/2. + P(3,1)*ProjP(2,1) + P(3,2)*ProjP(2,1) + P(3,4)*ProjP(2,1)')

FFVS41 = Lorentz(name = 'FFVS41',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/3. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) + (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/3.')

FFVS42 = Lorentz(name = 'FFVS42',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) + P(3,1)*ProjP(2,1) + P(3,2)*ProjP(2,1) + P(3,4)*ProjP(2,1)')

FFVS43 = Lorentz(name = 'FFVS43',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS44 = Lorentz(name = 'FFVS44',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS45 = Lorentz(name = 'FFVS45',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS46 = Lorentz(name = 'FFVS46',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(11*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/13. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/13. + (11*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/13. + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) + (5*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/13. + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS47 = Lorentz(name = 'FFVS47',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/3. + P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1) + (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/3. + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS48 = Lorentz(name = 'FFVS48',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,-2,-3)*Gamma(3,2,-2)*ProjP(-3,1)')

FFVV1 = Lorentz(name = 'FFVV1',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Metric(3,4)*ProjM(2,1)')

FFVV2 = Lorentz(name = 'FFVV2',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVV3 = Lorentz(name = 'FFVV3',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)')

FFVV4 = Lorentz(name = 'FFVV4',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Metric(3,4)*ProjP(2,1)')

FFVV5 = Lorentz(name = 'FFVV5',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV6 = Lorentz(name = 'FFVV6',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)')

VVSS1 = Lorentz(name = 'VVSS1',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Metric(1,2)')

VVVV1 = Lorentz(name = 'VVVV1',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV2 = Lorentz(name = 'VVVV2',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVV3 = Lorentz(name = 'VVVV3',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV4 = Lorentz(name = 'VVVV4',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVV5 = Lorentz(name = 'VVVV5',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

FFSSS1 = Lorentz(name = 'FFSSS1',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjM(2,1)')

FFSSS2 = Lorentz(name = 'FFSSS2',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjP(2,1)')

FFVSS1 = Lorentz(name = 'FFVSS1',
                 spins = [ 2, 2, 3, 1, 1 ],
                 structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFVSS2 = Lorentz(name = 'FFVSS2',
                 spins = [ 2, 2, 3, 1, 1 ],
                 structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFVVS1 = Lorentz(name = 'FFVVS1',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVVS2 = Lorentz(name = 'FFVVS2',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)')

FFVVS3 = Lorentz(name = 'FFVVS3',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVVS4 = Lorentz(name = 'FFVVS4',
                 spins = [ 2, 2, 3, 3, 1 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)')


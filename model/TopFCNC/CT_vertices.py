# This file was automatically created by FeynRules 2.4.35
# Mathematica version: 10.1.0  for Mac OS X x86 (64-bit) (March 24, 2015)
# Date: Thu 11 Feb 2016 16:03:48


from object_library import all_vertices, all_CTvertices, Vertex, CTVertex
import particles as P
import CT_couplings as C
import lorentz as L


V_1 = CTVertex(name = 'V_1',
               type = 'R2',
               particles = [ P.g, P.g, P.g ],
               color = [ 'f(1,2,3)' ],
               lorentz = [ L.VVV7 ],
               loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ] ],
               couplings = {(0,0,0):C.R2GC_2984_2423,(0,0,1):C.R2GC_2984_2424})

V_2 = CTVertex(name = 'V_2',
               type = 'R2',
               particles = [ P.t__tilde__, P.b, P.G__plus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS18 ],
               loop_particles = [ [ [P.b, P.g, P.t] ] ],
               couplings = {(0,0,0):C.R2GC_3064_2490})

V_3 = CTVertex(name = 'V_3',
               type = 'R2',
               particles = [ P.b__tilde__, P.d, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS6 ],
               loop_particles = [ [ [P.b, P.d, P.g] ] ],
               couplings = {(0,0,0):C.R2GC_2058_934})

V_4 = CTVertex(name = 'V_4',
               type = 'R2',
               particles = [ P.b__tilde__, P.s, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS6 ],
               loop_particles = [ [ [P.b, P.g, P.s] ] ],
               couplings = {(0,0,0):C.R2GC_2080_960})

V_5 = CTVertex(name = 'V_5',
               type = 'R2',
               particles = [ P.d__tilde__, P.b, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS6 ],
               loop_particles = [ [ [P.b, P.d, P.g] ] ],
               couplings = {(0,0,0):C.R2GC_2073_951})

V_6 = CTVertex(name = 'V_6',
               type = 'R2',
               particles = [ P.s__tilde__, P.b, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS6 ],
               loop_particles = [ [ [P.b, P.g, P.s] ] ],
               couplings = {(0,0,0):C.R2GC_2095_977})

V_7 = CTVertex(name = 'V_7',
               type = 'R2',
               particles = [ P.t__tilde__, P.t, P.G0 ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS1 ],
               loop_particles = [ [ [P.g, P.t] ] ],
               couplings = {(0,0,0):C.R2GC_890_3024})

V_8 = CTVertex(name = 'V_8',
               type = 'R2',
               particles = [ P.t__tilde__, P.t, P.H ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS11, L.FFS18, L.FFS2, L.FFS24, L.FFS27, L.FFS3, L.FFS8 ],
               loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
               couplings = {(0,2,1):C.R2GC_889_3023,(0,5,0):C.R2GC_3621_2926,(0,5,2):C.R2GC_3621_2927,(0,1,0):C.R2GC_3666_2932,(0,1,2):C.R2GC_3666_2933,(0,6,0):C.R2GC_1086_102,(0,6,2):C.R2GC_1086_103,(0,0,0):C.R2GC_1090_109,(0,0,2):C.R2GC_1090_110,(0,4,0):C.R2GC_1082_95,(0,4,2):C.R2GC_1082_96,(0,3,0):C.R2GC_1085_100,(0,3,2):C.R2GC_1085_101})

V_9 = CTVertex(name = 'V_9',
               type = 'R2',
               particles = [ P.b__tilde__, P.t, P.G__minus__ ],
               color = [ 'Identity(1,2)' ],
               lorentz = [ L.FFS3 ],
               loop_particles = [ [ [P.b, P.g, P.t] ] ],
               couplings = {(0,0,0):C.R2GC_3065_2491})

V_10 = CTVertex(name = 'V_10',
                type = 'R2',
                particles = [ P.u__tilde__, P.t, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS14, L.FFS18, L.FFS19, L.FFS22, L.FFS23, L.FFS3, L.FFS30, L.FFS4, L.FFS6, L.FFS7 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,8,1):C.R2GC_3334_2651,(0,3,1):C.R2GC_3321_2641,(0,5,0):C.R2GC_3464_2800,(0,1,0):C.R2GC_3034_2477,(0,9,0):C.R2GC_928_3062,(0,4,0):C.R2GC_872_3006,(0,0,0):C.R2GC_925_3059,(0,6,0):C.R2GC_855_2989,(0,7,2):C.R2GC_1759_555,(0,2,2):C.R2GC_1685_464})

V_11 = CTVertex(name = 'V_11',
                type = 'R2',
                particles = [ P.u__tilde__, P.t, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS14, L.FFS18, L.FFS19, L.FFS21, L.FFS3, L.FFS30, L.FFS4, L.FFS5 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,4,0):C.R2GC_3513_2843,(0,4,1):C.R2GC_3513_2844,(0,1,0):C.R2GC_3233_2612,(0,1,1):C.R2GC_3233_2613,(0,7,0):C.R2GC_929_3063,(0,3,0):C.R2GC_871_3005,(0,0,0):C.R2GC_924_3058,(0,5,0):C.R2GC_856_2990,(0,6,2):C.R2GC_1758_554,(0,2,2):C.R2GC_1686_465})

V_12 = CTVertex(name = 'V_12',
                type = 'R2',
                particles = [ P.d__tilde__, P.t, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS10, L.FFS14, L.FFS3, L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,2,2):C.R2GC_1597_427,(0,0,2):C.R2GC_1592_422,(0,1,2):C.R2GC_923_3057,(0,3,0):C.R2GC_1390_179,(0,4,1):C.R2GC_2433_1391})

V_13 = CTVertex(name = 'V_13',
                type = 'R2',
                particles = [ P.s__tilde__, P.t, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS10, L.FFS14, L.FFS3, L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,2,2):C.R2GC_1556_372,(0,0,2):C.R2GC_1551_367,(0,1,2):C.R2GC_909_3043,(0,3,0):C.R2GC_1396_182,(0,4,1):C.R2GC_1882_688})

V_14 = CTVertex(name = 'V_14',
                type = 'R2',
                particles = [ P.c__tilde__, P.t, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS14, L.FFS18, L.FFS19, L.FFS22, L.FFS23, L.FFS3, L.FFS30, L.FFS4, L.FFS6, L.FFS7 ],
                loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,8,1):C.R2GC_3326_2646,(0,3,1):C.R2GC_3314_2636,(0,5,2):C.R2GC_3381_2718,(0,1,2):C.R2GC_3032_2475,(0,9,2):C.R2GC_914_3048,(0,4,2):C.R2GC_866_3000,(0,0,2):C.R2GC_911_3045,(0,6,2):C.R2GC_853_2987,(0,7,0):C.R2GC_1384_176,(0,2,0):C.R2GC_1345_143})

V_15 = CTVertex(name = 'V_15',
                type = 'R2',
                particles = [ P.c__tilde__, P.t, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS14, L.FFS18, L.FFS19, L.FFS21, L.FFS3, L.FFS30, L.FFS4, L.FFS5 ],
                loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,4,2):C.R2GC_3430_2760,(0,4,1):C.R2GC_3430_2761,(0,1,2):C.R2GC_3230_2606,(0,1,1):C.R2GC_3230_2607,(0,7,2):C.R2GC_915_3049,(0,3,2):C.R2GC_865_2999,(0,0,2):C.R2GC_910_3044,(0,5,2):C.R2GC_854_2988,(0,6,0):C.R2GC_1383_175,(0,2,0):C.R2GC_1346_144})

V_16 = CTVertex(name = 'V_16',
                type = 'R2',
                particles = [ P.t__tilde__, P.u, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS12, L.FFS14, L.FFS18, L.FFS19, L.FFS22, L.FFS28, L.FFS3, L.FFS30, L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,9,1):C.R2GC_3070_2495,(0,4,1):C.R2GC_3067_2493,(0,6,0):C.R2GC_1659_461,(0,2,0):C.R2GC_1609_438,(0,0,0):C.R2GC_870_3004,(0,5,0):C.R2GC_939_3073,(0,1,2):C.R2GC_969_3099,(0,7,2):C.R2GC_947_3079,(0,8,0):C.R2GC_1603_433,(0,3,0):C.R2GC_1402_187})

V_17 = CTVertex(name = 'V_17',
                type = 'R2',
                particles = [ P.t__tilde__, P.u, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS10, L.FFS14, L.FFS18, L.FFS19, L.FFS26, L.FFS3, L.FFS30, L.FFS4 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,5,0):C.R2GC_3560_2892,(0,5,1):C.R2GC_3560_2893,(0,2,0):C.R2GC_3232_2610,(0,2,1):C.R2GC_3232_2611,(0,0,0):C.R2GC_869_3003,(0,4,0):C.R2GC_940_3074,(0,1,2):C.R2GC_968_3098,(0,6,2):C.R2GC_948_3080,(0,7,0):C.R2GC_1602_432,(0,3,0):C.R2GC_1403_188})

V_18 = CTVertex(name = 'V_18',
                type = 'R2',
                particles = [ P.b__tilde__, P.u, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS11, L.FFS14, L.FFS3, L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,2,1):C.R2GC_2172_1074,(0,0,1):C.R2GC_984_3114,(0,1,3):C.R2GC_967_3097,(0,3,0):C.R2GC_1341_141,(0,4,2):C.R2GC_2413_1367})

V_19 = CTVertex(name = 'V_19',
                type = 'R2',
                particles = [ P.t__tilde__, P.c, P.G0 ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS12, L.FFS14, L.FFS18, L.FFS19, L.FFS22, L.FFS28, L.FFS3, L.FFS30, L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,9,1):C.R2GC_3068_2494,(0,4,1):C.R2GC_3066_2492,(0,6,2):C.R2GC_1634_459,(0,2,2):C.R2GC_1608_437,(0,0,2):C.R2GC_868_3002,(0,5,2):C.R2GC_903_3037,(0,1,0):C.R2GC_827_2971,(0,7,0):C.R2GC_811_2957,(0,8,2):C.R2GC_1503_310,(0,3,2):C.R2GC_1399_184})

V_20 = CTVertex(name = 'V_20',
                type = 'R2',
                particles = [ P.t__tilde__, P.c, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS10, L.FFS14, L.FFS18, L.FFS19, L.FFS26, L.FFS3, L.FFS30, L.FFS4 ],
                loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,5,2):C.R2GC_3357_2686,(0,5,1):C.R2GC_3357_2687,(0,2,2):C.R2GC_3231_2608,(0,2,1):C.R2GC_3231_2609,(0,0,2):C.R2GC_867_3001,(0,4,2):C.R2GC_904_3038,(0,1,0):C.R2GC_826_2970,(0,6,0):C.R2GC_812_2958,(0,7,2):C.R2GC_1502_309,(0,3,2):C.R2GC_1400_185})

V_21 = CTVertex(name = 'V_21',
                type = 'R2',
                particles = [ P.b__tilde__, P.c, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS11, L.FFS14, L.FFS3, L.FFS4, L.FFS6 ],
                loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                couplings = {(0,2,2):C.R2GC_2141_1035,(0,0,2):C.R2GC_983_3113,(0,1,3):C.R2GC_825_2969,(0,3,1):C.R2GC_1340_140,(0,4,0):C.R2GC_1868_674})

V_22 = CTVertex(name = 'V_22',
                type = 'R2',
                particles = [ P.t__tilde__, P.d, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS18, L.FFS19, L.FFS30, L.FFS5, L.FFS6 ],
                loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,0,2):C.R2GC_1447_228,(0,3,2):C.R2GC_1433_214,(0,2,0):C.R2GC_839_2981,(0,1,2):C.R2GC_1401_186,(0,4,1):C.R2GC_2413_1367})

V_23 = CTVertex(name = 'V_23',
                type = 'R2',
                particles = [ P.t__tilde__, P.s, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS18, L.FFS19, L.FFS30, L.FFS5, L.FFS6 ],
                loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,0,2):C.R2GC_1446_227,(0,3,2):C.R2GC_1432_213,(0,2,0):C.R2GC_846_2984,(0,1,2):C.R2GC_1398_183,(0,4,1):C.R2GC_1868_674})

V_24 = CTVertex(name = 'V_24',
                type = 'R2',
                particles = [ P.u__tilde__, P.b, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS18, L.FFS19, L.FFS30, L.FFS6, L.FFS8 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,0,1):C.R2GC_3033_2476,(0,4,1):C.R2GC_1001_2,(0,2,0):C.R2GC_803_2951,(0,1,3):C.R2GC_1684_463,(0,3,2):C.R2GC_2433_1391})

V_25 = CTVertex(name = 'V_25',
                type = 'R2',
                particles = [ P.c__tilde__, P.b, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFS18, L.FFS19, L.FFS30, L.FFS6, L.FFS8 ],
                loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                couplings = {(0,0,2):C.R2GC_3031_2474,(0,4,2):C.R2GC_997_3127,(0,2,1):C.R2GC_802_2950,(0,1,3):C.R2GC_1344_142,(0,3,0):C.R2GC_1882_688})

V_26 = CTVertex(name = 'V_26',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_809_2956})

V_27 = CTVertex(name = 'V_27',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_809_2956})

V_28 = CTVertex(name = 'V_28',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_809_2956})

V_29 = CTVertex(name = 'V_29',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_801_2949})

V_30 = CTVertex(name = 'V_30',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.c, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_801_2949})

V_31 = CTVertex(name = 'V_31',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_801_2949})

V_32 = CTVertex(name = 'V_32',
                type = 'R2',
                particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV5 ],
                loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ] ],
                couplings = {(0,1,0):C.R2GC_1633_458,(0,1,1):C.R2GC_1907_721,(0,0,0):C.R2GC_1014_15})

V_33 = CTVertex(name = 'V_33',
                type = 'R2',
                particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV5 ],
                loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s, P.t] ] ],
                couplings = {(0,1,0):C.R2GC_1907_721,(0,1,1):C.R2GC_1622_449,(0,0,1):C.R2GC_1018_19})

V_34 = CTVertex(name = 'V_34',
                type = 'R2',
                particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV5 ],
                loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_1076_85,(0,0,2):C.R2GC_1076_86,(0,1,1):C.R2GC_1907_721})

V_35 = CTVertex(name = 'V_35',
                type = 'R2',
                particles = [ P.u__tilde__, P.u, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV12, L.FFV22, L.FFV30, L.FFV37, L.FFV38, L.FFV5 ],
                loop_particles = [ [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,6,1):C.R2GC_3036_2478,(0,6,0):C.R2GC_3497_2840,(0,3,1):C.R2GC_3047_2481,(0,2,0):C.R2GC_3311_2635,(0,1,0):C.R2GC_1083_97,(0,0,0):C.R2GC_1089_108,(0,5,0):C.R2GC_1078_89,(0,4,0):C.R2GC_1079_90})

V_36 = CTVertex(name = 'V_36',
                type = 'R2',
                particles = [ P.c__tilde__, P.c, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV12, L.FFV22, L.FFV30, L.FFV37, L.FFV38, L.FFV5 ],
                loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ] ],
                couplings = {(0,6,0):C.R2GC_3036_2478,(0,6,1):C.R2GC_3413_2756,(0,3,0):C.R2GC_3047_2481,(0,2,1):C.R2GC_3308_2630,(0,0,1):C.R2GC_1073_81,(0,1,1):C.R2GC_1074_82,(0,5,1):C.R2GC_1071_79,(0,4,1):C.R2GC_1072_80})

V_37 = CTVertex(name = 'V_37',
                type = 'R2',
                particles = [ P.t__tilde__, P.t, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV12, L.FFV22, L.FFV30, L.FFV37, L.FFV38, L.FFV5 ],
                loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                couplings = {(0,6,1):C.R2GC_3036_2478,(0,6,0):C.R2GC_3615_2924,(0,6,2):C.R2GC_3615_2925,(0,3,1):C.R2GC_3047_2481,(0,2,0):C.R2GC_3671_2944,(0,2,2):C.R2GC_3671_2945,(0,1,0):C.R2GC_1088_106,(0,1,2):C.R2GC_1088_107,(0,0,0):C.R2GC_1084_98,(0,0,2):C.R2GC_1084_99,(0,4,0):C.R2GC_1077_87,(0,4,2):C.R2GC_1077_88,(0,5,0):C.R2GC_1080_91,(0,5,2):C.R2GC_1080_92})

V_38 = CTVertex(name = 'V_38',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_800_2948})

V_39 = CTVertex(name = 'V_39',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_800_2948})

V_40 = CTVertex(name = 'V_40',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_800_2948})

V_41 = CTVertex(name = 'V_41',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_801_2949})

V_42 = CTVertex(name = 'V_42',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_801_2949})

V_43 = CTVertex(name = 'V_43',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.g ],
                color = [ 'T(3,2,1)' ],
                lorentz = [ L.FFV1 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_801_2949})

V_44 = CTVertex(name = 'V_44',
                type = 'R2',
                particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV38, L.FFV5 ],
                loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ] ],
                couplings = {(0,1,0):C.R2GC_3495_2837,(0,1,1):C.R2GC_1907_721,(0,0,0):C.R2GC_1009_10})

V_45 = CTVertex(name = 'V_45',
                type = 'R2',
                particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV38, L.FFV5 ],
                loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s, P.t] ] ],
                couplings = {(0,1,0):C.R2GC_1907_721,(0,1,1):C.R2GC_3411_2754,(0,0,1):C.R2GC_1015_16})

V_46 = CTVertex(name = 'V_46',
                type = 'R2',
                particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV38, L.FFV5 ],
                loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                couplings = {(0,0,0):C.R2GC_1075_83,(0,0,2):C.R2GC_1075_84,(0,1,1):C.R2GC_1907_721})

V_47 = CTVertex(name = 'V_47',
                type = 'R2',
                particles = [ P.d__tilde__, P.d, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV23, L.FFV5 ],
                loop_particles = [ [ [P.d, P.g] ] ],
                couplings = {(0,1,0):C.R2GC_805_2953,(0,0,0):C.R2GC_3047_2481})

V_48 = CTVertex(name = 'V_48',
                type = 'R2',
                particles = [ P.b__tilde__, P.d, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV5 ],
                loop_particles = [ [ [P.b, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2070_948})

V_49 = CTVertex(name = 'V_49',
                type = 'R2',
                particles = [ P.s__tilde__, P.s, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV23, L.FFV5 ],
                loop_particles = [ [ [P.g, P.s] ] ],
                couplings = {(0,1,0):C.R2GC_805_2953,(0,0,0):C.R2GC_3047_2481})

V_50 = CTVertex(name = 'V_50',
                type = 'R2',
                particles = [ P.b__tilde__, P.s, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV5 ],
                loop_particles = [ [ [P.b, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_2092_974})

V_51 = CTVertex(name = 'V_51',
                type = 'R2',
                particles = [ P.d__tilde__, P.b, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV5 ],
                loop_particles = [ [ [P.b, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2075_953})

V_52 = CTVertex(name = 'V_52',
                type = 'R2',
                particles = [ P.s__tilde__, P.b, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV5 ],
                loop_particles = [ [ [P.b, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_2099_983})

V_53 = CTVertex(name = 'V_53',
                type = 'R2',
                particles = [ P.b__tilde__, P.b, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV23, L.FFV5 ],
                loop_particles = [ [ [P.b, P.g] ] ],
                couplings = {(0,1,0):C.R2GC_805_2953,(0,0,0):C.R2GC_3047_2481})

V_54 = CTVertex(name = 'V_54',
                type = 'R2',
                particles = [ P.t__tilde__, P.u, P.a, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS13, L.FFVS21, L.FFVS22, L.FFVS23, L.FFVS29, L.FFVS30, L.FFVS37, L.FFVS44, L.FFVS45, L.FFVS46, L.FFVS5, L.FFVS6 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,11,0):C.R2GC_1438_219,(0,5,0):C.R2GC_1604_434,(0,0,0):C.R2GC_937_3071,(0,3,2):C.R2GC_971_3101,(0,6,0):C.R2GC_862_2996,(0,9,2):C.R2GC_949_3081,(0,1,1):C.R2GC_2849_2063,(0,7,1):C.R2GC_2552_1577,(0,10,1):C.R2GC_2851_2067,(0,4,1):C.R2GC_2554_1579,(0,2,0):C.R2GC_938_3072,(0,8,0):C.R2GC_863_2997})

V_55 = CTVertex(name = 'V_55',
                type = 'R2',
                particles = [ P.t__tilde__, P.u, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV10, L.FFV12, L.FFV17, L.FFV18, L.FFV21, L.FFV22, L.FFV35, L.FFV38, L.FFV4, L.FFV44, L.FFV45, L.FFV5 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,11,0):C.R2GC_3602_2915,(0,5,0):C.R2GC_3642_2929,(0,0,0):C.R2GC_943_3077,(0,1,2):C.R2GC_974_3104,(0,6,0):C.R2GC_894_3028,(0,7,2):C.R2GC_958_3088,(0,2,1):C.R2GC_2856_2074,(0,9,1):C.R2GC_2579_1622,(0,8,1):C.R2GC_2857_2075,(0,4,1):C.R2GC_2580_1623,(0,3,0):C.R2GC_944_3078,(0,10,0):C.R2GC_895_3029})

V_56 = CTVertex(name = 'V_56',
                type = 'R2',
                particles = [ P.t__tilde__, P.u, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS16, L.FFVS21, L.FFVS23, L.FFVS29, L.FFVS30, L.FFVS37, L.FFVS44, L.FFVS45, L.FFVS46, L.FFVS5, L.FFVS6 ],
                loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,10,2):C.R2GC_1462_241,(0,10,0):C.R2GC_2905_2176,(0,10,3):[ C.R2GC_2905_2177, C.R2GC_3206_2602 ],(0,10,1):C.R2GC_2905_2178,(0,4,2):C.R2GC_1607_436,(0,4,0):C.R2GC_2920_2221,(0,4,3):[ C.R2GC_2920_2222, C.R2GC_3203_2599 ],(0,4,1):C.R2GC_2920_2223,(0,0,2):C.R2GC_1057_58,(0,2,4):C.R2GC_973_3103,(0,5,2):C.R2GC_885_3019,(0,8,4):C.R2GC_1060_61,(0,1,3):C.R2GC_2853_2069,(0,6,3):C.R2GC_2571_1610,(0,9,3):C.R2GC_2855_2073,(0,3,3):C.R2GC_2573_1614,(0,7,2):C.R2GC_886_3020})

V_57 = CTVertex(name = 'V_57',
                type = 'R2',
                particles = [ P.t__tilde__, P.u, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV12, L.FFV17, L.FFV21, L.FFV22, L.FFV35, L.FFV38, L.FFV4, L.FFV44, L.FFV45, L.FFV5 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,10,0):C.R2GC_3605_2917,(0,10,1):C.R2GC_3295_2617,(0,4,0):C.R2GC_3653_2931,(0,4,1):C.R2GC_3293_2615,(0,0,0):C.R2GC_1059_60,(0,1,2):C.R2GC_975_3105,(0,5,0):C.R2GC_899_3033,(0,6,2):C.R2GC_1061_62,(0,2,1):C.R2GC_2859_2078,(0,8,1):C.R2GC_2591_1648,(0,7,1):C.R2GC_2860_2079,(0,3,1):C.R2GC_2592_1649,(0,9,0):C.R2GC_900_3034})

V_58 = CTVertex(name = 'V_58',
                type = 'R2',
                particles = [ P.t__tilde__, P.u, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS30, L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t, P.u] ] ],
                couplings = {(0,3,2):C.R2GC_2060_936,(0,0,0):C.R2GC_2176_1078,(0,0,2):C.R2GC_3172_2590,(0,2,1):C.R2GC_1022_23,(0,1,2):C.R2GC_2569_1608})

V_59 = CTVertex(name = 'V_59',
                type = 'R2',
                particles = [ P.t__tilde__, P.u, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS15, L.FFVS17, L.FFVS30, L.FFVS6, L.FFVS8 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                couplings = {(0,3,0):C.R2GC_993_3123,(0,3,1):C.R2GC_2059_935,(0,2,0):C.R2GC_2175_1077,(0,2,1):C.R2GC_3173_2591,(0,0,0):C.R2GC_1006_7,(0,1,1):C.R2GC_2852_2068,(0,4,0):C.R2GC_1007_8})

V_60 = CTVertex(name = 'V_60',
                type = 'R2',
                particles = [ P.u__tilde__, P.t, P.a, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS13, L.FFVS21, L.FFVS22, L.FFVS23, L.FFVS29, L.FFVS30, L.FFVS37, L.FFVS44, L.FFVS45, L.FFVS46, L.FFVS5, L.FFVS6 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,11,0):C.R2GC_1594_424,(0,5,0):C.R2GC_1439_220,(0,3,0):C.R2GC_927_3061,(0,0,2):C.R2GC_963_3093,(0,9,0):C.R2GC_864_2998,(0,6,2):C.R2GC_951_3083,(0,1,1):C.R2GC_2776_1947,(0,7,1):C.R2GC_2556_1583,(0,10,1):C.R2GC_2778_1951,(0,4,1):C.R2GC_2558_1585,(0,2,2):C.R2GC_964_3094,(0,8,2):C.R2GC_952_3084})

V_61 = CTVertex(name = 'V_61',
                type = 'R2',
                particles = [ P.u__tilde__, P.t, P.a ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV10, L.FFV12, L.FFV17, L.FFV18, L.FFV21, L.FFV22, L.FFV35, L.FFV38, L.FFV4, L.FFV44, L.FFV45, L.FFV5 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,11,0):C.R2GC_3473_2818,(0,5,0):C.R2GC_3149_2565,(0,1,0):C.R2GC_935_3069,(0,0,2):C.R2GC_965_3095,(0,7,0):C.R2GC_896_3030,(0,6,2):C.R2GC_959_3089,(0,2,1):C.R2GC_2783_1958,(0,9,1):C.R2GC_2581_1624,(0,8,1):C.R2GC_2784_1959,(0,4,1):C.R2GC_2582_1625,(0,3,2):C.R2GC_966_3096,(0,10,2):C.R2GC_960_3090})

V_62 = CTVertex(name = 'V_62',
                type = 'R2',
                particles = [ P.u__tilde__, P.t, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS16, L.FFVS21, L.FFVS23, L.FFVS29, L.FFVS30, L.FFVS37, L.FFVS44, L.FFVS45, L.FFVS46, L.FFVS5, L.FFVS6 ],
                loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,10,2):C.R2GC_1601_431,(0,10,0):C.R2GC_2916_2209,(0,10,3):[ C.R2GC_2916_2210, C.R2GC_3338_2653 ],(0,10,1):C.R2GC_2916_2211,(0,4,2):C.R2GC_1463_242,(0,4,0):C.R2GC_2909_2188,(0,4,3):[ C.R2GC_2909_2189, C.R2GC_3324_2644 ],(0,4,1):C.R2GC_2909_2190,(0,2,2):C.R2GC_933_3067,(0,0,4):C.R2GC_1062_63,(0,8,2):C.R2GC_1049_50,(0,5,4):C.R2GC_956_3086,(0,1,3):C.R2GC_2780_1953,(0,6,3):C.R2GC_2575_1616,(0,9,3):C.R2GC_2782_1957,(0,3,3):C.R2GC_2577_1620,(0,7,4):C.R2GC_957_3087})

V_63 = CTVertex(name = 'V_63',
                type = 'R2',
                particles = [ P.u__tilde__, P.t, P.Z ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV12, L.FFV17, L.FFV21, L.FFV22, L.FFV35, L.FFV38, L.FFV4, L.FFV44, L.FFV45, L.FFV5 ],
                loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,10,0):C.R2GC_3477_2822,(0,10,1):C.R2GC_3341_2655,(0,4,0):C.R2GC_3229_2605,(0,4,1):C.R2GC_3325_2645,(0,1,0):C.R2GC_936_3070,(0,0,2):C.R2GC_1063_64,(0,6,0):C.R2GC_1053_54,(0,5,2):C.R2GC_961_3091,(0,2,1):C.R2GC_2786_1962,(0,8,1):C.R2GC_2593_1650,(0,7,1):C.R2GC_2787_1963,(0,3,1):C.R2GC_2594_1651,(0,9,2):C.R2GC_962_3092})

V_64 = CTVertex(name = 'V_64',
                type = 'R2',
                particles = [ P.u__tilde__, P.t, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS15, L.FFVS17, L.FFVS30, L.FFVS6, L.FFVS8 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t, P.u] ] ],
                couplings = {(0,3,2):C.R2GC_2071_949,(0,2,0):C.R2GC_2120_1008,(0,2,2):C.R2GC_3323_2643,(0,0,1):C.R2GC_1025_26,(0,1,2):C.R2GC_2779_1952,(0,4,1):C.R2GC_1026_27})

V_65 = CTVertex(name = 'V_65',
                type = 'R2',
                particles = [ P.u__tilde__, P.t, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS30, L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                couplings = {(0,3,0):C.R2GC_1003_4,(0,3,1):C.R2GC_2072_950,(0,0,0):C.R2GC_2121_1009,(0,0,1):C.R2GC_3322_2642,(0,2,0):C.R2GC_987_3117,(0,1,1):C.R2GC_2570_1609})

V_66 = CTVertex(name = 'V_66',
                type = 'R2',
                particles = [ P.b__tilde__, P.d, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2060_936,(0,0,1):C.R2GC_992_3122})

V_67 = CTVertex(name = 'V_67',
                type = 'R2',
                particles = [ P.b__tilde__, P.s, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g, P.s] ], [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2082_962,(0,0,1):C.R2GC_989_3119})

V_68 = CTVertex(name = 'V_68',
                type = 'R2',
                particles = [ P.d__tilde__, P.b, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2072_950})

V_69 = CTVertex(name = 'V_69',
                type = 'R2',
                particles = [ P.s__tilde__, P.b, P.W__minus__, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_2094_976})

V_70 = CTVertex(name = 'V_70',
                type = 'R2',
                particles = [ P.b__tilde__, P.d, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2059_935})

V_71 = CTVertex(name = 'V_71',
                type = 'R2',
                particles = [ P.b__tilde__, P.s, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_2081_961})

V_72 = CTVertex(name = 'V_72',
                type = 'R2',
                particles = [ P.d__tilde__, P.b, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2071_949,(0,0,1):C.R2GC_1004_5})

V_73 = CTVertex(name = 'V_73',
                type = 'R2',
                particles = [ P.s__tilde__, P.b, P.W__plus__, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g, P.s] ], [ [P.b, P.g, P.t] ] ],
                couplings = {(0,0,0):C.R2GC_2093_975,(0,0,1):C.R2GC_1000_1})

V_74 = CTVertex(name = 'V_74',
                type = 'R2',
                particles = [ P.b__tilde__, P.d, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2061_937})

V_75 = CTVertex(name = 'V_75',
                type = 'R2',
                particles = [ P.b__tilde__, P.s, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_2083_963})

V_76 = CTVertex(name = 'V_76',
                type = 'R2',
                particles = [ P.d__tilde__, P.b, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.d, P.g] ] ],
                couplings = {(0,0,0):C.R2GC_2074_952})

V_77 = CTVertex(name = 'V_77',
                type = 'R2',
                particles = [ P.s__tilde__, P.b, P.Z, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g, P.s] ] ],
                couplings = {(0,0,0):C.R2GC_2096_978})

V_78 = CTVertex(name = 'V_78',
                type = 'R2',
                particles = [ P.b__tilde__, P.u, P.a, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS16, L.FFVS17, L.FFVS23, L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,3,1):C.R2GC_2105_991,(0,3,2):C.R2GC_2414_1368,(0,1,2):C.R2GC_2481_1470,(0,0,0):C.R2GC_808_2955,(0,2,3):C.R2GC_970_3100})

V_79 = CTVertex(name = 'V_79',
                type = 'R2',
                particles = [ P.b__tilde__, P.u, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS16, L.FFVS17, L.FFVS6 ],
                loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                couplings = {(0,2,1):C.R2GC_991_3121,(0,2,3):C.R2GC_2418_1374,(0,2,0):C.R2GC_1094_117,(0,2,2):C.R2GC_1094_118,(0,1,3):C.R2GC_2482_1471,(0,0,1):C.R2GC_1005_6})

V_80 = CTVertex(name = 'V_80',
                type = 'R2',
                particles = [ P.b__tilde__, P.u, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV13, L.FFV5 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                couplings = {(0,2,0):C.R2GC_1613_442,(0,2,1):C.R2GC_2432_1390,(0,1,1):C.R2GC_2490_1482,(0,0,0):C.R2GC_1008_9})

V_81 = CTVertex(name = 'V_81',
                type = 'R2',
                particles = [ P.b__tilde__, P.u, P.Z, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS16, L.FFVS17, L.FFVS23, L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,3,1):C.R2GC_2123_1011,(0,3,2):C.R2GC_2423_1381,(0,1,2):C.R2GC_2486_1476,(0,0,0):C.R2GC_1039_40,(0,2,3):C.R2GC_972_3102})

V_82 = CTVertex(name = 'V_82',
                type = 'R2',
                particles = [ P.t__tilde__, P.d, P.a, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS39, L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,3,2):C.R2GC_1437_218,(0,3,1):C.R2GC_3073_2497,(0,1,1):C.R2GC_2180_1082,(0,2,0):C.R2GC_840_2982,(0,0,2):C.R2GC_861_2995})

V_83 = CTVertex(name = 'V_83',
                type = 'R2',
                particles = [ P.t__tilde__, P.s, P.a, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS39, L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,3,2):C.R2GC_1435_216,(0,3,1):C.R2GC_3072_2496,(0,1,1):C.R2GC_2298_1214,(0,2,0):C.R2GC_847_2985,(0,0,2):C.R2GC_858_2992})

V_84 = CTVertex(name = 'V_84',
                type = 'R2',
                particles = [ P.t__tilde__, P.d, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.c, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ], [ [P.d, P.g, P.u] ] ],
                couplings = {(0,0,1):C.R2GC_2186_1092,(0,1,3):C.R2GC_1021_22,(0,2,1):C.R2GC_2418_1374,(0,2,0):C.R2GC_1097_123,(0,2,2):C.R2GC_1097_124})

V_85 = CTVertex(name = 'V_85',
                type = 'R2',
                particles = [ P.t__tilde__, P.s, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.s, P.t, P.u] ] ],
                couplings = {(0,0,2):C.R2GC_2305_1226,(0,1,0):C.R2GC_976_3106,(0,2,2):C.R2GC_1870_676,(0,2,1):C.R2GC_1099_127,(0,2,3):C.R2GC_1099_128})

V_86 = CTVertex(name = 'V_86',
                type = 'R2',
                particles = [ P.t__tilde__, P.d, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV38, L.FFV39, L.FFV5 ],
                loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ] ],
                couplings = {(0,1,0):C.R2GC_2188_1094,(0,0,1):C.R2GC_1023_24,(0,2,0):C.R2GC_2432_1390})

V_87 = CTVertex(name = 'V_87',
                type = 'R2',
                particles = [ P.t__tilde__, P.s, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV38, L.FFV39, L.FFV5 ],
                loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s, P.t] ] ],
                couplings = {(0,1,1):C.R2GC_2308_1230,(0,0,0):C.R2GC_978_3108,(0,2,1):C.R2GC_1881_687})

V_88 = CTVertex(name = 'V_88',
                type = 'R2',
                particles = [ P.t__tilde__, P.d, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS39, L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,3,2):C.R2GC_1461_240,(0,3,1):C.R2GC_3207_2603,(0,1,1):C.R2GC_2187_1093,(0,2,0):C.R2GC_1044_45,(0,0,2):C.R2GC_884_3018})

V_89 = CTVertex(name = 'V_89',
                type = 'R2',
                particles = [ P.t__tilde__, P.s, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS39, L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,3,2):C.R2GC_1459_238,(0,3,1):C.R2GC_3205_2601,(0,1,1):C.R2GC_2306_1227,(0,2,0):C.R2GC_1046_47,(0,0,2):C.R2GC_881_3015})

V_90 = CTVertex(name = 'V_90',
                type = 'R2',
                particles = [ P.u__tilde__, P.b, P.a, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS39, L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,3,1):C.R2GC_2169_1071,(0,3,2):C.R2GC_2434_1392,(0,1,2):C.R2GC_2402_1350,(0,2,0):C.R2GC_804_2952,(0,0,3):C.R2GC_950_3082})

V_91 = CTVertex(name = 'V_91',
                type = 'R2',
                particles = [ P.u__tilde__, P.b, P.W__plus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                couplings = {(0,2,1):C.R2GC_1002_3,(0,2,3):C.R2GC_2435_1393,(0,2,0):C.R2GC_1092_113,(0,2,2):C.R2GC_1092_114,(0,0,3):C.R2GC_2411_1365,(0,1,1):C.R2GC_986_3116})

V_92 = CTVertex(name = 'V_92',
                type = 'R2',
                particles = [ P.u__tilde__, P.b, P.W__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV38, L.FFV39, L.FFV5 ],
                loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                couplings = {(0,2,0):C.R2GC_3476_2821,(0,2,1):C.R2GC_2445_1411,(0,1,1):C.R2GC_2419_1375,(0,0,0):C.R2GC_994_3124})

V_93 = CTVertex(name = 'V_93',
                type = 'R2',
                particles = [ P.u__tilde__, P.b, P.Z, P.G__plus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS39, L.FFVS40, L.FFVS46, L.FFVS6 ],
                loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                couplings = {(0,3,1):C.R2GC_2171_1073,(0,3,2):C.R2GC_2436_1394,(0,1,2):C.R2GC_2412_1366,(0,2,0):C.R2GC_1038_39,(0,0,3):C.R2GC_955_3085})

V_94 = CTVertex(name = 'V_94',
                type = 'R2',
                particles = [ P.d__tilde__, P.t, P.a, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS16, L.FFVS17, L.FFVS23, L.FFVS6 ],
                loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,3,2):C.R2GC_1593_423,(0,3,1):C.R2GC_3336_2652,(0,1,1):C.R2GC_2240_1148,(0,0,0):C.R2GC_843_2983,(0,2,2):C.R2GC_926_3060})

V_95 = CTVertex(name = 'V_95',
                type = 'R2',
                particles = [ P.s__tilde__, P.t, P.a, P.G__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS16, L.FFVS17, L.FFVS23, L.FFVS6 ],
                loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                couplings = {(0,3,2):C.R2GC_1552_368,(0,3,1):C.R2GC_3328_2647,(0,1,1):C.R2GC_2374_1316,(0,0,0):C.R2GC_850_2986,(0,2,2):C.R2GC_912_3046})

V_96 = CTVertex(name = 'V_96',
                type = 'R2',
                particles = [ P.d__tilde__, P.t, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS16, L.FFVS17, L.FFVS6 ],
                loop_particles = [ [ [P.c, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ], [ [P.d, P.g, P.u] ] ],
                couplings = {(0,1,1):C.R2GC_2241_1149,(0,0,3):C.R2GC_1024_25,(0,2,1):C.R2GC_2435_1393,(0,2,0):C.R2GC_1098_125,(0,2,2):C.R2GC_1098_126})

V_97 = CTVertex(name = 'V_97',
                type = 'R2',
                particles = [ P.s__tilde__, P.t, P.W__minus__, P.H ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFVS16, L.FFVS17, L.FFVS6 ],
                loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.s, P.t, P.u] ] ],
                couplings = {(0,1,2):C.R2GC_2375_1317,(0,0,0):C.R2GC_979_3109,(0,2,2):C.R2GC_1884_690,(0,2,1):C.R2GC_1100_129,(0,2,3):C.R2GC_1100_130})

V_98 = CTVertex(name = 'V_98',
                type = 'R2',
                particles = [ P.d__tilde__, P.t, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV13, L.FFV5 ],
                loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ] ],
                couplings = {(0,1,0):C.R2GC_2243_1151,(0,0,1):C.R2GC_1027_28,(0,2,0):C.R2GC_2445_1411})

V_99 = CTVertex(name = 'V_99',
                type = 'R2',
                particles = [ P.s__tilde__, P.t, P.W__minus__ ],
                color = [ 'Identity(1,2)' ],
                lorentz = [ L.FFV11, L.FFV13, L.FFV5 ],
                loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s, P.t] ] ],
                couplings = {(0,1,1):C.R2GC_2379_1323,(0,0,0):C.R2GC_982_3112,(0,2,1):C.R2GC_1886_692})

V_100 = CTVertex(name = 'V_100',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.t, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS16, L.FFVS17, L.FFVS23, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,2):C.R2GC_1600_430,(0,3,1):C.R2GC_3339_2654,(0,1,1):C.R2GC_2242_1150,(0,0,0):C.R2GC_1045_46,(0,2,2):C.R2GC_932_3066})

V_101 = CTVertex(name = 'V_101',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.t, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS16, L.FFVS17, L.FFVS23, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,2):C.R2GC_1561_377,(0,3,1):C.R2GC_3331_2649,(0,1,1):C.R2GC_2377_1320,(0,0,0):C.R2GC_1047_48,(0,2,2):C.R2GC_918_3052})

V_102 = CTVertex(name = 'V_102',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS13, L.FFVS21, L.FFVS22, L.FFVS23, L.FFVS29, L.FFVS30, L.FFVS37, L.FFVS44, L.FFVS45, L.FFVS46, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,11,2):C.R2GC_1436_217,(0,5,2):C.R2GC_1510_315,(0,3,0):C.R2GC_829_2973,(0,0,2):C.R2GC_901_3035,(0,9,0):C.R2GC_815_2961,(0,6,2):C.R2GC_859_2993,(0,1,1):C.R2GC_1995_851,(0,7,1):C.R2GC_1923_741,(0,10,1):C.R2GC_1997_855,(0,4,1):C.R2GC_1925_743,(0,2,2):C.R2GC_902_3036,(0,8,2):C.R2GC_860_2994})

V_103 = CTVertex(name = 'V_103',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV10, L.FFV12, L.FFV17, L.FFV18, L.FFV21, L.FFV22, L.FFV35, L.FFV38, L.FFV4, L.FFV44, L.FFV45, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,11,2):C.R2GC_3593_2907,(0,5,2):C.R2GC_3598_2911,(0,1,0):C.R2GC_831_2975,(0,0,2):C.R2GC_907_3041,(0,7,0):C.R2GC_822_2966,(0,6,2):C.R2GC_892_3026,(0,2,1):C.R2GC_2002_862,(0,9,1):C.R2GC_1948_782,(0,8,1):C.R2GC_2003_863,(0,4,1):C.R2GC_1949_783,(0,3,2):C.R2GC_908_3042,(0,10,2):C.R2GC_893_3027})

V_104 = CTVertex(name = 'V_104',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS16, L.FFVS21, L.FFVS23, L.FFVS29, L.FFVS30, L.FFVS37, L.FFVS44, L.FFVS45, L.FFVS46, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,10,3):C.R2GC_1460_239,(0,10,1):[ C.R2GC_3204_2600, C.R2GC_2917_2212 ],(0,10,4):C.R2GC_2917_2213,(0,10,2):C.R2GC_2917_2214,(0,4,3):C.R2GC_1515_319,(0,4,1):[ C.R2GC_3202_2598, C.R2GC_2908_2185 ],(0,4,4):C.R2GC_2908_2186,(0,4,2):C.R2GC_2908_2187,(0,2,0):C.R2GC_830_2974,(0,0,3):C.R2GC_1054_55,(0,8,0):C.R2GC_1040_41,(0,5,3):C.R2GC_882_3016,(0,1,1):C.R2GC_1999_857,(0,6,1):C.R2GC_1942_774,(0,9,1):C.R2GC_2001_861,(0,3,1):C.R2GC_1944_778,(0,7,3):C.R2GC_883_3017})

V_105 = CTVertex(name = 'V_105',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV11, L.FFV12, L.FFV17, L.FFV21, L.FFV22, L.FFV35, L.FFV38, L.FFV4, L.FFV44, L.FFV45, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,10,2):C.R2GC_3596_2909,(0,10,1):C.R2GC_3294_2616,(0,4,2):C.R2GC_3600_2913,(0,4,1):C.R2GC_3292_2614,(0,1,0):C.R2GC_832_2976,(0,0,2):C.R2GC_1056_57,(0,6,0):C.R2GC_1041_42,(0,5,2):C.R2GC_897_3031,(0,2,1):C.R2GC_2004_864,(0,8,1):C.R2GC_1959_806,(0,7,1):C.R2GC_2005_865,(0,3,1):C.R2GC_1960_807,(0,9,2):C.R2GC_898_3032})

V_106 = CTVertex(name = 'V_106',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS40, L.FFVS46, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,3,2):C.R2GC_2082_962,(0,0,2):C.R2GC_3168_2588,(0,0,0):C.R2GC_2151_1047,(0,2,1):C.R2GC_977_3107,(0,1,2):C.R2GC_1937_767})

V_107 = CTVertex(name = 'V_107',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS15, L.FFVS17, L.FFVS30, L.FFVS6, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,3,1):C.R2GC_2081_961,(0,3,0):C.R2GC_990_3120,(0,2,1):C.R2GC_3169_2589,(0,2,0):C.R2GC_2150_1046,(0,0,0):C.R2GC_995_3125,(0,1,1):C.R2GC_1998_856,(0,4,0):C.R2GC_996_3126})

V_108 = CTVertex(name = 'V_108',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS13, L.FFVS21, L.FFVS22, L.FFVS23, L.FFVS29, L.FFVS30, L.FFVS37, L.FFVS44, L.FFVS45, L.FFVS46, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,11,2):C.R2GC_1553_369,(0,5,2):C.R2GC_1434_215,(0,0,0):C.R2GC_833_2977,(0,3,2):C.R2GC_913_3047,(0,6,0):C.R2GC_813_2959,(0,9,2):C.R2GC_857_2991,(0,1,1):C.R2GC_2039_909,(0,7,1):C.R2GC_1919_735,(0,10,1):C.R2GC_2041_913,(0,4,1):C.R2GC_1921_737,(0,2,0):C.R2GC_834_2978,(0,8,0):C.R2GC_814_2960})

V_109 = CTVertex(name = 'V_109',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV10, L.FFV12, L.FFV17, L.FFV18, L.FFV21, L.FFV22, L.FFV35, L.FFV38, L.FFV4, L.FFV44, L.FFV45, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,11,2):C.R2GC_3390_2736,(0,5,2):C.R2GC_3148_2564,(0,0,0):C.R2GC_835_2979,(0,1,2):C.R2GC_921_3055,(0,6,0):C.R2GC_820_2964,(0,7,2):C.R2GC_891_3025,(0,2,1):C.R2GC_2046_920,(0,9,1):C.R2GC_1946_780,(0,8,1):C.R2GC_2047_921,(0,4,1):C.R2GC_1947_781,(0,3,0):C.R2GC_836_2980,(0,10,0):C.R2GC_821_2965})

V_110 = CTVertex(name = 'V_110',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS16, L.FFVS21, L.FFVS23, L.FFVS29, L.FFVS30, L.FFVS37, L.FFVS44, L.FFVS45, L.FFVS46, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,10,3):C.R2GC_1562_378,(0,10,1):[ C.R2GC_3330_2648, C.R2GC_2912_2197 ],(0,10,4):C.R2GC_2912_2198,(0,10,2):C.R2GC_2912_2199,(0,4,3):C.R2GC_1458_237,(0,4,1):[ C.R2GC_3317_2639, C.R2GC_2913_2200 ],(0,4,4):C.R2GC_2913_2201,(0,4,2):C.R2GC_2913_2202,(0,0,0):C.R2GC_1042_43,(0,2,3):C.R2GC_919_3053,(0,5,0):C.R2GC_818_2962,(0,8,3):C.R2GC_1048_49,(0,1,1):C.R2GC_2043_915,(0,6,1):C.R2GC_1938_768,(0,9,1):C.R2GC_2045_919,(0,3,1):C.R2GC_1940_772,(0,7,0):C.R2GC_819_2963})

V_111 = CTVertex(name = 'V_111',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV11, L.FFV12, L.FFV17, L.FFV21, L.FFV22, L.FFV35, L.FFV38, L.FFV4, L.FFV44, L.FFV45, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,10,2):C.R2GC_3394_2740,(0,10,1):C.R2GC_3333_2650,(0,4,2):C.R2GC_3228_2604,(0,4,1):C.R2GC_3318_2640,(0,0,0):C.R2GC_1043_44,(0,1,2):C.R2GC_922_3056,(0,5,0):C.R2GC_823_2967,(0,6,2):C.R2GC_1052_53,(0,2,1):C.R2GC_2049_924,(0,8,1):C.R2GC_1957_804,(0,7,1):C.R2GC_2050_925,(0,3,1):C.R2GC_1958_805,(0,9,0):C.R2GC_824_2968})

V_112 = CTVertex(name = 'V_112',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS15, L.FFVS17, L.FFVS30, L.FFVS6, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,3,2):C.R2GC_2093_975,(0,2,2):C.R2GC_3316_2638,(0,2,0):C.R2GC_2118_1006,(0,0,1):C.R2GC_980_3110,(0,1,2):C.R2GC_2042_914,(0,4,1):C.R2GC_981_3111})

V_113 = CTVertex(name = 'V_113',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS40, L.FFVS46, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,3,1):C.R2GC_2094_976,(0,3,0):C.R2GC_999_3129,(0,0,1):C.R2GC_3315_2637,(0,0,0):C.R2GC_2119_1007,(0,2,0):C.R2GC_985_3115,(0,1,1):C.R2GC_1936_766})

V_114 = CTVertex(name = 'V_114',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.c, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS16, L.FFVS17, L.FFVS23, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,3,0):C.R2GC_1869_675,(0,3,2):C.R2GC_2104_990,(0,1,0):C.R2GC_1896_707,(0,0,1):C.R2GC_807_2954,(0,2,3):C.R2GC_828_2972})

V_115 = CTVertex(name = 'V_115',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.c, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2146_1041,(0,2,2):C.R2GC_2147_1043,(0,7,0):C.R2GC_2146_1041,(0,7,2):C.R2GC_2146_1042,(0,5,0):C.R2GC_1870_676,(0,5,2):C.R2GC_988_3118,(0,5,1):C.R2GC_1093_115,(0,5,3):C.R2GC_1093_116,(0,0,0):C.R2GC_1897_708,(0,3,0):C.R2GC_1897_708,(0,4,0):C.R2GC_1897_708,(0,1,2):C.R2GC_2148_1044,(0,6,2):C.R2GC_2149_1045})

V_116 = CTVertex(name = 'V_116',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.c, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV2, L.FFV3, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,6,0):C.R2GC_2156_1055,(0,6,1):C.R2GC_2157_1057,(0,4,0):C.R2GC_2156_1055,(0,4,1):C.R2GC_2156_1056,(0,2,0):C.R2GC_1881_687,(0,2,1):C.R2GC_1612_441,(0,0,0):C.R2GC_1902_716,(0,1,0):C.R2GC_1902_716,(0,5,1):C.R2GC_2158_1058,(0,3,1):C.R2GC_2159_1059})

V_117 = CTVertex(name = 'V_117',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.c, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,2,1):C.R2GC_1901_714,(0,2,3):C.R2GC_1901_715,(0,2,0):C.R2GC_1900_713,(0,8,1):C.R2GC_1900_711,(0,8,3):C.R2GC_1900_712,(0,8,0):C.R2GC_1900_713,(0,6,0):C.R2GC_1872_678,(0,6,2):C.R2GC_2122_1010,(0,0,0):C.R2GC_1899_710,(0,4,0):C.R2GC_1899_710,(0,5,0):C.R2GC_1899_710,(0,1,1):C.R2GC_1378_172,(0,1,3):C.R2GC_1376_170,(0,7,1):C.R2GC_1377_171,(0,7,3):C.R2GC_1375_169,(0,3,3):C.R2GC_1376_170,(0,9,3):C.R2GC_1375_169})

V_118 = CTVertex(name = 'V_118',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.b, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,7,1):C.R2GC_1856_653,(0,7,3):C.R2GC_1856_654,(0,7,0):C.R2GC_1855_652,(0,4,1):C.R2GC_1855_650,(0,4,3):C.R2GC_1855_651,(0,4,0):C.R2GC_1855_652,(0,9,0):C.R2GC_1883_689,(0,9,2):C.R2GC_2166_1068,(0,0,0):C.R2GC_1857_655,(0,1,0):C.R2GC_1857_655,(0,2,0):C.R2GC_1857_655,(0,6,1):C.R2GC_1331_132,(0,6,3):C.R2GC_1347_145,(0,3,1):C.R2GC_1332_133,(0,3,3):C.R2GC_1348_146,(0,8,1):C.R2GC_1331_132,(0,5,1):C.R2GC_1332_133})

V_119 = CTVertex(name = 'V_119',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.b, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ] ],
                 couplings = {(0,7,0):C.R2GC_2108_994,(0,7,2):C.R2GC_2109_996,(0,4,0):C.R2GC_2108_994,(0,4,2):C.R2GC_2108_995,(0,9,0):C.R2GC_1884_690,(0,9,2):C.R2GC_998_3128,(0,9,1):C.R2GC_1091_111,(0,9,3):C.R2GC_1091_112,(0,0,0):C.R2GC_1864_667,(0,1,0):C.R2GC_1864_667,(0,2,0):C.R2GC_1864_667,(0,6,2):C.R2GC_2111_998,(0,3,2):C.R2GC_2110_997,(0,8,2):C.R2GC_2111_998,(0,5,2):C.R2GC_2110_997})

V_120 = CTVertex(name = 'V_120',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.b, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV19, L.FFV20, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,5,0):C.R2GC_2128_1019,(0,5,1):C.R2GC_2129_1021,(0,3,0):C.R2GC_2128_1019,(0,3,1):C.R2GC_2128_1020,(0,6,0):C.R2GC_1886_692,(0,6,1):C.R2GC_3393_2739,(0,0,0):C.R2GC_1871_677,(0,1,0):C.R2GC_1871_677,(0,4,1):C.R2GC_2131_1023,(0,2,1):C.R2GC_2130_1022})

V_121 = CTVertex(name = 'V_121',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.b, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,7,1):C.R2GC_1865_668,(0,7,3):C.R2GC_1865_669,(0,7,0):C.R2GC_1865_670,(0,4,1):C.R2GC_1866_671,(0,4,3):C.R2GC_1866_672,(0,4,0):C.R2GC_1865_670,(0,9,0):C.R2GC_1885_691,(0,9,2):C.R2GC_2168_1070,(0,0,0):C.R2GC_1867_673,(0,1,0):C.R2GC_1867_673,(0,2,0):C.R2GC_1867_673,(0,6,1):C.R2GC_1337_138,(0,6,3):C.R2GC_1361_156,(0,3,1):C.R2GC_1338_139,(0,3,3):C.R2GC_1360_155,(0,8,1):C.R2GC_1337_138,(0,5,1):C.R2GC_1338_139})

V_122 = CTVertex(name = 'V_122',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_3563_2897,(0,2,2):C.R2GC_3563_2898,(0,2,1):C.R2GC_2850_2066,(0,13,0):C.R2GC_2998_2429,(0,13,2):C.R2GC_2998_2430,(0,13,1):C.R2GC_2551_1576,(0,18,0):C.R2GC_2850_2064,(0,18,2):C.R2GC_2850_2065,(0,18,1):C.R2GC_2850_2066,(0,10,0):C.R2GC_2551_1574,(0,10,2):C.R2GC_2551_1575,(0,10,1):C.R2GC_2551_1576,(0,16,0):C.R2GC_875_3009,(0,8,0):C.R2GC_941_3075,(0,1,0):C.R2GC_1789_596,(0,1,2):C.R2GC_1788_595,(0,12,0):C.R2GC_1687_466,(0,12,2):C.R2GC_1688_467,(0,17,0):C.R2GC_1788_595,(0,17,2):C.R2GC_1789_596,(0,9,0):C.R2GC_1688_467,(0,9,2):C.R2GC_1687_466,(0,3,2):C.R2GC_1788_595,(0,14,2):C.R2GC_1688_467,(0,19,2):C.R2GC_1789_596,(0,11,2):C.R2GC_1687_466,(0,0,1):C.R2GC_3564_2899,(0,4,1):C.R2GC_3564_2899,(0,15,1):C.R2GC_3564_2899,(0,5,1):C.R2GC_2553_1578,(0,6,1):C.R2GC_2553_1578,(0,7,1):C.R2GC_2553_1578})

V_123 = CTVertex(name = 'V_123',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_3567_2900,(0,2,2):C.R2GC_3567_2901,(0,2,1):C.R2GC_2854_2072,(0,13,0):C.R2GC_3052_2484,(0,13,2):C.R2GC_3052_2485,(0,13,1):C.R2GC_2572_1613,(0,18,0):C.R2GC_2854_2070,(0,18,2):C.R2GC_2854_2071,(0,18,1):C.R2GC_2854_2072,(0,10,0):C.R2GC_2572_1611,(0,10,2):C.R2GC_2572_1612,(0,10,1):C.R2GC_2572_1613,(0,16,0):C.R2GC_888_3022,(0,8,0):C.R2GC_1058_59,(0,1,0):C.R2GC_1807_612,(0,1,2):C.R2GC_1805_610,(0,12,0):C.R2GC_1713_484,(0,12,2):C.R2GC_1711_482,(0,17,0):C.R2GC_1806_611,(0,17,2):C.R2GC_1804_609,(0,9,0):C.R2GC_1714_485,(0,9,2):C.R2GC_1712_483,(0,3,2):C.R2GC_1805_610,(0,14,2):C.R2GC_1711_482,(0,19,2):C.R2GC_1804_609,(0,11,2):C.R2GC_1712_483,(0,0,1):C.R2GC_3568_2902,(0,4,1):C.R2GC_3568_2902,(0,15,1):C.R2GC_3568_2902,(0,5,1):C.R2GC_2574_1615,(0,6,1):C.R2GC_2574_1615,(0,7,1):C.R2GC_2574_1615})

V_124 = CTVertex(name = 'V_124',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_3572_2905,(0,2,6):C.R2GC_2858_2077,(0,2,3):C.R2GC_2399_1344,(0,2,5):C.R2GC_2399_1345,(0,1,1):C.R2GC_2858_2076,(0,1,6):C.R2GC_2858_2077,(0,1,3):C.R2GC_2401_1348,(0,1,5):C.R2GC_2401_1349,(0,5,2):C.R2GC_2885_2128,(0,5,0):C.R2GC_2885_2129,(0,5,4):C.R2GC_3159_2578,(0,5,6):C.R2GC_2589_1645,(0,4,2):C.R2GC_2887_2132,(0,4,0):C.R2GC_2887_2133,(0,4,4):C.R2GC_2589_1644,(0,4,6):C.R2GC_2589_1645,(0,0,1):C.R2GC_3571_2903,(0,0,6):C.R2GC_3571_2904,(0,0,3):C.R2GC_2400_1346,(0,0,5):C.R2GC_2400_1347,(0,3,2):C.R2GC_2886_2130,(0,3,0):C.R2GC_2886_2131,(0,3,4):C.R2GC_3158_2576,(0,3,6):C.R2GC_3158_2577})

V_125 = CTVertex(name = 'V_125',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.d, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,7,0):C.R2GC_2495_1488,(0,7,1):C.R2GC_3044_2480,(0,4,0):C.R2GC_2495_1488,(0,4,1):C.R2GC_2495_1489,(0,0,0):C.R2GC_2185_1091,(0,1,0):C.R2GC_2185_1091,(0,2,0):C.R2GC_2185_1091,(0,6,1):C.R2GC_2497_1491,(0,3,1):C.R2GC_2496_1490,(0,8,1):C.R2GC_2497_1491,(0,5,1):C.R2GC_2496_1490})

V_126 = CTVertex(name = 'V_126',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.s, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,7,0):C.R2GC_3040_2479,(0,7,1):C.R2GC_2304_1225,(0,4,0):C.R2GC_2304_1224,(0,4,1):C.R2GC_2304_1225,(0,0,1):C.R2GC_2303_1223,(0,1,1):C.R2GC_2303_1223,(0,2,1):C.R2GC_2303_1223,(0,6,0):C.R2GC_1909_723,(0,3,0):C.R2GC_1908_722,(0,8,0):C.R2GC_1909_723,(0,5,0):C.R2GC_1908_722})

V_127 = CTVertex(name = 'V_127',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.d, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2499_1494,(0,2,1):C.R2GC_3180_2595,(0,1,0):C.R2GC_2499_1494,(0,1,1):C.R2GC_2499_1495,(0,0,0):C.R2GC_3181_2596,(0,0,1):C.R2GC_3181_2597})

V_128 = CTVertex(name = 'V_128',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.s, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3178_2592,(0,2,1):C.R2GC_2309_1232,(0,1,0):C.R2GC_2309_1231,(0,1,1):C.R2GC_2309_1232,(0,0,0):C.R2GC_3179_2593,(0,0,1):C.R2GC_3179_2594})

V_129 = CTVertex(name = 'V_129',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.d, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.d, P.g] ], [ [P.b, P.d, P.g, P.u] ], [ [P.c, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,2,3):C.R2GC_2498_1492,(0,2,5):C.R2GC_3165_2587,(0,2,0):C.R2GC_2955_2342,(0,2,1):C.R2GC_2955_2343,(0,2,2):C.R2GC_2955_2344,(0,2,4):C.R2GC_2955_2345,(0,1,3):C.R2GC_2498_1492,(0,1,5):C.R2GC_2498_1493,(0,1,0):C.R2GC_2956_2346,(0,1,1):C.R2GC_2956_2347,(0,1,2):C.R2GC_2956_2348,(0,1,4):C.R2GC_2956_2349,(0,0,3):C.R2GC_3164_2585,(0,0,5):C.R2GC_3164_2586,(0,0,0):C.R2GC_2957_2350,(0,0,1):C.R2GC_2957_2351,(0,0,2):C.R2GC_2957_2352,(0,0,4):C.R2GC_2957_2353})

V_130 = CTVertex(name = 'V_130',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.s, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.s] ], [ [P.b, P.g, P.s, P.u] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.s, P.t, P.u] ] ],
                 couplings = {(0,2,2):C.R2GC_3163_2584,(0,2,4):C.R2GC_2307_1229,(0,2,0):C.R2GC_2976_2396,(0,2,1):C.R2GC_2976_2397,(0,2,3):C.R2GC_2976_2398,(0,2,5):C.R2GC_2976_2399,(0,1,2):C.R2GC_2307_1228,(0,1,4):C.R2GC_2307_1229,(0,1,0):C.R2GC_2977_2400,(0,1,1):C.R2GC_2977_2401,(0,1,3):C.R2GC_2977_2402,(0,1,5):C.R2GC_2977_2403,(0,0,2):C.R2GC_3162_2582,(0,0,4):C.R2GC_3162_2583,(0,0,0):C.R2GC_2978_2404,(0,0,1):C.R2GC_2978_2405,(0,0,3):C.R2GC_2978_2406,(0,0,5):C.R2GC_2978_2407})

V_131 = CTVertex(name = 'V_131',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_3538_2867,(0,2,2):C.R2GC_3538_2868,(0,2,1):C.R2GC_2788_1966,(0,13,0):C.R2GC_3023_2462,(0,13,2):C.R2GC_3023_2463,(0,13,1):C.R2GC_2560_1591,(0,18,0):C.R2GC_2788_1964,(0,18,2):C.R2GC_2788_1965,(0,18,1):C.R2GC_2788_1966,(0,10,0):C.R2GC_2560_1589,(0,10,2):C.R2GC_2560_1590,(0,10,1):C.R2GC_2560_1591,(0,16,0):C.R2GC_879_3013,(0,0,0):C.R2GC_3535_2862,(0,0,2):C.R2GC_3535_2863,(0,0,1):C.R2GC_3535_2864,(0,4,0):C.R2GC_3536_2865,(0,4,2):C.R2GC_3537_2866,(0,4,1):C.R2GC_3535_2864,(0,15,0):C.R2GC_3536_2865,(0,15,2):C.R2GC_3535_2863,(0,15,1):C.R2GC_3535_2864,(0,8,0):C.R2GC_942_3076,(0,5,0):C.R2GC_2563_1596,(0,5,2):C.R2GC_2562_1595,(0,5,1):C.R2GC_2561_1594,(0,6,0):C.R2GC_2561_1592,(0,6,2):C.R2GC_2561_1593,(0,6,1):C.R2GC_2561_1594,(0,7,0):C.R2GC_2561_1592,(0,7,2):C.R2GC_2562_1595,(0,7,1):C.R2GC_2561_1594,(0,1,0):C.R2GC_1797_602,(0,1,2):C.R2GC_1796_601,(0,12,0):C.R2GC_1701_475,(0,12,2):C.R2GC_1700_474,(0,17,0):C.R2GC_1803_608,(0,17,2):C.R2GC_1802_607,(0,9,0):C.R2GC_1695_469,(0,9,2):C.R2GC_1694_468,(0,3,2):C.R2GC_1796_601,(0,14,2):C.R2GC_1700_474,(0,19,2):C.R2GC_1802_607,(0,11,2):C.R2GC_1694_468})

V_132 = CTVertex(name = 'V_132',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_3539_2869,(0,2,2):C.R2GC_3539_2870,(0,2,1):C.R2GC_2789_1969,(0,13,0):C.R2GC_3019_2455,(0,13,2):C.R2GC_3019_2456,(0,13,1):C.R2GC_2559_1588,(0,18,0):C.R2GC_2789_1967,(0,18,2):C.R2GC_2789_1968,(0,18,1):C.R2GC_2789_1969,(0,10,0):C.R2GC_2559_1586,(0,10,2):C.R2GC_2559_1587,(0,10,1):C.R2GC_2559_1588,(0,16,0):C.R2GC_1444_225,(0,0,0):C.R2GC_3540_2871,(0,0,2):C.R2GC_3540_2872,(0,0,1):C.R2GC_3540_2873,(0,4,0):C.R2GC_3541_2874,(0,4,2):C.R2GC_3542_2875,(0,4,1):C.R2GC_3540_2873,(0,15,0):C.R2GC_3541_2874,(0,15,2):C.R2GC_3540_2872,(0,15,1):C.R2GC_3540_2873,(0,8,0):C.R2GC_1605_435,(0,5,0):C.R2GC_3020_2457,(0,5,2):C.R2GC_3020_2458,(0,5,1):C.R2GC_3020_2459,(0,6,0):C.R2GC_3021_2460,(0,6,2):C.R2GC_3022_2461,(0,6,1):C.R2GC_3020_2459,(0,7,0):C.R2GC_3021_2460,(0,7,2):C.R2GC_3020_2458,(0,7,1):C.R2GC_3020_2459,(0,1,0):C.R2GC_1801_606,(0,1,2):C.R2GC_1800_605,(0,12,0):C.R2GC_1699_473,(0,12,2):C.R2GC_1698_472,(0,17,0):C.R2GC_1799_604,(0,17,2):C.R2GC_1798_603,(0,9,0):C.R2GC_1697_471,(0,9,2):C.R2GC_1696_470,(0,3,2):C.R2GC_1800_605,(0,14,2):C.R2GC_1698_472,(0,19,2):C.R2GC_1798_603,(0,11,2):C.R2GC_1696_470})

V_133 = CTVertex(name = 'V_133',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV22, L.FFV3, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,13,0):C.R2GC_3543_2876,(0,13,2):C.R2GC_3543_2877,(0,13,1):C.R2GC_2790_1972,(0,8,0):C.R2GC_3084_2512,(0,8,2):C.R2GC_3084_2513,(0,8,1):C.R2GC_2583_1628,(0,11,0):C.R2GC_2790_1970,(0,11,2):C.R2GC_2790_1971,(0,11,1):C.R2GC_2790_1972,(0,6,0):C.R2GC_2583_1626,(0,6,2):C.R2GC_2583_1627,(0,6,1):C.R2GC_2583_1628,(0,9,0):C.R2GC_3603_2916,(0,1,0):C.R2GC_3544_2878,(0,1,2):C.R2GC_3544_2879,(0,1,1):C.R2GC_3544_2880,(0,4,0):C.R2GC_3545_2881,(0,4,2):C.R2GC_3545_2882,(0,4,1):C.R2GC_3544_2880,(0,3,0):C.R2GC_3643_2930,(0,0,0):C.R2GC_3085_2514,(0,0,2):C.R2GC_3085_2515,(0,0,1):C.R2GC_3085_2516,(0,2,0):C.R2GC_3086_2517,(0,2,2):C.R2GC_3086_2518,(0,2,1):C.R2GC_3085_2516,(0,12,0):C.R2GC_1813_620,(0,12,2):C.R2GC_1813_621,(0,7,0):C.R2GC_1728_500,(0,7,2):C.R2GC_1728_501,(0,10,0):C.R2GC_1812_618,(0,10,2):C.R2GC_1812_619,(0,5,0):C.R2GC_1727_498,(0,5,2):C.R2GC_1727_499})

V_134 = CTVertex(name = 'V_134',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_1817_628,(0,2,3):C.R2GC_1817_629,(2,2,0):C.R2GC_3548_2888,(2,2,1):C.R2GC_2791_1973,(2,2,3):C.R2GC_3548_2889,(2,2,2):C.R2GC_2792_1978,(1,2,0):C.R2GC_3546_2883,(1,2,1):C.R2GC_2792_1976,(1,2,3):C.R2GC_3546_2884,(1,2,2):C.R2GC_2791_1975,(0,1,1):C.R2GC_1817_628,(0,1,3):C.R2GC_1817_629,(2,1,1):C.R2GC_2792_1976,(2,1,3):C.R2GC_2792_1977,(2,1,2):C.R2GC_2792_1978,(1,1,1):C.R2GC_2791_1973,(1,1,3):C.R2GC_2791_1974,(1,1,2):C.R2GC_2791_1975,(2,0,0):C.R2GC_3546_2883,(2,0,1):C.R2GC_3547_2885,(2,0,3):C.R2GC_3547_2886,(2,0,2):C.R2GC_3547_2887,(1,0,0):C.R2GC_3548_2888,(1,0,1):C.R2GC_3547_2885,(1,0,3):C.R2GC_3549_2890,(1,0,2):C.R2GC_3549_2891,(0,0,1):C.R2GC_1818_630,(0,0,3):C.R2GC_1818_631,(0,5,1):C.R2GC_1737_518,(0,5,3):C.R2GC_1737_519,(2,5,0):C.R2GC_3100_2549,(2,5,1):C.R2GC_2585_1632,(2,5,3):C.R2GC_3100_2550,(2,5,2):C.R2GC_2586_1637,(1,5,0):C.R2GC_3098_2544,(1,5,1):C.R2GC_2586_1635,(1,5,3):C.R2GC_3098_2545,(1,5,2):C.R2GC_2585_1634,(0,4,1):C.R2GC_1737_518,(0,4,3):C.R2GC_1737_519,(2,4,1):C.R2GC_2586_1635,(2,4,3):C.R2GC_2586_1636,(2,4,2):C.R2GC_2586_1637,(1,4,1):C.R2GC_2585_1632,(1,4,3):C.R2GC_2585_1633,(1,4,2):C.R2GC_2585_1634,(2,3,0):C.R2GC_3098_2544,(2,3,1):C.R2GC_3099_2546,(2,3,3):C.R2GC_3099_2547,(2,3,2):C.R2GC_3099_2548,(1,3,0):C.R2GC_3100_2549,(1,3,1):C.R2GC_3099_2546,(1,3,3):C.R2GC_3101_2551,(1,3,2):C.R2GC_3101_2552,(0,3,1):C.R2GC_1738_520,(0,3,3):C.R2GC_1738_521})

V_135 = CTVertex(name = 'V_135',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.d, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,7,0):C.R2GC_3017_2453,(0,7,2):C.R2GC_3017_2454,(0,7,1):C.R2GC_2181_1085,(0,4,0):C.R2GC_2181_1083,(0,4,2):C.R2GC_2181_1084,(0,4,1):C.R2GC_2181_1085,(0,9,2):C.R2GC_1443_224,(0,0,0):C.R2GC_2183_1089,(0,0,2):C.R2GC_2184_1090,(0,0,1):C.R2GC_2182_1088,(0,1,0):C.R2GC_2182_1086,(0,1,2):C.R2GC_2182_1087,(0,1,1):C.R2GC_2182_1088,(0,2,0):C.R2GC_2183_1089,(0,2,2):C.R2GC_2182_1087,(0,2,1):C.R2GC_2182_1088,(0,6,0):C.R2GC_1388_178,(0,6,2):C.R2GC_1426_207,(0,3,0):C.R2GC_1387_177,(0,3,2):C.R2GC_1427_208,(0,8,0):C.R2GC_1388_178,(0,5,0):C.R2GC_1387_177})

V_136 = CTVertex(name = 'V_136',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.s, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,7,0):C.R2GC_3009_2442,(0,7,2):C.R2GC_3009_2443,(0,7,1):C.R2GC_2299_1217,(0,4,0):C.R2GC_2299_1215,(0,4,2):C.R2GC_2299_1216,(0,4,1):C.R2GC_2299_1217,(0,9,2):C.R2GC_1441_222,(0,0,0):C.R2GC_2301_1221,(0,0,2):C.R2GC_2302_1222,(0,0,1):C.R2GC_2300_1220,(0,1,0):C.R2GC_2300_1218,(0,1,2):C.R2GC_2300_1219,(0,1,1):C.R2GC_2300_1220,(0,2,0):C.R2GC_2301_1221,(0,2,2):C.R2GC_2300_1219,(0,2,1):C.R2GC_2300_1220,(0,6,0):C.R2GC_1394_181,(0,6,2):C.R2GC_1420_201,(0,3,0):C.R2GC_1393_180,(0,3,2):C.R2GC_1421_202,(0,8,0):C.R2GC_1394_181,(0,5,0):C.R2GC_1393_180})

V_137 = CTVertex(name = 'V_137',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3358_2688,(0,2,2):C.R2GC_3358_2689,(0,2,1):C.R2GC_1996_854,(0,13,0):C.R2GC_2994_2427,(0,13,2):C.R2GC_2994_2428,(0,13,1):C.R2GC_1922_740,(0,18,0):C.R2GC_1996_852,(0,18,2):C.R2GC_1996_853,(0,18,1):C.R2GC_1996_854,(0,10,0):C.R2GC_1922_738,(0,10,2):C.R2GC_1922_739,(0,10,1):C.R2GC_1922_740,(0,16,2):C.R2GC_874_3008,(0,8,2):C.R2GC_905_3039,(0,1,0):C.R2GC_1365_159,(0,1,2):C.R2GC_1366_160,(0,12,0):C.R2GC_1351_148,(0,12,2):C.R2GC_1350_147,(0,17,0):C.R2GC_1366_160,(0,17,2):C.R2GC_1365_159,(0,9,0):C.R2GC_1350_147,(0,9,2):C.R2GC_1351_148,(0,3,0):C.R2GC_1365_159,(0,14,0):C.R2GC_1351_148,(0,19,0):C.R2GC_1366_160,(0,11,0):C.R2GC_1350_147,(0,0,1):C.R2GC_3359_2690,(0,4,1):C.R2GC_3359_2690,(0,15,1):C.R2GC_3359_2690,(0,5,1):C.R2GC_1924_742,(0,6,1):C.R2GC_1924_742,(0,7,1):C.R2GC_1924_742})

V_138 = CTVertex(name = 'V_138',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3362_2691,(0,2,2):C.R2GC_3362_2692,(0,2,1):C.R2GC_2000_860,(0,13,0):C.R2GC_3056_2486,(0,13,2):C.R2GC_3056_2487,(0,13,1):C.R2GC_1943_777,(0,18,0):C.R2GC_2000_858,(0,18,2):C.R2GC_2000_859,(0,18,1):C.R2GC_2000_860,(0,10,0):C.R2GC_1943_775,(0,10,2):C.R2GC_1943_776,(0,10,1):C.R2GC_1943_777,(0,16,2):C.R2GC_887_3021,(0,8,2):C.R2GC_1055_56,(0,1,0):C.R2GC_1380_174,(0,1,2):C.R2GC_1513_317,(0,12,0):C.R2GC_1362_157,(0,12,2):C.R2GC_1454_234,(0,17,0):C.R2GC_1379_173,(0,17,2):C.R2GC_1514_318,(0,9,0):C.R2GC_1363_158,(0,9,2):C.R2GC_1453_233,(0,3,0):C.R2GC_1380_174,(0,14,0):C.R2GC_1362_157,(0,19,0):C.R2GC_1379_173,(0,11,0):C.R2GC_1363_158,(0,0,1):C.R2GC_3363_2693,(0,4,1):C.R2GC_3363_2693,(0,15,1):C.R2GC_3363_2693,(0,5,1):C.R2GC_1945_779,(0,6,1):C.R2GC_1945_779,(0,7,1):C.R2GC_1945_779})

V_139 = CTVertex(name = 'V_139',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,2):C.R2GC_3367_2696,(0,2,4):C.R2GC_2153_1051,(0,2,5):C.R2GC_2366_1299,(0,2,6):C.R2GC_2366_1300,(0,1,2):C.R2GC_2153_1050,(0,1,4):C.R2GC_2153_1051,(0,1,5):C.R2GC_2368_1303,(0,1,6):C.R2GC_2368_1304,(0,5,0):C.R2GC_2888_2134,(0,5,1):C.R2GC_2888_2135,(0,5,3):C.R2GC_3157_2575,(0,5,4):C.R2GC_1956_803,(0,4,0):C.R2GC_2890_2138,(0,4,1):C.R2GC_2890_2139,(0,4,3):C.R2GC_1956_802,(0,4,4):C.R2GC_1956_803,(0,0,2):C.R2GC_3366_2694,(0,0,4):C.R2GC_3366_2695,(0,0,5):C.R2GC_2367_1301,(0,0,6):C.R2GC_2367_1302,(0,3,0):C.R2GC_2889_2136,(0,3,1):C.R2GC_2889_2137,(0,3,3):C.R2GC_3156_2573,(0,3,4):C.R2GC_3156_2574})

V_140 = CTVertex(name = 'V_140',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3345_2661,(0,2,2):C.R2GC_3345_2662,(0,2,1):C.R2GC_1985_833,(0,13,0):C.R2GC_3015_2451,(0,13,2):C.R2GC_3015_2452,(0,13,1):C.R2GC_1932_760,(0,18,0):C.R2GC_1985_831,(0,18,2):C.R2GC_1985_832,(0,18,1):C.R2GC_1985_833,(0,10,0):C.R2GC_1932_758,(0,10,2):C.R2GC_1932_759,(0,10,1):C.R2GC_1932_760,(0,16,2):C.R2GC_878_3012,(0,0,0):C.R2GC_3342_2656,(0,0,2):C.R2GC_3342_2657,(0,0,1):C.R2GC_3342_2658,(0,4,0):C.R2GC_3344_2660,(0,4,2):C.R2GC_3343_2659,(0,4,1):C.R2GC_3342_2658,(0,15,0):C.R2GC_3342_2656,(0,15,2):C.R2GC_3343_2659,(0,15,1):C.R2GC_3342_2658,(0,8,2):C.R2GC_906_3040,(0,5,0):C.R2GC_1934_764,(0,5,2):C.R2GC_1935_765,(0,5,1):C.R2GC_1933_763,(0,6,0):C.R2GC_1933_761,(0,6,2):C.R2GC_1933_762,(0,6,1):C.R2GC_1933_763,(0,7,0):C.R2GC_1934_764,(0,7,2):C.R2GC_1933_762,(0,7,1):C.R2GC_1933_763,(0,1,0):C.R2GC_1371_165,(0,1,2):C.R2GC_1509_314,(0,12,0):C.R2GC_1358_154,(0,12,2):C.R2GC_1422_203,(0,17,0):C.R2GC_1374_168,(0,17,2):C.R2GC_1506_311,(0,9,0):C.R2GC_1355_151,(0,9,2):C.R2GC_1425_206,(0,3,0):C.R2GC_1371_165,(0,14,0):C.R2GC_1358_154,(0,19,0):C.R2GC_1374_168,(0,11,0):C.R2GC_1355_151})

V_141 = CTVertex(name = 'V_141',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3346_2663,(0,2,2):C.R2GC_3346_2664,(0,2,1):C.R2GC_1986_836,(0,13,0):C.R2GC_3011_2444,(0,13,2):C.R2GC_3011_2445,(0,13,1):C.R2GC_1931_757,(0,18,0):C.R2GC_1986_834,(0,18,2):C.R2GC_1986_835,(0,18,1):C.R2GC_1986_836,(0,10,0):C.R2GC_1931_755,(0,10,2):C.R2GC_1931_756,(0,10,1):C.R2GC_1931_757,(0,16,2):C.R2GC_1442_223,(0,0,0):C.R2GC_3347_2665,(0,0,2):C.R2GC_3347_2666,(0,0,1):C.R2GC_3347_2667,(0,4,0):C.R2GC_3349_2669,(0,4,2):C.R2GC_3348_2668,(0,4,1):C.R2GC_3347_2667,(0,15,0):C.R2GC_3347_2665,(0,15,2):C.R2GC_3348_2668,(0,15,1):C.R2GC_3347_2667,(0,8,2):C.R2GC_1511_316,(0,5,0):C.R2GC_3012_2446,(0,5,2):C.R2GC_3012_2447,(0,5,1):C.R2GC_3012_2448,(0,6,0):C.R2GC_3014_2450,(0,6,2):C.R2GC_3013_2449,(0,6,1):C.R2GC_3012_2448,(0,7,0):C.R2GC_3012_2446,(0,7,2):C.R2GC_3013_2449,(0,7,1):C.R2GC_3012_2448,(0,1,0):C.R2GC_1373_167,(0,1,2):C.R2GC_1508_313,(0,12,0):C.R2GC_1357_153,(0,12,2):C.R2GC_1424_205,(0,17,0):C.R2GC_1372_166,(0,17,2):C.R2GC_1507_312,(0,9,0):C.R2GC_1356_152,(0,9,2):C.R2GC_1423_204,(0,3,0):C.R2GC_1373_167,(0,14,0):C.R2GC_1357_153,(0,19,0):C.R2GC_1372_166,(0,11,0):C.R2GC_1356_152})

V_142 = CTVertex(name = 'V_142',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV22, L.FFV3, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,13,0):C.R2GC_3350_2670,(0,13,2):C.R2GC_3350_2671,(0,13,1):C.R2GC_1987_839,(0,8,0):C.R2GC_3081_2505,(0,8,2):C.R2GC_3081_2506,(0,8,1):C.R2GC_1951_789,(0,11,0):C.R2GC_1987_837,(0,11,2):C.R2GC_1987_838,(0,11,1):C.R2GC_1987_839,(0,6,0):C.R2GC_1951_787,(0,6,2):C.R2GC_1951_788,(0,6,1):C.R2GC_1951_789,(0,9,2):C.R2GC_3594_2908,(0,1,0):C.R2GC_3351_2672,(0,1,2):C.R2GC_3351_2673,(0,1,1):C.R2GC_3351_2674,(0,4,0):C.R2GC_3352_2675,(0,4,2):C.R2GC_3352_2676,(0,4,1):C.R2GC_3351_2674,(0,3,2):C.R2GC_3599_2912,(0,0,0):C.R2GC_3082_2507,(0,0,2):C.R2GC_3082_2508,(0,0,1):C.R2GC_3082_2509,(0,2,0):C.R2GC_3083_2510,(0,2,2):C.R2GC_3083_2511,(0,2,1):C.R2GC_3082_2509,(0,12,0):C.R2GC_1521_327,(0,12,2):C.R2GC_1521_328,(0,7,0):C.R2GC_1475_259,(0,7,2):C.R2GC_1475_260,(0,10,0):C.R2GC_1520_325,(0,10,2):C.R2GC_1520_326,(0,5,0):C.R2GC_1474_257,(0,5,2):C.R2GC_1474_258})

V_143 = CTVertex(name = 'V_143',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1525_335,(0,2,3):C.R2GC_1525_336,(2,2,2):C.R2GC_3355_2682,(2,2,0):C.R2GC_3355_2683,(2,2,3):C.R2GC_1988_841,(2,2,1):C.R2GC_1989_845,(1,2,2):C.R2GC_3353_2677,(1,2,0):C.R2GC_3353_2678,(1,2,3):C.R2GC_1989_844,(1,2,1):C.R2GC_1988_842,(0,1,0):C.R2GC_1525_335,(0,1,3):C.R2GC_1525_336,(2,1,0):C.R2GC_1989_843,(2,1,3):C.R2GC_1989_844,(2,1,1):C.R2GC_1989_845,(1,1,0):C.R2GC_1988_840,(1,1,3):C.R2GC_1988_841,(1,1,1):C.R2GC_1988_842,(2,0,2):C.R2GC_3353_2677,(2,0,0):C.R2GC_3354_2679,(2,0,3):C.R2GC_3354_2680,(2,0,1):C.R2GC_3354_2681,(1,0,2):C.R2GC_3355_2682,(1,0,0):C.R2GC_3356_2684,(1,0,3):C.R2GC_3354_2680,(1,0,1):C.R2GC_3356_2685,(0,0,0):C.R2GC_1526_337,(0,0,3):C.R2GC_1526_338,(0,5,0):C.R2GC_1484_277,(0,5,3):C.R2GC_1484_278,(2,5,2):C.R2GC_3096_2540,(2,5,0):C.R2GC_3096_2541,(2,5,3):C.R2GC_1954_797,(2,5,1):C.R2GC_1955_801,(1,5,2):C.R2GC_3094_2535,(1,5,0):C.R2GC_3094_2536,(1,5,3):C.R2GC_1955_800,(1,5,1):C.R2GC_1954_798,(0,4,0):C.R2GC_1484_277,(0,4,3):C.R2GC_1484_278,(2,4,0):C.R2GC_1955_799,(2,4,3):C.R2GC_1955_800,(2,4,1):C.R2GC_1955_801,(1,4,0):C.R2GC_1954_796,(1,4,3):C.R2GC_1954_797,(1,4,1):C.R2GC_1954_798,(2,3,2):C.R2GC_3094_2535,(2,3,0):C.R2GC_3095_2537,(2,3,3):C.R2GC_3095_2538,(2,3,1):C.R2GC_3095_2539,(1,3,2):C.R2GC_3096_2540,(1,3,0):C.R2GC_3097_2542,(1,3,3):C.R2GC_3095_2538,(1,3,1):C.R2GC_3097_2543,(0,3,0):C.R2GC_1485_279,(0,3,3):C.R2GC_1485_280})

V_144 = CTVertex(name = 'V_144',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_3515_2845,(0,2,2):C.R2GC_3515_2846,(0,2,1):C.R2GC_2777_1950,(0,13,0):C.R2GC_3001_2431,(0,13,2):C.R2GC_3001_2432,(0,13,1):C.R2GC_2555_1582,(0,18,0):C.R2GC_2777_1948,(0,18,2):C.R2GC_2777_1949,(0,18,1):C.R2GC_2777_1950,(0,10,0):C.R2GC_2555_1580,(0,10,2):C.R2GC_2555_1581,(0,10,1):C.R2GC_2555_1582,(0,16,0):C.R2GC_930_3064,(0,8,0):C.R2GC_876_3010,(0,1,0):C.R2GC_1582_412,(0,1,2):C.R2GC_1583_413,(0,12,0):C.R2GC_1411_192,(0,12,2):C.R2GC_1410_191,(0,17,0):C.R2GC_1583_413,(0,17,2):C.R2GC_1582_412,(0,9,0):C.R2GC_1410_191,(0,9,2):C.R2GC_1411_192,(0,3,0):C.R2GC_1582_412,(0,14,0):C.R2GC_1411_192,(0,19,0):C.R2GC_1583_413,(0,11,0):C.R2GC_1410_191,(0,0,1):C.R2GC_3516_2847,(0,4,1):C.R2GC_3516_2847,(0,15,1):C.R2GC_3516_2847,(0,5,1):C.R2GC_2557_1584,(0,6,1):C.R2GC_2557_1584,(0,7,1):C.R2GC_2557_1584})

V_145 = CTVertex(name = 'V_145',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_3523_2850,(0,2,2):C.R2GC_3523_2851,(0,2,1):C.R2GC_2781_1956,(0,13,0):C.R2GC_3059_2488,(0,13,2):C.R2GC_3059_2489,(0,13,1):C.R2GC_2576_1619,(0,18,0):C.R2GC_2781_1954,(0,18,2):C.R2GC_2781_1955,(0,18,1):C.R2GC_2781_1956,(0,10,0):C.R2GC_2576_1617,(0,10,2):C.R2GC_2576_1618,(0,10,1):C.R2GC_2576_1619,(0,16,0):C.R2GC_934_3068,(0,8,0):C.R2GC_1051_52,(0,1,0):C.R2GC_1599_429,(0,1,2):C.R2GC_1767_560,(0,12,0):C.R2GC_1455_235,(0,12,2):C.R2GC_1718_487,(0,17,0):C.R2GC_1598_428,(0,17,2):C.R2GC_1768_561,(0,9,0):C.R2GC_1456_236,(0,9,2):C.R2GC_1717_486,(0,3,0):C.R2GC_1599_429,(0,14,0):C.R2GC_1455_235,(0,19,0):C.R2GC_1598_428,(0,11,0):C.R2GC_1456_236,(0,0,1):C.R2GC_3524_2852,(0,4,1):C.R2GC_3524_2852,(0,15,1):C.R2GC_3524_2852,(0,5,1):C.R2GC_2578_1621,(0,6,1):C.R2GC_2578_1621,(0,7,1):C.R2GC_2578_1621})

V_146 = CTVertex(name = 'V_146',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,2):C.R2GC_2891_2140,(0,2,0):C.R2GC_2891_2141,(0,2,4):C.R2GC_3528_2855,(0,2,6):C.R2GC_2785_1961,(0,1,2):C.R2GC_2893_2144,(0,1,0):C.R2GC_2893_2145,(0,1,4):C.R2GC_2785_1960,(0,1,6):C.R2GC_2785_1961,(0,5,1):C.R2GC_3161_2581,(0,5,6):C.R2GC_2590_1647,(0,5,3):C.R2GC_2349_1279,(0,5,5):C.R2GC_2349_1280,(0,4,1):C.R2GC_2590_1646,(0,4,6):C.R2GC_2590_1647,(0,4,3):C.R2GC_2351_1283,(0,4,5):C.R2GC_2351_1284,(0,0,2):C.R2GC_2892_2142,(0,0,0):C.R2GC_2892_2143,(0,0,4):C.R2GC_3527_2853,(0,0,6):C.R2GC_3527_2854,(0,3,1):C.R2GC_3160_2579,(0,3,6):C.R2GC_3160_2580,(0,3,3):C.R2GC_2350_1281,(0,3,5):C.R2GC_2350_1282})

V_147 = CTVertex(name = 'V_147',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.b, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,7,0):C.R2GC_2410_1364,(0,7,1):C.R2GC_2409_1363,(0,4,0):C.R2GC_2409_1362,(0,4,1):C.R2GC_2409_1363,(0,0,1):C.R2GC_2408_1361,(0,1,1):C.R2GC_2408_1361,(0,2,1):C.R2GC_2408_1361,(0,6,0):C.R2GC_2117_1005,(0,3,0):C.R2GC_2116_1004,(0,8,0):C.R2GC_2117_1005,(0,5,0):C.R2GC_2116_1004})

V_148 = CTVertex(name = 'V_148',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.b, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2420_1376,(0,2,1):C.R2GC_2420_1377,(0,1,0):C.R2GC_2421_1378,(0,1,1):C.R2GC_2420_1377,(0,0,0):C.R2GC_2422_1379,(0,0,1):C.R2GC_2422_1380})

V_149 = CTVertex(name = 'V_149',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.b, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.t] ], [ [P.b, P.g, P.s, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,3):C.R2GC_2416_1371,(0,2,5):C.R2GC_2415_1370,(0,2,0):C.R2GC_2928_2258,(0,2,1):C.R2GC_2928_2259,(0,2,2):C.R2GC_2928_2260,(0,2,4):C.R2GC_2928_2261,(0,1,3):C.R2GC_2415_1369,(0,1,5):C.R2GC_2415_1370,(0,1,0):C.R2GC_2929_2262,(0,1,1):C.R2GC_2929_2263,(0,1,2):C.R2GC_2929_2264,(0,1,4):C.R2GC_2929_2265,(0,0,3):C.R2GC_2417_1372,(0,0,5):C.R2GC_2417_1373,(0,0,0):C.R2GC_2930_2266,(0,0,1):C.R2GC_2930_2267,(0,0,2):C.R2GC_2930_2268,(0,0,4):C.R2GC_2930_2269})

V_150 = CTVertex(name = 'V_150',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_3459_2791,(0,2,2):C.R2GC_3459_2792,(0,2,1):C.R2GC_2746_1905,(0,13,0):C.R2GC_3029_2471,(0,13,2):C.R2GC_3029_2472,(0,13,1):C.R2GC_2565_1602,(0,18,0):C.R2GC_2746_1903,(0,18,2):C.R2GC_2746_1904,(0,18,1):C.R2GC_2746_1905,(0,10,0):C.R2GC_2565_1600,(0,10,2):C.R2GC_2565_1601,(0,10,1):C.R2GC_2565_1602,(0,16,0):C.R2GC_931_3065,(0,0,0):C.R2GC_3456_2786,(0,0,2):C.R2GC_3456_2787,(0,0,1):C.R2GC_3456_2788,(0,4,0):C.R2GC_3458_2790,(0,4,2):C.R2GC_3457_2789,(0,4,1):C.R2GC_3456_2788,(0,15,0):C.R2GC_3456_2786,(0,15,2):C.R2GC_3457_2789,(0,15,1):C.R2GC_3456_2788,(0,8,0):C.R2GC_880_3014,(0,5,0):C.R2GC_2567_1606,(0,5,2):C.R2GC_2568_1607,(0,5,1):C.R2GC_2566_1605,(0,6,0):C.R2GC_2566_1603,(0,6,2):C.R2GC_2566_1604,(0,6,1):C.R2GC_2566_1605,(0,7,0):C.R2GC_2567_1606,(0,7,2):C.R2GC_2566_1604,(0,7,1):C.R2GC_2566_1605,(0,1,0):C.R2GC_1588_418,(0,1,2):C.R2GC_1765_559,(0,12,0):C.R2GC_1431_212,(0,12,2):C.R2GC_1705_478,(0,17,0):C.R2GC_1591_421,(0,17,2):C.R2GC_1762_556,(0,9,0):C.R2GC_1428_209,(0,9,2):C.R2GC_1708_481,(0,3,0):C.R2GC_1588_418,(0,14,0):C.R2GC_1431_212,(0,19,0):C.R2GC_1591_421,(0,11,0):C.R2GC_1428_209})

V_151 = CTVertex(name = 'V_151',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_3460_2793,(0,2,2):C.R2GC_3460_2794,(0,2,1):C.R2GC_2747_1908,(0,13,0):C.R2GC_3025_2464,(0,13,2):C.R2GC_3025_2465,(0,13,1):C.R2GC_2564_1599,(0,18,0):C.R2GC_2747_1906,(0,18,2):C.R2GC_2747_1907,(0,18,1):C.R2GC_2747_1908,(0,10,0):C.R2GC_2564_1597,(0,10,2):C.R2GC_2564_1598,(0,10,1):C.R2GC_2564_1599,(0,16,0):C.R2GC_1596_426,(0,0,0):C.R2GC_3461_2795,(0,0,2):C.R2GC_3461_2796,(0,0,1):C.R2GC_3461_2797,(0,4,0):C.R2GC_3463_2799,(0,4,2):C.R2GC_3462_2798,(0,4,1):C.R2GC_3461_2797,(0,15,0):C.R2GC_3461_2795,(0,15,2):C.R2GC_3462_2798,(0,15,1):C.R2GC_3461_2797,(0,8,0):C.R2GC_1445_226,(0,5,0):C.R2GC_3026_2466,(0,5,2):C.R2GC_3026_2467,(0,5,1):C.R2GC_3026_2468,(0,6,0):C.R2GC_3028_2470,(0,6,2):C.R2GC_3027_2469,(0,6,1):C.R2GC_3026_2468,(0,7,0):C.R2GC_3026_2466,(0,7,2):C.R2GC_3027_2469,(0,7,1):C.R2GC_3026_2468,(0,1,0):C.R2GC_1590_420,(0,1,2):C.R2GC_1764_558,(0,12,0):C.R2GC_1430_211,(0,12,2):C.R2GC_1707_480,(0,17,0):C.R2GC_1589_419,(0,17,2):C.R2GC_1763_557,(0,9,0):C.R2GC_1429_210,(0,9,2):C.R2GC_1706_479,(0,3,0):C.R2GC_1590_420,(0,14,0):C.R2GC_1430_211,(0,19,0):C.R2GC_1589_419,(0,11,0):C.R2GC_1429_210})

V_152 = CTVertex(name = 'V_152',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV22, L.FFV3, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,13,0):C.R2GC_3465_2801,(0,13,2):C.R2GC_3465_2802,(0,13,1):C.R2GC_2748_1911,(0,8,0):C.R2GC_3087_2519,(0,8,2):C.R2GC_3087_2520,(0,8,1):C.R2GC_2584_1631,(0,11,0):C.R2GC_2748_1909,(0,11,2):C.R2GC_2748_1910,(0,11,1):C.R2GC_2748_1911,(0,6,0):C.R2GC_2584_1629,(0,6,2):C.R2GC_2584_1630,(0,6,1):C.R2GC_2584_1631,(0,9,0):C.R2GC_3474_2819,(0,1,0):C.R2GC_3466_2803,(0,1,2):C.R2GC_3466_2804,(0,1,1):C.R2GC_3466_2805,(0,4,0):C.R2GC_3467_2806,(0,4,2):C.R2GC_3467_2807,(0,4,1):C.R2GC_3466_2805,(0,3,0):C.R2GC_3151_2567,(0,0,0):C.R2GC_3088_2521,(0,0,2):C.R2GC_3088_2522,(0,0,1):C.R2GC_3088_2523,(0,2,0):C.R2GC_3089_2524,(0,2,2):C.R2GC_3089_2525,(0,2,1):C.R2GC_3088_2523,(0,12,0):C.R2GC_1774_569,(0,12,2):C.R2GC_1774_570,(0,7,0):C.R2GC_1730_504,(0,7,2):C.R2GC_1730_505,(0,10,0):C.R2GC_1773_567,(0,10,2):C.R2GC_1773_568,(0,5,0):C.R2GC_1729_502,(0,5,2):C.R2GC_1729_503})

V_153 = CTVertex(name = 'V_153',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_1778_577,(0,2,3):C.R2GC_1778_578,(2,2,0):C.R2GC_3470_2813,(2,2,1):C.R2GC_3470_2814,(2,2,3):C.R2GC_2749_1913,(2,2,2):C.R2GC_2750_1917,(1,2,0):C.R2GC_3468_2808,(1,2,1):C.R2GC_3468_2809,(1,2,3):C.R2GC_2750_1916,(1,2,2):C.R2GC_2749_1914,(0,1,1):C.R2GC_1778_577,(0,1,3):C.R2GC_1778_578,(2,1,1):C.R2GC_2750_1915,(2,1,3):C.R2GC_2750_1916,(2,1,2):C.R2GC_2750_1917,(1,1,1):C.R2GC_2749_1912,(1,1,3):C.R2GC_2749_1913,(1,1,2):C.R2GC_2749_1914,(2,0,0):C.R2GC_3468_2808,(2,0,1):C.R2GC_3469_2810,(2,0,3):C.R2GC_3469_2811,(2,0,2):C.R2GC_3469_2812,(1,0,0):C.R2GC_3470_2813,(1,0,1):C.R2GC_3471_2815,(1,0,3):C.R2GC_3469_2811,(1,0,2):C.R2GC_3471_2816,(0,0,1):C.R2GC_1779_579,(0,0,3):C.R2GC_1779_580,(0,5,1):C.R2GC_1739_522,(0,5,3):C.R2GC_1739_523,(2,5,0):C.R2GC_3104_2558,(2,5,1):C.R2GC_3104_2559,(2,5,3):C.R2GC_2587_1639,(2,5,2):C.R2GC_2588_1643,(1,5,0):C.R2GC_3102_2553,(1,5,1):C.R2GC_3102_2554,(1,5,3):C.R2GC_2588_1642,(1,5,2):C.R2GC_2587_1640,(0,4,1):C.R2GC_1739_522,(0,4,3):C.R2GC_1739_523,(2,4,1):C.R2GC_2588_1641,(2,4,3):C.R2GC_2588_1642,(2,4,2):C.R2GC_2588_1643,(1,4,1):C.R2GC_2587_1638,(1,4,3):C.R2GC_2587_1639,(1,4,2):C.R2GC_2587_1640,(2,3,0):C.R2GC_3102_2553,(2,3,1):C.R2GC_3103_2555,(2,3,3):C.R2GC_3103_2556,(2,3,2):C.R2GC_3103_2557,(1,3,0):C.R2GC_3104_2558,(1,3,1):C.R2GC_3105_2560,(1,3,3):C.R2GC_3103_2556,(1,3,2):C.R2GC_3105_2561,(0,3,1):C.R2GC_1740_524,(0,3,3):C.R2GC_1740_525})

V_154 = CTVertex(name = 'V_154',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.b, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,7,0):C.R2GC_2403_1351,(0,7,3):C.R2GC_2403_1352,(0,7,2):C.R2GC_2403_1353,(0,4,0):C.R2GC_2404_1354,(0,4,3):C.R2GC_2404_1355,(0,4,2):C.R2GC_2403_1353,(0,9,1):C.R2GC_2170_1072,(0,0,0):C.R2GC_2406_1359,(0,0,3):C.R2GC_2407_1360,(0,0,2):C.R2GC_2405_1358,(0,1,0):C.R2GC_2405_1356,(0,1,3):C.R2GC_2405_1357,(0,1,2):C.R2GC_2405_1358,(0,2,0):C.R2GC_2406_1359,(0,2,3):C.R2GC_2405_1357,(0,2,2):C.R2GC_2405_1358,(0,6,0):C.R2GC_1336_137,(0,6,3):C.R2GC_1702_476,(0,3,0):C.R2GC_1335_136,(0,3,3):C.R2GC_1704_477,(0,8,0):C.R2GC_1336_137,(0,5,0):C.R2GC_1335_136})

V_155 = CTVertex(name = 'V_155',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3432_2762,(0,2,2):C.R2GC_3432_2763,(0,2,1):C.R2GC_2040_912,(0,13,0):C.R2GC_2990_2425,(0,13,2):C.R2GC_2990_2426,(0,13,1):C.R2GC_1918_734,(0,18,0):C.R2GC_2040_910,(0,18,2):C.R2GC_2040_911,(0,18,1):C.R2GC_2040_912,(0,10,0):C.R2GC_1918_732,(0,10,2):C.R2GC_1918_733,(0,10,1):C.R2GC_1918_734,(0,16,2):C.R2GC_916_3050,(0,8,2):C.R2GC_873_3007,(0,1,0):C.R2GC_1536_354,(0,1,2):C.R2GC_1535_353,(0,12,0):C.R2GC_1404_189,(0,12,2):C.R2GC_1405_190,(0,17,0):C.R2GC_1535_353,(0,17,2):C.R2GC_1536_354,(0,9,0):C.R2GC_1405_190,(0,9,2):C.R2GC_1404_189,(0,3,2):C.R2GC_1535_353,(0,14,2):C.R2GC_1405_190,(0,19,2):C.R2GC_1536_354,(0,11,2):C.R2GC_1404_189,(0,0,1):C.R2GC_3433_2764,(0,4,1):C.R2GC_3433_2764,(0,15,1):C.R2GC_3433_2764,(0,5,1):C.R2GC_1920_736,(0,6,1):C.R2GC_1920_736,(0,7,1):C.R2GC_1920_736})

V_156 = CTVertex(name = 'V_156',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3440_2767,(0,2,2):C.R2GC_3440_2768,(0,2,1):C.R2GC_2044_918,(0,13,0):C.R2GC_3048_2482,(0,13,2):C.R2GC_3048_2483,(0,13,1):C.R2GC_1939_771,(0,18,0):C.R2GC_2044_916,(0,18,2):C.R2GC_2044_917,(0,18,1):C.R2GC_2044_918,(0,10,0):C.R2GC_1939_769,(0,10,2):C.R2GC_1939_770,(0,10,1):C.R2GC_1939_771,(0,16,2):C.R2GC_920_3054,(0,8,2):C.R2GC_1050_51,(0,1,0):C.R2GC_1560_376,(0,1,2):C.R2GC_1558_374,(0,12,0):C.R2GC_1450_231,(0,12,2):C.R2GC_1448_229,(0,17,0):C.R2GC_1559_375,(0,17,2):C.R2GC_1557_373,(0,9,0):C.R2GC_1451_232,(0,9,2):C.R2GC_1449_230,(0,3,2):C.R2GC_1558_374,(0,14,2):C.R2GC_1448_229,(0,19,2):C.R2GC_1557_373,(0,11,2):C.R2GC_1449_230,(0,0,1):C.R2GC_3441_2769,(0,4,1):C.R2GC_3441_2769,(0,15,1):C.R2GC_3441_2769,(0,5,1):C.R2GC_1941_773,(0,6,1):C.R2GC_1941_773,(0,7,1):C.R2GC_1941_773})

V_157 = CTVertex(name = 'V_157',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2894_2146,(0,2,1):C.R2GC_2894_2147,(0,2,3):C.R2GC_3445_2772,(0,2,4):C.R2GC_2048_923,(0,1,0):C.R2GC_2896_2150,(0,1,1):C.R2GC_2896_2151,(0,1,3):C.R2GC_2048_922,(0,1,4):C.R2GC_2048_923,(0,5,2):C.R2GC_3155_2572,(0,5,4):C.R2GC_2124_1013,(0,5,5):C.R2GC_2346_1273,(0,5,6):C.R2GC_2346_1274,(0,4,2):C.R2GC_2124_1012,(0,4,4):C.R2GC_2124_1013,(0,4,5):C.R2GC_2348_1277,(0,4,6):C.R2GC_2348_1278,(0,0,0):C.R2GC_2895_2148,(0,0,1):C.R2GC_2895_2149,(0,0,3):C.R2GC_3444_2770,(0,0,4):C.R2GC_3444_2771,(0,3,2):C.R2GC_3154_2570,(0,3,4):C.R2GC_3154_2571,(0,3,5):C.R2GC_2347_1275,(0,3,6):C.R2GC_2347_1276})

V_158 = CTVertex(name = 'V_158',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.b, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,7,0):C.R2GC_2112_999,(0,7,1):C.R2GC_2113_1001,(0,4,0):C.R2GC_2112_999,(0,4,1):C.R2GC_2112_1000,(0,0,0):C.R2GC_1863_666,(0,1,0):C.R2GC_1863_666,(0,2,0):C.R2GC_1863_666,(0,6,1):C.R2GC_2115_1003,(0,3,1):C.R2GC_2114_1002,(0,8,1):C.R2GC_2115_1003,(0,5,1):C.R2GC_2114_1002})

V_159 = CTVertex(name = 'V_159',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.b, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2132_1024,(0,2,1):C.R2GC_2132_1025,(0,1,0):C.R2GC_2132_1024,(0,1,1):C.R2GC_2133_1026,(0,0,0):C.R2GC_2134_1027,(0,0,1):C.R2GC_2134_1028})

V_160 = CTVertex(name = 'V_160',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.b, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.t] ], [ [P.b, P.g, P.s, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2125_1014,(0,2,4):C.R2GC_2126_1016,(0,2,1):C.R2GC_2925_2246,(0,2,2):C.R2GC_2925_2247,(0,2,3):C.R2GC_2925_2248,(0,2,5):C.R2GC_2925_2249,(0,1,0):C.R2GC_2125_1014,(0,1,4):C.R2GC_2125_1015,(0,1,1):C.R2GC_2926_2250,(0,1,2):C.R2GC_2926_2251,(0,1,3):C.R2GC_2926_2252,(0,1,5):C.R2GC_2926_2253,(0,0,0):C.R2GC_2127_1017,(0,0,4):C.R2GC_2127_1018,(0,0,1):C.R2GC_2927_2254,(0,0,2):C.R2GC_2927_2255,(0,0,3):C.R2GC_2927_2256,(0,0,5):C.R2GC_2927_2257})

V_161 = CTVertex(name = 'V_161',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3376_2709,(0,2,2):C.R2GC_3376_2710,(0,2,1):C.R2GC_2006_868,(0,13,0):C.R2GC_3007_2440,(0,13,2):C.R2GC_3007_2441,(0,13,1):C.R2GC_1927_749,(0,18,0):C.R2GC_2006_866,(0,18,2):C.R2GC_2006_867,(0,18,1):C.R2GC_2006_868,(0,10,0):C.R2GC_1927_747,(0,10,2):C.R2GC_1927_748,(0,10,1):C.R2GC_1927_749,(0,16,2):C.R2GC_917_3051,(0,0,0):C.R2GC_3373_2704,(0,0,2):C.R2GC_3373_2705,(0,0,1):C.R2GC_3373_2706,(0,4,0):C.R2GC_3374_2707,(0,4,2):C.R2GC_3375_2708,(0,4,1):C.R2GC_3373_2706,(0,15,0):C.R2GC_3374_2707,(0,15,2):C.R2GC_3373_2705,(0,15,1):C.R2GC_3373_2706,(0,8,2):C.R2GC_877_3011,(0,5,0):C.R2GC_1930_754,(0,5,2):C.R2GC_1929_753,(0,5,1):C.R2GC_1928_752,(0,6,0):C.R2GC_1928_750,(0,6,2):C.R2GC_1928_751,(0,6,1):C.R2GC_1928_752,(0,7,0):C.R2GC_1928_750,(0,7,2):C.R2GC_1929_753,(0,7,1):C.R2GC_1928_752,(0,1,0):C.R2GC_1544_360,(0,1,2):C.R2GC_1543_359,(0,12,0):C.R2GC_1419_200,(0,12,2):C.R2GC_1418_199,(0,17,0):C.R2GC_1550_366,(0,17,2):C.R2GC_1549_365,(0,9,0):C.R2GC_1413_194,(0,9,2):C.R2GC_1412_193,(0,3,2):C.R2GC_1543_359,(0,14,2):C.R2GC_1418_199,(0,19,2):C.R2GC_1549_365,(0,11,2):C.R2GC_1412_193})

V_162 = CTVertex(name = 'V_162',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3377_2711,(0,2,2):C.R2GC_3377_2712,(0,2,1):C.R2GC_2007_871,(0,13,0):C.R2GC_3003_2433,(0,13,2):C.R2GC_3003_2434,(0,13,1):C.R2GC_1926_746,(0,18,0):C.R2GC_2007_869,(0,18,2):C.R2GC_2007_870,(0,18,1):C.R2GC_2007_871,(0,10,0):C.R2GC_1926_744,(0,10,2):C.R2GC_1926_745,(0,10,1):C.R2GC_1926_746,(0,16,2):C.R2GC_1555_371,(0,0,0):C.R2GC_3378_2713,(0,0,2):C.R2GC_3378_2714,(0,0,1):C.R2GC_3378_2715,(0,4,0):C.R2GC_3379_2716,(0,4,2):C.R2GC_3380_2717,(0,4,1):C.R2GC_3378_2715,(0,15,0):C.R2GC_3379_2716,(0,15,2):C.R2GC_3378_2714,(0,15,1):C.R2GC_3378_2715,(0,8,2):C.R2GC_1440_221,(0,5,0):C.R2GC_3004_2435,(0,5,2):C.R2GC_3004_2436,(0,5,1):C.R2GC_3004_2437,(0,6,0):C.R2GC_3005_2438,(0,6,2):C.R2GC_3006_2439,(0,6,1):C.R2GC_3004_2437,(0,7,0):C.R2GC_3005_2438,(0,7,2):C.R2GC_3004_2436,(0,7,1):C.R2GC_3004_2437,(0,1,0):C.R2GC_1548_364,(0,1,2):C.R2GC_1547_363,(0,12,0):C.R2GC_1417_198,(0,12,2):C.R2GC_1416_197,(0,17,0):C.R2GC_1546_362,(0,17,2):C.R2GC_1545_361,(0,9,0):C.R2GC_1415_196,(0,9,2):C.R2GC_1414_195,(0,3,2):C.R2GC_1547_363,(0,14,2):C.R2GC_1416_197,(0,19,2):C.R2GC_1545_361,(0,11,2):C.R2GC_1414_195})

V_163 = CTVertex(name = 'V_163',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV22, L.FFV3, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,13,0):C.R2GC_3382_2719,(0,13,2):C.R2GC_3382_2720,(0,13,1):C.R2GC_2008_874,(0,8,0):C.R2GC_3078_2498,(0,8,2):C.R2GC_3078_2499,(0,8,1):C.R2GC_1950_786,(0,11,0):C.R2GC_2008_872,(0,11,2):C.R2GC_2008_873,(0,11,1):C.R2GC_2008_874,(0,6,0):C.R2GC_1950_784,(0,6,2):C.R2GC_1950_785,(0,6,1):C.R2GC_1950_786,(0,9,2):C.R2GC_3391_2737,(0,1,0):C.R2GC_3383_2721,(0,1,2):C.R2GC_3383_2722,(0,1,1):C.R2GC_3383_2723,(0,4,0):C.R2GC_3384_2724,(0,4,2):C.R2GC_3384_2725,(0,4,1):C.R2GC_3383_2723,(0,3,2):C.R2GC_3150_2566,(0,0,0):C.R2GC_3079_2500,(0,0,2):C.R2GC_3079_2501,(0,0,1):C.R2GC_3079_2502,(0,2,0):C.R2GC_3080_2503,(0,2,2):C.R2GC_3080_2504,(0,2,1):C.R2GC_3079_2502,(0,12,0):C.R2GC_1568_386,(0,12,2):C.R2GC_1568_387,(0,7,0):C.R2GC_1473_255,(0,7,2):C.R2GC_1473_256,(0,10,0):C.R2GC_1567_384,(0,10,2):C.R2GC_1567_385,(0,5,0):C.R2GC_1472_253,(0,5,2):C.R2GC_1472_254})

V_164 = CTVertex(name = 'V_164',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1572_394,(0,2,3):C.R2GC_1572_395,(2,2,2):C.R2GC_3387_2731,(2,2,0):C.R2GC_2009_875,(2,2,3):C.R2GC_3387_2732,(2,2,1):C.R2GC_2010_880,(1,2,2):C.R2GC_3385_2726,(1,2,0):C.R2GC_2010_878,(1,2,3):C.R2GC_3385_2727,(1,2,1):C.R2GC_2009_877,(0,1,0):C.R2GC_1572_394,(0,1,3):C.R2GC_1572_395,(2,1,0):C.R2GC_2010_878,(2,1,3):C.R2GC_2010_879,(2,1,1):C.R2GC_2010_880,(1,1,0):C.R2GC_2009_875,(1,1,3):C.R2GC_2009_876,(1,1,1):C.R2GC_2009_877,(2,0,2):C.R2GC_3385_2726,(2,0,0):C.R2GC_3386_2728,(2,0,3):C.R2GC_3386_2729,(2,0,1):C.R2GC_3386_2730,(1,0,2):C.R2GC_3387_2731,(1,0,0):C.R2GC_3386_2728,(1,0,3):C.R2GC_3388_2733,(1,0,1):C.R2GC_3388_2734,(0,0,0):C.R2GC_1573_396,(0,0,3):C.R2GC_1573_397,(0,5,0):C.R2GC_1482_273,(0,5,3):C.R2GC_1482_274,(2,5,2):C.R2GC_3092_2531,(2,5,0):C.R2GC_1952_790,(2,5,3):C.R2GC_3092_2532,(2,5,1):C.R2GC_1953_795,(1,5,2):C.R2GC_3090_2526,(1,5,0):C.R2GC_1953_793,(1,5,3):C.R2GC_3090_2527,(1,5,1):C.R2GC_1952_792,(0,4,0):C.R2GC_1482_273,(0,4,3):C.R2GC_1482_274,(2,4,0):C.R2GC_1953_793,(2,4,3):C.R2GC_1953_794,(2,4,1):C.R2GC_1953_795,(1,4,0):C.R2GC_1952_790,(1,4,3):C.R2GC_1952_791,(1,4,1):C.R2GC_1952_792,(2,3,2):C.R2GC_3090_2526,(2,3,0):C.R2GC_3091_2528,(2,3,3):C.R2GC_3091_2529,(2,3,1):C.R2GC_3091_2530,(1,3,2):C.R2GC_3092_2531,(1,3,0):C.R2GC_3091_2528,(1,3,3):C.R2GC_3093_2533,(1,3,1):C.R2GC_3093_2534,(0,3,0):C.R2GC_1483_275,(0,3,3):C.R2GC_1483_276})

V_165 = CTVertex(name = 'V_165',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.b, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,7,1):C.R2GC_1858_656,(0,7,3):C.R2GC_1858_657,(0,7,0):C.R2GC_1858_658,(0,4,1):C.R2GC_1859_659,(0,4,3):C.R2GC_1859_660,(0,4,0):C.R2GC_1858_658,(0,9,2):C.R2GC_2167_1069,(0,0,1):C.R2GC_1861_664,(0,0,3):C.R2GC_1862_665,(0,0,0):C.R2GC_1860_663,(0,1,1):C.R2GC_1860_661,(0,1,3):C.R2GC_1860_662,(0,1,0):C.R2GC_1860_663,(0,2,1):C.R2GC_1861_664,(0,2,3):C.R2GC_1860_662,(0,2,0):C.R2GC_1860_663,(0,6,1):C.R2GC_1334_135,(0,6,3):C.R2GC_1352_149,(0,3,1):C.R2GC_1333_134,(0,3,3):C.R2GC_1354_150,(0,8,1):C.R2GC_1334_135,(0,5,1):C.R2GC_1333_134})

V_166 = CTVertex(name = 'V_166',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.t, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2505_1501,(0,2,1):C.R2GC_3519_2849,(0,6,0):C.R2GC_2505_1501,(0,6,1):C.R2GC_2505_1502,(0,0,0):C.R2GC_3518_2848,(0,3,0):C.R2GC_3518_2848,(0,4,0):C.R2GC_3518_2848,(0,1,1):C.R2GC_2504_1500,(0,5,1):C.R2GC_2503_1499})

V_167 = CTVertex(name = 'V_167',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.t, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3435_2765,(0,2,1):C.R2GC_2376_1319,(0,6,0):C.R2GC_2376_1318,(0,6,1):C.R2GC_2376_1319,(0,0,1):C.R2GC_3436_2766,(0,3,1):C.R2GC_3436_2766,(0,4,1):C.R2GC_3436_2766,(0,1,0):C.R2GC_1914_728,(0,5,0):C.R2GC_1913_727})

V_168 = CTVertex(name = 'V_168',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.t, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2507_1505,(0,2,1):C.R2GC_3534_2861,(0,1,0):C.R2GC_2507_1505,(0,1,1):C.R2GC_2507_1506,(0,0,0):C.R2GC_3533_2859,(0,0,1):C.R2GC_3533_2860})

V_169 = CTVertex(name = 'V_169',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.t, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3451_2778,(0,2,1):C.R2GC_2380_1325,(0,1,0):C.R2GC_2380_1324,(0,1,1):C.R2GC_2380_1325,(0,0,0):C.R2GC_3450_2776,(0,0,1):C.R2GC_3450_2777})

V_170 = CTVertex(name = 'V_170',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.t, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.d, P.g] ], [ [P.b, P.d, P.g, P.u] ], [ [P.c, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ], [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,2,3):C.R2GC_2506_1503,(0,2,5):C.R2GC_3529_2856,(0,2,0):C.R2GC_2968_2376,(0,2,1):C.R2GC_2968_2377,(0,2,2):C.R2GC_2968_2378,(0,2,4):C.R2GC_2968_2379,(0,1,3):C.R2GC_2506_1503,(0,1,5):C.R2GC_2506_1504,(0,1,0):C.R2GC_2967_2372,(0,1,1):C.R2GC_2967_2373,(0,1,2):C.R2GC_2967_2374,(0,1,4):C.R2GC_2967_2375,(0,0,3):C.R2GC_3530_2857,(0,0,5):C.R2GC_3530_2858,(0,0,0):C.R2GC_2969_2380,(0,0,1):C.R2GC_2969_2381,(0,0,2):C.R2GC_2969_2382,(0,0,4):C.R2GC_2969_2383})

V_171 = CTVertex(name = 'V_171',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.t, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.s] ], [ [P.b, P.g, P.s, P.u] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.s, P.t, P.u] ] ],
                 couplings = {(0,2,2):C.R2GC_3446_2773,(0,2,4):C.R2GC_2378_1322,(0,2,0):C.R2GC_2980_2412,(0,2,1):C.R2GC_2980_2413,(0,2,3):C.R2GC_2980_2414,(0,2,5):C.R2GC_2980_2415,(0,1,2):C.R2GC_2378_1321,(0,1,4):C.R2GC_2378_1322,(0,1,0):C.R2GC_2979_2408,(0,1,1):C.R2GC_2979_2409,(0,1,3):C.R2GC_2979_2410,(0,1,5):C.R2GC_2979_2411,(0,0,2):C.R2GC_3447_2774,(0,0,4):C.R2GC_3447_2775,(0,0,0):C.R2GC_2981_2416,(0,0,1):C.R2GC_2981_2417,(0,0,3):C.R2GC_2981_2418,(0,0,5):C.R2GC_2981_2419})

V_172 = CTVertex(name = 'V_172',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.t, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3455_2784,(0,2,2):C.R2GC_3455_2785,(0,2,1):C.R2GC_2239_1147,(0,8,0):C.R2GC_2239_1145,(0,8,2):C.R2GC_2239_1146,(0,8,1):C.R2GC_2239_1147,(0,6,2):C.R2GC_1595_425,(0,0,0):C.R2GC_3452_2779,(0,0,2):C.R2GC_3452_2780,(0,0,1):C.R2GC_3452_2781,(0,4,0):C.R2GC_3453_2782,(0,4,2):C.R2GC_3454_2783,(0,4,1):C.R2GC_3452_2781,(0,5,0):C.R2GC_3453_2782,(0,5,2):C.R2GC_3452_2780,(0,5,1):C.R2GC_3452_2781,(0,1,0):C.R2GC_1585_415,(0,1,2):C.R2GC_1584_414,(0,7,0):C.R2GC_1587_417,(0,7,2):C.R2GC_1586_416,(0,3,2):C.R2GC_1584_414,(0,9,2):C.R2GC_1586_416})

V_173 = CTVertex(name = 'V_173',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.t, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_3372_2702,(0,2,2):C.R2GC_3372_2703,(0,2,1):C.R2GC_2369_1307,(0,8,0):C.R2GC_2369_1305,(0,8,2):C.R2GC_2369_1306,(0,8,1):C.R2GC_2369_1307,(0,6,2):C.R2GC_1554_370,(0,0,0):C.R2GC_3369_2697,(0,0,2):C.R2GC_3369_2698,(0,0,1):C.R2GC_3369_2699,(0,4,0):C.R2GC_3370_2700,(0,4,2):C.R2GC_3371_2701,(0,4,1):C.R2GC_3369_2699,(0,5,0):C.R2GC_3370_2700,(0,5,2):C.R2GC_3369_2698,(0,5,1):C.R2GC_3369_2699,(0,1,0):C.R2GC_1540_356,(0,1,2):C.R2GC_1539_355,(0,7,0):C.R2GC_1542_358,(0,7,2):C.R2GC_1541_357,(0,3,2):C.R2GC_1539_355,(0,9,2):C.R2GC_1541_357})

V_174 = CTVertex(name = 'V_174',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.u, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2483_1472,(0,2,1):C.R2GC_2483_1473,(0,6,0):C.R2GC_2484_1474,(0,6,1):C.R2GC_2483_1473,(0,0,1):C.R2GC_2485_1475,(0,3,1):C.R2GC_2485_1475,(0,4,1):C.R2GC_2485_1475,(0,1,0):C.R2GC_2174_1076,(0,5,0):C.R2GC_2173_1075})

V_175 = CTVertex(name = 'V_175',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.u, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2491_1483,(0,2,1):C.R2GC_2491_1484,(0,1,0):C.R2GC_2492_1485,(0,1,1):C.R2GC_2491_1484,(0,0,0):C.R2GC_2493_1486,(0,0,1):C.R2GC_2493_1487})

V_176 = CTVertex(name = 'V_176',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.u, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.t] ], [ [P.b, P.g, P.s, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,3):C.R2GC_2487_1477,(0,2,5):C.R2GC_2487_1478,(0,2,0):C.R2GC_2938_2304,(0,2,1):C.R2GC_2938_2305,(0,2,2):C.R2GC_2938_2306,(0,2,4):C.R2GC_2938_2307,(0,1,3):C.R2GC_2488_1479,(0,1,5):C.R2GC_2487_1478,(0,1,0):C.R2GC_2937_2300,(0,1,1):C.R2GC_2937_2301,(0,1,2):C.R2GC_2937_2302,(0,1,4):C.R2GC_2937_2303,(0,0,3):C.R2GC_2489_1480,(0,0,5):C.R2GC_2489_1481,(0,0,0):C.R2GC_2939_2308,(0,0,1):C.R2GC_2939_2309,(0,0,2):C.R2GC_2939_2310,(0,0,4):C.R2GC_2939_2311})

V_177 = CTVertex(name = 'V_177',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.u, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2476_1464,(0,2,3):C.R2GC_2476_1465,(0,2,2):C.R2GC_2475_1463,(0,8,0):C.R2GC_2475_1461,(0,8,3):C.R2GC_2475_1462,(0,8,2):C.R2GC_2475_1463,(0,6,1):C.R2GC_2107_993,(0,0,0):C.R2GC_2472_1456,(0,0,3):C.R2GC_2472_1457,(0,0,2):C.R2GC_2472_1458,(0,4,0):C.R2GC_2473_1459,(0,4,3):C.R2GC_2474_1460,(0,4,2):C.R2GC_2472_1458,(0,5,0):C.R2GC_2473_1459,(0,5,3):C.R2GC_2472_1457,(0,5,2):C.R2GC_2472_1458,(0,1,0):C.R2GC_1793_598,(0,1,3):C.R2GC_1792_597,(0,7,0):C.R2GC_1795_600,(0,7,3):C.R2GC_1794_599,(0,3,3):C.R2GC_1792_597,(0,9,3):C.R2GC_1794_599})

V_178 = CTVertex(name = 'V_178',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.c, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2144_1038,(0,2,1):C.R2GC_2144_1039,(0,6,0):C.R2GC_2144_1038,(0,6,1):C.R2GC_2145_1040,(0,0,0):C.R2GC_1898_709,(0,3,0):C.R2GC_1898_709,(0,4,0):C.R2GC_1898_709,(0,1,1):C.R2GC_2143_1037,(0,5,1):C.R2GC_2142_1036})

V_179 = CTVertex(name = 'V_179',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.c, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2160_1060,(0,2,1):C.R2GC_2160_1061,(0,1,0):C.R2GC_2160_1060,(0,1,1):C.R2GC_2162_1064,(0,0,0):C.R2GC_2161_1062,(0,0,1):C.R2GC_2161_1063})

V_180 = CTVertex(name = 'V_180',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.c, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.t] ], [ [P.b, P.g, P.s, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2154_1052,(0,2,4):C.R2GC_2154_1053,(0,2,1):C.R2GC_2932_2274,(0,2,2):C.R2GC_2932_2275,(0,2,3):C.R2GC_2932_2276,(0,2,5):C.R2GC_2932_2277,(0,1,0):C.R2GC_2154_1052,(0,1,4):C.R2GC_2155_1054,(0,1,1):C.R2GC_2931_2270,(0,1,2):C.R2GC_2931_2271,(0,1,3):C.R2GC_2931_2272,(0,1,5):C.R2GC_2931_2273,(0,0,0):C.R2GC_2152_1048,(0,0,4):C.R2GC_2152_1049,(0,0,1):C.R2GC_2933_2278,(0,0,2):C.R2GC_2933_2279,(0,0,3):C.R2GC_2933_2280,(0,0,5):C.R2GC_2933_2281})

V_181 = CTVertex(name = 'V_181',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.c, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,2,1):C.R2GC_1891_701,(0,2,3):C.R2GC_1891_702,(0,2,0):C.R2GC_1890_700,(0,8,1):C.R2GC_1890_698,(0,8,3):C.R2GC_1890_699,(0,8,0):C.R2GC_1890_700,(0,6,2):C.R2GC_2106_992,(0,0,1):C.R2GC_1887_693,(0,0,3):C.R2GC_1887_694,(0,0,0):C.R2GC_1887_695,(0,4,1):C.R2GC_1888_696,(0,4,3):C.R2GC_1889_697,(0,4,0):C.R2GC_1887_695,(0,5,1):C.R2GC_1888_696,(0,5,3):C.R2GC_1887_694,(0,5,0):C.R2GC_1887_695,(0,1,1):C.R2GC_1368_162,(0,1,3):C.R2GC_1367_161,(0,7,1):C.R2GC_1370_164,(0,7,3):C.R2GC_1369_163,(0,3,3):C.R2GC_1367_161,(0,9,3):C.R2GC_1369_163})

V_182 = CTVertex(name = 'V_182',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5, L.FF6, L.FF7 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t], [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1660_462,(0,3,0):C.R2GC_1611_440,(0,2,0):C.R2GC_3601_2914,(0,5,0):C.R2GC_3641_2928,(0,1,1):C.R2GC_1808_613,(0,4,1):C.R2GC_1719_488})

V_183 = CTVertex(name = 'V_183',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5, L.FF6, L.FF7 ],
                 loop_particles = [ [ [P.c, P.g], [P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_1635_460,(0,3,1):C.R2GC_1610_439,(0,2,1):C.R2GC_3592_2906,(0,5,1):C.R2GC_3597_2910,(0,1,0):C.R2GC_1516_320,(0,4,0):C.R2GC_1465_244})

V_184 = CTVertex(name = 'V_184',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.b ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1330_131})

V_185 = CTVertex(name = 'V_185',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1330_131})

V_186 = CTVertex(name = 'V_186',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.R2GC_1330_131})

V_187 = CTVertex(name = 'V_187',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.R2GC_1330_131})

V_188 = CTVertex(name = 'V_188',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5, L.FF6, L.FF7 ],
                 loop_particles = [ [ [P.c, P.g], [P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_3392_2738,(0,3,1):C.R2GC_3152_2568,(0,2,1):C.R2GC_3389_2735,(0,5,1):C.R2GC_3146_2562,(0,1,0):C.R2GC_1563_379,(0,4,0):C.R2GC_1464_243})

V_189 = CTVertex(name = 'V_189',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF4, L.FF5, L.FF7 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_3030_2473,(0,2,0):C.R2GC_3030_2473,(0,1,0):C.R2GC_1330_131,(0,3,0):C.R2GC_1330_131})

V_190 = CTVertex(name = 'V_190',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5, L.FF6, L.FF7 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t], [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_3475_2820,(0,3,0):C.R2GC_3153_2569,(0,2,0):C.R2GC_3472_2817,(0,5,0):C.R2GC_3147_2563,(0,1,1):C.R2GC_1769_562,(0,4,1):C.R2GC_1720_489})

V_191 = CTVertex(name = 'V_191',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1330_131})

V_192 = CTVertex(name = 'V_192',
                 type = 'R2',
                 particles = [ P.g, P.g ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VV1, L.VV2, L.VV3 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.t], [P.u] ], [ [P.g] ], [ [P.t] ] ],
                 couplings = {(0,0,1):C.R2GC_2983_2422,(0,1,2):C.R2GC_798_2946,(0,2,0):C.R2GC_2982_2420,(0,2,1):C.R2GC_2982_2421})

V_193 = CTVertex(name = 'V_193',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV22, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,9,0):C.R2GC_2735_1881,(0,9,1):C.R2GC_2735_1882,(0,4,0):C.R2GC_2616_1686,(0,4,1):C.R2GC_2616_1687,(0,5,0):C.R2GC_3414_2757,(0,5,1):C.R2GC_3414_2758,(0,0,0):C.R2GC_3309_2631,(0,0,1):C.R2GC_3309_2632,(0,8,0):C.R2GC_2737_1885,(0,8,1):C.R2GC_2737_1886,(0,3,0):C.R2GC_2618_1690,(0,3,1):C.R2GC_2618_1691,(0,6,0):C.R2GC_2736_1883,(0,6,1):C.R2GC_2736_1884,(0,1,0):C.R2GC_2617_1688,(0,1,1):C.R2GC_2617_1689,(0,7,0):C.R2GC_2734_1879,(0,7,1):C.R2GC_2734_1880,(0,2,0):C.R2GC_2615_1684,(0,2,1):C.R2GC_2615_1685})

V_194 = CTVertex(name = 'V_194',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV22, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,9,0):C.R2GC_2703_1829,(0,9,1):C.R2GC_2703_1830,(0,4,0):C.R2GC_2651_1740,(0,4,1):C.R2GC_2651_1741,(0,5,0):C.R2GC_3496_2838,(0,5,1):C.R2GC_3496_2839,(0,0,0):C.R2GC_3310_2633,(0,0,1):C.R2GC_3310_2634,(0,8,0):C.R2GC_2704_1831,(0,8,1):C.R2GC_2704_1832,(0,3,0):C.R2GC_2652_1742,(0,3,1):C.R2GC_2652_1743,(0,6,0):C.R2GC_2705_1833,(0,6,1):C.R2GC_2705_1834,(0,1,0):C.R2GC_2653_1744,(0,1,1):C.R2GC_2653_1745,(0,7,0):C.R2GC_2702_1827,(0,7,1):C.R2GC_2702_1828,(0,2,0):C.R2GC_2650_1738,(0,2,1):C.R2GC_2650_1739})

V_195 = CTVertex(name = 'V_195',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.d, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,4,0):C.R2GC_3412_2755,(0,2,0):C.R2GC_2208_1114,(0,0,0):C.R2GC_2207_1113,(0,3,0):C.R2GC_2206_1112,(0,1,0):C.R2GC_2205_1111})

V_196 = CTVertex(name = 'V_196',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.s, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,4,0):C.R2GC_3494_2836,(0,2,0):C.R2GC_2329_1252,(0,0,0):C.R2GC_2328_1251,(0,3,0):C.R2GC_2327_1250,(0,1,0):C.R2GC_2326_1249})

V_197 = CTVertex(name = 'V_197',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.c, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1632_457,(0,3,0):C.R2GC_2231_1137,(0,1,0):C.R2GC_2232_1138,(0,4,0):C.R2GC_2230_1136,(0,2,0):C.R2GC_2229_1135})

V_198 = CTVertex(name = 'V_198',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.u, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1623_450,(0,3,0):C.R2GC_2391_1336,(0,1,0):C.R2GC_2392_1337,(0,4,0):C.R2GC_2390_1335,(0,2,0):C.R2GC_2389_1334})

V_199 = CTVertex(name = 'V_199',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS21, L.FFS22, L.FFS3, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,3,0):C.R2GC_3420_2759,(0,0,0):C.R2GC_3304_2624,(0,4,0):C.R2GC_2034_903,(0,1,0):C.R2GC_1991_847,(0,5,0):C.R2GC_2033_902,(0,2,0):C.R2GC_1990_846})

V_200 = CTVertex(name = 'V_200',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS21, L.FFS22, L.FFS3, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,3,0):C.R2GC_3561_2894,(0,3,1):C.R2GC_3561_2895,(0,0,0):C.R2GC_3305_2625,(0,0,1):C.R2GC_3305_2626,(0,4,0):C.R2GC_2731_1873,(0,4,1):C.R2GC_2731_1874,(0,1,0):C.R2GC_2837_2049,(0,1,1):C.R2GC_2837_2050,(0,5,0):C.R2GC_2035_904,(0,5,1):C.R2GC_2035_905,(0,2,0):C.R2GC_2057_932,(0,2,1):C.R2GC_2057_933})

V_201 = CTVertex(name = 'V_201',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS21, L.FFS22, L.FFS3, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,3,0):C.R2GC_3503_2841,(0,3,1):C.R2GC_3503_2842,(0,0,0):C.R2GC_3306_2627,(0,0,1):C.R2GC_3306_2628,(0,4,0):C.R2GC_2763_1932,(0,4,1):C.R2GC_2763_1933,(0,1,0):C.R2GC_2690_1807,(0,1,1):C.R2GC_2690_1808,(0,5,0):C.R2GC_2762_1930,(0,5,1):C.R2GC_2762_1931,(0,2,0):C.R2GC_2689_1805,(0,2,1):C.R2GC_2689_1806})

V_202 = CTVertex(name = 'V_202',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS21, L.FFS22, L.FFS3, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,3,0):C.R2GC_3562_2896,(0,0,0):C.R2GC_3307_2629,(0,4,0):C.R2GC_2766_1935,(0,1,0):C.R2GC_2839_2051,(0,5,0):C.R2GC_2764_1934,(0,2,0):C.R2GC_2793_1979})

V_203 = CTVertex(name = 'V_203',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_2036_906,(0,0,0):C.R2GC_1992_848})

V_204 = CTVertex(name = 'V_204',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2732_1875,(0,1,1):C.R2GC_2732_1876,(0,0,0):C.R2GC_2840_2052,(0,0,1):C.R2GC_2840_2053})

V_205 = CTVertex(name = 'V_205',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2767_1936,(0,1,1):C.R2GC_2767_1937,(0,0,0):C.R2GC_2691_1809,(0,0,1):C.R2GC_2691_1810})

V_206 = CTVertex(name = 'V_206',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2768_1938,(0,0,0):C.R2GC_2841_2054})

V_207 = CTVertex(name = 'V_207',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_1992_848,(0,1,1):C.R2GC_2841_2054,(0,0,0):C.R2GC_2036_906,(0,0,1):C.R2GC_2768_1938})

V_208 = CTVertex(name = 'V_208',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,10,0):[ C.R2GC_3408_2749, C.R2GC_2038_908 ],(0,3,0):[ C.R2GC_3296_2618, C.R2GC_1994_850 ],(0,1,0):C.R2GC_2014_884,(0,8,0):C.R2GC_1974_821,(0,0,0):C.R2GC_2018_888,(0,7,0):C.R2GC_1975_822,(0,11,0):C.R2GC_2017_887,(0,4,0):C.R2GC_1976_823,(0,12,0):C.R2GC_2013_883,(0,5,0):C.R2GC_1973_820,(0,2,0):C.R2GC_2016_886,(0,9,0):C.R2GC_1962_809,(0,13,0):C.R2GC_2015_885,(0,6,0):C.R2GC_1961_808})

V_209 = CTVertex(name = 'V_209',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,10,0):[ C.R2GC_3410_2752, C.R2GC_2738_1887 ],(0,10,1):[ C.R2GC_3410_2753, C.R2GC_2738_1888 ],(0,3,0):[ C.R2GC_2846_2058, C.R2GC_3297_2619 ],(0,3,1):[ C.R2GC_2846_2059, C.R2GC_3297_2620 ],(0,1,0):C.R2GC_2720_1859,(0,1,1):C.R2GC_2720_1860,(0,8,0):C.R2GC_2596_1654,(0,8,1):C.R2GC_2596_1655,(0,0,0):C.R2GC_2724_1864,(0,0,1):C.R2GC_2722_1862,(0,7,0):C.R2GC_2600_1659,(0,7,1):C.R2GC_2598_1657,(0,11,0):C.R2GC_2723_1863,(0,11,1):C.R2GC_2721_1861,(0,4,0):C.R2GC_2599_1658,(0,4,1):C.R2GC_2597_1656,(0,12,0):C.R2GC_2719_1857,(0,12,1):C.R2GC_2719_1858,(0,5,0):C.R2GC_2595_1652,(0,5,1):C.R2GC_2595_1653,(0,2,1):C.R2GC_2722_1862,(0,9,1):C.R2GC_2598_1657,(0,13,1):C.R2GC_2721_1861,(0,6,1):C.R2GC_2597_1656})

V_210 = CTVertex(name = 'V_210',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,10,0):[ C.R2GC_3491_2831, C.R2GC_2773_1942 ],(0,10,1):[ C.R2GC_3491_2832, C.R2GC_2773_1943 ],(0,3,0):[ C.R2GC_3300_2621, C.R2GC_2693_1813 ],(0,3,1):[ C.R2GC_3300_2622, C.R2GC_2693_1814 ],(0,1,0):C.R2GC_2699_1823,(0,1,1):C.R2GC_2699_1824,(0,8,0):C.R2GC_2635_1720,(0,8,1):C.R2GC_2635_1721,(0,0,0):C.R2GC_2054_929,(0,0,1):C.R2GC_2700_1825,(0,7,0):C.R2GC_1966_813,(0,7,1):C.R2GC_2636_1722,(0,11,0):C.R2GC_2053_928,(0,11,1):C.R2GC_2701_1826,(0,4,0):C.R2GC_1965_812,(0,4,1):C.R2GC_2637_1723,(0,12,0):C.R2GC_2698_1821,(0,12,1):C.R2GC_2698_1822,(0,5,0):C.R2GC_2634_1718,(0,5,1):C.R2GC_2634_1719,(0,2,0):C.R2GC_2054_929,(0,9,0):C.R2GC_1966_813,(0,13,0):C.R2GC_2053_928,(0,6,0):C.R2GC_1965_812})

V_211 = CTVertex(name = 'V_211',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,10,0):[ C.R2GC_3493_2835, C.R2GC_2775_1946 ],(0,3,0):[ C.R2GC_3301_2623, C.R2GC_2847_2060 ],(0,1,0):C.R2GC_2826_2039,(0,8,0):C.R2GC_2647_1735,(0,0,0):C.R2GC_2827_2040,(0,7,0):C.R2GC_2648_1736,(0,11,0):C.R2GC_2828_2041,(0,4,0):C.R2GC_2649_1737,(0,12,0):C.R2GC_2825_2038,(0,5,0):C.R2GC_2646_1734,(0,2,0):C.R2GC_2758_1927,(0,9,0):C.R2GC_2612_1681,(0,13,0):C.R2GC_2757_1926,(0,6,0):C.R2GC_2611_1680})

V_212 = CTVertex(name = 'V_212',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,10,0):[ C.R2GC_3613_2922, C.R2GC_2848_2061 ],(0,10,1):[ C.R2GC_3613_2923, C.R2GC_2848_2062 ],(0,3,0):[ C.R2GC_3670_2942, C.R2GC_2774_1944 ],(0,3,1):[ C.R2GC_3670_2943, C.R2GC_2774_1945 ],(0,1,0):C.R2GC_2802_1998,(0,1,1):C.R2GC_2802_1999,(0,8,0):C.R2GC_2670_1769,(0,8,1):C.R2GC_2670_1770,(0,0,0):C.R2GC_2806_2006,(0,0,1):C.R2GC_2806_2007,(0,7,0):C.R2GC_2674_1777,(0,7,1):C.R2GC_2674_1778,(0,11,0):C.R2GC_2805_2004,(0,11,1):C.R2GC_2805_2005,(0,4,0):C.R2GC_2673_1775,(0,4,1):C.R2GC_2673_1776,(0,12,0):C.R2GC_2801_1996,(0,12,1):C.R2GC_2801_1997,(0,5,0):C.R2GC_2669_1767,(0,5,1):C.R2GC_2669_1768,(0,2,0):C.R2GC_2804_2002,(0,2,1):C.R2GC_2804_2003,(0,9,0):C.R2GC_2672_1773,(0,9,1):C.R2GC_2672_1774,(0,13,0):C.R2GC_2803_2000,(0,13,1):C.R2GC_2803_2001,(0,6,0):C.R2GC_2671_1771,(0,6,1):C.R2GC_2671_1772})

V_213 = CTVertex(name = 'V_213',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,6,1):C.R2GC_3405_2745,(0,6,0):C.R2GC_1028_29,(0,3,1):C.R2GC_2313_1236,(0,0,1):C.R2GC_2312_1235,(0,4,1):C.R2GC_2311_1234,(0,1,1):C.R2GC_2310_1233,(0,5,1):C.R2GC_2313_1236,(0,2,1):C.R2GC_2312_1235})

V_214 = CTVertex(name = 'V_214',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.d, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,6,0):C.R2GC_3406_2746,(0,6,1):C.R2GC_1032_33,(0,3,0):C.R2GC_2192_1098,(0,0,0):C.R2GC_2191_1097,(0,4,0):C.R2GC_2190_1096,(0,1,0):C.R2GC_2189_1095,(0,5,0):C.R2GC_2192_1098,(0,2,0):C.R2GC_2191_1097})

V_215 = CTVertex(name = 'V_215',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.s, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,6,1):C.R2GC_3488_2827,(0,6,0):C.R2GC_1031_32,(0,3,1):C.R2GC_2321_1244,(0,0,1):C.R2GC_2320_1243,(0,4,1):C.R2GC_2319_1242,(0,1,1):C.R2GC_2318_1241,(0,5,1):C.R2GC_2321_1244,(0,2,1):C.R2GC_2320_1243})

V_216 = CTVertex(name = 'V_216',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,6,0):C.R2GC_3489_2828,(0,6,1):C.R2GC_1033_34,(0,3,0):C.R2GC_2200_1106,(0,0,0):C.R2GC_2199_1105,(0,4,0):C.R2GC_2198_1104,(0,1,0):C.R2GC_2197_1103,(0,5,0):C.R2GC_2200_1106,(0,2,0):C.R2GC_2199_1105})

V_217 = CTVertex(name = 'V_217',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,3,0):C.R2GC_2440_1401,(0,3,3):C.R2GC_2440_1402,(0,0,0):C.R2GC_2439_1399,(0,0,3):C.R2GC_2439_1400,(0,4,0):C.R2GC_2438_1397,(0,4,3):C.R2GC_2438_1398,(0,1,0):C.R2GC_2437_1395,(0,1,3):C.R2GC_2437_1396,(0,5,0):C.R2GC_2440_1401,(0,5,3):C.R2GC_2440_1402,(0,2,0):C.R2GC_2439_1399,(0,2,3):C.R2GC_2439_1400,(0,6,1):C.R2GC_1095_119,(0,6,2):C.R2GC_1095_120})

V_218 = CTVertex(name = 'V_218',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.c, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_1628_451,(0,2,1):C.R2GC_1034_35,(0,0,0):C.R2GC_2227_1133,(0,3,0):C.R2GC_2228_1134,(0,1,0):C.R2GC_2226_1132,(0,4,0):C.R2GC_2225_1131})

V_219 = CTVertex(name = 'V_219',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_1629_452,(0,2,1):C.R2GC_1035_36,(0,0,0):C.R2GC_2250_1158,(0,3,0):C.R2GC_2251_1159,(0,1,0):C.R2GC_2249_1157,(0,4,0):C.R2GC_2248_1156})

V_220 = CTVertex(name = 'V_220',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_1618_443,(0,2,0):C.R2GC_1029_30,(0,0,1):C.R2GC_2358_1291,(0,3,1):C.R2GC_2359_1292,(0,1,1):C.R2GC_2357_1290,(0,4,1):C.R2GC_2356_1289})

V_221 = CTVertex(name = 'V_221',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.u, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,1):C.R2GC_1619_444,(0,2,0):C.R2GC_1030_31,(0,0,1):C.R2GC_2387_1332,(0,3,1):C.R2GC_2388_1333,(0,1,1):C.R2GC_2386_1331,(0,4,1):C.R2GC_2385_1330})

V_222 = CTVertex(name = 'V_222',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2463_1440,(0,0,3):C.R2GC_2463_1441,(0,3,0):C.R2GC_2464_1442,(0,3,3):C.R2GC_2464_1443,(0,1,0):C.R2GC_2462_1438,(0,1,3):C.R2GC_2462_1439,(0,4,0):C.R2GC_2461_1436,(0,4,3):C.R2GC_2461_1437,(0,2,1):C.R2GC_1096_121,(0,2,2):C.R2GC_1096_122})

V_223 = CTVertex(name = 'V_223',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_2037_907,(0,0,0):C.R2GC_1993_849})

V_224 = CTVertex(name = 'V_224',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2733_1877,(0,1,1):C.R2GC_2733_1878,(0,0,0):C.R2GC_2843_2055,(0,0,1):C.R2GC_2843_2056})

V_225 = CTVertex(name = 'V_225',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2770_1939,(0,1,1):C.R2GC_2770_1940,(0,0,0):C.R2GC_2692_1811,(0,0,1):C.R2GC_2692_1812})

V_226 = CTVertex(name = 'V_226',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2771_1941,(0,0,0):C.R2GC_2844_2057})

V_227 = CTVertex(name = 'V_227',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_1993_849,(0,1,1):C.R2GC_2844_2057,(0,0,0):C.R2GC_2037_907,(0,0,1):C.R2GC_2771_1941})

V_228 = CTVertex(name = 'V_228',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_2020_890,(0,7,0):C.R2GC_1971_818,(0,0,0):C.R2GC_2024_894,(0,6,0):C.R2GC_1970_817,(0,9,0):C.R2GC_2023_893,(0,3,0):C.R2GC_1969_816,(0,10,0):C.R2GC_2019_889,(0,4,0):C.R2GC_1972_819,(0,2,0):C.R2GC_2022_892,(0,8,0):C.R2GC_1964_811,(0,11,0):C.R2GC_2021_891,(0,5,0):C.R2GC_1963_810})

V_229 = CTVertex(name = 'V_229',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2726_1867,(0,1,1):C.R2GC_2726_1868,(0,7,0):C.R2GC_2602_1662,(0,7,1):C.R2GC_2602_1663,(0,0,0):C.R2GC_2730_1872,(0,0,1):C.R2GC_2728_1870,(0,6,0):C.R2GC_2606_1667,(0,6,1):C.R2GC_2604_1665,(0,9,0):C.R2GC_2729_1871,(0,9,1):C.R2GC_2727_1869,(0,3,0):C.R2GC_2605_1666,(0,3,1):C.R2GC_2603_1664,(0,10,0):C.R2GC_2725_1865,(0,10,1):C.R2GC_2725_1866,(0,4,0):C.R2GC_2601_1660,(0,4,1):C.R2GC_2601_1661,(0,2,1):C.R2GC_2728_1870,(0,8,1):C.R2GC_2604_1665,(0,11,1):C.R2GC_2727_1869,(0,5,1):C.R2GC_2603_1664})

V_230 = CTVertex(name = 'V_230',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2696_1817,(0,1,1):C.R2GC_2696_1818,(0,7,0):C.R2GC_2632_1714,(0,7,1):C.R2GC_2632_1715,(0,0,0):C.R2GC_2056_931,(0,0,1):C.R2GC_2695_1816,(0,6,0):C.R2GC_1968_815,(0,6,1):C.R2GC_2631_1713,(0,9,0):C.R2GC_2055_930,(0,9,1):C.R2GC_2694_1815,(0,3,0):C.R2GC_1967_814,(0,3,1):C.R2GC_2630_1712,(0,10,0):C.R2GC_2697_1819,(0,10,1):C.R2GC_2697_1820,(0,4,0):C.R2GC_2633_1716,(0,4,1):C.R2GC_2633_1717,(0,2,0):C.R2GC_2056_931,(0,8,0):C.R2GC_1968_815,(0,11,0):C.R2GC_2055_930,(0,5,0):C.R2GC_1967_814})

V_231 = CTVertex(name = 'V_231',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2823_2036,(0,7,0):C.R2GC_2644_1732,(0,0,0):C.R2GC_2822_2035,(0,6,0):C.R2GC_2643_1731,(0,9,0):C.R2GC_2821_2034,(0,3,0):C.R2GC_2642_1730,(0,10,0):C.R2GC_2824_2037,(0,4,0):C.R2GC_2645_1733,(0,2,0):C.R2GC_2760_1929,(0,8,0):C.R2GC_2614_1683,(0,11,0):C.R2GC_2759_1928,(0,5,0):C.R2GC_2613_1682})

V_232 = CTVertex(name = 'V_232',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,0):C.R2GC_2808_2010,(0,1,1):C.R2GC_2808_2011,(0,8,0):C.R2GC_2676_1781,(0,8,1):C.R2GC_2676_1782,(0,10,0):C.R2GC_1087_104,(0,10,1):C.R2GC_1087_105,(0,3,0):C.R2GC_1081_93,(0,3,1):C.R2GC_1081_94,(0,0,0):C.R2GC_2812_2018,(0,0,1):C.R2GC_2812_2019,(0,7,0):C.R2GC_2680_1789,(0,7,1):C.R2GC_2680_1790,(0,11,0):C.R2GC_2811_2016,(0,11,1):C.R2GC_2811_2017,(0,4,0):C.R2GC_2679_1787,(0,4,1):C.R2GC_2679_1788,(0,12,0):C.R2GC_2807_2008,(0,12,1):C.R2GC_2807_2009,(0,5,0):C.R2GC_2675_1779,(0,5,1):C.R2GC_2675_1780,(0,2,0):C.R2GC_2810_2014,(0,2,1):C.R2GC_2810_2015,(0,9,0):C.R2GC_2678_1785,(0,9,1):C.R2GC_2678_1786,(0,13,0):C.R2GC_2809_2012,(0,13,1):C.R2GC_2809_2013,(0,6,0):C.R2GC_2677_1783,(0,6,1):C.R2GC_2677_1784})

V_233 = CTVertex(name = 'V_233',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,3,0):C.R2GC_2317_1240,(0,0,0):C.R2GC_2316_1239,(0,4,0):C.R2GC_2315_1238,(0,1,0):C.R2GC_2314_1237,(0,5,0):C.R2GC_2317_1240,(0,2,0):C.R2GC_2316_1239})

V_234 = CTVertex(name = 'V_234',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.d, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,3,0):C.R2GC_2196_1102,(0,0,0):C.R2GC_2195_1101,(0,4,0):C.R2GC_2194_1100,(0,1,0):C.R2GC_2193_1099,(0,5,0):C.R2GC_2196_1102,(0,2,0):C.R2GC_2195_1101})

V_235 = CTVertex(name = 'V_235',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.s, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,3,0):C.R2GC_2325_1248,(0,0,0):C.R2GC_2324_1247,(0,4,0):C.R2GC_2323_1246,(0,1,0):C.R2GC_2322_1245,(0,5,0):C.R2GC_2325_1248,(0,2,0):C.R2GC_2324_1247})

V_236 = CTVertex(name = 'V_236',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,3,0):C.R2GC_2204_1110,(0,0,0):C.R2GC_2203_1109,(0,4,0):C.R2GC_2202_1108,(0,1,0):C.R2GC_2201_1107,(0,5,0):C.R2GC_2204_1110,(0,2,0):C.R2GC_2203_1109})

V_237 = CTVertex(name = 'V_237',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,3,0):C.R2GC_2444_1409,(0,3,1):C.R2GC_2444_1410,(0,0,0):C.R2GC_2443_1407,(0,0,1):C.R2GC_2443_1408,(0,4,0):C.R2GC_2442_1405,(0,4,1):C.R2GC_2442_1406,(0,1,0):C.R2GC_2441_1403,(0,1,1):C.R2GC_2441_1404,(0,5,0):C.R2GC_2444_1409,(0,5,1):C.R2GC_2444_1410,(0,2,0):C.R2GC_2443_1407,(0,2,1):C.R2GC_2443_1408})

V_238 = CTVertex(name = 'V_238',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.c, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2222_1128,(0,2,0):C.R2GC_2221_1127,(0,1,0):C.R2GC_2223_1129,(0,3,0):C.R2GC_2224_1130})

V_239 = CTVertex(name = 'V_239',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2245_1153,(0,2,0):C.R2GC_2244_1152,(0,1,0):C.R2GC_2246_1154,(0,3,0):C.R2GC_2247_1155})

V_240 = CTVertex(name = 'V_240',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2353_1286,(0,2,0):C.R2GC_2352_1285,(0,1,0):C.R2GC_2354_1287,(0,3,0):C.R2GC_2355_1288})

V_241 = CTVertex(name = 'V_241',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.u, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_2382_1327,(0,2,0):C.R2GC_2381_1326,(0,1,0):C.R2GC_2383_1328,(0,3,0):C.R2GC_2384_1329})

V_242 = CTVertex(name = 'V_242',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_2458_1430,(0,0,1):C.R2GC_2458_1431,(0,2,0):C.R2GC_2457_1428,(0,2,1):C.R2GC_2457_1429,(0,1,0):C.R2GC_2459_1432,(0,1,1):C.R2GC_2459_1433,(0,3,0):C.R2GC_2460_1434,(0,3,1):C.R2GC_2460_1435})

V_243 = CTVertex(name = 'V_243',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.s] ], [ [P.b, P.g, P.s, P.t] ], [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,6,2):C.R2GC_3407_2747,(0,6,1):C.R2GC_3407_2748,(0,3,0):C.R2GC_2084_964,(0,3,2):C.R2GC_2086_966,(0,0,0):C.R2GC_2085_965,(0,0,2):C.R2GC_2087_967,(0,4,0):C.R2GC_2088_968,(0,4,2):C.R2GC_2088_969,(0,1,0):C.R2GC_2089_970,(0,1,2):C.R2GC_2089_971,(0,5,0):C.R2GC_2084_964,(0,2,0):C.R2GC_2085_965})

V_244 = CTVertex(name = 'V_244',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.d, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.d, P.g, P.t] ], [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,6,2):C.R2GC_3409_2750,(0,6,1):C.R2GC_3409_2751,(0,3,0):C.R2GC_2062_938,(0,3,2):C.R2GC_2064_940,(0,0,0):C.R2GC_2063_939,(0,0,2):C.R2GC_2065_941,(0,4,0):C.R2GC_2066_942,(0,4,2):C.R2GC_2066_943,(0,1,0):C.R2GC_2067_944,(0,1,2):C.R2GC_2067_945,(0,5,0):C.R2GC_2062_938,(0,2,0):C.R2GC_2063_939})

V_245 = CTVertex(name = 'V_245',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.s, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.s] ], [ [P.b, P.g, P.s, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,6,2):C.R2GC_3490_2829,(0,6,1):C.R2GC_3490_2830,(0,3,0):C.R2GC_2090_972,(0,3,2):C.R2GC_2627_1707,(0,0,0):C.R2GC_2091_973,(0,0,2):C.R2GC_2626_1706,(0,4,0):C.R2GC_2628_1708,(0,4,2):C.R2GC_2628_1709,(0,1,0):C.R2GC_2629_1710,(0,1,2):C.R2GC_2629_1711,(0,5,0):C.R2GC_2090_972,(0,2,0):C.R2GC_2091_973})

V_246 = CTVertex(name = 'V_246',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.d, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,6,2):C.R2GC_3492_2833,(0,6,1):C.R2GC_3492_2834,(0,3,0):C.R2GC_2068_946,(0,3,2):C.R2GC_2639_1725,(0,0,0):C.R2GC_2069_947,(0,0,2):C.R2GC_2638_1724,(0,4,0):C.R2GC_2640_1726,(0,4,2):C.R2GC_2640_1727,(0,1,0):C.R2GC_2641_1728,(0,1,2):C.R2GC_2641_1729,(0,5,0):C.R2GC_2068_946,(0,2,0):C.R2GC_2069_947})

V_247 = CTVertex(name = 'V_247',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g] ], [ [P.b, P.g, P.s] ], [ [P.b, P.g, P.t, P.u] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,6,4):C.R2GC_3612_2918,(0,6,5):C.R2GC_3612_2919,(0,6,0):C.R2GC_3612_2920,(0,6,3):C.R2GC_3612_2921,(0,3,1):C.R2GC_2097_979,(0,3,2):C.R2GC_2097_980,(0,3,4):C.R2GC_2608_1670,(0,3,5):C.R2GC_2608_1671,(0,0,1):C.R2GC_2098_981,(0,0,2):C.R2GC_2098_982,(0,0,4):C.R2GC_2607_1668,(0,0,5):C.R2GC_2607_1669,(0,4,1):C.R2GC_2609_1672,(0,4,2):C.R2GC_2609_1673,(0,4,4):C.R2GC_2609_1674,(0,4,5):C.R2GC_2609_1675,(0,1,1):C.R2GC_2610_1676,(0,1,2):C.R2GC_2610_1677,(0,1,4):C.R2GC_2610_1678,(0,1,5):C.R2GC_2610_1679,(0,5,1):C.R2GC_2097_979,(0,5,2):C.R2GC_2097_980,(0,2,1):C.R2GC_2098_981,(0,2,2):C.R2GC_2098_982})

V_248 = CTVertex(name = 'V_248',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.s, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1010_11})

V_249 = CTVertex(name = 'V_249',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1011_12})

V_250 = CTVertex(name = 'V_250',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1016_17})

V_251 = CTVertex(name = 'V_251',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.d, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1017_18})

V_252 = CTVertex(name = 'V_252',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ] ],
                 couplings = {(0,4,0):C.R2GC_1874_680,(0,6,1):C.R2GC_3401_2741,(0,3,0):C.R2GC_1876_682,(0,0,0):C.R2GC_1875_681,(0,1,0):C.R2GC_1873_679,(0,5,0):C.R2GC_1876_682,(0,2,0):C.R2GC_1875_681})

V_253 = CTVertex(name = 'V_253',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,4,1):C.R2GC_2425_1383,(0,6,0):C.R2GC_3403_2743,(0,3,1):C.R2GC_2427_1385,(0,0,1):C.R2GC_2426_1384,(0,1,1):C.R2GC_2424_1382,(0,5,1):C.R2GC_2427_1385,(0,2,1):C.R2GC_2426_1384})

V_254 = CTVertex(name = 'V_254',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ] ],
                 couplings = {(0,4,0):C.R2GC_1878_684,(0,6,1):C.R2GC_3484_2823,(0,3,0):C.R2GC_1880_686,(0,0,0):C.R2GC_1879_685,(0,1,0):C.R2GC_1877_683,(0,5,0):C.R2GC_1880_686,(0,2,0):C.R2GC_1879_685})

V_255 = CTVertex(name = 'V_255',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,4,1):C.R2GC_2429_1387,(0,6,0):C.R2GC_3486_2825,(0,3,1):C.R2GC_2431_1389,(0,0,1):C.R2GC_2430_1388,(0,1,1):C.R2GC_2428_1386,(0,5,1):C.R2GC_2431_1389,(0,2,1):C.R2GC_2430_1388})

V_256 = CTVertex(name = 'V_256',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.d, P.g, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,5,2):C.R2GC_2343_1267,(0,5,3):C.R2GC_2343_1268,(0,0,2):C.R2GC_3668_2934,(0,0,3):C.R2GC_3668_2935,(0,0,0):C.R2GC_3668_2936,(0,0,1):C.R2GC_3668_2937,(0,4,2):C.R2GC_2345_1271,(0,4,3):C.R2GC_2345_1272,(0,1,2):C.R2GC_2344_1269,(0,1,3):C.R2GC_2344_1270,(0,2,2):C.R2GC_2342_1265,(0,2,3):C.R2GC_2342_1266,(0,6,2):C.R2GC_2345_1271,(0,6,3):C.R2GC_2345_1272,(0,3,2):C.R2GC_2344_1269,(0,3,3):C.R2GC_2344_1270})

V_257 = CTVertex(name = 'V_257',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.c, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.d, P.g, P.t] ], [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,3,2):C.R2GC_1630_453,(0,3,1):C.R2GC_1630_454,(0,0,0):C.R2GC_2078_958,(0,0,2):C.R2GC_2052_927,(0,4,0):C.R2GC_2079_959,(0,4,2):C.R2GC_2051_926,(0,1,0):C.R2GC_2077_956,(0,1,2):C.R2GC_2077_957,(0,5,0):C.R2GC_2076_954,(0,5,2):C.R2GC_2076_955,(0,2,2):C.R2GC_2052_927,(0,6,2):C.R2GC_2051_926})

V_258 = CTVertex(name = 'V_258',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.d, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,3,2):C.R2GC_1631_455,(0,3,1):C.R2GC_1631_456,(0,0,0):C.R2GC_2756_1925,(0,0,2):C.R2GC_2754_1923,(0,4,0):C.R2GC_2755_1924,(0,4,2):C.R2GC_2753_1922,(0,1,0):C.R2GC_2752_1920,(0,1,2):C.R2GC_2752_1921,(0,5,0):C.R2GC_2751_1918,(0,5,2):C.R2GC_2751_1919,(0,2,2):C.R2GC_2754_1923,(0,6,2):C.R2GC_2753_1922})

V_259 = CTVertex(name = 'V_259',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.g, P.s] ], [ [P.b, P.g, P.s, P.t] ], [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,3,2):C.R2GC_1620_445,(0,3,1):C.R2GC_1620_446,(0,0,0):C.R2GC_2102_988,(0,0,2):C.R2GC_2012_882,(0,4,0):C.R2GC_2103_989,(0,4,2):C.R2GC_2011_881,(0,1,0):C.R2GC_2101_986,(0,1,2):C.R2GC_2101_987,(0,5,0):C.R2GC_2100_984,(0,5,2):C.R2GC_2100_985,(0,2,2):C.R2GC_2012_882,(0,6,2):C.R2GC_2011_881})

V_260 = CTVertex(name = 'V_260',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.u, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.g, P.s] ], [ [P.b, P.g, P.s, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,3,2):C.R2GC_1621_447,(0,3,1):C.R2GC_1621_448,(0,0,0):C.R2GC_2718_1856,(0,0,2):C.R2GC_2716_1854,(0,4,0):C.R2GC_2717_1855,(0,4,2):C.R2GC_2715_1853,(0,1,0):C.R2GC_2714_1851,(0,1,2):C.R2GC_2714_1852,(0,5,0):C.R2GC_2713_1849,(0,5,2):C.R2GC_2713_1850,(0,2,2):C.R2GC_2716_1854,(0,6,2):C.R2GC_2715_1853})

V_261 = CTVertex(name = 'V_261',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g] ], [ [P.b, P.g, P.s] ], [ [P.b, P.g, P.t, P.u] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,3,4):C.R2GC_2921_2224,(0,3,5):C.R2GC_2921_2225,(0,3,0):C.R2GC_2921_2226,(0,3,3):C.R2GC_2921_2227,(0,0,1):C.R2GC_2800_1994,(0,0,2):C.R2GC_2800_1995,(0,0,4):C.R2GC_2798_1990,(0,0,5):C.R2GC_2798_1991,(0,4,1):C.R2GC_2799_1992,(0,4,2):C.R2GC_2799_1993,(0,4,4):C.R2GC_2797_1988,(0,4,5):C.R2GC_2797_1989,(0,1,1):C.R2GC_2796_1984,(0,1,2):C.R2GC_2796_1985,(0,1,4):C.R2GC_2796_1986,(0,1,5):C.R2GC_2796_1987,(0,5,1):C.R2GC_2795_1980,(0,5,2):C.R2GC_2795_1981,(0,5,4):C.R2GC_2795_1982,(0,5,5):C.R2GC_2795_1983,(0,2,4):C.R2GC_2798_1990,(0,2,5):C.R2GC_2798_1991,(0,6,4):C.R2GC_2797_1988,(0,6,5):C.R2GC_2797_1989})

V_262 = CTVertex(name = 'V_262',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.s, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1020_21})

V_263 = CTVertex(name = 'V_263',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.d, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1013_14})

V_264 = CTVertex(name = 'V_264',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.s, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1019_20})

V_265 = CTVertex(name = 'V_265',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.d, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1012_13})

V_266 = CTVertex(name = 'V_266',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1894_705,(0,2,1):C.R2GC_3402_2742,(0,0,0):C.R2GC_1893_704,(0,3,0):C.R2GC_1892_703,(0,4,0):C.R2GC_1895_706})

V_267 = CTVertex(name = 'V_267',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ] ],
                 couplings = {(0,1,0):C.R2GC_1905_719,(0,2,1):C.R2GC_3404_2744,(0,0,0):C.R2GC_1904_718,(0,3,0):C.R2GC_1903_717,(0,4,0):C.R2GC_1906_720})

V_268 = CTVertex(name = 'V_268',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_2455_1426,(0,2,0):C.R2GC_3485_2824,(0,0,1):C.R2GC_2454_1425,(0,3,1):C.R2GC_2453_1424,(0,4,1):C.R2GC_2456_1427})

V_269 = CTVertex(name = 'V_269',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,1,1):C.R2GC_2479_1468,(0,2,0):C.R2GC_3487_2826,(0,0,1):C.R2GC_2478_1467,(0,3,1):C.R2GC_2477_1466,(0,4,1):C.R2GC_2480_1469})

V_270 = CTVertex(name = 'V_270',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS30, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.d, P.g, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,1,2):C.R2GC_2372_1312,(0,1,3):C.R2GC_2372_1313,(0,2,2):C.R2GC_3669_2938,(0,2,3):C.R2GC_3669_2939,(0,2,0):C.R2GC_3669_2940,(0,2,1):C.R2GC_3669_2941,(0,0,2):C.R2GC_2371_1310,(0,0,3):C.R2GC_2371_1311,(0,3,2):C.R2GC_2370_1308,(0,3,3):C.R2GC_2370_1309,(0,4,2):C.R2GC_2373_1314,(0,4,3):C.R2GC_2373_1315})

V_271 = CTVertex(name = 'V_271',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2330_1253,(0,1,0):C.R2GC_2331_1254,(0,0,0):C.R2GC_2332_1255})

V_272 = CTVertex(name = 'V_272',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.d, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2209_1115,(0,1,0):C.R2GC_2210_1116,(0,0,0):C.R2GC_2211_1117})

V_273 = CTVertex(name = 'V_273',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.s, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2333_1256,(0,1,0):C.R2GC_2334_1257,(0,0,0):C.R2GC_2335_1258})

V_274 = CTVertex(name = 'V_274',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2212_1118,(0,1,0):C.R2GC_2213_1119,(0,0,0):C.R2GC_2214_1120})

V_275 = CTVertex(name = 'V_275',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2447_1412,(0,2,1):C.R2GC_2447_1413,(0,1,0):C.R2GC_2448_1414,(0,1,1):C.R2GC_2448_1415,(0,0,0):C.R2GC_2449_1416,(0,0,1):C.R2GC_2449_1417})

V_276 = CTVertex(name = 'V_276',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.s, P.t] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,2):C.R2GC_2874_2095,(0,2,0):C.R2GC_2874_2096,(0,2,1):C.R2GC_2874_2097,(0,1,2):C.R2GC_2873_2092,(0,1,0):C.R2GC_2873_2093,(0,1,1):C.R2GC_2873_2094,(0,0,2):C.R2GC_2875_2098,(0,0,0):C.R2GC_2875_2099,(0,0,1):C.R2GC_2875_2100})

V_277 = CTVertex(name = 'V_277',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.d, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_2946_2318,(0,2,0):C.R2GC_2946_2319,(0,2,2):C.R2GC_2946_2320,(0,1,1):C.R2GC_2947_2321,(0,1,0):C.R2GC_2947_2322,(0,1,2):C.R2GC_2947_2323,(0,0,1):C.R2GC_2948_2324,(0,0,0):C.R2GC_2948_2325,(0,0,2):C.R2GC_2948_2326})

V_278 = CTVertex(name = 'V_278',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.s, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.s, P.t] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,2):C.R2GC_2877_2104,(0,2,0):C.R2GC_2877_2105,(0,2,1):C.R2GC_2877_2106,(0,1,2):C.R2GC_2876_2101,(0,1,0):C.R2GC_2876_2102,(0,1,1):C.R2GC_2876_2103,(0,0,2):C.R2GC_2878_2107,(0,0,0):C.R2GC_2878_2108,(0,0,1):C.R2GC_2878_2109})

V_279 = CTVertex(name = 'V_279',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_2949_2327,(0,2,0):C.R2GC_2949_2328,(0,2,2):C.R2GC_2949_2329,(0,1,1):C.R2GC_2950_2330,(0,1,0):C.R2GC_2950_2331,(0,1,2):C.R2GC_2950_2332,(0,0,1):C.R2GC_2951_2333,(0,0,0):C.R2GC_2951_2334,(0,0,2):C.R2GC_2951_2335})

V_280 = CTVertex(name = 'V_280',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.s] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.u] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2922_2228,(0,2,5):C.R2GC_2922_2229,(0,2,1):C.R2GC_2922_2230,(0,2,2):C.R2GC_2922_2231,(0,2,3):C.R2GC_2922_2232,(0,2,4):C.R2GC_2922_2233,(0,1,0):C.R2GC_2923_2234,(0,1,5):C.R2GC_2923_2235,(0,1,1):C.R2GC_2923_2236,(0,1,2):C.R2GC_2923_2237,(0,1,3):C.R2GC_2923_2238,(0,1,4):C.R2GC_2923_2239,(0,0,0):C.R2GC_2924_2240,(0,0,5):C.R2GC_2924_2241,(0,0,1):C.R2GC_2924_2242,(0,0,2):C.R2GC_2924_2243,(0,0,3):C.R2GC_2924_2244,(0,0,4):C.R2GC_2924_2245})

V_281 = CTVertex(name = 'V_281',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.c, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2233_1139,(0,1,0):C.R2GC_2235_1141,(0,0,0):C.R2GC_2234_1140})

V_282 = CTVertex(name = 'V_282',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2252_1160,(0,1,0):C.R2GC_2254_1162,(0,0,0):C.R2GC_2253_1161})

V_283 = CTVertex(name = 'V_283',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2360_1293,(0,1,0):C.R2GC_2362_1295,(0,0,0):C.R2GC_2361_1294})

V_284 = CTVertex(name = 'V_284',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.u, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2393_1338,(0,1,0):C.R2GC_2395_1340,(0,0,0):C.R2GC_2394_1339})

V_285 = CTVertex(name = 'V_285',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2466_1444,(0,2,1):C.R2GC_2466_1445,(0,1,0):C.R2GC_2468_1448,(0,1,1):C.R2GC_2468_1449,(0,0,0):C.R2GC_2467_1446,(0,0,1):C.R2GC_2467_1447})

V_286 = CTVertex(name = 'V_286',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.c, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_2962_2360,(0,2,0):C.R2GC_2962_2361,(0,2,2):C.R2GC_2962_2362,(0,1,1):C.R2GC_2961_2357,(0,1,0):C.R2GC_2961_2358,(0,1,2):C.R2GC_2961_2359,(0,0,1):C.R2GC_2963_2363,(0,0,0):C.R2GC_2963_2364,(0,0,2):C.R2GC_2963_2365})

V_287 = CTVertex(name = 'V_287',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_2974_2390,(0,2,0):C.R2GC_2974_2391,(0,2,2):C.R2GC_2974_2392,(0,1,1):C.R2GC_2973_2387,(0,1,0):C.R2GC_2973_2388,(0,1,2):C.R2GC_2973_2389,(0,0,1):C.R2GC_2975_2393,(0,0,0):C.R2GC_2975_2394,(0,0,2):C.R2GC_2975_2395})

V_288 = CTVertex(name = 'V_288',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.g, P.s, P.t] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,2):C.R2GC_2879_2110,(0,2,0):C.R2GC_2879_2111,(0,2,1):C.R2GC_2879_2112,(0,1,2):C.R2GC_2880_2113,(0,1,0):C.R2GC_2880_2114,(0,1,1):C.R2GC_2880_2115,(0,0,2):C.R2GC_2881_2116,(0,0,0):C.R2GC_2881_2117,(0,0,1):C.R2GC_2881_2118})

V_289 = CTVertex(name = 'V_289',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.u, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.g, P.s, P.t] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,2):C.R2GC_2882_2119,(0,2,0):C.R2GC_2882_2120,(0,2,1):C.R2GC_2882_2121,(0,1,2):C.R2GC_2883_2122,(0,1,0):C.R2GC_2883_2123,(0,1,1):C.R2GC_2883_2124,(0,0,2):C.R2GC_2884_2125,(0,0,0):C.R2GC_2884_2126,(0,0,1):C.R2GC_2884_2127})

V_290 = CTVertex(name = 'V_290',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.s] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.u] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2935_2288,(0,2,5):C.R2GC_2935_2289,(0,2,1):C.R2GC_2935_2290,(0,2,2):C.R2GC_2935_2291,(0,2,3):C.R2GC_2935_2292,(0,2,4):C.R2GC_2935_2293,(0,1,0):C.R2GC_2934_2282,(0,1,5):C.R2GC_2934_2283,(0,1,1):C.R2GC_2934_2284,(0,1,2):C.R2GC_2934_2285,(0,1,3):C.R2GC_2934_2286,(0,1,4):C.R2GC_2934_2287,(0,0,0):C.R2GC_2936_2294,(0,0,5):C.R2GC_2936_2295,(0,0,1):C.R2GC_2936_2296,(0,0,2):C.R2GC_2936_2297,(0,0,3):C.R2GC_2936_2298,(0,0,4):C.R2GC_2936_2299})

V_291 = CTVertex(name = 'V_291',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t], [P.c, P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2867_2086,(0,0,0):C.R2GC_2868_2087,(0,5,0):C.R2GC_2861_2080,(0,3,0):C.R2GC_2862_2081,(0,1,0):C.R2GC_2869_2088,(0,4,0):C.R2GC_2863_2082})

V_292 = CTVertex(name = 'V_292',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t], [P.c, P.g, P.s, P.t] ], [ [P.b, P.g, P.t, P.u], [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2870_2089,(0,0,0):C.R2GC_2871_2090,(0,5,1):C.R2GC_2940_2312,(0,3,1):C.R2GC_2941_2313,(0,1,0):C.R2GC_2872_2091,(0,4,1):C.R2GC_2942_2314})

V_293 = CTVertex(name = 'V_293',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t], [P.c, P.g, P.s, P.t] ], [ [P.b, P.g, P.t, P.u], [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,1):C.R2GC_2958_2354,(0,0,1):C.R2GC_2959_2355,(0,5,0):C.R2GC_2864_2083,(0,3,0):C.R2GC_2865_2084,(0,1,1):C.R2GC_2960_2356,(0,4,0):C.R2GC_2866_2085})

V_294 = CTVertex(name = 'V_294',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.t, P.u], [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2970_2384,(0,0,0):C.R2GC_2971_2385,(0,5,0):C.R2GC_2943_2315,(0,3,0):C.R2GC_2944_2316,(0,1,0):C.R2GC_2972_2386,(0,4,0):C.R2GC_2945_2317})

V_295 = CTVertex(name = 'V_295',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t], [P.c, P.g, P.s, P.t] ], [ [P.b, P.g, P.t, P.u], [P.d, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2964_2366,(0,2,1):C.R2GC_2964_2367,(0,0,0):C.R2GC_2965_2368,(0,0,1):C.R2GC_2965_2369,(0,5,0):C.R2GC_2952_2336,(0,5,1):C.R2GC_2952_2337,(0,3,0):C.R2GC_2953_2338,(0,3,1):C.R2GC_2953_2339,(0,1,0):C.R2GC_2966_2370,(0,1,1):C.R2GC_2966_2371,(0,4,0):C.R2GC_2954_2340,(0,4,1):C.R2GC_2954_2341})

V_296 = CTVertex(name = 'V_296',
                 type = 'R2',
                 particles = [ P.g, P.g, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVV1, L.VVV2, L.VVV3, L.VVV4, L.VVV5, L.VVV6 ],
                 loop_particles = [ [ [P.b], [P.d], [P.s] ], [ [P.c, P.t] ], [ [P.c], [P.t], [P.u] ], [ [P.t, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1036_38,(0,0,2):C.R2GC_1036_37,(0,0,1):C.R2GC_1067_71,(0,0,3):C.R2GC_1067_72,(0,1,0):C.R2GC_1036_37,(0,1,2):C.R2GC_1036_38,(0,1,1):C.R2GC_1064_65,(0,1,3):C.R2GC_1064_66,(0,2,1):C.R2GC_1065_67,(0,2,3):C.R2GC_1065_68,(0,3,1):C.R2GC_1065_67,(0,3,3):C.R2GC_1065_68,(0,4,1):C.R2GC_1066_69,(0,4,3):C.R2GC_1066_70,(0,5,1):C.R2GC_1066_69,(0,5,3):C.R2GC_1066_70})

V_297 = CTVertex(name = 'V_297',
                 type = 'R2',
                 particles = [ P.g, P.g, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVS1, L.VVS10, L.VVS11, L.VVS2, L.VVS3, L.VVS4, L.VVS5, L.VVS6, L.VVS7, L.VVS8, L.VVS9 ],
                 loop_particles = [ [ [P.c, P.t] ], [ [P.t] ], [ [P.t, P.u] ] ],
                 couplings = {(0,3,0):C.R2GC_1854_648,(0,3,2):C.R2GC_1854_649,(0,0,0):C.R2GC_1850_646,(0,0,2):C.R2GC_1850_647,(0,6,0):C.R2GC_1068_73,(0,6,2):C.R2GC_1068_74,(0,7,0):C.R2GC_1068_73,(0,7,2):C.R2GC_1068_74,(0,4,0):C.R2GC_1070_77,(0,4,2):C.R2GC_1070_78,(0,5,0):C.R2GC_1070_77,(0,5,2):C.R2GC_1070_78,(0,8,1):C.R2GC_799_2947,(0,9,0):C.R2GC_1070_77,(0,9,2):C.R2GC_1070_78,(0,1,0):C.R2GC_1069_75,(0,1,2):C.R2GC_1069_76,(0,10,0):C.R2GC_1069_75,(0,10,2):C.R2GC_1069_76,(0,2,0):C.R2GC_1850_646,(0,2,2):C.R2GC_1850_647})

V_298 = CTVertex(name = 'V_298',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_1810_614,(0,2,1):C.R2GC_1810_615,(0,1,0):C.R2GC_1810_614,(0,1,1):C.R2GC_1810_615,(0,5,0):C.R2GC_1723_490,(0,5,1):C.R2GC_1723_491,(0,4,0):C.R2GC_1723_490,(0,4,1):C.R2GC_1723_491,(0,0,0):C.R2GC_1811_616,(0,0,1):C.R2GC_1811_617,(0,3,0):C.R2GC_1724_492,(0,3,1):C.R2GC_1724_493})

V_299 = CTVertex(name = 'V_299',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_1771_563,(0,2,1):C.R2GC_1771_564,(0,1,0):C.R2GC_1771_563,(0,1,1):C.R2GC_1771_564,(0,5,0):C.R2GC_1725_494,(0,5,1):C.R2GC_1725_495,(0,4,0):C.R2GC_1725_494,(0,4,1):C.R2GC_1725_495,(0,0,0):C.R2GC_1772_565,(0,0,1):C.R2GC_1772_566,(0,3,0):C.R2GC_1726_496,(0,3,1):C.R2GC_1726_497})

V_300 = CTVertex(name = 'V_300',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1518_321,(0,2,1):C.R2GC_1518_322,(0,1,0):C.R2GC_1518_321,(0,1,1):C.R2GC_1518_322,(0,5,0):C.R2GC_1470_249,(0,5,1):C.R2GC_1470_250,(0,4,0):C.R2GC_1470_249,(0,4,1):C.R2GC_1470_250,(0,0,0):C.R2GC_1519_323,(0,0,1):C.R2GC_1519_324,(0,3,0):C.R2GC_1471_251,(0,3,1):C.R2GC_1471_252})

V_301 = CTVertex(name = 'V_301',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1565_380,(0,2,1):C.R2GC_1565_381,(0,1,0):C.R2GC_1565_380,(0,1,1):C.R2GC_1565_381,(0,5,0):C.R2GC_1468_245,(0,5,1):C.R2GC_1468_246,(0,4,0):C.R2GC_1468_245,(0,4,1):C.R2GC_1468_246,(0,0,0):C.R2GC_1566_382,(0,0,1):C.R2GC_1566_383,(0,3,0):C.R2GC_1469_247,(0,3,1):C.R2GC_1469_248})

V_302 = CTVertex(name = 'V_302',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1821_634,(0,0,1):C.R2GC_1821_635,(0,5,0):C.R2GC_1744_526,(0,5,1):C.R2GC_1744_527,(0,4,0):C.R2GC_1744_526,(0,4,1):C.R2GC_1744_527,(0,2,0):C.R2GC_1820_632,(0,2,1):C.R2GC_1820_633,(0,1,0):C.R2GC_1820_632,(0,1,1):C.R2GC_1820_633,(0,3,0):C.R2GC_1745_528,(0,3,1):C.R2GC_1745_529})

V_303 = CTVertex(name = 'V_303',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.R2GC_1782_583,(0,0,1):C.R2GC_1782_584,(0,5,0):C.R2GC_1746_530,(0,5,1):C.R2GC_1746_531,(0,4,0):C.R2GC_1746_530,(0,4,1):C.R2GC_1746_531,(0,2,0):C.R2GC_1781_581,(0,2,1):C.R2GC_1781_582,(0,1,0):C.R2GC_1781_581,(0,1,1):C.R2GC_1781_582,(0,3,0):C.R2GC_1747_532,(0,3,1):C.R2GC_1747_533})

V_304 = CTVertex(name = 'V_304',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1529_341,(0,0,1):C.R2GC_1529_342,(0,5,0):C.R2GC_1490_285,(0,5,1):C.R2GC_1490_286,(0,4,0):C.R2GC_1490_285,(0,4,1):C.R2GC_1490_286,(0,2,0):C.R2GC_1528_339,(0,2,1):C.R2GC_1528_340,(0,1,0):C.R2GC_1528_339,(0,1,1):C.R2GC_1528_340,(0,3,0):C.R2GC_1491_287,(0,3,1):C.R2GC_1491_288})

V_305 = CTVertex(name = 'V_305',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.R2GC_1576_400,(0,0,1):C.R2GC_1576_401,(0,5,0):C.R2GC_1488_281,(0,5,1):C.R2GC_1488_282,(0,4,0):C.R2GC_1488_281,(0,4,1):C.R2GC_1488_282,(0,2,0):C.R2GC_1575_398,(0,2,1):C.R2GC_1575_399,(0,1,0):C.R2GC_1575_398,(0,1,1):C.R2GC_1575_399,(0,3,0):C.R2GC_1489_283,(0,3,1):C.R2GC_1489_284})

V_306 = CTVertex(name = 'V_306',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2830_2042,(0,1,0):C.R2GC_2830_2042,(0,0,0):C.R2GC_2831_2043,(0,5,0):C.R2GC_2657_1750,(0,4,0):C.R2GC_2657_1750,(0,3,0):C.R2GC_2658_1751})

V_307 = CTVertex(name = 'V_307',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2739_1889,(0,2,1):C.R2GC_2739_1890,(0,1,0):C.R2GC_2739_1889,(0,1,1):C.R2GC_2739_1890,(0,0,0):C.R2GC_2740_1891,(0,0,1):C.R2GC_2740_1892,(0,5,0):C.R2GC_2619_1692,(0,5,1):C.R2GC_2619_1693,(0,4,0):C.R2GC_2619_1692,(0,4,1):C.R2GC_2619_1693,(0,3,0):C.R2GC_2620_1694,(0,3,1):C.R2GC_2620_1695})

V_308 = CTVertex(name = 'V_308',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2706_1835,(0,2,1):C.R2GC_2706_1836,(0,1,0):C.R2GC_2706_1835,(0,1,1):C.R2GC_2706_1836,(0,0,0):C.R2GC_2707_1837,(0,0,1):C.R2GC_2707_1838,(0,5,0):C.R2GC_2655_1746,(0,5,1):C.R2GC_2655_1747,(0,4,0):C.R2GC_2655_1746,(0,4,1):C.R2GC_2655_1747,(0,3,0):C.R2GC_2656_1748,(0,3,1):C.R2GC_2656_1749})

V_309 = CTVertex(name = 'V_309',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2814_2020,(0,2,1):C.R2GC_2814_2021,(0,1,0):C.R2GC_2814_2020,(0,1,1):C.R2GC_2814_2021,(0,0,0):C.R2GC_2815_2022,(0,0,1):C.R2GC_2815_2023,(0,5,0):C.R2GC_2682_1791,(0,5,1):C.R2GC_2682_1792,(0,4,0):C.R2GC_2682_1791,(0,4,1):C.R2GC_2682_1792,(0,3,0):C.R2GC_2683_1793,(0,3,1):C.R2GC_2683_1794})

V_310 = CTVertex(name = 'V_310',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2026_895,(0,1,0):C.R2GC_2026_895,(0,0,0):C.R2GC_2027_896,(0,5,0):C.R2GC_1978_824,(0,4,0):C.R2GC_1978_824,(0,3,0):C.R2GC_1979_825})

V_311 = CTVertex(name = 'V_311',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,2):C.R2GC_1825_642,(0,2,4):C.R2GC_1825_643,(0,2,0):C.R2GC_2907_2182,(0,2,3):C.R2GC_2907_2183,(0,2,1):C.R2GC_2907_2184,(0,1,2):C.R2GC_1825_642,(0,1,4):C.R2GC_1825_643,(0,1,0):C.R2GC_2907_2182,(0,1,3):C.R2GC_2907_2183,(0,1,1):C.R2GC_2907_2184,(0,0,2):C.R2GC_1826_644,(0,0,4):C.R2GC_1826_645,(0,0,0):C.R2GC_2906_2179,(0,0,3):C.R2GC_2906_2180,(0,0,1):C.R2GC_2906_2181,(0,5,2):C.R2GC_1755_548,(0,5,4):C.R2GC_1755_549,(0,5,0):C.R2GC_2902_2167,(0,5,3):C.R2GC_2902_2168,(0,5,1):C.R2GC_2902_2169,(0,4,2):C.R2GC_1755_548,(0,4,4):C.R2GC_1755_549,(0,4,0):C.R2GC_2902_2167,(0,4,3):C.R2GC_2902_2168,(0,4,1):C.R2GC_2902_2169,(0,3,2):C.R2GC_1754_546,(0,3,4):C.R2GC_1754_547,(0,3,0):C.R2GC_2901_2164,(0,3,3):C.R2GC_2901_2165,(0,3,1):C.R2GC_2901_2166})

V_312 = CTVertex(name = 'V_312',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,2):C.R2GC_1786_591,(0,2,4):C.R2GC_1786_592,(0,2,0):C.R2GC_2910_2191,(0,2,3):C.R2GC_2910_2192,(0,2,1):C.R2GC_2910_2193,(0,1,2):C.R2GC_1786_591,(0,1,4):C.R2GC_1786_592,(0,1,0):C.R2GC_2910_2191,(0,1,3):C.R2GC_2910_2192,(0,1,1):C.R2GC_2910_2193,(0,0,2):C.R2GC_1787_593,(0,0,4):C.R2GC_1787_594,(0,0,0):C.R2GC_2911_2194,(0,0,3):C.R2GC_2911_2195,(0,0,1):C.R2GC_2911_2196,(0,5,2):C.R2GC_1756_550,(0,5,4):C.R2GC_1756_551,(0,5,0):C.R2GC_2899_2158,(0,5,3):C.R2GC_2899_2159,(0,5,1):C.R2GC_2899_2160,(0,4,2):C.R2GC_1756_550,(0,4,4):C.R2GC_1756_551,(0,4,0):C.R2GC_2899_2158,(0,4,3):C.R2GC_2899_2159,(0,4,1):C.R2GC_2899_2160,(0,3,2):C.R2GC_1757_552,(0,3,4):C.R2GC_1757_553,(0,3,0):C.R2GC_2900_2161,(0,3,3):C.R2GC_2900_2162,(0,3,1):C.R2GC_2900_2163})

V_313 = CTVertex(name = 'V_313',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_1533_349,(0,2,3):C.R2GC_1533_350,(0,2,1):C.R2GC_2919_2218,(0,2,4):C.R2GC_2919_2219,(0,2,2):C.R2GC_2919_2220,(0,1,0):C.R2GC_1533_349,(0,1,3):C.R2GC_1533_350,(0,1,1):C.R2GC_2919_2218,(0,1,4):C.R2GC_2919_2219,(0,1,2):C.R2GC_2919_2220,(0,0,0):C.R2GC_1534_351,(0,0,3):C.R2GC_1534_352,(0,0,1):C.R2GC_2918_2215,(0,0,4):C.R2GC_2918_2216,(0,0,2):C.R2GC_2918_2217,(0,5,0):C.R2GC_1500_305,(0,5,3):C.R2GC_1500_306,(0,5,1):C.R2GC_2904_2173,(0,5,4):C.R2GC_2904_2174,(0,5,2):C.R2GC_2904_2175,(0,4,0):C.R2GC_1500_305,(0,4,3):C.R2GC_1500_306,(0,4,1):C.R2GC_2904_2173,(0,4,4):C.R2GC_2904_2174,(0,4,2):C.R2GC_2904_2175,(0,3,0):C.R2GC_1501_307,(0,3,3):C.R2GC_1501_308,(0,3,1):C.R2GC_2903_2170,(0,3,4):C.R2GC_2903_2171,(0,3,2):C.R2GC_2903_2172})

V_314 = CTVertex(name = 'V_314',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_1580_408,(0,2,3):C.R2GC_1580_409,(0,2,1):C.R2GC_2914_2203,(0,2,4):C.R2GC_2914_2204,(0,2,2):C.R2GC_2914_2205,(0,1,0):C.R2GC_1580_408,(0,1,3):C.R2GC_1580_409,(0,1,1):C.R2GC_2914_2203,(0,1,4):C.R2GC_2914_2204,(0,1,2):C.R2GC_2914_2205,(0,0,0):C.R2GC_1581_410,(0,0,3):C.R2GC_1581_411,(0,0,1):C.R2GC_2915_2206,(0,0,4):C.R2GC_2915_2207,(0,0,2):C.R2GC_2915_2208,(0,5,0):C.R2GC_1499_303,(0,5,3):C.R2GC_1499_304,(0,5,1):C.R2GC_2897_2152,(0,5,4):C.R2GC_2897_2153,(0,5,2):C.R2GC_2897_2154,(0,4,0):C.R2GC_1499_303,(0,4,3):C.R2GC_1499_304,(0,4,1):C.R2GC_2897_2152,(0,4,4):C.R2GC_2897_2153,(0,4,2):C.R2GC_2897_2154,(0,3,0):C.R2GC_1498_301,(0,3,3):C.R2GC_1498_302,(0,3,1):C.R2GC_2898_2155,(0,3,4):C.R2GC_2898_2156,(0,3,2):C.R2GC_2898_2157})

V_315 = CTVertex(name = 'V_315',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2835_2047,(0,1,0):C.R2GC_2835_2047,(0,0,0):C.R2GC_2836_2048,(0,5,0):C.R2GC_2667_1765,(0,4,0):C.R2GC_2667_1765,(0,3,0):C.R2GC_2668_1766})

V_316 = CTVertex(name = 'V_316',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2744_1899,(0,2,1):C.R2GC_2744_1900,(0,1,0):C.R2GC_2744_1899,(0,1,1):C.R2GC_2744_1900,(0,0,0):C.R2GC_2745_1901,(0,0,1):C.R2GC_2745_1902,(0,5,0):C.R2GC_2624_1702,(0,5,1):C.R2GC_2624_1703,(0,4,0):C.R2GC_2624_1702,(0,4,1):C.R2GC_2624_1703,(0,3,0):C.R2GC_2625_1704,(0,3,1):C.R2GC_2625_1705})

V_317 = CTVertex(name = 'V_317',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2711_1845,(0,2,1):C.R2GC_2711_1846,(0,1,0):C.R2GC_2711_1845,(0,1,1):C.R2GC_2711_1846,(0,0,0):C.R2GC_2712_1847,(0,0,1):C.R2GC_2712_1848,(0,5,0):C.R2GC_2665_1761,(0,5,1):C.R2GC_2665_1762,(0,4,0):C.R2GC_2665_1761,(0,4,1):C.R2GC_2665_1762,(0,3,0):C.R2GC_2666_1763,(0,3,1):C.R2GC_2666_1764})

V_318 = CTVertex(name = 'V_318',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2819_2030,(0,2,1):C.R2GC_2819_2031,(0,1,0):C.R2GC_2819_2030,(0,1,1):C.R2GC_2819_2031,(0,0,0):C.R2GC_2820_2032,(0,0,1):C.R2GC_2820_2033,(0,5,0):C.R2GC_2687_1801,(0,5,1):C.R2GC_2687_1802,(0,4,0):C.R2GC_2687_1801,(0,4,1):C.R2GC_2687_1802,(0,3,0):C.R2GC_2688_1803,(0,3,1):C.R2GC_2688_1804})

V_319 = CTVertex(name = 'V_319',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2031_900,(0,1,0):C.R2GC_2031_900,(0,0,0):C.R2GC_2032_901,(0,5,0):C.R2GC_1983_829,(0,4,0):C.R2GC_1983_829,(0,3,0):C.R2GC_1984_830})

V_320 = CTVertex(name = 'V_320',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_1815_624,(0,2,1):C.R2GC_1815_625,(0,1,0):C.R2GC_1814_622,(0,1,1):C.R2GC_1814_623,(0,0,0):C.R2GC_1816_626,(0,0,1):C.R2GC_1816_627,(0,5,0):C.R2GC_1732_508,(0,5,1):C.R2GC_1732_509,(0,4,0):C.R2GC_1731_506,(0,4,1):C.R2GC_1731_507,(0,3,0):C.R2GC_1733_510,(0,3,1):C.R2GC_1733_511})

V_321 = CTVertex(name = 'V_321',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_1777_575,(0,2,1):C.R2GC_1777_576,(0,1,0):C.R2GC_1776_573,(0,1,1):C.R2GC_1776_574,(0,0,0):C.R2GC_1775_571,(0,0,1):C.R2GC_1775_572,(0,5,0):C.R2GC_1736_516,(0,5,1):C.R2GC_1736_517,(0,4,0):C.R2GC_1735_514,(0,4,1):C.R2GC_1735_515,(0,3,0):C.R2GC_1734_512,(0,3,1):C.R2GC_1734_513})

V_322 = CTVertex(name = 'V_322',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1524_333,(0,2,1):C.R2GC_1524_334,(0,1,0):C.R2GC_1523_331,(0,1,1):C.R2GC_1523_332,(0,0,0):C.R2GC_1522_329,(0,0,1):C.R2GC_1522_330,(0,5,0):C.R2GC_1481_271,(0,5,1):C.R2GC_1481_272,(0,4,0):C.R2GC_1480_269,(0,4,1):C.R2GC_1480_270,(0,3,0):C.R2GC_1479_267,(0,3,1):C.R2GC_1479_268})

V_323 = CTVertex(name = 'V_323',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1570_390,(0,2,1):C.R2GC_1570_391,(0,1,0):C.R2GC_1569_388,(0,1,1):C.R2GC_1569_389,(0,0,0):C.R2GC_1571_392,(0,0,1):C.R2GC_1571_393,(0,5,0):C.R2GC_1477_263,(0,5,1):C.R2GC_1477_264,(0,4,0):C.R2GC_1476_261,(0,4,1):C.R2GC_1476_262,(0,3,0):C.R2GC_1478_265,(0,3,1):C.R2GC_1478_266})

V_324 = CTVertex(name = 'V_324',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_1822_636,(0,2,1):C.R2GC_1822_637,(0,1,0):C.R2GC_1823_638,(0,1,1):C.R2GC_1823_639,(0,0,0):C.R2GC_1824_640,(0,0,1):C.R2GC_1824_641,(0,5,0):C.R2GC_1748_534,(0,5,1):C.R2GC_1748_535,(0,4,0):C.R2GC_1749_536,(0,4,1):C.R2GC_1749_537,(0,3,0):C.R2GC_1750_538,(0,3,1):C.R2GC_1750_539})

V_325 = CTVertex(name = 'V_325',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.t, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_1784_587,(0,2,1):C.R2GC_1784_588,(0,1,0):C.R2GC_1785_589,(0,1,1):C.R2GC_1785_590,(0,0,0):C.R2GC_1783_585,(0,0,1):C.R2GC_1783_586,(0,5,0):C.R2GC_1752_542,(0,5,1):C.R2GC_1752_543,(0,4,0):C.R2GC_1753_544,(0,4,1):C.R2GC_1753_545,(0,3,0):C.R2GC_1751_540,(0,3,1):C.R2GC_1751_541})

V_326 = CTVertex(name = 'V_326',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1531_345,(0,2,1):C.R2GC_1531_346,(0,1,0):C.R2GC_1532_347,(0,1,1):C.R2GC_1532_348,(0,0,0):C.R2GC_1530_343,(0,0,1):C.R2GC_1530_344,(0,5,0):C.R2GC_1496_297,(0,5,1):C.R2GC_1496_298,(0,4,0):C.R2GC_1497_299,(0,4,1):C.R2GC_1497_300,(0,3,0):C.R2GC_1495_295,(0,3,1):C.R2GC_1495_296})

V_327 = CTVertex(name = 'V_327',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.t, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_1577_402,(0,2,1):C.R2GC_1577_403,(0,1,0):C.R2GC_1578_404,(0,1,1):C.R2GC_1578_405,(0,0,0):C.R2GC_1579_406,(0,0,1):C.R2GC_1579_407,(0,5,0):C.R2GC_1492_289,(0,5,1):C.R2GC_1492_290,(0,4,0):C.R2GC_1493_291,(0,4,1):C.R2GC_1493_292,(0,3,0):C.R2GC_1494_293,(0,3,1):C.R2GC_1494_294})

V_328 = CTVertex(name = 'V_328',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.u, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2833_2045,(0,1,0):C.R2GC_2834_2046,(0,0,0):C.R2GC_2832_2044,(0,5,0):C.R2GC_2663_1759,(0,4,0):C.R2GC_2664_1760,(0,3,0):C.R2GC_2662_1758})

V_329 = CTVertex(name = 'V_329',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.u, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2741_1893,(0,2,1):C.R2GC_2741_1894,(0,1,0):C.R2GC_2742_1895,(0,1,1):C.R2GC_2742_1896,(0,0,0):C.R2GC_2743_1897,(0,0,1):C.R2GC_2743_1898,(0,5,0):C.R2GC_2621_1696,(0,5,1):C.R2GC_2621_1697,(0,4,0):C.R2GC_2622_1698,(0,4,1):C.R2GC_2622_1699,(0,3,0):C.R2GC_2623_1700,(0,3,1):C.R2GC_2623_1701})

V_330 = CTVertex(name = 'V_330',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.c, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2709_1841,(0,2,1):C.R2GC_2709_1842,(0,1,0):C.R2GC_2710_1843,(0,1,1):C.R2GC_2710_1844,(0,0,0):C.R2GC_2708_1839,(0,0,1):C.R2GC_2708_1840,(0,5,0):C.R2GC_2660_1754,(0,5,1):C.R2GC_2660_1755,(0,4,0):C.R2GC_2661_1756,(0,4,1):C.R2GC_2661_1757,(0,3,0):C.R2GC_2659_1752,(0,3,1):C.R2GC_2659_1753})

V_331 = CTVertex(name = 'V_331',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.t, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2816_2024,(0,2,1):C.R2GC_2816_2025,(0,1,0):C.R2GC_2817_2026,(0,1,1):C.R2GC_2817_2027,(0,0,0):C.R2GC_2818_2028,(0,0,1):C.R2GC_2818_2029,(0,5,0):C.R2GC_2684_1795,(0,5,1):C.R2GC_2684_1796,(0,4,0):C.R2GC_2685_1797,(0,4,1):C.R2GC_2685_1798,(0,3,0):C.R2GC_2686_1799,(0,3,1):C.R2GC_2686_1800})

V_332 = CTVertex(name = 'V_332',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.c, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2028_897,(0,1,0):C.R2GC_2029_898,(0,0,0):C.R2GC_2030_899,(0,5,0):C.R2GC_1981_827,(0,4,0):C.R2GC_1982_828,(0,3,0):C.R2GC_1980_826})

V_333 = CTVertex(name = 'V_333',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.d, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2500_1496,(0,1,0):C.R2GC_2501_1497,(0,0,0):C.R2GC_2502_1498})

V_334 = CTVertex(name = 'V_334',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.b, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2450_1418,(0,2,1):C.R2GC_2450_1419,(0,1,0):C.R2GC_2451_1420,(0,1,1):C.R2GC_2451_1421,(0,0,0):C.R2GC_2452_1422,(0,0,1):C.R2GC_2452_1423})

V_335 = CTVertex(name = 'V_335',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.s, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,2,0):C.R2GC_1910_724,(0,1,0):C.R2GC_1911_725,(0,0,0):C.R2GC_1912_726})

V_336 = CTVertex(name = 'V_336',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.d, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2218_1124,(0,1,0):C.R2GC_2219_1125,(0,0,0):C.R2GC_2220_1126})

V_337 = CTVertex(name = 'V_337',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.d, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2215_1121,(0,1,0):C.R2GC_2216_1122,(0,0,0):C.R2GC_2217_1123})

V_338 = CTVertex(name = 'V_338',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.s, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2339_1262,(0,1,0):C.R2GC_2340_1263,(0,0,0):C.R2GC_2341_1264})

V_339 = CTVertex(name = 'V_339',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.s, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2336_1259,(0,1,0):C.R2GC_2337_1260,(0,0,0):C.R2GC_2338_1261})

V_340 = CTVertex(name = 'V_340',
                 type = 'R2',
                 particles = [ P.u__tilde__, P.b, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2138_1032,(0,1,0):C.R2GC_2139_1033,(0,0,0):C.R2GC_2140_1034})

V_341 = CTVertex(name = 'V_341',
                 type = 'R2',
                 particles = [ P.c__tilde__, P.b, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2135_1029,(0,1,0):C.R2GC_2136_1030,(0,0,0):C.R2GC_2137_1031})

V_342 = CTVertex(name = 'V_342',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.t, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2509_1508,(0,1,0):C.R2GC_2510_1509,(0,0,0):C.R2GC_2508_1507})

V_343 = CTVertex(name = 'V_343',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.R2GC_2470_1452,(0,2,1):C.R2GC_2470_1453,(0,1,0):C.R2GC_2471_1454,(0,1,1):C.R2GC_2471_1455,(0,0,0):C.R2GC_2469_1450,(0,0,1):C.R2GC_2469_1451})

V_344 = CTVertex(name = 'V_344',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.t, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,2,0):C.R2GC_1916_730,(0,1,0):C.R2GC_1917_731,(0,0,0):C.R2GC_1915_729})

V_345 = CTVertex(name = 'V_345',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.u, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2256_1164,(0,1,0):C.R2GC_2257_1165,(0,0,0):C.R2GC_2255_1163})

V_346 = CTVertex(name = 'V_346',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.c, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2237_1143,(0,1,0):C.R2GC_2238_1144,(0,0,0):C.R2GC_2236_1142})

V_347 = CTVertex(name = 'V_347',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.u, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2397_1342,(0,1,0):C.R2GC_2398_1343,(0,0,0):C.R2GC_2396_1341})

V_348 = CTVertex(name = 'V_348',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.c, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2364_1297,(0,1,0):C.R2GC_2365_1298,(0,0,0):C.R2GC_2363_1296})

V_349 = CTVertex(name = 'V_349',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.u, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2178_1080,(0,1,0):C.R2GC_2179_1081,(0,0,0):C.R2GC_2177_1079})

V_350 = CTVertex(name = 'V_350',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.c, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.R2GC_2164_1066,(0,1,0):C.R2GC_2165_1067,(0,0,0):C.R2GC_2163_1065})

V_351 = CTVertex(name = 'V_351',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.u__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF1, L.FFFF5, L.FFFF6, L.FFFF7 ],
                 loop_particles = [ [ [P.g, P.H, P.t] ], [ [P.g, P.H, P.u] ] ],
                 couplings = {(1,0,0):C.R2GC_2550_1572,(1,0,1):C.R2GC_2550_1573,(0,0,0):C.R2GC_2549_1570,(0,0,1):C.R2GC_2549_1571,(1,1,0):C.R2GC_2532_1546,(1,1,1):C.R2GC_2532_1547,(0,1,0):C.R2GC_2531_1544,(0,1,1):C.R2GC_2531_1545,(1,2,0):C.R2GC_2542_1562,(1,2,1):C.R2GC_2542_1563,(0,2,0):C.R2GC_2541_1560,(0,2,1):C.R2GC_2541_1561,(1,3,0):C.R2GC_2520_1524,(1,3,1):C.R2GC_2520_1525,(0,3,0):C.R2GC_2519_1522,(0,3,1):C.R2GC_2519_1523})

V_352 = CTVertex(name = 'V_352',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.u__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF1, L.FFFF5, L.FFFF6, L.FFFF7 ],
                 loop_particles = [ [ [P.c, P.g, P.H] ], [ [P.g, P.H, P.t] ], [ [P.g, P.H, P.u] ] ],
                 couplings = {(1,0,0):C.R2GC_2534_1551,(1,0,1):C.R2GC_2534_1552,(1,0,2):C.R2GC_2534_1553,(0,0,0):C.R2GC_2533_1548,(0,0,1):C.R2GC_2533_1549,(0,0,2):C.R2GC_2533_1550,(1,1,0):C.R2GC_2530_1541,(1,1,1):C.R2GC_2530_1542,(1,1,2):C.R2GC_2530_1543,(0,1,0):C.R2GC_2529_1538,(0,1,1):C.R2GC_2529_1539,(0,1,2):C.R2GC_2529_1540,(1,2,0):C.R2GC_2522_1529,(1,2,1):C.R2GC_2522_1530,(1,2,2):C.R2GC_2522_1531,(0,2,0):C.R2GC_2521_1526,(0,2,1):C.R2GC_2521_1527,(0,2,2):C.R2GC_2521_1528,(1,3,0):C.R2GC_2518_1519,(1,3,1):C.R2GC_2518_1520,(1,3,2):C.R2GC_2518_1521,(0,3,0):C.R2GC_2517_1516,(0,3,1):C.R2GC_2517_1517,(0,3,2):C.R2GC_2517_1518})

V_353 = CTVertex(name = 'V_353',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.u, P.c__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF1, L.FFFF5, L.FFFF6, L.FFFF7 ],
                 loop_particles = [ [ [P.c, P.g, P.H] ], [ [P.g, P.H, P.t] ], [ [P.g, P.H, P.u] ] ],
                 couplings = {(1,0,0):C.R2GC_2546_1567,(1,0,1):C.R2GC_2546_1568,(1,0,2):C.R2GC_2546_1569,(0,0,0):C.R2GC_2545_1564,(0,0,1):C.R2GC_2545_1565,(0,0,2):C.R2GC_2545_1566,(1,1,0):C.R2GC_2526_1535,(1,1,1):C.R2GC_2526_1536,(1,1,2):C.R2GC_2526_1537,(0,1,0):C.R2GC_2525_1532,(0,1,1):C.R2GC_2525_1533,(0,1,2):C.R2GC_2525_1534,(1,2,0):C.R2GC_2538_1557,(1,2,1):C.R2GC_2538_1558,(1,2,2):C.R2GC_2538_1559,(0,2,0):C.R2GC_2537_1554,(0,2,1):C.R2GC_2537_1555,(0,2,2):C.R2GC_2537_1556,(1,3,0):C.R2GC_2514_1513,(1,3,1):C.R2GC_2514_1514,(1,3,2):C.R2GC_2514_1515,(0,3,0):C.R2GC_2513_1510,(0,3,1):C.R2GC_2513_1511,(0,3,2):C.R2GC_2513_1512})

V_354 = CTVertex(name = 'V_354',
                 type = 'R2',
                 particles = [ P.t__tilde__, P.c, P.c__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF1, L.FFFF5, L.FFFF6, L.FFFF7 ],
                 loop_particles = [ [ [P.c, P.g, P.H] ], [ [P.g, P.H, P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_2293_1212,(1,0,1):C.R2GC_2293_1213,(0,0,0):C.R2GC_2292_1210,(0,0,1):C.R2GC_2292_1211,(1,1,0):C.R2GC_2289_1208,(1,1,1):C.R2GC_2289_1209,(0,1,0):C.R2GC_2288_1206,(0,1,1):C.R2GC_2288_1207,(1,2,0):C.R2GC_2285_1204,(1,2,1):C.R2GC_2285_1205,(0,2,0):C.R2GC_2284_1202,(0,2,1):C.R2GC_2284_1203,(1,3,0):C.R2GC_2277_1200,(1,3,1):C.R2GC_2277_1201,(0,3,0):C.R2GC_2276_1198,(0,3,1):C.R2GC_2276_1199})

V_355 = CTVertex(name = 'V_355',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.u, P.t__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_2273_1196,(1,0,1):C.R2GC_2273_1197,(0,0,0):C.R2GC_2272_1194,(0,0,1):C.R2GC_2272_1195})

V_356 = CTVertex(name = 'V_356',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.c, P.t__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_2267_1184,(1,0,1):C.R2GC_2267_1185,(0,0,0):C.R2GC_2266_1182,(0,0,1):C.R2GC_2266_1183})

V_357 = CTVertex(name = 'V_357',
                 type = 'R2',
                 particles = [ P.d__tilde__, P.t, P.t__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_2271_1192,(1,0,1):C.R2GC_2271_1193,(0,0,0):C.R2GC_2270_1190,(0,0,1):C.R2GC_2270_1191})

V_358 = CTVertex(name = 'V_358',
                 type = 'R2',
                 particles = [ P.s__tilde__, P.t, P.t__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_2269_1188,(1,0,1):C.R2GC_2269_1189,(0,0,0):C.R2GC_2268_1186,(0,0,1):C.R2GC_2268_1187})

V_359 = CTVertex(name = 'V_359',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.t__tilde__, P.d ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_2263_1176,(1,0,1):C.R2GC_2263_1177,(0,0,0):C.R2GC_2262_1174,(0,0,1):C.R2GC_2262_1175})

V_360 = CTVertex(name = 'V_360',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.t__tilde__, P.s ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_2261_1172,(1,0,1):C.R2GC_2261_1173,(0,0,0):C.R2GC_2260_1170,(0,0,1):C.R2GC_2260_1171})

V_361 = CTVertex(name = 'V_361',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.u__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_2265_1180,(1,0,1):C.R2GC_2265_1181,(0,0,0):C.R2GC_2264_1178,(0,0,1):C.R2GC_2264_1179})

V_362 = CTVertex(name = 'V_362',
                 type = 'R2',
                 particles = [ P.b__tilde__, P.t, P.c__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,0,0):C.R2GC_2259_1168,(1,0,1):C.R2GC_2259_1169,(0,0,0):C.R2GC_2258_1166,(0,0,1):C.R2GC_2258_1167})

V_363 = CTVertex(name = 'V_363',
                 type = 'UV',
                 particles = [ P.g, P.g, P.g ],
                 color = [ 'f(1,2,3)' ],
                 lorentz = [ L.VVV7 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2984_2638,(0,0,1):C.UVGC_2984_2639,(0,0,2):C.UVGC_2984_2640,(0,0,3):C.UVGC_2984_2641})

V_364 = CTVertex(name = 'V_364',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS6 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_3064_2817,(0,0,3):C.UVGC_3064_2818,(0,0,1):C.UVGC_3064_2819,(0,1,2):C.UVGC_3608_3835,(0,1,3):C.UVGC_3608_3836,(0,1,4):C.UVGC_3608_3837})

V_365 = CTVertex(name = 'V_365',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.d, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2058_1166,(0,0,0):C.UVGC_2058_1167})

V_366 = CTVertex(name = 'V_366',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.s, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2080_1198,(0,0,1):C.UVGC_2080_1199})

V_367 = CTVertex(name = 'V_367',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.b, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2073_1187,(0,0,0):C.UVGC_2073_1188})

V_368 = CTVertex(name = 'V_368',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.b, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2095_1219,(0,0,1):C.UVGC_2095_1220})

V_369 = CTVertex(name = 'V_369',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS15, L.FFS16, L.FFS22, L.FFS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,1,1):C.UVGC_1162_50,(0,0,1):C.UVGC_3063_2816,(0,2,0):C.UVGC_3667_4020,(0,2,1):C.UVGC_3667_4021,(0,2,2):C.UVGC_3667_4022,(0,3,0):C.UVGC_3609_3838,(0,3,1):C.UVGC_3609_3839,(0,3,2):C.UVGC_3609_3840})

V_370 = CTVertex(name = 'V_370',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS10, L.FFS17, L.FFS18, L.FFS20, L.FFS21, L.FFS26, L.FFS3, L.FFS5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,6,0):C.UVGC_3621_3882,(0,6,2):C.UVGC_3621_3883,(0,6,4):C.UVGC_3621_3884,(0,6,1):C.UVGC_3621_3885,(0,6,3):C.UVGC_3621_3886,(0,2,0):C.UVGC_3666_4015,(0,2,2):C.UVGC_3666_4016,(0,2,4):C.UVGC_3666_4017,(0,2,1):C.UVGC_3666_4018,(0,2,3):C.UVGC_3666_4019,(0,1,2):C.UVGC_1161_49,(0,3,2):C.UVGC_3062_2815,(0,7,1):C.UVGC_1990_1084,(0,7,3):C.UVGC_2793_2253,(0,0,1):C.UVGC_2838_2313,(0,0,3):C.UVGC_2838_2314,(0,5,1):C.UVGC_2761_2200,(0,5,3):C.UVGC_2761_2201,(0,4,1):C.UVGC_2033_1140,(0,4,3):C.UVGC_2764_2204})

V_371 = CTVertex(name = 'V_371',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS3, L.FFS6 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_3065_2820,(0,0,3):C.UVGC_3065_2821,(0,0,1):C.UVGC_3065_2822,(0,1,2):C.UVGC_1834_812,(0,1,3):C.UVGC_1834_813,(0,1,4):C.UVGC_1834_814})

V_372 = CTVertex(name = 'V_372',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS13, L.FFS18, L.FFS19, L.FFS22, L.FFS29, L.FFS3, L.FFS4, L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,7,0):[ C.UVGC_3334_3303, C.UVGC_1181_67 ],(0,7,2):C.UVGC_3334_3304,(0,7,1):C.UVGC_3334_3305,(0,3,0):[ C.UVGC_3321_3270, C.UVGC_1154_42 ],(0,3,2):C.UVGC_3321_3271,(0,3,1):C.UVGC_3321_3272,(0,5,0):C.UVGC_3464_3571,(0,1,0):C.UVGC_3034_2761,(0,6,2):C.UVGC_1759_707,(0,2,2):C.UVGC_1685_604,(0,0,0):C.UVGC_1177_63,(0,4,0):C.UVGC_1140_30})

V_373 = CTVertex(name = 'V_373',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS13, L.FFS18, L.FFS19, L.FFS25, L.FFS29, L.FFS3, L.FFS4, L.FFS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,5,0):C.UVGC_3513_3644,(0,5,2):C.UVGC_3513_3645,(0,5,1):C.UVGC_3513_3646,(0,1,0):C.UVGC_3233_3156,(0,1,2):C.UVGC_3233_3157,(0,1,1):C.UVGC_3233_3158,(0,6,2):C.UVGC_1758_706,(0,2,2):C.UVGC_1686_605,(0,7,0):C.UVGC_1180_66,(0,3,0):C.UVGC_1155_43,(0,0,0):C.UVGC_1176_62,(0,4,0):C.UVGC_1141_31})

V_374 = CTVertex(name = 'V_374',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS10, L.FFS13, L.FFS3, L.FFS4, L.FFS6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.UVGC_1597_517,(0,0,2):C.UVGC_1592_512,(0,3,0):C.UVGC_1390_262,(0,1,2):C.UVGC_1175_61,(0,4,0):C.UVGC_2433_1654,(0,4,2):C.UVGC_3335_3306,(0,4,1):C.UVGC_2433_1655})

V_375 = CTVertex(name = 'V_375',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS10, L.FFS13, L.FFS3, L.FFS4, L.FFS6 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.UVGC_1556_462,(0,0,2):C.UVGC_1551_457,(0,3,0):C.UVGC_1396_267,(0,1,2):C.UVGC_1166_53,(0,4,0):C.UVGC_1882_911,(0,4,2):C.UVGC_3327_3288,(0,4,1):C.UVGC_1882_912})

V_376 = CTVertex(name = 'V_376',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS13, L.FFS18, L.FFS19, L.FFS22, L.FFS29, L.FFS3, L.FFS4, L.FFS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,7,0):C.UVGC_3326_3285,(0,7,2):[ C.UVGC_3326_3286, C.UVGC_1172_58 ],(0,7,1):C.UVGC_3326_3287,(0,3,0):C.UVGC_3314_3253,(0,3,2):[ C.UVGC_3314_3254, C.UVGC_1148_36 ],(0,3,1):C.UVGC_3314_3255,(0,5,2):C.UVGC_3381_3417,(0,1,2):C.UVGC_3032_2758,(0,6,0):C.UVGC_1384_257,(0,2,0):C.UVGC_1345_223,(0,0,2):C.UVGC_1168_55,(0,4,2):C.UVGC_1138_28})

V_377 = CTVertex(name = 'V_377',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS13, L.FFS18, L.FFS19, L.FFS25, L.FFS29, L.FFS3, L.FFS4, L.FFS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,0):C.UVGC_3430_3489,(0,5,2):C.UVGC_3430_3490,(0,5,1):C.UVGC_3430_3491,(0,1,0):C.UVGC_3230_3147,(0,1,2):C.UVGC_3230_3148,(0,1,1):C.UVGC_3230_3149,(0,6,0):C.UVGC_1383_256,(0,2,0):C.UVGC_1346_224,(0,7,2):C.UVGC_1171_57,(0,3,2):C.UVGC_1149_37,(0,0,2):C.UVGC_1167_54,(0,4,2):C.UVGC_1139_29})

V_378 = CTVertex(name = 'V_378',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS13, L.FFS18, L.FFS19, L.FFS22, L.FFS29, L.FFS3, L.FFS4, L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,7,0):[ C.UVGC_3070_2833, C.UVGC_1153_41 ],(0,7,2):C.UVGC_3070_2834,(0,7,1):C.UVGC_3070_2835,(0,3,0):[ C.UVGC_3067_2826, C.UVGC_1185_71 ],(0,3,2):C.UVGC_3067_2827,(0,3,1):C.UVGC_3067_2828,(0,5,0):C.UVGC_1659_579,(0,1,0):C.UVGC_1609_528,(0,6,0):C.UVGC_1603_522,(0,2,0):C.UVGC_1402_273,(0,0,2):C.UVGC_1200_80,(0,4,2):C.UVGC_1190_73})

V_379 = CTVertex(name = 'V_379',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS13, L.FFS18, L.FFS19, L.FFS25, L.FFS29, L.FFS3, L.FFS4, L.FFS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,5,0):C.UVGC_3560_3749,(0,5,2):C.UVGC_3560_3750,(0,5,1):C.UVGC_3560_3751,(0,1,0):C.UVGC_3232_3153,(0,1,2):C.UVGC_3232_3154,(0,1,1):C.UVGC_3232_3155,(0,6,0):C.UVGC_1602_521,(0,2,0):C.UVGC_1403_274,(0,7,0):C.UVGC_1152_40,(0,3,0):C.UVGC_1186_72,(0,0,2):C.UVGC_1199_79,(0,4,2):C.UVGC_1191_74})

V_380 = CTVertex(name = 'V_380',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS10, L.FFS13, L.FFS3, L.FFS4, L.FFS6 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,4):C.UVGC_2172_1319,(0,2,2):C.UVGC_2172_1320,(0,3,0):C.UVGC_1341_220,(0,0,2):C.UVGC_1244_123,(0,1,5):C.UVGC_1198_78,(0,4,1):C.UVGC_2413_1623,(0,4,3):C.UVGC_2413_1624})

V_381 = CTVertex(name = 'V_381',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS13, L.FFS18, L.FFS19, L.FFS22, L.FFS29, L.FFS3, L.FFS4, L.FFS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,7,0):C.UVGC_3068_2829,(0,7,2):[ C.UVGC_3068_2830, C.UVGC_1151_39 ],(0,7,1):C.UVGC_3068_2831,(0,3,0):C.UVGC_3066_2823,(0,3,2):[ C.UVGC_3066_2824, C.UVGC_1164_51 ],(0,3,1):C.UVGC_3066_2825,(0,5,2):C.UVGC_1634_553,(0,1,2):C.UVGC_1608_527,(0,6,2):C.UVGC_1503_398,(0,2,2):C.UVGC_1399_270,(0,0,0):C.UVGC_1119_17,(0,4,0):C.UVGC_1112_10})

V_382 = CTVertex(name = 'V_382',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS13, L.FFS18, L.FFS19, L.FFS25, L.FFS29, L.FFS3, L.FFS4, L.FFS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,0):C.UVGC_3357_3361,(0,5,2):C.UVGC_3357_3362,(0,5,1):C.UVGC_3357_3363,(0,1,0):C.UVGC_3231_3150,(0,1,2):C.UVGC_3231_3151,(0,1,1):C.UVGC_3231_3152,(0,6,2):C.UVGC_1502_397,(0,2,2):C.UVGC_1400_271,(0,7,2):C.UVGC_1150_38,(0,3,2):C.UVGC_1165_52,(0,0,0):C.UVGC_1118_16,(0,4,0):C.UVGC_1113_11})

V_383 = CTVertex(name = 'V_383',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS10, L.FFS13, L.FFS3, L.FFS4, L.FFS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,5):C.UVGC_2141_1278,(0,2,3):C.UVGC_2141_1279,(0,3,1):C.UVGC_1340_219,(0,0,3):C.UVGC_1243_122,(0,1,4):C.UVGC_1117_15,(0,4,2):C.UVGC_1868_893,(0,4,0):C.UVGC_1868_894})

V_384 = CTVertex(name = 'V_384',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.d, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS19, L.FFS29, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_1447_312,(0,3,2):C.UVGC_1433_298,(0,1,2):C.UVGC_1401_272,(0,2,0):C.UVGC_1127_22,(0,4,0):C.UVGC_2413_1623,(0,4,2):C.UVGC_3071_2836,(0,4,1):C.UVGC_2413_1624})

V_385 = CTVertex(name = 'V_385',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.s, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS19, L.FFS29, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_1446_311,(0,3,2):C.UVGC_1432_297,(0,1,2):C.UVGC_1398_269,(0,2,0):C.UVGC_1133_25,(0,4,0):C.UVGC_1868_893,(0,4,2):C.UVGC_3069_2832,(0,4,1):C.UVGC_1868_894})

V_386 = CTVertex(name = 'V_386',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.b, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS19, L.FFS29, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,4):C.UVGC_3033_2759,(0,0,2):C.UVGC_3033_2760,(0,1,5):C.UVGC_1684_603,(0,3,2):C.UVGC_1251_130,(0,2,0):C.UVGC_1105_5,(0,4,1):C.UVGC_2433_1654,(0,4,3):C.UVGC_2433_1655})

V_387 = CTVertex(name = 'V_387',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.b, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS19, L.FFS29, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,5):C.UVGC_3031_2756,(0,0,3):C.UVGC_3031_2757,(0,1,4):C.UVGC_1344_222,(0,3,3):C.UVGC_1250_129,(0,2,1):C.UVGC_1104_4,(0,4,2):C.UVGC_1882_911,(0,4,0):C.UVGC_1882_912})

V_388 = CTVertex(name = 'V_388',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV15, L.FFV24, L.FFV41 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,1,1):C.UVGC_1110_9,(0,0,0):C.UVGC_3588_3797,(0,2,0):C.UVGC_3241_3166})

V_389 = CTVertex(name = 'V_389',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV15, L.FFV24, L.FFV41 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_1110_9,(0,0,1):C.UVGC_3426_3485,(0,2,1):C.UVGC_3238_3163})

V_390 = CTVertex(name = 'V_390',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV15, L.FFV27, L.FFV29, L.FFV41 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,1):C.UVGC_1110_9,(0,1,1):C.UVGC_2986_2643,(0,0,0):C.UVGC_3627_3902,(0,0,1):C.UVGC_3627_3903,(0,0,2):C.UVGC_3627_3904,(0,3,0):C.UVGC_3654_3979,(0,3,1):C.UVGC_3654_3980,(0,3,2):C.UVGC_3654_3981})

V_391 = CTVertex(name = 'V_391',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV15, L.FFV25, L.FFV27, L.FFV41 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,1,4):C.UVGC_1103_3,(0,2,0):C.UVGC_1329_208,(0,2,1):C.UVGC_1329_209,(0,2,2):C.UVGC_1329_210,(0,0,3):C.UVGC_3557_3746,(0,3,3):C.UVGC_3245_3170})

V_392 = CTVertex(name = 'V_392',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV15, L.FFV25, L.FFV27, L.FFV41 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.g] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,1):C.UVGC_1103_3,(0,2,0):C.UVGC_1329_208,(0,2,2):C.UVGC_1329_209,(0,2,3):C.UVGC_1329_210,(0,0,4):C.UVGC_3417_3475,(0,3,4):C.UVGC_3242_3167})

V_393 = CTVertex(name = 'V_393',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV15, L.FFV26, L.FFV27, L.FFV36, L.FFV41, L.FFV42 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.c, P.g] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,1,4):C.UVGC_1103_3,(0,2,0):C.UVGC_1329_208,(0,2,2):C.UVGC_1329_209,(0,2,3):C.UVGC_1329_210,(0,2,4):C.UVGC_2987_2644,(0,5,1):C.UVGC_1827_801,(0,5,4):C.UVGC_1827_802,(0,5,5):C.UVGC_1827_803,(0,3,1):C.UVGC_1828_804,(0,3,5):C.UVGC_1828_805,(0,0,1):C.UVGC_3618_3873,(0,0,4):C.UVGC_3618_3874,(0,0,5):C.UVGC_3618_3875,(0,4,1):C.UVGC_3655_3982,(0,4,4):C.UVGC_3655_3983,(0,4,5):C.UVGC_3655_3984})

V_394 = CTVertex(name = 'V_394',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV5 ],
                 loop_particles = [ [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,3):C.UVGC_1633_552,(0,2,0):C.UVGC_1907_952,(0,2,2):C.UVGC_1907_953,(0,0,1):C.UVGC_1256_135,(0,1,3):C.UVGC_1680_600})

V_395 = CTVertex(name = 'V_395',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_1907_952,(0,2,1):C.UVGC_1907_953,(0,2,3):C.UVGC_1622_541,(0,0,2):C.UVGC_1306_185,(0,1,3):C.UVGC_1645_565})

V_396 = CTVertex(name = 'V_396',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV5 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,1):C.UVGC_1907_952,(0,2,5):[ C.UVGC_3035_2762, C.UVGC_1837_822 ],(0,2,2):C.UVGC_1907_953,(0,2,4):C.UVGC_1837_821,(0,2,6):C.UVGC_1837_823,(0,0,0):C.UVGC_2465_1705,(0,0,3):C.UVGC_2465_1706,(0,1,4):C.UVGC_1846_848,(0,1,5):C.UVGC_1846_849,(0,1,6):C.UVGC_1846_850})

V_397 = CTVertex(name = 'V_397',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV22, L.FFV40, L.FFV41, L.FFV5 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,5,0):C.UVGC_3497_3626,(0,1,0):C.UVGC_3591_3800,(0,4,0):C.UVGC_3302_3235,(0,2,0):C.UVGC_3311_3250,(0,0,1):C.UVGC_2829_2304,(0,3,1):C.UVGC_2654_2035})

V_398 = CTVertex(name = 'V_398',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV22, L.FFV40, L.FFV41, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,1):C.UVGC_3413_3471,(0,1,1):C.UVGC_3429_3488,(0,4,1):C.UVGC_3299_3232,(0,2,1):C.UVGC_3308_3247,(0,0,0):C.UVGC_2025_1132,(0,3,0):C.UVGC_1977_1056})

V_399 = CTVertex(name = 'V_399',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV22, L.FFV30, L.FFV40, L.FFV41, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,6,2):[ C.UVGC_3036_2763, C.UVGC_3615_3863 ],(0,6,0):C.UVGC_3615_3862,(0,6,4):C.UVGC_3615_3864,(0,6,1):C.UVGC_3615_3865,(0,6,3):C.UVGC_3615_3866,(0,3,2):C.UVGC_3047_2786,(0,1,0):C.UVGC_3630_3911,(0,1,2):C.UVGC_3630_3912,(0,1,4):C.UVGC_3630_3913,(0,5,0):C.UVGC_3665_4012,(0,5,2):C.UVGC_3665_4013,(0,5,4):C.UVGC_3665_4014,(0,2,0):C.UVGC_3671_4042,(0,2,2):C.UVGC_3671_4043,(0,2,4):C.UVGC_3671_4044,(0,2,1):C.UVGC_3671_4045,(0,2,3):C.UVGC_3671_4046,(0,0,1):C.UVGC_2813_2282,(0,0,3):C.UVGC_2813_2283,(0,4,1):C.UVGC_2681_2073,(0,4,3):C.UVGC_2681_2074})

V_400 = CTVertex(name = 'V_400',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV24 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1102_2})

V_401 = CTVertex(name = 'V_401',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV24 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1102_2})

V_402 = CTVertex(name = 'V_402',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV24 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1102_2})

V_403 = CTVertex(name = 'V_403',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV25, L.FFV27 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.d, P.g] ], [ [P.g] ], [ [P.ghG] ] ],
                 couplings = {(0,0,1):C.UVGC_1103_3,(0,1,0):C.UVGC_1329_208,(0,1,2):C.UVGC_1329_209,(0,1,3):C.UVGC_1329_210})

V_404 = CTVertex(name = 'V_404',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV25, L.FFV27 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.s] ] ],
                 couplings = {(0,0,3):C.UVGC_1103_3,(0,1,0):C.UVGC_1329_208,(0,1,1):C.UVGC_1329_209,(0,1,2):C.UVGC_1329_210})

V_405 = CTVertex(name = 'V_405',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV25, L.FFV27 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b, P.g] ], [ [P.g] ], [ [P.ghG] ] ],
                 couplings = {(0,0,1):C.UVGC_1103_3,(0,1,0):C.UVGC_1329_208,(0,1,2):C.UVGC_1329_209,(0,1,3):C.UVGC_1329_210})

V_406 = CTVertex(name = 'V_406',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV40, L.FFV41, L.FFV5 ],
                 loop_particles = [ [ [P.d, P.g], [P.g, P.u] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,3):C.UVGC_3495_3624,(0,2,0):C.UVGC_1907_952,(0,2,2):C.UVGC_1907_953,(0,0,1):C.UVGC_1255_134,(0,1,3):C.UVGC_3283_3208})

V_407 = CTVertex(name = 'V_407',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV40, L.FFV41, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g], [P.g, P.s] ], [ [P.c, P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_1907_952,(0,2,1):C.UVGC_1907_953,(0,2,3):C.UVGC_3411_3469,(0,0,2):C.UVGC_1305_184,(0,1,3):C.UVGC_3276_3201})

V_408 = CTVertex(name = 'V_408',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV40, L.FFV41, L.FFV5 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,1):C.UVGC_1907_952,(0,2,5):[ C.UVGC_3035_2762, C.UVGC_3614_3860 ],(0,2,2):C.UVGC_1907_953,(0,2,4):C.UVGC_3614_3859,(0,2,6):C.UVGC_3614_3861,(0,0,0):C.UVGC_2446_1676,(0,0,3):C.UVGC_2446_1677,(0,1,4):C.UVGC_3662_4003,(0,1,5):C.UVGC_3662_4004,(0,1,6):C.UVGC_3662_4005})

V_409 = CTVertex(name = 'V_409',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.d, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2070_1181,(0,0,0):C.UVGC_2070_1182})

V_410 = CTVertex(name = 'V_410',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.s, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2092_1213,(0,0,1):C.UVGC_2092_1214})

V_411 = CTVertex(name = 'V_411',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.b, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2075_1191,(0,0,0):C.UVGC_2075_1192})

V_412 = CTVertex(name = 'V_412',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.b, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV5 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2099_1225,(0,0,1):C.UVGC_2099_1226})

V_413 = CTVertex(name = 'V_413',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS18, L.FFVS19, L.FFVS21, L.FFVS24, L.FFVS29, L.FFVS30, L.FFVS41, L.FFVS42, L.FFVS44, L.FFVS47, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,11,0):C.UVGC_1438_303,(0,5,0):C.UVGC_1604_523,(0,2,1):C.UVGC_2849_2327,(0,8,1):C.UVGC_2552_1875,(0,1,0):C.UVGC_3565_3762,(0,1,2):C.UVGC_3565_3763,(0,7,0):C.UVGC_2997_2663,(0,7,2):C.UVGC_2997_2664,(0,0,0):C.UVGC_1184_70,(0,3,2):C.UVGC_1184_70,(0,6,0):C.UVGC_1146_34,(0,9,2):C.UVGC_1146_34,(0,10,1):C.UVGC_2851_2330,(0,4,1):C.UVGC_2554_1877})

V_414 = CTVertex(name = 'V_414',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV17, L.FFV21, L.FFV22, L.FFV4, L.FFV40, L.FFV41, L.FFV44, L.FFV5 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t], [P.g, P.u] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,9,0):C.UVGC_3602_3821,(0,9,3):C.UVGC_3602_3822,(0,4,0):C.UVGC_3642_3946,(0,4,3):C.UVGC_3642_3947,(0,2,2):C.UVGC_2856_2337,(0,8,2):C.UVGC_2579_1923,(0,1,0):C.UVGC_3570_3773,(0,1,3):C.UVGC_3570_3774,(0,7,0):C.UVGC_3076_2847,(0,7,3):C.UVGC_3076_2848,(0,0,1):C.UVGC_1809_766,(0,6,1):C.UVGC_1721_635,(0,5,2):C.UVGC_2857_2338,(0,3,2):C.UVGC_2580_1924})

V_415 = CTVertex(name = 'V_415',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS18, L.FFVS19, L.FFVS21, L.FFVS24, L.FFVS29, L.FFVS30, L.FFVS41, L.FFVS42, L.FFVS44, L.FFVS47, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,11,2):[ C.UVGC_1462_323, C.UVGC_3206_3113 ],(0,11,0):C.UVGC_2905_2427,(0,11,3):[ C.UVGC_2905_2428, C.UVGC_3206_3115 ],(0,11,1):C.UVGC_2905_2429,(0,11,4):C.UVGC_3206_3114,(0,5,2):[ C.UVGC_1607_526, C.UVGC_3203_3104 ],(0,5,0):C.UVGC_2920_2472,(0,5,3):[ C.UVGC_2920_2473, C.UVGC_3203_3106 ],(0,5,1):C.UVGC_2920_2474,(0,5,4):C.UVGC_3203_3105,(0,2,3):C.UVGC_2853_2332,(0,8,3):C.UVGC_2571_1911,(0,1,2):C.UVGC_3569_3771,(0,1,4):C.UVGC_3569_3772,(0,7,2):C.UVGC_3053_2797,(0,7,4):C.UVGC_3053_2798,(0,0,2):C.UVGC_1606_525,(0,3,4):C.UVGC_1204_83,(0,6,2):C.UVGC_1160_48,(0,9,4):C.UVGC_1715_630,(0,10,3):C.UVGC_2855_2336,(0,4,3):C.UVGC_2573_1915})

V_416 = CTVertex(name = 'V_416',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV17, L.FFV21, L.FFV22, L.FFV4, L.FFV40, L.FFV41, L.FFV44, L.FFV5 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,9,0):[ C.UVGC_3605_3827, C.UVGC_3295_3226 ],(0,9,2):[ C.UVGC_3605_3828, C.UVGC_3295_3227 ],(0,9,1):C.UVGC_3295_3228,(0,4,0):[ C.UVGC_3653_3977, C.UVGC_3293_3220 ],(0,4,2):[ C.UVGC_3653_3978, C.UVGC_3293_3221 ],(0,4,1):C.UVGC_3293_3222,(0,2,1):C.UVGC_2859_2341,(0,8,1):C.UVGC_2591_1955,(0,1,0):C.UVGC_3573_3781,(0,1,2):C.UVGC_3573_3782,(0,7,0):C.UVGC_3209_3121,(0,7,2):C.UVGC_3209_3122,(0,0,0):C.UVGC_1819_785,(0,0,2):C.UVGC_1819_786,(0,6,0):C.UVGC_1742_674,(0,6,2):C.UVGC_1742_675,(0,5,1):C.UVGC_2860_2342,(0,3,1):C.UVGC_2592_1956})

V_417 = CTVertex(name = 'V_417',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS38, L.FFVS40, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,2):C.UVGC_3170_3053,(0,4,4):C.UVGC_2060_1170,(0,4,3):C.UVGC_2060_1171,(0,0,0):C.UVGC_2176_1323,(0,0,2):C.UVGC_3172_3055,(0,0,4):C.UVGC_3172_3056,(0,0,3):C.UVGC_3172_3057,(0,1,2):C.UVGC_3045_2782,(0,1,4):C.UVGC_3045_2783,(0,3,1):C.UVGC_1308_187,(0,2,3):C.UVGC_2569_1909})

V_418 = CTVertex(name = 'V_418',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,1):C.UVGC_3171_3054,(0,4,3):C.UVGC_2059_1168,(0,4,2):C.UVGC_2059_1169,(0,3,0):C.UVGC_2175_1322,(0,3,1):C.UVGC_3173_3058,(0,3,3):C.UVGC_3173_3059,(0,3,2):C.UVGC_3173_3060,(0,2,1):C.UVGC_3566_3764,(0,2,3):C.UVGC_3566_3765,(0,1,0):C.UVGC_1253_132,(0,0,2):C.UVGC_2852_2331})

V_419 = CTVertex(name = 'V_419',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS18, L.FFVS19, L.FFVS21, L.FFVS24, L.FFVS29, L.FFVS30, L.FFVS41, L.FFVS42, L.FFVS44, L.FFVS47, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,11,0):C.UVGC_1594_514,(0,5,0):C.UVGC_1439_304,(0,2,1):C.UVGC_2776_2217,(0,8,1):C.UVGC_2556_1880,(0,1,0):C.UVGC_3517_3654,(0,1,2):C.UVGC_3517_3655,(0,7,0):C.UVGC_3000_2669,(0,7,2):C.UVGC_3000_2670,(0,3,0):C.UVGC_1179_65,(0,0,2):C.UVGC_1179_65,(0,9,0):C.UVGC_1147_35,(0,6,2):C.UVGC_1147_35,(0,10,1):C.UVGC_2778_2220,(0,4,1):C.UVGC_2558_1882})

V_420 = CTVertex(name = 'V_420',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV17, L.FFV21, L.FFV22, L.FFV4, L.FFV40, L.FFV41, L.FFV44, L.FFV5 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t], [P.g, P.u] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,9,0):C.UVGC_3473_3599,(0,9,3):C.UVGC_3473_3600,(0,4,0):C.UVGC_3149_2995,(0,4,3):C.UVGC_3149_2996,(0,2,2):C.UVGC_2783_2227,(0,8,2):C.UVGC_2581_1925,(0,1,0):C.UVGC_3526_3673,(0,1,3):C.UVGC_3526_3674,(0,7,0):C.UVGC_3077_2849,(0,7,3):C.UVGC_3077_2850,(0,0,1):C.UVGC_1770_715,(0,6,1):C.UVGC_1722_636,(0,5,2):C.UVGC_2784_2228,(0,3,2):C.UVGC_2582_1926})

V_421 = CTVertex(name = 'V_421',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS18, L.FFVS19, L.FFVS21, L.FFVS24, L.FFVS29, L.FFVS30, L.FFVS41, L.FFVS42, L.FFVS44, L.FFVS47, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,11,2):[ C.UVGC_1601_520, C.UVGC_3338_3311 ],(0,11,0):C.UVGC_2916_2460,(0,11,3):[ C.UVGC_2916_2461, C.UVGC_3338_3313 ],(0,11,1):C.UVGC_2916_2462,(0,11,4):C.UVGC_3338_3312,(0,5,2):[ C.UVGC_1463_324, C.UVGC_3324_3279 ],(0,5,0):C.UVGC_2909_2439,(0,5,3):[ C.UVGC_2909_2440, C.UVGC_3324_3281 ],(0,5,1):C.UVGC_2909_2441,(0,5,4):C.UVGC_3324_3280,(0,2,3):C.UVGC_2780_2222,(0,8,3):C.UVGC_2575_1917,(0,1,2):C.UVGC_3525_3671,(0,1,4):C.UVGC_3525_3672,(0,7,2):C.UVGC_3060_2811,(0,7,4):C.UVGC_3060_2812,(0,3,2):C.UVGC_1183_69,(0,0,4):C.UVGC_1766_712,(0,9,2):C.UVGC_1457_318,(0,6,4):C.UVGC_1196_77,(0,10,3):C.UVGC_2782_2226,(0,4,3):C.UVGC_2577_1921})

V_422 = CTVertex(name = 'V_422',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV17, L.FFV21, L.FFV22, L.FFV4, L.FFV40, L.FFV41, L.FFV44, L.FFV5 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,9,0):[ C.UVGC_3477_3605, C.UVGC_3341_3318 ],(0,9,2):[ C.UVGC_3477_3606, C.UVGC_3341_3319 ],(0,9,1):C.UVGC_3341_3320,(0,4,0):[ C.UVGC_3229_3145, C.UVGC_3325_3282 ],(0,4,2):[ C.UVGC_3229_3146, C.UVGC_3325_3283 ],(0,4,1):C.UVGC_3325_3284,(0,2,1):C.UVGC_2786_2231,(0,8,1):C.UVGC_2593_1957,(0,1,0):C.UVGC_3532_3690,(0,1,2):C.UVGC_3532_3691,(0,7,0):C.UVGC_3211_3125,(0,7,2):C.UVGC_3211_3126,(0,0,0):C.UVGC_1780_734,(0,0,2):C.UVGC_1780_735,(0,6,0):C.UVGC_1743_676,(0,6,2):C.UVGC_1743_677,(0,5,1):C.UVGC_2787_2232,(0,3,1):C.UVGC_2594_1958})

V_423 = CTVertex(name = 'V_423',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,2):C.UVGC_3320_3269,(0,4,4):C.UVGC_2071_1183,(0,4,3):C.UVGC_2071_1184,(0,3,0):C.UVGC_2120_1248,(0,3,2):C.UVGC_3323_3276,(0,3,4):C.UVGC_3323_3277,(0,3,3):C.UVGC_3323_3278,(0,2,2):C.UVGC_3521_3662,(0,2,4):C.UVGC_3521_3663,(0,1,1):C.UVGC_1311_190,(0,0,3):C.UVGC_2779_2221})

V_424 = CTVertex(name = 'V_424',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS38, L.FFVS40, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,1):C.UVGC_3319_3268,(0,4,3):C.UVGC_2072_1185,(0,4,2):C.UVGC_2072_1186,(0,0,0):C.UVGC_2121_1249,(0,0,1):C.UVGC_3322_3273,(0,0,3):C.UVGC_3322_3274,(0,0,2):C.UVGC_3322_3275,(0,1,1):C.UVGC_3046_2784,(0,1,3):C.UVGC_3046_2785,(0,3,0):C.UVGC_1247_126,(0,2,2):C.UVGC_2570_1910})

V_425 = CTVertex(name = 'V_425',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.d, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2060_1170,(0,0,0):C.UVGC_2060_1171})

V_426 = CTVertex(name = 'V_426',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.s, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2082_1202,(0,0,1):C.UVGC_2082_1203})

V_427 = CTVertex(name = 'V_427',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.b, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2072_1185,(0,0,0):C.UVGC_2072_1186})

V_428 = CTVertex(name = 'V_428',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.b, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2094_1217,(0,0,1):C.UVGC_2094_1218})

V_429 = CTVertex(name = 'V_429',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.d, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2059_1168,(0,0,0):C.UVGC_2059_1169})

V_430 = CTVertex(name = 'V_430',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.s, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2081_1200,(0,0,1):C.UVGC_2081_1201})

V_431 = CTVertex(name = 'V_431',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.b, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2071_1183,(0,0,0):C.UVGC_2071_1184})

V_432 = CTVertex(name = 'V_432',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.b, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2093_1215,(0,0,1):C.UVGC_2093_1216})

V_433 = CTVertex(name = 'V_433',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.d, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2061_1172,(0,0,0):C.UVGC_2061_1173})

V_434 = CTVertex(name = 'V_434',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.s, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2083_1204,(0,0,1):C.UVGC_2083_1205})

V_435 = CTVertex(name = 'V_435',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.b, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.b, P.g], [P.d, P.g] ] ],
                 couplings = {(0,0,1):C.UVGC_2074_1189,(0,0,0):C.UVGC_2074_1190})

V_436 = CTVertex(name = 'V_436',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.b, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.s] ], [ [P.b, P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_2096_1221,(0,0,1):C.UVGC_2096_1222})

V_437 = CTVertex(name = 'V_437',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS24, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,2):C.UVGC_2105_1233,(0,4,1):C.UVGC_2414_1625,(0,4,3):C.UVGC_2414_1626,(0,0,3):C.UVGC_2481_1734,(0,2,1):C.UVGC_1829_806,(0,1,0):C.UVGC_1108_8,(0,3,4):C.UVGC_1201_81})

V_438 = CTVertex(name = 'V_438',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,3,0):C.UVGC_2418_1634,(0,3,2):C.UVGC_2418_1635,(0,0,2):C.UVGC_2482_1735,(0,2,0):C.UVGC_1830_807,(0,1,1):C.UVGC_1252_131})

V_439 = CTVertex(name = 'V_439',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV13, L.FFV14, L.FFV15, L.FFV5 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,3):C.UVGC_1613_532,(0,3,0):C.UVGC_2432_1652,(0,3,2):C.UVGC_2432_1653,(0,0,2):C.UVGC_2490_1749,(0,2,0):C.UVGC_1832_809,(0,1,1):C.UVGC_1254_133})

V_440 = CTVertex(name = 'V_440',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS14, L.FFVS17, L.FFVS18, L.FFVS24, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,2):C.UVGC_2123_1251,(0,4,1):C.UVGC_2423_1644,(0,4,3):C.UVGC_2423_1645,(0,1,3):C.UVGC_2486_1741,(0,0,1):C.UVGC_1831_808,(0,2,0):C.UVGC_1342_221,(0,3,4):C.UVGC_1203_82})

V_441 = CTVertex(name = 'V_441',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.d, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS38, L.FFVS40, L.FFVS41, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,2):[ C.UVGC_1437_302, C.UVGC_3073_2841 ],(0,4,0):C.UVGC_3073_2840,(0,4,1):C.UVGC_3073_2842,(0,1,1):C.UVGC_2180_1327,(0,0,0):C.UVGC_2995_2659,(0,0,2):C.UVGC_2995_2660,(0,3,0):C.UVGC_1128_23,(0,2,2):C.UVGC_1145_33})

V_442 = CTVertex(name = 'V_442',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.s, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS38, L.FFVS40, L.FFVS41, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,2):[ C.UVGC_1435_300, C.UVGC_3072_2838 ],(0,4,0):C.UVGC_3072_2837,(0,4,1):C.UVGC_3072_2839,(0,1,1):C.UVGC_2298_1483,(0,0,0):C.UVGC_2991_2651,(0,0,2):C.UVGC_2991_2652,(0,3,0):C.UVGC_1134_26,(0,2,2):C.UVGC_1143_32})

V_443 = CTVertex(name = 'V_443',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.d, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS40, L.FFVS42, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2186_1338,(0,1,0):C.UVGC_3043_2777,(0,1,3):C.UVGC_3043_2778,(0,2,2):C.UVGC_1307_186,(0,3,0):C.UVGC_2418_1634,(0,3,3):C.UVGC_3175_3062,(0,3,1):C.UVGC_2418_1635})

V_444 = CTVertex(name = 'V_444',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.s, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS40, L.FFVS42, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2305_1496,(0,1,1):C.UVGC_3039_2768,(0,1,3):C.UVGC_3039_2769,(0,2,0):C.UVGC_1237_116,(0,3,1):C.UVGC_1870_897,(0,3,3):C.UVGC_3174_3061,(0,3,2):C.UVGC_1870_898})

V_445 = CTVertex(name = 'V_445',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.d, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV39, L.FFV40, L.FFV41, L.FFV5 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,1):C.UVGC_2188_1340,(0,3,0):C.UVGC_2432_1652,(0,3,3):[ C.UVGC_3271_3196, C.UVGC_3604_3825 ],(0,3,1):C.UVGC_2432_1653,(0,3,4):C.UVGC_3604_3826,(0,2,0):C.UVGC_3177_3065,(0,2,3):C.UVGC_3177_3066,(0,1,2):C.UVGC_1309_188})

V_446 = CTVertex(name = 'V_446',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.s, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV39, L.FFV40, L.FFV41, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.s] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,3):C.UVGC_2308_1500,(0,3,2):C.UVGC_1881_909,(0,3,4):[ C.UVGC_3270_3195, C.UVGC_3595_3808 ],(0,3,3):C.UVGC_1881_910,(0,3,0):C.UVGC_3595_3807,(0,2,2):C.UVGC_3176_3063,(0,2,4):C.UVGC_3176_3064,(0,1,1):C.UVGC_1239_118})

V_447 = CTVertex(name = 'V_447',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.d, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS40, L.FFVS41, L.FFVS42, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,2):[ C.UVGC_1461_322, C.UVGC_3207_3117 ],(0,4,0):C.UVGC_3207_3116,(0,4,1):C.UVGC_3207_3118,(0,0,1):C.UVGC_2187_1339,(0,2,0):C.UVGC_3051_2793,(0,2,2):C.UVGC_3051_2794,(0,3,0):C.UVGC_1389_261,(0,1,2):C.UVGC_1159_47})

V_448 = CTVertex(name = 'V_448',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.s, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS40, L.FFVS41, L.FFVS42, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,2):[ C.UVGC_1459_320, C.UVGC_3205_3111 ],(0,4,0):C.UVGC_3205_3110,(0,4,1):C.UVGC_3205_3112,(0,0,1):C.UVGC_2306_1497,(0,2,0):C.UVGC_3055_2801,(0,2,2):C.UVGC_3055_2802,(0,3,0):C.UVGC_1395_266,(0,1,2):C.UVGC_1157_45})

V_449 = CTVertex(name = 'V_449',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.b, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS38, L.FFVS40, L.FFVS41, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,2):C.UVGC_2169_1316,(0,4,1):C.UVGC_2434_1656,(0,4,3):C.UVGC_2434_1657,(0,1,3):C.UVGC_2402_1603,(0,0,1):C.UVGC_1693_607,(0,3,0):C.UVGC_1106_6,(0,2,4):C.UVGC_1193_75})

V_450 = CTVertex(name = 'V_450',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.b, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS40, L.FFVS42, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,3,0):C.UVGC_2435_1658,(0,3,2):C.UVGC_2435_1659,(0,0,2):C.UVGC_2411_1621,(0,1,0):C.UVGC_1710_627,(0,2,1):C.UVGC_1246_125})

V_451 = CTVertex(name = 'V_451',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.b, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV39, L.FFV40, L.FFV41, L.FFV5 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,3):C.UVGC_3476_3604,(0,3,0):C.UVGC_2445_1674,(0,3,2):C.UVGC_2445_1675,(0,0,2):C.UVGC_2419_1636,(0,2,0):C.UVGC_1741_673,(0,1,1):C.UVGC_1248_127})

V_452 = CTVertex(name = 'V_452',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.b, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS40, L.FFVS41, L.FFVS42, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g] ], [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,2):C.UVGC_2171_1318,(0,4,1):C.UVGC_2436_1660,(0,4,3):C.UVGC_2436_1661,(0,0,3):C.UVGC_2412_1622,(0,2,1):C.UVGC_1716_631,(0,3,0):C.UVGC_1339_218,(0,1,4):C.UVGC_1195_76})

V_453 = CTVertex(name = 'V_453',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS24, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,2):[ C.UVGC_1593_513, C.UVGC_3336_3308 ],(0,4,0):C.UVGC_3336_3307,(0,4,1):C.UVGC_3336_3309,(0,0,1):C.UVGC_2240_1387,(0,2,0):C.UVGC_3514_3647,(0,2,2):C.UVGC_3514_3648,(0,1,0):C.UVGC_1129_24,(0,3,2):C.UVGC_1178_64})

V_454 = CTVertex(name = 'V_454',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS24, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,2):[ C.UVGC_1552_458, C.UVGC_3328_3290 ],(0,4,0):C.UVGC_3328_3289,(0,4,1):C.UVGC_3328_3291,(0,0,1):C.UVGC_2374_1574,(0,2,0):C.UVGC_3431_3492,(0,2,2):C.UVGC_3431_3493,(0,1,0):C.UVGC_1135_27,(0,3,2):C.UVGC_1169_56})

V_455 = CTVertex(name = 'V_455',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_2241_1388,(0,2,0):C.UVGC_3520_3660,(0,2,3):C.UVGC_3520_3661,(0,1,2):C.UVGC_1310_189,(0,3,0):C.UVGC_2435_1658,(0,3,3):C.UVGC_3337_3310,(0,3,1):C.UVGC_2435_1659})

V_456 = CTVertex(name = 'V_456',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_2375_1575,(0,2,1):C.UVGC_3437_3505,(0,2,3):C.UVGC_3437_3506,(0,1,0):C.UVGC_1240_119,(0,3,1):C.UVGC_1884_915,(0,3,3):C.UVGC_3329_3292,(0,3,2):C.UVGC_1884_916})

V_457 = CTVertex(name = 'V_457',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV13, L.FFV14, L.FFV15, L.FFV5 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,1):C.UVGC_2243_1390,(0,3,0):C.UVGC_2445_1674,(0,3,3):[ C.UVGC_3340_3317, C.UVGC_1833_810 ],(0,3,1):C.UVGC_2445_1675,(0,3,4):C.UVGC_1833_811,(0,2,0):C.UVGC_3531_3688,(0,2,3):C.UVGC_3531_3689,(0,1,2):C.UVGC_1312_191})

V_458 = CTVertex(name = 'V_458',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV13, L.FFV14, L.FFV15, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.s] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,3):C.UVGC_2379_1581,(0,3,2):C.UVGC_1886_919,(0,3,4):[ C.UVGC_3332_3299, C.UVGC_1636_556 ],(0,3,3):C.UVGC_1886_920,(0,3,0):C.UVGC_1636_555,(0,2,2):C.UVGC_3448_3533,(0,2,4):C.UVGC_3448_3534,(0,1,1):C.UVGC_1242_121})

V_459 = CTVertex(name = 'V_459',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS14, L.FFVS17, L.FFVS18, L.FFVS24, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,2):[ C.UVGC_1600_519, C.UVGC_3339_3315 ],(0,4,0):C.UVGC_3339_3314,(0,4,1):C.UVGC_3339_3316,(0,1,1):C.UVGC_2242_1389,(0,0,0):C.UVGC_3522_3664,(0,0,2):C.UVGC_3522_3665,(0,2,0):C.UVGC_1391_263,(0,3,2):C.UVGC_1182_68})

V_460 = CTVertex(name = 'V_460',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS14, L.FFVS17, L.FFVS18, L.FFVS24, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,2):[ C.UVGC_1561_465, C.UVGC_3331_3297 ],(0,4,0):C.UVGC_3331_3296,(0,4,1):C.UVGC_3331_3298,(0,1,1):C.UVGC_2377_1578,(0,0,0):C.UVGC_3439_3509,(0,0,2):C.UVGC_3439_3510,(0,2,0):C.UVGC_1397_268,(0,3,2):C.UVGC_1173_59})

V_461 = CTVertex(name = 'V_461',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS18, L.FFVS19, L.FFVS21, L.FFVS24, L.FFVS29, L.FFVS30, L.FFVS41, L.FFVS42, L.FFVS44, L.FFVS47, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,11,2):C.UVGC_1436_301,(0,5,2):C.UVGC_1510_403,(0,2,1):C.UVGC_1995_1089,(0,8,1):C.UVGC_1923_969,(0,1,0):C.UVGC_3360_3369,(0,1,2):C.UVGC_3360_3370,(0,7,0):C.UVGC_2993_2655,(0,7,2):C.UVGC_2993_2656,(0,3,0):C.UVGC_1121_19,(0,0,2):C.UVGC_1121_19,(0,9,0):C.UVGC_1115_13,(0,6,2):C.UVGC_1115_13,(0,10,1):C.UVGC_1997_1092,(0,4,1):C.UVGC_1925_971})

V_462 = CTVertex(name = 'V_462',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV17, L.FFV21, L.FFV22, L.FFV4, L.FFV40, L.FFV41, L.FFV44, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g], [P.g, P.t] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,0):C.UVGC_3593_3803,(0,9,3):C.UVGC_3593_3804,(0,4,0):C.UVGC_3598_3813,(0,4,3):C.UVGC_3598_3814,(0,2,2):C.UVGC_2002_1099,(0,8,2):C.UVGC_1948_1014,(0,1,0):C.UVGC_3365_3380,(0,1,3):C.UVGC_3365_3381,(0,7,0):C.UVGC_3075_2845,(0,7,3):C.UVGC_3075_2846,(0,0,1):C.UVGC_1517_409,(0,6,1):C.UVGC_1467_328,(0,5,2):C.UVGC_2003_1100,(0,3,2):C.UVGC_1949_1015})

V_463 = CTVertex(name = 'V_463',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS18, L.FFVS19, L.FFVS21, L.FFVS24, L.FFVS29, L.FFVS30, L.FFVS41, L.FFVS42, L.FFVS44, L.FFVS47, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,11,3):[ C.UVGC_1460_321, C.UVGC_3204_3108 ],(0,11,0):C.UVGC_3204_3107,(0,11,1):[ C.UVGC_3204_3109, C.UVGC_2917_2463 ],(0,11,4):C.UVGC_2917_2464,(0,11,2):C.UVGC_2917_2465,(0,5,3):[ C.UVGC_1515_407, C.UVGC_3202_3102 ],(0,5,0):C.UVGC_3202_3101,(0,5,1):[ C.UVGC_3202_3103, C.UVGC_2908_2436 ],(0,5,4):C.UVGC_2908_2437,(0,5,2):C.UVGC_2908_2438,(0,2,1):C.UVGC_1999_1094,(0,8,1):C.UVGC_1942_1006,(0,1,0):C.UVGC_3364_3378,(0,1,3):C.UVGC_3364_3379,(0,7,0):C.UVGC_3057_2805,(0,7,3):C.UVGC_3057_2806,(0,3,0):C.UVGC_1122_20,(0,0,3):C.UVGC_1512_405,(0,9,0):C.UVGC_1364_241,(0,6,3):C.UVGC_1158_46,(0,10,1):C.UVGC_2001_1098,(0,4,1):C.UVGC_1944_1010})

V_464 = CTVertex(name = 'V_464',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV17, L.FFV21, L.FFV22, L.FFV4, L.FFV40, L.FFV41, L.FFV44, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,0):[ C.UVGC_3596_3809, C.UVGC_3294_3223 ],(0,9,2):[ C.UVGC_3596_3810, C.UVGC_3294_3224 ],(0,9,1):C.UVGC_3294_3225,(0,4,0):[ C.UVGC_3600_3817, C.UVGC_3292_3217 ],(0,4,2):[ C.UVGC_3600_3818, C.UVGC_3292_3218 ],(0,4,1):C.UVGC_3292_3219,(0,2,1):C.UVGC_2004_1101,(0,8,1):C.UVGC_1959_1044,(0,1,0):C.UVGC_3368_3388,(0,1,2):C.UVGC_3368_3389,(0,7,0):C.UVGC_3210_3123,(0,7,2):C.UVGC_3210_3124,(0,0,0):C.UVGC_1527_428,(0,0,2):C.UVGC_1527_429,(0,6,0):C.UVGC_1487_367,(0,6,2):C.UVGC_1487_368,(0,5,1):C.UVGC_2005_1102,(0,3,1):C.UVGC_1960_1045})

V_465 = CTVertex(name = 'V_465',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS38, L.FFVS40, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,1):C.UVGC_2082_1202,(0,4,4):C.UVGC_3166_3045,(0,4,3):C.UVGC_2082_1203,(0,0,1):C.UVGC_3168_3047,(0,0,4):C.UVGC_3168_3048,(0,0,3):C.UVGC_3168_3049,(0,0,0):C.UVGC_2151_1289,(0,1,1):C.UVGC_3041_2773,(0,1,4):C.UVGC_3041_2774,(0,3,2):C.UVGC_1238_117,(0,2,3):C.UVGC_1937_999})

V_466 = CTVertex(name = 'V_466',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,1):C.UVGC_2081_1200,(0,4,3):C.UVGC_3167_3046,(0,4,2):C.UVGC_2081_1201,(0,3,1):C.UVGC_3169_3050,(0,3,3):C.UVGC_3169_3051,(0,3,2):C.UVGC_3169_3052,(0,3,0):C.UVGC_2150_1288,(0,2,1):C.UVGC_3361_3371,(0,2,3):C.UVGC_3361_3372,(0,1,0):C.UVGC_1249_128,(0,0,2):C.UVGC_1998_1093})

V_467 = CTVertex(name = 'V_467',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS18, L.FFVS19, L.FFVS21, L.FFVS24, L.FFVS29, L.FFVS30, L.FFVS41, L.FFVS42, L.FFVS44, L.FFVS47, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,11,2):C.UVGC_1553_459,(0,5,2):C.UVGC_1434_299,(0,2,1):C.UVGC_2039_1146,(0,8,1):C.UVGC_1919_964,(0,1,0):C.UVGC_3434_3499,(0,1,2):C.UVGC_3434_3500,(0,7,0):C.UVGC_2989_2647,(0,7,2):C.UVGC_2989_2648,(0,0,0):C.UVGC_1123_21,(0,3,2):C.UVGC_1123_21,(0,6,0):C.UVGC_1114_12,(0,9,2):C.UVGC_1114_12,(0,10,1):C.UVGC_2041_1149,(0,4,1):C.UVGC_1921_966})

V_468 = CTVertex(name = 'V_468',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV17, L.FFV21, L.FFV22, L.FFV4, L.FFV40, L.FFV41, L.FFV44, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g], [P.g, P.t] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,0):C.UVGC_3390_3445,(0,9,3):C.UVGC_3390_3446,(0,4,0):C.UVGC_3148_2993,(0,4,3):C.UVGC_3148_2994,(0,2,2):C.UVGC_2046_1156,(0,8,2):C.UVGC_1946_1012,(0,1,0):C.UVGC_3443_3518,(0,1,3):C.UVGC_3443_3519,(0,7,0):C.UVGC_3074_2843,(0,7,3):C.UVGC_3074_2844,(0,0,1):C.UVGC_1564_468,(0,6,1):C.UVGC_1466_327,(0,5,2):C.UVGC_2047_1157,(0,3,2):C.UVGC_1947_1013})

V_469 = CTVertex(name = 'V_469',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS18, L.FFVS19, L.FFVS21, L.FFVS24, L.FFVS29, L.FFVS30, L.FFVS41, L.FFVS42, L.FFVS44, L.FFVS47, L.FFVS5, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,11,3):[ C.UVGC_1562_466, C.UVGC_3330_3294 ],(0,11,0):C.UVGC_3330_3293,(0,11,1):[ C.UVGC_3330_3295, C.UVGC_2912_2448 ],(0,11,4):C.UVGC_2912_2449,(0,11,2):C.UVGC_2912_2450,(0,5,3):[ C.UVGC_1458_319, C.UVGC_3317_3263 ],(0,5,0):C.UVGC_3317_3262,(0,5,1):[ C.UVGC_3317_3264, C.UVGC_2913_2451 ],(0,5,4):C.UVGC_2913_2452,(0,5,2):C.UVGC_2913_2453,(0,2,1):C.UVGC_2043_1151,(0,8,1):C.UVGC_1938_1000,(0,1,0):C.UVGC_3442_3516,(0,1,3):C.UVGC_3442_3517,(0,7,0):C.UVGC_3049_2789,(0,7,3):C.UVGC_3049_2790,(0,0,0):C.UVGC_1385_258,(0,3,3):C.UVGC_1174_60,(0,6,0):C.UVGC_1116_14,(0,9,3):C.UVGC_1452_315,(0,10,1):C.UVGC_2045_1155,(0,4,1):C.UVGC_1940_1004})

V_470 = CTVertex(name = 'V_470',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV14, L.FFV15, L.FFV17, L.FFV21, L.FFV22, L.FFV4, L.FFV40, L.FFV41, L.FFV44, L.FFV5 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,0):[ C.UVGC_3394_3451, C.UVGC_3333_3300 ],(0,9,2):[ C.UVGC_3394_3452, C.UVGC_3333_3301 ],(0,9,1):C.UVGC_3333_3302,(0,4,0):[ C.UVGC_3228_3143, C.UVGC_3318_3265 ],(0,4,2):[ C.UVGC_3228_3144, C.UVGC_3318_3266 ],(0,4,1):C.UVGC_3318_3267,(0,2,1):C.UVGC_2049_1160,(0,8,1):C.UVGC_1957_1042,(0,1,0):C.UVGC_3449_3535,(0,1,2):C.UVGC_3449_3536,(0,7,0):C.UVGC_3208_3119,(0,7,2):C.UVGC_3208_3120,(0,0,0):C.UVGC_1574_487,(0,0,2):C.UVGC_1574_488,(0,6,0):C.UVGC_1486_365,(0,6,2):C.UVGC_1486_366,(0,5,1):C.UVGC_2050_1161,(0,3,1):C.UVGC_1958_1043})

V_471 = CTVertex(name = 'V_471',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS30, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,1):C.UVGC_2093_1215,(0,4,4):C.UVGC_3313_3252,(0,4,3):C.UVGC_2093_1216,(0,3,1):C.UVGC_3316_3259,(0,3,4):C.UVGC_3316_3260,(0,3,3):C.UVGC_3316_3261,(0,3,0):C.UVGC_2118_1246,(0,2,1):C.UVGC_3438_3507,(0,2,4):C.UVGC_3438_3508,(0,1,2):C.UVGC_1241_120,(0,0,3):C.UVGC_2042_1150})

V_472 = CTVertex(name = 'V_472',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS30, L.FFVS38, L.FFVS40, L.FFVS47, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,1):C.UVGC_2094_1217,(0,4,3):C.UVGC_3312_3251,(0,4,2):C.UVGC_2094_1218,(0,0,1):C.UVGC_3315_3256,(0,0,3):C.UVGC_3315_3257,(0,0,2):C.UVGC_3315_3258,(0,0,0):C.UVGC_2119_1247,(0,1,1):C.UVGC_3037_2764,(0,1,3):C.UVGC_3037_2765,(0,3,0):C.UVGC_1245_124,(0,2,2):C.UVGC_1936_998})

V_473 = CTVertex(name = 'V_473',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS17, L.FFVS18, L.FFVS19, L.FFVS24, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,4,2):C.UVGC_1869_895,(0,4,0):C.UVGC_1869_896,(0,4,3):C.UVGC_2104_1232,(0,0,0):C.UVGC_1896_936,(0,2,2):C.UVGC_1381_254,(0,1,1):C.UVGC_1107_7,(0,3,4):C.UVGC_1120_18})

V_474 = CTVertex(name = 'V_474',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_1897_937,(0,2,0):C.UVGC_2146_1284,(0,2,2):C.UVGC_2147_1286,(0,7,0):C.UVGC_2146_1284,(0,7,2):C.UVGC_2146_1285,(0,5,1):C.UVGC_1870_897,(0,5,0):C.UVGC_1870_898,(0,0,1):C.UVGC_1897_937,(0,0,0):C.UVGC_1897_938,(0,3,1):C.UVGC_1897_937,(0,3,0):C.UVGC_1897_938,(0,4,1):C.UVGC_1897_937,(0,4,0):C.UVGC_1897_938,(0,1,2):C.UVGC_2148_1287,(0,6,2):C.UVGC_2148_1287})

V_475 = CTVertex(name = 'V_475',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV2, L.FFV3, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,6,1):C.UVGC_1902_947,(0,6,0):C.UVGC_2156_1299,(0,6,2):C.UVGC_2157_1301,(0,4,0):C.UVGC_2156_1299,(0,4,2):C.UVGC_2156_1300,(0,2,1):C.UVGC_1881_909,(0,2,0):C.UVGC_1881_910,(0,2,3):C.UVGC_1612_531,(0,0,1):C.UVGC_1902_947,(0,0,0):C.UVGC_1902_948,(0,1,1):C.UVGC_1902_947,(0,1,0):C.UVGC_1902_948,(0,5,2):C.UVGC_2158_1302,(0,3,2):C.UVGC_2158_1302})

V_476 = CTVertex(name = 'V_476',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS3, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,2,1):C.UVGC_1901_945,(0,2,4):C.UVGC_1901_946,(0,2,0):C.UVGC_1900_944,(0,9,1):C.UVGC_1900_942,(0,9,4):C.UVGC_1900_943,(0,9,0):C.UVGC_1900_944,(0,7,2):C.UVGC_1872_901,(0,7,0):C.UVGC_1872_902,(0,7,3):C.UVGC_2122_1250,(0,0,0):C.UVGC_1899_941,(0,4,0):C.UVGC_1899_941,(0,6,0):C.UVGC_1899_941,(0,1,1):C.UVGC_1377_252,(0,1,4):C.UVGC_1375_251,(0,8,1):C.UVGC_1377_252,(0,8,4):C.UVGC_1375_251,(0,3,4):C.UVGC_1375_251,(0,10,4):C.UVGC_1375_251,(0,5,2):C.UVGC_1382_255})

V_477 = CTVertex(name = 'V_477',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.b, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,8,1):C.UVGC_1856_868,(0,8,4):C.UVGC_1856_869,(0,8,0):C.UVGC_1855_867,(0,5,1):C.UVGC_1855_865,(0,5,4):C.UVGC_1855_866,(0,5,0):C.UVGC_1855_867,(0,10,2):C.UVGC_1883_913,(0,10,0):C.UVGC_1883_914,(0,10,3):C.UVGC_2166_1313,(0,0,0):C.UVGC_1857_870,(0,1,0):C.UVGC_1857_870,(0,3,0):C.UVGC_1857_870,(0,7,1):C.UVGC_1331_212,(0,7,4):C.UVGC_1347_225,(0,4,1):C.UVGC_1331_212,(0,4,4):C.UVGC_1347_225,(0,9,1):C.UVGC_1331_212,(0,6,1):C.UVGC_1331_212,(0,2,2):C.UVGC_1349_226})

V_478 = CTVertex(name = 'V_478',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.b, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,7,1):C.UVGC_1864_884,(0,7,0):C.UVGC_2108_1236,(0,7,2):C.UVGC_2109_1238,(0,4,0):C.UVGC_2108_1236,(0,4,2):C.UVGC_2108_1237,(0,9,1):C.UVGC_1884_915,(0,9,0):C.UVGC_1884_916,(0,0,1):C.UVGC_1864_884,(0,0,0):C.UVGC_1864_885,(0,1,1):C.UVGC_1864_884,(0,1,0):C.UVGC_1864_885,(0,2,1):C.UVGC_1864_884,(0,2,0):C.UVGC_1864_885,(0,6,2):C.UVGC_2110_1239,(0,3,2):C.UVGC_2110_1239,(0,8,2):C.UVGC_2110_1239,(0,5,2):C.UVGC_2110_1239})

V_479 = CTVertex(name = 'V_479',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.b, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV19, L.FFV20, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,1):C.UVGC_1871_899,(0,5,0):C.UVGC_2128_1261,(0,5,2):C.UVGC_2129_1263,(0,3,0):C.UVGC_2128_1261,(0,3,2):C.UVGC_2128_1262,(0,6,1):C.UVGC_1886_919,(0,6,0):C.UVGC_1886_920,(0,6,3):C.UVGC_3393_3450,(0,0,1):C.UVGC_1871_899,(0,0,0):C.UVGC_1871_900,(0,1,1):C.UVGC_1871_899,(0,1,0):C.UVGC_1871_900,(0,4,2):C.UVGC_2130_1264,(0,2,2):C.UVGC_2130_1264})

V_480 = CTVertex(name = 'V_480',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.b, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ] ],
                 couplings = {(0,7,1):C.UVGC_1865_886,(0,7,4):C.UVGC_1865_887,(0,7,0):C.UVGC_1865_888,(0,4,1):C.UVGC_1866_889,(0,4,4):C.UVGC_1866_890,(0,4,0):C.UVGC_1865_888,(0,9,2):C.UVGC_1885_917,(0,9,0):C.UVGC_1885_918,(0,9,3):C.UVGC_2168_1315,(0,0,2):C.UVGC_1867_891,(0,0,0):C.UVGC_1867_892,(0,1,2):C.UVGC_1867_891,(0,1,0):C.UVGC_1867_892,(0,2,2):C.UVGC_1867_891,(0,2,0):C.UVGC_1867_892,(0,6,1):C.UVGC_1337_217,(0,6,4):C.UVGC_1360_239,(0,3,1):C.UVGC_1337_217,(0,3,4):C.UVGC_1360_239,(0,8,1):C.UVGC_1337_217,(0,5,1):C.UVGC_1337_217})

V_481 = CTVertex(name = 'V_481',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t], [P.g, P.u] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_3563_3757,(0,2,3):C.UVGC_3563_3758,(0,2,2):C.UVGC_2850_2329,(0,13,0):C.UVGC_2998_2665,(0,13,3):C.UVGC_2998_2666,(0,13,2):C.UVGC_2551_1874,(0,17,1):C.UVGC_2850_2328,(0,17,2):C.UVGC_2850_2329,(0,10,1):C.UVGC_2551_1873,(0,10,2):C.UVGC_2551_1874,(0,1,1):C.UVGC_1788_750,(0,12,1):C.UVGC_1687_606,(0,16,1):C.UVGC_1788_750,(0,9,1):C.UVGC_1687_606,(0,3,3):C.UVGC_1788_750,(0,14,3):C.UVGC_1687_606,(0,18,3):C.UVGC_1788_750,(0,11,3):C.UVGC_1687_606,(0,0,0):C.UVGC_3564_3759,(0,0,3):C.UVGC_3564_3760,(0,0,2):C.UVGC_3564_3761,(0,4,0):C.UVGC_3564_3759,(0,4,3):C.UVGC_3564_3760,(0,4,2):C.UVGC_3564_3761,(0,15,0):C.UVGC_3564_3759,(0,15,3):C.UVGC_3564_3760,(0,15,2):C.UVGC_3564_3761,(0,7,0):C.UVGC_2996_2661,(0,7,3):C.UVGC_2996_2662,(0,5,2):C.UVGC_2553_1876,(0,6,2):C.UVGC_2553_1876,(0,8,2):C.UVGC_2553_1876})

V_482 = CTVertex(name = 'V_482',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_3567_3766,(0,2,2):C.UVGC_3567_3767,(0,2,1):C.UVGC_2854_2335,(0,13,0):C.UVGC_3052_2795,(0,13,2):C.UVGC_3052_2796,(0,13,1):C.UVGC_2572_1914,(0,17,0):C.UVGC_2854_2333,(0,17,2):C.UVGC_2854_2334,(0,17,1):C.UVGC_2854_2335,(0,10,0):C.UVGC_2572_1912,(0,10,2):C.UVGC_2572_1913,(0,10,1):C.UVGC_2572_1914,(0,1,0):C.UVGC_1806_764,(0,1,2):C.UVGC_1804_763,(0,12,0):C.UVGC_1713_629,(0,12,2):C.UVGC_1711_628,(0,16,0):C.UVGC_1806_764,(0,16,2):C.UVGC_1804_763,(0,9,0):C.UVGC_1713_629,(0,9,2):C.UVGC_1711_628,(0,3,2):C.UVGC_1804_763,(0,14,2):C.UVGC_1711_628,(0,18,2):C.UVGC_1804_763,(0,11,2):C.UVGC_1711_628,(0,0,0):C.UVGC_3568_3768,(0,0,2):C.UVGC_3568_3769,(0,0,1):C.UVGC_3568_3770,(0,4,0):C.UVGC_3568_3768,(0,4,2):C.UVGC_3568_3769,(0,4,1):C.UVGC_3568_3770,(0,15,0):C.UVGC_3568_3768,(0,15,2):C.UVGC_3568_3769,(0,15,1):C.UVGC_3568_3770,(0,7,0):C.UVGC_3054_2799,(0,7,2):C.UVGC_3054_2800,(0,5,1):C.UVGC_2574_1916,(0,6,1):C.UVGC_2574_1916,(0,8,1):C.UVGC_2574_1916})

V_483 = CTVertex(name = 'V_483',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,6):C.UVGC_3572_3778,(0,2,8):C.UVGC_3572_3779,(0,2,1):C.UVGC_3572_3780,(0,2,7):C.UVGC_2858_2340,(0,2,3):C.UVGC_2399_1599,(0,2,5):C.UVGC_2399_1600,(0,1,1):C.UVGC_2858_2339,(0,1,7):C.UVGC_2858_2340,(0,1,3):C.UVGC_2400_1601,(0,1,5):C.UVGC_2400_1602,(0,5,2):C.UVGC_2885_2383,(0,5,0):C.UVGC_2885_2384,(0,5,6):C.UVGC_3159_3021,(0,5,8):C.UVGC_3159_3022,(0,5,4):C.UVGC_3159_3023,(0,5,7):C.UVGC_2589_1952,(0,4,2):C.UVGC_2887_2387,(0,4,0):C.UVGC_2887_2388,(0,4,4):C.UVGC_2589_1951,(0,4,7):C.UVGC_2589_1952,(0,0,6):C.UVGC_3571_3775,(0,0,8):C.UVGC_3571_3776,(0,0,1):C.UVGC_2858_2339,(0,0,7):C.UVGC_3571_3777,(0,0,3):C.UVGC_2400_1601,(0,0,5):C.UVGC_2400_1602,(0,3,2):C.UVGC_2886_2385,(0,3,0):C.UVGC_2886_2386,(0,3,6):C.UVGC_3158_3017,(0,3,8):C.UVGC_3158_3018,(0,3,4):C.UVGC_3158_3019,(0,3,7):C.UVGC_3158_3020})

V_484 = CTVertex(name = 'V_484',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.d, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,8,0):C.UVGC_3044_2779,(0,8,3):C.UVGC_3044_2780,(0,8,1):C.UVGC_2495_1757,(0,8,2):C.UVGC_3044_2781,(0,5,1):C.UVGC_2495_1757,(0,5,2):C.UVGC_2495_1758,(0,0,1):C.UVGC_2185_1337,(0,1,1):C.UVGC_2185_1337,(0,3,1):C.UVGC_2185_1337,(0,7,2):C.UVGC_2496_1759,(0,4,2):C.UVGC_2496_1759,(0,9,2):C.UVGC_2496_1759,(0,6,2):C.UVGC_2496_1759,(0,2,0):C.UVGC_3042_2775,(0,2,3):C.UVGC_3042_2776})

V_485 = CTVertex(name = 'V_485',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.s, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS48 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,8,1):C.UVGC_3040_2770,(0,8,3):C.UVGC_3040_2771,(0,8,0):C.UVGC_3040_2772,(0,8,2):C.UVGC_2304_1495,(0,5,0):C.UVGC_2304_1494,(0,5,2):C.UVGC_2304_1495,(0,0,2):C.UVGC_2303_1493,(0,1,2):C.UVGC_2303_1493,(0,3,2):C.UVGC_2303_1493,(0,7,0):C.UVGC_1908_954,(0,4,0):C.UVGC_1908_954,(0,9,0):C.UVGC_1908_954,(0,6,0):C.UVGC_1908_954,(0,2,1):C.UVGC_3038_2766,(0,2,3):C.UVGC_3038_2767,(0,10,3):C.UVGC_1156_44})

V_486 = CTVertex(name = 'V_486',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.d, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3180_3074,(0,2,3):C.UVGC_3180_3075,(0,2,1):C.UVGC_2499_1762,(0,2,2):C.UVGC_3180_3076,(0,1,1):C.UVGC_2499_1762,(0,1,2):C.UVGC_2499_1763,(0,0,0):C.UVGC_3181_3077,(0,0,3):C.UVGC_3181_3078,(0,0,1):C.UVGC_3181_3079,(0,0,2):C.UVGC_3181_3080})

V_487 = CTVertex(name = 'V_487',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.s, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_3178_3067,(0,2,3):C.UVGC_3178_3068,(0,2,0):C.UVGC_3178_3069,(0,2,2):C.UVGC_2309_1502,(0,1,0):C.UVGC_2309_1501,(0,1,2):C.UVGC_2309_1502,(0,0,1):C.UVGC_3179_3070,(0,0,3):C.UVGC_3179_3071,(0,0,0):C.UVGC_3179_3072,(0,0,2):C.UVGC_3179_3073})

V_488 = CTVertex(name = 'V_488',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.d, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.d, P.g] ], [ [P.b, P.d, P.g, P.u] ], [ [P.c, P.d, P.g, P.t] ], [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,3):C.UVGC_3165_3042,(0,2,7):C.UVGC_3165_3043,(0,2,4):C.UVGC_2498_1760,(0,2,6):C.UVGC_3165_3044,(0,2,0):C.UVGC_2955_2576,(0,2,1):C.UVGC_2955_2577,(0,2,2):C.UVGC_2955_2578,(0,2,5):C.UVGC_2955_2579,(0,1,4):C.UVGC_2498_1760,(0,1,6):C.UVGC_2498_1761,(0,1,0):C.UVGC_2956_2580,(0,1,1):C.UVGC_2956_2581,(0,1,2):C.UVGC_2956_2582,(0,1,5):C.UVGC_2956_2583,(0,0,3):C.UVGC_3164_3038,(0,0,7):C.UVGC_3164_3039,(0,0,4):C.UVGC_3164_3040,(0,0,6):C.UVGC_3164_3041,(0,0,0):C.UVGC_2957_2584,(0,0,1):C.UVGC_2957_2585,(0,0,2):C.UVGC_2956_2582,(0,0,5):C.UVGC_2956_2583})

V_489 = CTVertex(name = 'V_489',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.s, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.s] ], [ [P.b, P.g, P.s, P.u] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.s, P.t, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,4):C.UVGC_3163_3035,(0,2,7):C.UVGC_3163_3036,(0,2,2):C.UVGC_3163_3037,(0,2,5):C.UVGC_2307_1499,(0,2,0):C.UVGC_2976_2616,(0,2,1):C.UVGC_2976_2617,(0,2,3):C.UVGC_2976_2618,(0,2,6):C.UVGC_2976_2619,(0,1,2):C.UVGC_2307_1498,(0,1,5):C.UVGC_2307_1499,(0,1,0):C.UVGC_2977_2620,(0,1,1):C.UVGC_2977_2621,(0,1,3):C.UVGC_2977_2622,(0,1,6):C.UVGC_2977_2623,(0,0,4):C.UVGC_3162_3031,(0,0,7):C.UVGC_3162_3032,(0,0,2):C.UVGC_3162_3033,(0,0,5):C.UVGC_3162_3034,(0,0,0):C.UVGC_2978_2624,(0,0,1):C.UVGC_2978_2625,(0,0,3):C.UVGC_2977_2622,(0,0,6):C.UVGC_2977_2623})

V_490 = CTVertex(name = 'V_490',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.g, P.t, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,1):C.UVGC_2788_2236,(0,2,4):C.UVGC_3535_3700,(0,2,5):C.UVGC_3538_3706,(0,2,7):C.UVGC_3535_3699,(0,13,1):C.UVGC_2560_1890,(0,13,4):C.UVGC_3023_2735,(0,13,5):C.UVGC_3023_2736,(0,13,7):C.UVGC_3023_2737,(0,17,2):C.UVGC_2788_2233,(0,17,5):C.UVGC_2788_2234,(0,17,7):C.UVGC_2788_2235,(0,17,6):C.UVGC_2788_2236,(0,10,2):C.UVGC_2560_1887,(0,10,5):C.UVGC_2560_1888,(0,10,7):C.UVGC_2560_1889,(0,10,6):C.UVGC_2560_1890,(0,0,0):C.UVGC_2788_2236,(0,0,2):C.UVGC_3535_3699,(0,0,3):C.UVGC_3535_3700,(0,0,5):C.UVGC_3535_3701,(0,0,7):C.UVGC_3535_3702,(0,0,6):C.UVGC_3535_3703,(0,4,0):C.UVGC_2788_2236,(0,4,2):C.UVGC_3535_3699,(0,4,3):C.UVGC_3535_3700,(0,4,5):C.UVGC_3536_3704,(0,4,7):C.UVGC_3537_3705,(0,4,6):C.UVGC_3535_3703,(0,15,0):C.UVGC_2788_2236,(0,15,2):C.UVGC_3535_3699,(0,15,3):C.UVGC_3535_3700,(0,15,5):C.UVGC_3536_3704,(0,15,7):C.UVGC_3535_3702,(0,15,6):C.UVGC_3535_3703,(0,5,5):C.UVGC_2563_1895,(0,5,7):C.UVGC_2562_1894,(0,5,6):C.UVGC_2561_1893,(0,6,5):C.UVGC_2561_1891,(0,6,7):C.UVGC_2561_1892,(0,6,6):C.UVGC_2561_1893,(0,8,5):C.UVGC_2561_1891,(0,8,7):C.UVGC_2562_1894,(0,8,6):C.UVGC_2561_1893,(0,1,5):C.UVGC_1797_756,(0,1,7):C.UVGC_1796_755,(0,12,5):C.UVGC_1701_615,(0,12,7):C.UVGC_1700_614,(0,16,5):C.UVGC_1803_762,(0,16,7):C.UVGC_1802_761,(0,9,5):C.UVGC_1695_609,(0,9,7):C.UVGC_1694_608,(0,3,7):C.UVGC_1796_755,(0,14,7):C.UVGC_1700_614,(0,18,7):C.UVGC_1802_761,(0,11,7):C.UVGC_1694_608,(0,7,0):C.UVGC_3018_2722,(0,7,2):C.UVGC_3018_2723,(0,7,3):C.UVGC_3018_2724,(0,7,5):C.UVGC_3018_2725,(0,7,7):C.UVGC_3018_2726})

V_491 = CTVertex(name = 'V_491',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.g, P.t, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,1):C.UVGC_2789_2240,(0,2,4):C.UVGC_3539_3707,(0,2,5):C.UVGC_3539_3708,(0,2,7):C.UVGC_3539_3709,(0,13,1):C.UVGC_2559_1886,(0,13,4):C.UVGC_3019_2727,(0,13,5):C.UVGC_3019_2728,(0,13,7):C.UVGC_3019_2729,(0,18,2):C.UVGC_2789_2237,(0,18,5):C.UVGC_2789_2238,(0,18,7):C.UVGC_2789_2239,(0,18,6):C.UVGC_2789_2240,(0,10,2):C.UVGC_2559_1883,(0,10,5):C.UVGC_2559_1884,(0,10,7):C.UVGC_2559_1885,(0,10,6):C.UVGC_2559_1886,(0,16,5):C.UVGC_1444_309,(0,0,0):C.UVGC_2789_2240,(0,0,2):C.UVGC_3539_3709,(0,0,3):C.UVGC_3539_3707,(0,0,5):C.UVGC_3540_3710,(0,0,7):C.UVGC_3540_3711,(0,0,6):C.UVGC_3540_3712,(0,4,0):C.UVGC_2789_2240,(0,4,2):C.UVGC_3539_3709,(0,4,3):C.UVGC_3539_3707,(0,4,5):C.UVGC_3541_3713,(0,4,7):C.UVGC_3542_3714,(0,4,6):C.UVGC_3540_3712,(0,15,0):C.UVGC_2789_2240,(0,15,2):C.UVGC_3539_3709,(0,15,3):C.UVGC_3539_3707,(0,15,5):C.UVGC_3541_3713,(0,15,7):C.UVGC_3540_3711,(0,15,6):C.UVGC_3540_3712,(0,8,5):C.UVGC_1605_524,(0,5,0):C.UVGC_2559_1886,(0,5,2):C.UVGC_3019_2729,(0,5,3):C.UVGC_3019_2727,(0,5,5):C.UVGC_3020_2730,(0,5,7):C.UVGC_3020_2731,(0,5,6):C.UVGC_3020_2732,(0,6,0):C.UVGC_2559_1886,(0,6,2):C.UVGC_3019_2729,(0,6,3):C.UVGC_3019_2727,(0,6,5):C.UVGC_3021_2733,(0,6,7):C.UVGC_3022_2734,(0,6,6):C.UVGC_3020_2732,(0,7,0):C.UVGC_2559_1886,(0,7,2):C.UVGC_3019_2729,(0,7,3):C.UVGC_3019_2727,(0,7,5):C.UVGC_3021_2733,(0,7,7):C.UVGC_3020_2731,(0,7,6):C.UVGC_3020_2732,(0,1,5):C.UVGC_1801_760,(0,1,7):C.UVGC_1800_759,(0,12,5):C.UVGC_1699_613,(0,12,7):C.UVGC_1698_612,(0,17,5):C.UVGC_1799_758,(0,17,7):C.UVGC_1798_757,(0,9,5):C.UVGC_1697_611,(0,9,7):C.UVGC_1696_610,(0,3,7):C.UVGC_1800_759,(0,14,7):C.UVGC_1698_612,(0,19,7):C.UVGC_1798_757,(0,11,7):C.UVGC_1696_610})

V_492 = CTVertex(name = 'V_492',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV22, L.FFV3, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.g, P.t, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,13,1):C.UVGC_2790_2244,(0,13,4):C.UVGC_3543_3715,(0,13,5):C.UVGC_3543_3716,(0,13,7):C.UVGC_3543_3717,(0,8,1):C.UVGC_2583_1930,(0,8,4):C.UVGC_3084_2868,(0,8,5):C.UVGC_3084_2869,(0,8,7):C.UVGC_3084_2870,(0,11,2):C.UVGC_2790_2241,(0,11,5):C.UVGC_2790_2242,(0,11,7):C.UVGC_2790_2243,(0,11,6):C.UVGC_2790_2244,(0,6,2):C.UVGC_2583_1927,(0,6,5):C.UVGC_2583_1928,(0,6,7):C.UVGC_2583_1929,(0,6,6):C.UVGC_2583_1930,(0,9,5):C.UVGC_3603_3823,(0,9,7):C.UVGC_3603_3824,(0,1,0):C.UVGC_2790_2244,(0,1,2):C.UVGC_3543_3717,(0,1,3):C.UVGC_3543_3715,(0,1,5):C.UVGC_3544_3718,(0,1,7):C.UVGC_3544_3719,(0,1,6):C.UVGC_3544_3720,(0,4,0):C.UVGC_2790_2244,(0,4,2):C.UVGC_3543_3717,(0,4,3):C.UVGC_3543_3715,(0,4,5):C.UVGC_3545_3721,(0,4,7):C.UVGC_3545_3722,(0,4,6):C.UVGC_3544_3720,(0,3,5):C.UVGC_3643_3948,(0,3,7):C.UVGC_3643_3949,(0,0,0):C.UVGC_2583_1930,(0,0,2):C.UVGC_3084_2870,(0,0,3):C.UVGC_3084_2868,(0,0,5):C.UVGC_3085_2871,(0,0,7):C.UVGC_3085_2872,(0,0,6):C.UVGC_3085_2873,(0,2,0):C.UVGC_2583_1930,(0,2,2):C.UVGC_3084_2870,(0,2,3):C.UVGC_3084_2868,(0,2,5):C.UVGC_3086_2874,(0,2,7):C.UVGC_3086_2875,(0,2,6):C.UVGC_3085_2873,(0,12,5):C.UVGC_1813_773,(0,12,7):C.UVGC_1813_774,(0,7,5):C.UVGC_1728_647,(0,7,7):C.UVGC_1728_648,(0,10,5):C.UVGC_1812_771,(0,10,7):C.UVGC_1812_772,(0,5,5):C.UVGC_1727_645,(0,5,7):C.UVGC_1727_646})

V_493 = CTVertex(name = 'V_493',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.g, P.t, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,4):C.UVGC_1817_781,(0,2,6):C.UVGC_1817_782,(2,2,0):C.UVGC_3548_3731,(2,2,2):C.UVGC_3548_3732,(2,2,3):C.UVGC_3548_3733,(2,2,4):C.UVGC_3548_3734,(2,2,6):C.UVGC_3548_3735,(2,2,5):C.UVGC_2792_2252,(1,2,0):C.UVGC_3546_3723,(1,2,2):C.UVGC_3546_3724,(1,2,3):C.UVGC_3546_3725,(1,2,4):C.UVGC_3546_3726,(1,2,6):C.UVGC_3546_3727,(1,2,5):C.UVGC_2791_2248,(0,1,4):C.UVGC_1817_781,(0,1,6):C.UVGC_1817_782,(2,1,2):C.UVGC_2792_2249,(2,1,4):C.UVGC_2792_2250,(2,1,6):C.UVGC_2792_2251,(2,1,5):C.UVGC_2792_2252,(1,1,2):C.UVGC_2791_2245,(1,1,4):C.UVGC_2791_2246,(1,1,6):C.UVGC_2791_2247,(1,1,5):C.UVGC_2791_2248,(2,0,1):C.UVGC_3546_3723,(2,0,2):C.UVGC_3547_3728,(2,0,3):C.UVGC_3546_3725,(2,0,4):C.UVGC_3547_3729,(2,0,6):C.UVGC_3547_3730,(1,0,1):C.UVGC_3548_3731,(1,0,2):C.UVGC_3549_3736,(1,0,3):C.UVGC_3548_3733,(1,0,4):C.UVGC_3549_3737,(1,0,6):C.UVGC_3549_3738,(0,0,4):C.UVGC_1818_783,(0,0,6):C.UVGC_1818_784,(0,5,4):C.UVGC_1737_665,(0,5,6):C.UVGC_1737_666,(2,5,0):C.UVGC_3100_2925,(2,5,2):C.UVGC_3100_2926,(2,5,3):C.UVGC_3100_2927,(2,5,4):C.UVGC_3100_2928,(2,5,6):C.UVGC_3100_2929,(2,5,5):C.UVGC_2586_1942,(1,5,0):C.UVGC_3098_2917,(1,5,2):C.UVGC_3098_2918,(1,5,3):C.UVGC_3098_2919,(1,5,4):C.UVGC_3098_2920,(1,5,6):C.UVGC_3098_2921,(1,5,5):C.UVGC_2585_1938,(0,4,4):C.UVGC_1737_665,(0,4,6):C.UVGC_1737_666,(2,4,2):C.UVGC_2586_1939,(2,4,4):C.UVGC_2586_1940,(2,4,6):C.UVGC_2586_1941,(2,4,5):C.UVGC_2586_1942,(1,4,2):C.UVGC_2585_1935,(1,4,4):C.UVGC_2585_1936,(1,4,6):C.UVGC_2585_1937,(1,4,5):C.UVGC_2585_1938,(2,3,1):C.UVGC_3098_2917,(2,3,2):C.UVGC_3099_2922,(2,3,3):C.UVGC_3098_2919,(2,3,4):C.UVGC_3099_2923,(2,3,6):C.UVGC_3099_2924,(1,3,1):C.UVGC_3100_2925,(1,3,2):C.UVGC_3101_2930,(1,3,3):C.UVGC_3100_2927,(1,3,4):C.UVGC_3101_2931,(1,3,6):C.UVGC_3101_2932,(0,3,4):C.UVGC_1738_667,(0,3,6):C.UVGC_1738_668})

V_494 = CTVertex(name = 'V_494',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.d, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.d, P.g, P.t] ], [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,8,1):C.UVGC_2181_1331,(0,8,6):C.UVGC_3017_2719,(0,8,2):C.UVGC_3017_2720,(0,8,7):C.UVGC_3017_2721,(0,5,4):C.UVGC_2181_1328,(0,5,2):C.UVGC_2181_1329,(0,5,7):C.UVGC_2181_1330,(0,5,3):C.UVGC_2181_1331,(0,10,7):C.UVGC_1443_308,(0,0,2):C.UVGC_2183_1335,(0,0,7):C.UVGC_2184_1336,(0,0,3):C.UVGC_2182_1334,(0,1,2):C.UVGC_2182_1332,(0,1,7):C.UVGC_2182_1333,(0,1,3):C.UVGC_2182_1334,(0,3,2):C.UVGC_2183_1335,(0,3,7):C.UVGC_2182_1333,(0,3,3):C.UVGC_2182_1334,(0,7,2):C.UVGC_1388_260,(0,7,7):C.UVGC_1426_291,(0,4,2):C.UVGC_1387_259,(0,4,7):C.UVGC_1427_292,(0,9,2):C.UVGC_1388_260,(0,6,2):C.UVGC_1387_259,(0,2,0):C.UVGC_3016_2714,(0,2,4):C.UVGC_3016_2715,(0,2,5):C.UVGC_3016_2716,(0,2,2):C.UVGC_3016_2717,(0,2,7):C.UVGC_3016_2718})

V_495 = CTVertex(name = 'V_495',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.s, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.g, P.s, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,8,1):C.UVGC_2299_1487,(0,8,4):C.UVGC_3009_2695,(0,8,5):C.UVGC_3009_2696,(0,8,7):C.UVGC_3009_2697,(0,5,2):C.UVGC_2299_1484,(0,5,5):C.UVGC_2299_1485,(0,5,7):C.UVGC_2299_1486,(0,5,6):C.UVGC_2299_1487,(0,10,7):C.UVGC_1441_306,(0,0,5):C.UVGC_2301_1491,(0,0,7):C.UVGC_2302_1492,(0,0,6):C.UVGC_2300_1490,(0,1,5):C.UVGC_2300_1488,(0,1,7):C.UVGC_2300_1489,(0,1,6):C.UVGC_2300_1490,(0,3,5):C.UVGC_2301_1491,(0,3,7):C.UVGC_2300_1489,(0,3,6):C.UVGC_2300_1490,(0,7,5):C.UVGC_1394_265,(0,7,7):C.UVGC_1420_285,(0,4,5):C.UVGC_1393_264,(0,4,7):C.UVGC_1421_286,(0,9,5):C.UVGC_1394_265,(0,6,5):C.UVGC_1393_264,(0,2,0):C.UVGC_3008_2690,(0,2,2):C.UVGC_3008_2691,(0,2,3):C.UVGC_3008_2692,(0,2,5):C.UVGC_3008_2693,(0,2,7):C.UVGC_3008_2694})

V_496 = CTVertex(name = 'V_496',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g], [P.g, P.t] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3358_3364,(0,2,3):C.UVGC_3358_3365,(0,2,2):C.UVGC_1996_1091,(0,13,0):C.UVGC_2994_2657,(0,13,3):C.UVGC_2994_2658,(0,13,2):C.UVGC_1922_968,(0,17,1):C.UVGC_1996_1090,(0,17,2):C.UVGC_1996_1091,(0,10,1):C.UVGC_1922_967,(0,10,2):C.UVGC_1922_968,(0,1,1):C.UVGC_1365_242,(0,12,1):C.UVGC_1350_227,(0,16,1):C.UVGC_1365_242,(0,9,1):C.UVGC_1350_227,(0,3,0):C.UVGC_1365_242,(0,14,0):C.UVGC_1350_227,(0,18,0):C.UVGC_1365_242,(0,11,0):C.UVGC_1350_227,(0,0,0):C.UVGC_3359_3366,(0,0,3):C.UVGC_3359_3367,(0,0,2):C.UVGC_3359_3368,(0,4,0):C.UVGC_3359_3366,(0,4,3):C.UVGC_3359_3367,(0,4,2):C.UVGC_3359_3368,(0,15,0):C.UVGC_3359_3366,(0,15,3):C.UVGC_3359_3367,(0,15,2):C.UVGC_3359_3368,(0,7,0):C.UVGC_2992_2653,(0,7,3):C.UVGC_2992_2654,(0,5,2):C.UVGC_1924_970,(0,6,2):C.UVGC_1924_970,(0,8,2):C.UVGC_1924_970})

V_497 = CTVertex(name = 'V_497',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3362_3373,(0,2,2):C.UVGC_3362_3374,(0,2,1):C.UVGC_2000_1097,(0,13,0):C.UVGC_3056_2803,(0,13,2):C.UVGC_3056_2804,(0,13,1):C.UVGC_1943_1009,(0,17,0):C.UVGC_2000_1095,(0,17,2):C.UVGC_2000_1096,(0,17,1):C.UVGC_2000_1097,(0,10,0):C.UVGC_1943_1007,(0,10,2):C.UVGC_1943_1008,(0,10,1):C.UVGC_1943_1009,(0,1,0):C.UVGC_1379_253,(0,1,2):C.UVGC_1513_406,(0,12,0):C.UVGC_1362_240,(0,12,2):C.UVGC_1453_316,(0,16,0):C.UVGC_1379_253,(0,16,2):C.UVGC_1513_406,(0,9,0):C.UVGC_1362_240,(0,9,2):C.UVGC_1453_316,(0,3,0):C.UVGC_1379_253,(0,14,0):C.UVGC_1362_240,(0,18,0):C.UVGC_1379_253,(0,11,0):C.UVGC_1362_240,(0,0,0):C.UVGC_3363_3375,(0,0,2):C.UVGC_3363_3376,(0,0,1):C.UVGC_3363_3377,(0,4,0):C.UVGC_3363_3375,(0,4,2):C.UVGC_3363_3376,(0,4,1):C.UVGC_3363_3377,(0,15,0):C.UVGC_3363_3375,(0,15,2):C.UVGC_3363_3376,(0,15,1):C.UVGC_3363_3377,(0,7,0):C.UVGC_3058_2807,(0,7,2):C.UVGC_3058_2808,(0,5,1):C.UVGC_1945_1011,(0,6,1):C.UVGC_1945_1011,(0,8,1):C.UVGC_1945_1011})

V_498 = CTVertex(name = 'V_498',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,3):C.UVGC_3367_3385,(0,2,8):C.UVGC_3367_3386,(0,2,2):C.UVGC_3367_3387,(0,2,5):C.UVGC_2153_1294,(0,2,6):C.UVGC_2366_1560,(0,2,7):C.UVGC_2366_1561,(0,1,2):C.UVGC_2153_1293,(0,1,5):C.UVGC_2153_1294,(0,1,6):C.UVGC_2367_1562,(0,1,7):C.UVGC_2367_1563,(0,5,0):C.UVGC_2888_2389,(0,5,1):C.UVGC_2888_2390,(0,5,3):C.UVGC_3157_3014,(0,5,8):C.UVGC_3157_3015,(0,5,4):C.UVGC_3157_3016,(0,5,5):C.UVGC_1956_1041,(0,4,0):C.UVGC_2890_2393,(0,4,1):C.UVGC_2890_2394,(0,4,4):C.UVGC_1956_1040,(0,4,5):C.UVGC_1956_1041,(0,0,3):C.UVGC_3366_3382,(0,0,8):C.UVGC_3366_3383,(0,0,2):C.UVGC_2153_1293,(0,0,5):C.UVGC_3366_3384,(0,0,6):C.UVGC_2367_1562,(0,0,7):C.UVGC_2367_1563,(0,3,0):C.UVGC_2889_2391,(0,3,1):C.UVGC_2889_2392,(0,3,3):C.UVGC_3156_3010,(0,3,8):C.UVGC_3156_3011,(0,3,4):C.UVGC_3156_3012,(0,3,5):C.UVGC_3156_3013})

V_499 = CTVertex(name = 'V_499',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.c, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_1985_1067,(0,2,6):C.UVGC_3342_3322,(0,2,2):C.UVGC_3342_3321,(0,2,7):C.UVGC_3345_3328,(0,13,1):C.UVGC_1932_992,(0,13,6):C.UVGC_3015_2711,(0,13,2):C.UVGC_3015_2712,(0,13,7):C.UVGC_3015_2713,(0,17,4):C.UVGC_1985_1064,(0,17,2):C.UVGC_1985_1065,(0,17,7):C.UVGC_1985_1066,(0,17,3):C.UVGC_1985_1067,(0,10,4):C.UVGC_1932_989,(0,10,2):C.UVGC_1932_990,(0,10,7):C.UVGC_1932_991,(0,10,3):C.UVGC_1932_992,(0,0,0):C.UVGC_1985_1067,(0,0,4):C.UVGC_3342_3321,(0,0,5):C.UVGC_3342_3322,(0,0,2):C.UVGC_3342_3323,(0,0,7):C.UVGC_3342_3324,(0,0,3):C.UVGC_3342_3325,(0,4,0):C.UVGC_1985_1067,(0,4,4):C.UVGC_3342_3321,(0,4,5):C.UVGC_3342_3322,(0,4,2):C.UVGC_3344_3327,(0,4,7):C.UVGC_3343_3326,(0,4,3):C.UVGC_3342_3325,(0,15,0):C.UVGC_1985_1067,(0,15,4):C.UVGC_3342_3321,(0,15,5):C.UVGC_3342_3322,(0,15,2):C.UVGC_3342_3323,(0,15,7):C.UVGC_3343_3326,(0,15,3):C.UVGC_3342_3325,(0,5,2):C.UVGC_1934_996,(0,5,7):C.UVGC_1935_997,(0,5,3):C.UVGC_1933_995,(0,6,2):C.UVGC_1933_993,(0,6,7):C.UVGC_1933_994,(0,6,3):C.UVGC_1933_995,(0,8,2):C.UVGC_1934_996,(0,8,7):C.UVGC_1933_994,(0,8,3):C.UVGC_1933_995,(0,1,2):C.UVGC_1371_247,(0,1,7):C.UVGC_1509_402,(0,12,2):C.UVGC_1358_237,(0,12,7):C.UVGC_1422_287,(0,16,2):C.UVGC_1374_250,(0,16,7):C.UVGC_1506_399,(0,9,2):C.UVGC_1355_234,(0,9,7):C.UVGC_1425_290,(0,3,2):C.UVGC_1371_247,(0,14,2):C.UVGC_1358_237,(0,18,2):C.UVGC_1374_250,(0,11,2):C.UVGC_1355_234,(0,7,0):C.UVGC_3010_2698,(0,7,4):C.UVGC_3010_2699,(0,7,5):C.UVGC_3010_2700,(0,7,2):C.UVGC_3010_2701,(0,7,7):C.UVGC_3010_2702})

V_500 = CTVertex(name = 'V_500',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.c, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_1986_1071,(0,2,6):C.UVGC_3346_3329,(0,2,2):C.UVGC_3346_3330,(0,2,7):C.UVGC_3346_3331,(0,13,1):C.UVGC_1931_988,(0,13,6):C.UVGC_3011_2703,(0,13,2):C.UVGC_3011_2704,(0,13,7):C.UVGC_3011_2705,(0,18,4):C.UVGC_1986_1068,(0,18,2):C.UVGC_1986_1069,(0,18,7):C.UVGC_1986_1070,(0,18,3):C.UVGC_1986_1071,(0,10,4):C.UVGC_1931_985,(0,10,2):C.UVGC_1931_986,(0,10,7):C.UVGC_1931_987,(0,10,3):C.UVGC_1931_988,(0,16,7):C.UVGC_1442_307,(0,0,0):C.UVGC_1986_1071,(0,0,4):C.UVGC_3346_3330,(0,0,5):C.UVGC_3346_3329,(0,0,2):C.UVGC_3347_3332,(0,0,7):C.UVGC_3347_3333,(0,0,3):C.UVGC_3347_3334,(0,4,0):C.UVGC_1986_1071,(0,4,4):C.UVGC_3346_3330,(0,4,5):C.UVGC_3346_3329,(0,4,2):C.UVGC_3349_3336,(0,4,7):C.UVGC_3348_3335,(0,4,3):C.UVGC_3347_3334,(0,15,0):C.UVGC_1986_1071,(0,15,4):C.UVGC_3346_3330,(0,15,5):C.UVGC_3346_3329,(0,15,2):C.UVGC_3347_3332,(0,15,7):C.UVGC_3348_3335,(0,15,3):C.UVGC_3347_3334,(0,8,7):C.UVGC_1511_404,(0,5,0):C.UVGC_1931_988,(0,5,4):C.UVGC_3011_2704,(0,5,5):C.UVGC_3011_2703,(0,5,2):C.UVGC_3012_2706,(0,5,7):C.UVGC_3012_2707,(0,5,3):C.UVGC_3012_2708,(0,6,0):C.UVGC_1931_988,(0,6,4):C.UVGC_3011_2704,(0,6,5):C.UVGC_3011_2703,(0,6,2):C.UVGC_3014_2710,(0,6,7):C.UVGC_3013_2709,(0,6,3):C.UVGC_3012_2708,(0,7,0):C.UVGC_1931_988,(0,7,4):C.UVGC_3011_2704,(0,7,5):C.UVGC_3011_2703,(0,7,2):C.UVGC_3012_2706,(0,7,7):C.UVGC_3013_2709,(0,7,3):C.UVGC_3012_2708,(0,1,2):C.UVGC_1373_249,(0,1,7):C.UVGC_1508_401,(0,12,2):C.UVGC_1357_236,(0,12,7):C.UVGC_1424_289,(0,17,2):C.UVGC_1372_248,(0,17,7):C.UVGC_1507_400,(0,9,2):C.UVGC_1356_235,(0,9,7):C.UVGC_1423_288,(0,3,2):C.UVGC_1373_249,(0,14,2):C.UVGC_1357_236,(0,19,2):C.UVGC_1372_248,(0,11,2):C.UVGC_1356_235})

V_501 = CTVertex(name = 'V_501',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV22, L.FFV3, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.c, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,13,1):C.UVGC_1987_1075,(0,13,6):C.UVGC_3350_3337,(0,13,2):C.UVGC_3350_3338,(0,13,7):C.UVGC_3350_3339,(0,8,1):C.UVGC_1951_1023,(0,8,6):C.UVGC_3081_2860,(0,8,2):C.UVGC_3081_2861,(0,8,7):C.UVGC_3081_2862,(0,11,4):C.UVGC_1987_1072,(0,11,2):C.UVGC_1987_1073,(0,11,7):C.UVGC_1987_1074,(0,11,3):C.UVGC_1987_1075,(0,6,4):C.UVGC_1951_1020,(0,6,2):C.UVGC_1951_1021,(0,6,7):C.UVGC_1951_1022,(0,6,3):C.UVGC_1951_1023,(0,9,2):C.UVGC_3594_3805,(0,9,7):C.UVGC_3594_3806,(0,1,0):C.UVGC_1987_1075,(0,1,4):C.UVGC_3350_3338,(0,1,5):C.UVGC_3350_3337,(0,1,2):C.UVGC_3351_3340,(0,1,7):C.UVGC_3351_3341,(0,1,3):C.UVGC_3351_3342,(0,4,0):C.UVGC_1987_1075,(0,4,4):C.UVGC_3350_3338,(0,4,5):C.UVGC_3350_3337,(0,4,2):C.UVGC_3352_3343,(0,4,7):C.UVGC_3352_3344,(0,4,3):C.UVGC_3351_3342,(0,3,2):C.UVGC_3599_3815,(0,3,7):C.UVGC_3599_3816,(0,0,0):C.UVGC_1951_1023,(0,0,4):C.UVGC_3081_2861,(0,0,5):C.UVGC_3081_2860,(0,0,2):C.UVGC_3082_2863,(0,0,7):C.UVGC_3082_2864,(0,0,3):C.UVGC_3082_2865,(0,2,0):C.UVGC_1951_1023,(0,2,4):C.UVGC_3081_2861,(0,2,5):C.UVGC_3081_2860,(0,2,2):C.UVGC_3083_2866,(0,2,7):C.UVGC_3083_2867,(0,2,3):C.UVGC_3082_2865,(0,12,2):C.UVGC_1521_416,(0,12,7):C.UVGC_1521_417,(0,7,2):C.UVGC_1475_343,(0,7,7):C.UVGC_1475_344,(0,10,2):C.UVGC_1520_414,(0,10,7):C.UVGC_1520_415,(0,5,2):C.UVGC_1474_341,(0,5,7):C.UVGC_1474_342})

V_502 = CTVertex(name = 'V_502',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.c, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.UVGC_1525_424,(0,2,6):C.UVGC_1525_425,(2,2,0):C.UVGC_3355_3353,(2,2,4):C.UVGC_3355_3354,(2,2,5):C.UVGC_3355_3355,(2,2,2):C.UVGC_3355_3356,(2,2,6):C.UVGC_3355_3357,(2,2,3):C.UVGC_1989_1083,(1,2,0):C.UVGC_3353_3345,(1,2,4):C.UVGC_3353_3346,(1,2,5):C.UVGC_3353_3347,(1,2,2):C.UVGC_3353_3348,(1,2,6):C.UVGC_3353_3349,(1,2,3):C.UVGC_1988_1079,(0,1,2):C.UVGC_1525_424,(0,1,6):C.UVGC_1525_425,(2,1,4):C.UVGC_1989_1080,(2,1,2):C.UVGC_1989_1081,(2,1,6):C.UVGC_1989_1082,(2,1,3):C.UVGC_1989_1083,(1,1,4):C.UVGC_1988_1076,(1,1,2):C.UVGC_1988_1077,(1,1,6):C.UVGC_1988_1078,(1,1,3):C.UVGC_1988_1079,(2,0,1):C.UVGC_3353_3345,(2,0,4):C.UVGC_3354_3350,(2,0,5):C.UVGC_3353_3347,(2,0,2):C.UVGC_3354_3351,(2,0,6):C.UVGC_3354_3352,(1,0,1):C.UVGC_3355_3353,(1,0,4):C.UVGC_3356_3358,(1,0,5):C.UVGC_3355_3355,(1,0,2):C.UVGC_3356_3359,(1,0,6):C.UVGC_3356_3360,(0,0,2):C.UVGC_1526_426,(0,0,6):C.UVGC_1526_427,(0,5,2):C.UVGC_1484_361,(0,5,6):C.UVGC_1484_362,(2,5,0):C.UVGC_3096_2909,(2,5,4):C.UVGC_3096_2910,(2,5,5):C.UVGC_3096_2911,(2,5,2):C.UVGC_3096_2912,(2,5,6):C.UVGC_3096_2913,(2,5,3):C.UVGC_1955_1039,(1,5,0):C.UVGC_3094_2901,(1,5,4):C.UVGC_3094_2902,(1,5,5):C.UVGC_3094_2903,(1,5,2):C.UVGC_3094_2904,(1,5,6):C.UVGC_3094_2905,(1,5,3):C.UVGC_1954_1035,(0,4,2):C.UVGC_1484_361,(0,4,6):C.UVGC_1484_362,(2,4,4):C.UVGC_1955_1036,(2,4,2):C.UVGC_1955_1037,(2,4,6):C.UVGC_1955_1038,(2,4,3):C.UVGC_1955_1039,(1,4,4):C.UVGC_1954_1032,(1,4,2):C.UVGC_1954_1033,(1,4,6):C.UVGC_1954_1034,(1,4,3):C.UVGC_1954_1035,(2,3,1):C.UVGC_3094_2901,(2,3,4):C.UVGC_3095_2906,(2,3,5):C.UVGC_3094_2903,(2,3,2):C.UVGC_3095_2907,(2,3,6):C.UVGC_3095_2908,(1,3,1):C.UVGC_3096_2909,(1,3,4):C.UVGC_3097_2914,(1,3,5):C.UVGC_3096_2911,(1,3,2):C.UVGC_3097_2915,(1,3,6):C.UVGC_3097_2916,(0,3,2):C.UVGC_1485_363,(0,3,6):C.UVGC_1485_364})

V_503 = CTVertex(name = 'V_503',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t], [P.g, P.u] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_3515_3649,(0,2,3):C.UVGC_3515_3650,(0,2,2):C.UVGC_2777_2219,(0,13,0):C.UVGC_3001_2671,(0,13,3):C.UVGC_3001_2672,(0,13,2):C.UVGC_2555_1879,(0,17,1):C.UVGC_2777_2218,(0,17,2):C.UVGC_2777_2219,(0,10,1):C.UVGC_2555_1878,(0,10,2):C.UVGC_2555_1879,(0,1,1):C.UVGC_1582_503,(0,12,1):C.UVGC_1410_276,(0,16,1):C.UVGC_1582_503,(0,9,1):C.UVGC_1410_276,(0,3,0):C.UVGC_1582_503,(0,14,0):C.UVGC_1410_276,(0,18,0):C.UVGC_1582_503,(0,11,0):C.UVGC_1410_276,(0,0,0):C.UVGC_3516_3651,(0,0,3):C.UVGC_3516_3652,(0,0,2):C.UVGC_3516_3653,(0,4,0):C.UVGC_3516_3651,(0,4,3):C.UVGC_3516_3652,(0,4,2):C.UVGC_3516_3653,(0,15,0):C.UVGC_3516_3651,(0,15,3):C.UVGC_3516_3652,(0,15,2):C.UVGC_3516_3653,(0,7,0):C.UVGC_2999_2667,(0,7,3):C.UVGC_2999_2668,(0,5,2):C.UVGC_2557_1881,(0,6,2):C.UVGC_2557_1881,(0,8,2):C.UVGC_2557_1881})

V_504 = CTVertex(name = 'V_504',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_3523_3666,(0,2,2):C.UVGC_3523_3667,(0,2,1):C.UVGC_2781_2225,(0,13,0):C.UVGC_3059_2809,(0,13,2):C.UVGC_3059_2810,(0,13,1):C.UVGC_2576_1920,(0,17,0):C.UVGC_2781_2223,(0,17,2):C.UVGC_2781_2224,(0,17,1):C.UVGC_2781_2225,(0,10,0):C.UVGC_2576_1918,(0,10,2):C.UVGC_2576_1919,(0,10,1):C.UVGC_2576_1920,(0,1,0):C.UVGC_1598_518,(0,1,2):C.UVGC_1767_713,(0,12,0):C.UVGC_1455_317,(0,12,2):C.UVGC_1717_632,(0,16,0):C.UVGC_1598_518,(0,16,2):C.UVGC_1767_713,(0,9,0):C.UVGC_1455_317,(0,9,2):C.UVGC_1717_632,(0,3,0):C.UVGC_1598_518,(0,14,0):C.UVGC_1455_317,(0,18,0):C.UVGC_1598_518,(0,11,0):C.UVGC_1455_317,(0,0,0):C.UVGC_3524_3668,(0,0,2):C.UVGC_3524_3669,(0,0,1):C.UVGC_3524_3670,(0,4,0):C.UVGC_3524_3668,(0,4,2):C.UVGC_3524_3669,(0,4,1):C.UVGC_3524_3670,(0,15,0):C.UVGC_3524_3668,(0,15,2):C.UVGC_3524_3669,(0,15,1):C.UVGC_3524_3670,(0,7,0):C.UVGC_3061_2813,(0,7,2):C.UVGC_3061_2814,(0,5,1):C.UVGC_2578_1922,(0,6,1):C.UVGC_2578_1922,(0,8,1):C.UVGC_2578_1922})

V_505 = CTVertex(name = 'V_505',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,2):C.UVGC_2891_2395,(0,2,0):C.UVGC_2891_2396,(0,2,6):C.UVGC_3528_3678,(0,2,8):C.UVGC_3528_3679,(0,2,4):C.UVGC_3528_3680,(0,2,7):C.UVGC_2785_2230,(0,1,2):C.UVGC_2892_2397,(0,1,0):C.UVGC_2892_2398,(0,1,4):C.UVGC_2785_2229,(0,1,7):C.UVGC_2785_2230,(0,5,6):C.UVGC_3161_3028,(0,5,8):C.UVGC_3161_3029,(0,5,1):C.UVGC_3161_3030,(0,5,7):C.UVGC_2590_1954,(0,5,3):C.UVGC_2349_1542,(0,5,5):C.UVGC_2349_1543,(0,4,1):C.UVGC_2590_1953,(0,4,7):C.UVGC_2590_1954,(0,4,3):C.UVGC_2351_1546,(0,4,5):C.UVGC_2351_1547,(0,0,2):C.UVGC_2892_2397,(0,0,0):C.UVGC_2892_2398,(0,0,6):C.UVGC_3527_3675,(0,0,8):C.UVGC_3527_3676,(0,0,4):C.UVGC_2785_2229,(0,0,7):C.UVGC_3527_3677,(0,3,6):C.UVGC_3160_3024,(0,3,8):C.UVGC_3160_3025,(0,3,1):C.UVGC_3160_3026,(0,3,7):C.UVGC_3160_3027,(0,3,3):C.UVGC_2350_1544,(0,3,5):C.UVGC_2350_1545})

V_506 = CTVertex(name = 'V_506',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.b, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,8,0):C.UVGC_2410_1619,(0,8,1):C.UVGC_2410_1620,(0,8,2):C.UVGC_2409_1618,(0,5,1):C.UVGC_2409_1617,(0,5,2):C.UVGC_2409_1618,(0,0,2):C.UVGC_2408_1616,(0,1,2):C.UVGC_2408_1616,(0,3,2):C.UVGC_2408_1616,(0,7,1):C.UVGC_2116_1245,(0,4,1):C.UVGC_2116_1245,(0,9,1):C.UVGC_2116_1245,(0,6,1):C.UVGC_2116_1245,(0,2,0):C.UVGC_1709_626})

V_507 = CTVertex(name = 'V_507',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.b, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2420_1637,(0,2,1):C.UVGC_2420_1638,(0,2,2):C.UVGC_2420_1639,(0,1,1):C.UVGC_2421_1640,(0,1,2):C.UVGC_2420_1639,(0,0,0):C.UVGC_2422_1641,(0,0,1):C.UVGC_2422_1642,(0,0,2):C.UVGC_2422_1643})

V_508 = CTVertex(name = 'V_508',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.b, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.t] ], [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.s, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,2):C.UVGC_2416_1629,(0,2,4):C.UVGC_2416_1630,(0,2,6):C.UVGC_2415_1628,(0,2,0):C.UVGC_2928_2508,(0,2,1):C.UVGC_2928_2509,(0,2,3):C.UVGC_2928_2510,(0,2,5):C.UVGC_2928_2511,(0,1,4):C.UVGC_2415_1627,(0,1,6):C.UVGC_2415_1628,(0,1,0):C.UVGC_2929_2512,(0,1,1):C.UVGC_2929_2513,(0,1,3):C.UVGC_2929_2514,(0,1,5):C.UVGC_2929_2515,(0,0,2):C.UVGC_2417_1631,(0,0,4):C.UVGC_2417_1632,(0,0,6):C.UVGC_2417_1633,(0,0,0):C.UVGC_2929_2512,(0,0,1):C.UVGC_2930_2516,(0,0,3):C.UVGC_2930_2517,(0,0,5):C.UVGC_2929_2515})

V_509 = CTVertex(name = 'V_509',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.g, P.t, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,1):C.UVGC_2746_2175,(0,2,4):C.UVGC_3456_3554,(0,2,5):C.UVGC_3459_3560,(0,2,7):C.UVGC_3459_3561,(0,13,1):C.UVGC_2565_1903,(0,13,4):C.UVGC_3029_2752,(0,13,5):C.UVGC_3029_2753,(0,13,7):C.UVGC_3029_2754,(0,17,2):C.UVGC_2746_2172,(0,17,5):C.UVGC_2746_2173,(0,17,7):C.UVGC_2746_2174,(0,17,6):C.UVGC_2746_2175,(0,10,2):C.UVGC_2565_1900,(0,10,5):C.UVGC_2565_1901,(0,10,7):C.UVGC_2565_1902,(0,10,6):C.UVGC_2565_1903,(0,0,0):C.UVGC_2746_2175,(0,0,2):C.UVGC_3456_3553,(0,0,3):C.UVGC_3456_3554,(0,0,5):C.UVGC_3456_3555,(0,0,7):C.UVGC_3456_3556,(0,0,6):C.UVGC_3456_3557,(0,4,0):C.UVGC_2746_2175,(0,4,2):C.UVGC_3456_3553,(0,4,3):C.UVGC_3456_3554,(0,4,5):C.UVGC_3458_3559,(0,4,7):C.UVGC_3457_3558,(0,4,6):C.UVGC_3456_3557,(0,15,0):C.UVGC_2746_2175,(0,15,2):C.UVGC_3456_3553,(0,15,3):C.UVGC_3456_3554,(0,15,5):C.UVGC_3456_3555,(0,15,7):C.UVGC_3457_3558,(0,15,6):C.UVGC_3456_3557,(0,5,5):C.UVGC_2567_1907,(0,5,7):C.UVGC_2568_1908,(0,5,6):C.UVGC_2566_1906,(0,6,5):C.UVGC_2566_1904,(0,6,7):C.UVGC_2566_1905,(0,6,6):C.UVGC_2566_1906,(0,8,5):C.UVGC_2567_1907,(0,8,7):C.UVGC_2566_1905,(0,8,6):C.UVGC_2566_1906,(0,1,5):C.UVGC_1588_508,(0,1,7):C.UVGC_1765_711,(0,12,5):C.UVGC_1431_296,(0,12,7):C.UVGC_1705_622,(0,16,5):C.UVGC_1591_511,(0,16,7):C.UVGC_1762_708,(0,9,5):C.UVGC_1428_293,(0,9,7):C.UVGC_1708_625,(0,3,5):C.UVGC_1588_508,(0,14,5):C.UVGC_1431_296,(0,18,5):C.UVGC_1591_511,(0,11,5):C.UVGC_1428_293,(0,7,0):C.UVGC_3024_2738,(0,7,2):C.UVGC_3024_2739,(0,7,3):C.UVGC_3024_2740,(0,7,5):C.UVGC_3024_2741,(0,7,7):C.UVGC_3024_2742})

V_510 = CTVertex(name = 'V_510',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.g, P.t, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,1):C.UVGC_2747_2179,(0,2,4):C.UVGC_3460_3562,(0,2,5):C.UVGC_3460_3563,(0,2,7):C.UVGC_3460_3564,(0,13,1):C.UVGC_2564_1899,(0,13,4):C.UVGC_3025_2743,(0,13,5):C.UVGC_3025_2744,(0,13,7):C.UVGC_3025_2745,(0,18,2):C.UVGC_2747_2176,(0,18,5):C.UVGC_2747_2177,(0,18,7):C.UVGC_2747_2178,(0,18,6):C.UVGC_2747_2179,(0,10,2):C.UVGC_2564_1896,(0,10,5):C.UVGC_2564_1897,(0,10,7):C.UVGC_2564_1898,(0,10,6):C.UVGC_2564_1899,(0,16,5):C.UVGC_1596_516,(0,0,0):C.UVGC_2747_2179,(0,0,2):C.UVGC_3461_3565,(0,0,3):C.UVGC_3460_3562,(0,0,5):C.UVGC_3461_3566,(0,0,7):C.UVGC_3461_3567,(0,0,6):C.UVGC_3461_3568,(0,4,0):C.UVGC_2747_2179,(0,4,2):C.UVGC_3461_3565,(0,4,3):C.UVGC_3460_3562,(0,4,5):C.UVGC_3463_3570,(0,4,7):C.UVGC_3462_3569,(0,4,6):C.UVGC_3461_3568,(0,15,0):C.UVGC_2747_2179,(0,15,2):C.UVGC_3461_3565,(0,15,3):C.UVGC_3460_3562,(0,15,5):C.UVGC_3461_3566,(0,15,7):C.UVGC_3462_3569,(0,15,6):C.UVGC_3461_3568,(0,8,5):C.UVGC_1445_310,(0,5,0):C.UVGC_2564_1899,(0,5,2):C.UVGC_3026_2746,(0,5,3):C.UVGC_3025_2743,(0,5,5):C.UVGC_3026_2747,(0,5,7):C.UVGC_3026_2748,(0,5,6):C.UVGC_3026_2749,(0,6,0):C.UVGC_2564_1899,(0,6,2):C.UVGC_3026_2746,(0,6,3):C.UVGC_3025_2743,(0,6,5):C.UVGC_3028_2751,(0,6,7):C.UVGC_3027_2750,(0,6,6):C.UVGC_3026_2749,(0,7,0):C.UVGC_2564_1899,(0,7,2):C.UVGC_3026_2746,(0,7,3):C.UVGC_3025_2743,(0,7,5):C.UVGC_3026_2747,(0,7,7):C.UVGC_3027_2750,(0,7,6):C.UVGC_3026_2749,(0,1,5):C.UVGC_1590_510,(0,1,7):C.UVGC_1764_710,(0,12,5):C.UVGC_1430_295,(0,12,7):C.UVGC_1707_624,(0,17,5):C.UVGC_1589_509,(0,17,7):C.UVGC_1763_709,(0,9,5):C.UVGC_1429_294,(0,9,7):C.UVGC_1706_623,(0,3,5):C.UVGC_1590_510,(0,14,5):C.UVGC_1430_295,(0,19,5):C.UVGC_1589_509,(0,11,5):C.UVGC_1429_294})

V_511 = CTVertex(name = 'V_511',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV22, L.FFV3, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.g, P.t, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,13,1):C.UVGC_2748_2183,(0,13,4):C.UVGC_3465_3572,(0,13,5):C.UVGC_3465_3573,(0,13,7):C.UVGC_3465_3574,(0,8,1):C.UVGC_2584_1934,(0,8,4):C.UVGC_3087_2876,(0,8,5):C.UVGC_3087_2877,(0,8,7):C.UVGC_3087_2878,(0,11,2):C.UVGC_2748_2180,(0,11,5):C.UVGC_2748_2181,(0,11,7):C.UVGC_2748_2182,(0,11,6):C.UVGC_2748_2183,(0,6,2):C.UVGC_2584_1931,(0,6,5):C.UVGC_2584_1932,(0,6,7):C.UVGC_2584_1933,(0,6,6):C.UVGC_2584_1934,(0,9,5):C.UVGC_3474_3601,(0,9,7):C.UVGC_3474_3602,(0,1,0):C.UVGC_2748_2183,(0,1,2):C.UVGC_3466_3575,(0,1,3):C.UVGC_3465_3572,(0,1,5):C.UVGC_3466_3576,(0,1,7):C.UVGC_3466_3577,(0,1,6):C.UVGC_3466_3578,(0,4,0):C.UVGC_2748_2183,(0,4,2):C.UVGC_3466_3575,(0,4,3):C.UVGC_3465_3572,(0,4,5):C.UVGC_3467_3579,(0,4,7):C.UVGC_3467_3580,(0,4,6):C.UVGC_3466_3578,(0,3,5):C.UVGC_3151_2999,(0,3,7):C.UVGC_3151_3000,(0,0,0):C.UVGC_2584_1934,(0,0,2):C.UVGC_3088_2879,(0,0,3):C.UVGC_3087_2876,(0,0,5):C.UVGC_3088_2880,(0,0,7):C.UVGC_3088_2881,(0,0,6):C.UVGC_3088_2882,(0,2,0):C.UVGC_2584_1934,(0,2,2):C.UVGC_3088_2879,(0,2,3):C.UVGC_3087_2876,(0,2,5):C.UVGC_3089_2883,(0,2,7):C.UVGC_3089_2884,(0,2,6):C.UVGC_3088_2882,(0,12,5):C.UVGC_1774_722,(0,12,7):C.UVGC_1774_723,(0,7,5):C.UVGC_1730_651,(0,7,7):C.UVGC_1730_652,(0,10,5):C.UVGC_1773_720,(0,10,7):C.UVGC_1773_721,(0,5,5):C.UVGC_1729_649,(0,5,7):C.UVGC_1729_650})

V_512 = CTVertex(name = 'V_512',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.g, P.t, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,4):C.UVGC_1778_730,(0,2,6):C.UVGC_1778_731,(2,2,0):C.UVGC_3470_3589,(2,2,2):C.UVGC_3470_3590,(2,2,3):C.UVGC_3470_3591,(2,2,4):C.UVGC_3470_3592,(2,2,6):C.UVGC_3470_3593,(2,2,5):C.UVGC_2750_2191,(1,2,0):C.UVGC_3468_3581,(1,2,2):C.UVGC_3468_3582,(1,2,3):C.UVGC_3468_3583,(1,2,4):C.UVGC_3468_3584,(1,2,6):C.UVGC_3468_3585,(1,2,5):C.UVGC_2749_2187,(0,1,4):C.UVGC_1778_730,(0,1,6):C.UVGC_1778_731,(2,1,2):C.UVGC_2750_2188,(2,1,4):C.UVGC_2750_2189,(2,1,6):C.UVGC_2750_2190,(2,1,5):C.UVGC_2750_2191,(1,1,2):C.UVGC_2749_2184,(1,1,4):C.UVGC_2749_2185,(1,1,6):C.UVGC_2749_2186,(1,1,5):C.UVGC_2749_2187,(2,0,1):C.UVGC_3468_3581,(2,0,2):C.UVGC_3469_3586,(2,0,3):C.UVGC_3468_3583,(2,0,4):C.UVGC_3469_3587,(2,0,6):C.UVGC_3469_3588,(1,0,1):C.UVGC_3470_3589,(1,0,2):C.UVGC_3471_3594,(1,0,3):C.UVGC_3470_3591,(1,0,4):C.UVGC_3471_3595,(1,0,6):C.UVGC_3471_3596,(0,0,4):C.UVGC_1779_732,(0,0,6):C.UVGC_1779_733,(0,5,4):C.UVGC_1739_669,(0,5,6):C.UVGC_1739_670,(2,5,0):C.UVGC_3104_2941,(2,5,2):C.UVGC_3104_2942,(2,5,3):C.UVGC_3104_2943,(2,5,4):C.UVGC_3104_2944,(2,5,6):C.UVGC_3104_2945,(2,5,5):C.UVGC_2588_1950,(1,5,0):C.UVGC_3102_2933,(1,5,2):C.UVGC_3102_2934,(1,5,3):C.UVGC_3102_2935,(1,5,4):C.UVGC_3102_2936,(1,5,6):C.UVGC_3102_2937,(1,5,5):C.UVGC_2587_1946,(0,4,4):C.UVGC_1739_669,(0,4,6):C.UVGC_1739_670,(2,4,2):C.UVGC_2588_1947,(2,4,4):C.UVGC_2588_1948,(2,4,6):C.UVGC_2588_1949,(2,4,5):C.UVGC_2588_1950,(1,4,2):C.UVGC_2587_1943,(1,4,4):C.UVGC_2587_1944,(1,4,6):C.UVGC_2587_1945,(1,4,5):C.UVGC_2587_1946,(2,3,1):C.UVGC_3102_2933,(2,3,2):C.UVGC_3103_2938,(2,3,3):C.UVGC_3102_2935,(2,3,4):C.UVGC_3103_2939,(2,3,6):C.UVGC_3103_2940,(1,3,1):C.UVGC_3104_2941,(1,3,2):C.UVGC_3105_2946,(1,3,3):C.UVGC_3104_2943,(1,3,4):C.UVGC_3105_2947,(1,3,6):C.UVGC_3105_2948,(0,3,4):C.UVGC_1740_671,(0,3,6):C.UVGC_1740_672})

V_513 = CTVertex(name = 'V_513',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.b, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.b, P.g, P.u] ], [ [P.b, P.g] ], [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.u] ] ],
                 couplings = {(0,8,1):C.UVGC_2403_1604,(0,8,8):C.UVGC_2403_1605,(0,8,2):C.UVGC_2403_1606,(0,8,9):C.UVGC_2403_1607,(0,5,6):C.UVGC_2404_1608,(0,5,2):C.UVGC_2404_1609,(0,5,9):C.UVGC_2404_1610,(0,5,5):C.UVGC_2403_1604,(0,10,4):C.UVGC_2170_1317,(0,0,2):C.UVGC_2406_1614,(0,0,9):C.UVGC_2407_1615,(0,0,5):C.UVGC_2405_1613,(0,1,2):C.UVGC_2405_1611,(0,1,9):C.UVGC_2405_1612,(0,1,5):C.UVGC_2405_1613,(0,3,2):C.UVGC_2406_1614,(0,3,9):C.UVGC_2405_1612,(0,3,5):C.UVGC_2405_1613,(0,7,2):C.UVGC_1336_216,(0,7,9):C.UVGC_1702_616,(0,4,2):C.UVGC_1335_215,(0,4,9):C.UVGC_1704_621,(0,9,2):C.UVGC_1336_216,(0,6,2):C.UVGC_1335_215,(0,2,0):C.UVGC_1703_617,(0,2,6):C.UVGC_1703_618,(0,2,7):C.UVGC_1703_619,(0,2,3):C.UVGC_1703_620})

V_514 = CTVertex(name = 'V_514',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g], [P.g, P.t] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3432_3494,(0,2,3):C.UVGC_3432_3495,(0,2,2):C.UVGC_2040_1148,(0,13,0):C.UVGC_2990_2649,(0,13,3):C.UVGC_2990_2650,(0,13,2):C.UVGC_1918_963,(0,17,1):C.UVGC_2040_1147,(0,17,2):C.UVGC_2040_1148,(0,10,1):C.UVGC_1918_962,(0,10,2):C.UVGC_1918_963,(0,1,1):C.UVGC_1535_444,(0,12,1):C.UVGC_1404_275,(0,16,1):C.UVGC_1535_444,(0,9,1):C.UVGC_1404_275,(0,3,3):C.UVGC_1535_444,(0,14,3):C.UVGC_1404_275,(0,18,3):C.UVGC_1535_444,(0,11,3):C.UVGC_1404_275,(0,0,0):C.UVGC_3433_3496,(0,0,3):C.UVGC_3433_3497,(0,0,2):C.UVGC_3433_3498,(0,4,0):C.UVGC_3433_3496,(0,4,3):C.UVGC_3433_3497,(0,4,2):C.UVGC_3433_3498,(0,15,0):C.UVGC_3433_3496,(0,15,3):C.UVGC_3433_3497,(0,15,2):C.UVGC_3433_3498,(0,7,0):C.UVGC_2988_2645,(0,7,3):C.UVGC_2988_2646,(0,5,2):C.UVGC_1920_965,(0,6,2):C.UVGC_1920_965,(0,8,2):C.UVGC_1920_965})

V_515 = CTVertex(name = 'V_515',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3440_3511,(0,2,2):C.UVGC_3440_3512,(0,2,1):C.UVGC_2044_1154,(0,13,0):C.UVGC_3048_2787,(0,13,2):C.UVGC_3048_2788,(0,13,1):C.UVGC_1939_1003,(0,17,0):C.UVGC_2044_1152,(0,17,2):C.UVGC_2044_1153,(0,17,1):C.UVGC_2044_1154,(0,10,0):C.UVGC_1939_1001,(0,10,2):C.UVGC_1939_1002,(0,10,1):C.UVGC_1939_1003,(0,1,0):C.UVGC_1559_464,(0,1,2):C.UVGC_1557_463,(0,12,0):C.UVGC_1450_314,(0,12,2):C.UVGC_1448_313,(0,16,0):C.UVGC_1559_464,(0,16,2):C.UVGC_1557_463,(0,9,0):C.UVGC_1450_314,(0,9,2):C.UVGC_1448_313,(0,3,2):C.UVGC_1557_463,(0,14,2):C.UVGC_1448_313,(0,18,2):C.UVGC_1557_463,(0,11,2):C.UVGC_1448_313,(0,0,0):C.UVGC_3441_3513,(0,0,2):C.UVGC_3441_3514,(0,0,1):C.UVGC_3441_3515,(0,4,0):C.UVGC_3441_3513,(0,4,2):C.UVGC_3441_3514,(0,4,1):C.UVGC_3441_3515,(0,15,0):C.UVGC_3441_3513,(0,15,2):C.UVGC_3441_3514,(0,15,1):C.UVGC_3441_3515,(0,7,0):C.UVGC_3050_2791,(0,7,2):C.UVGC_3050_2792,(0,5,1):C.UVGC_1941_1005,(0,6,1):C.UVGC_1941_1005,(0,8,1):C.UVGC_1941_1005})

V_516 = CTVertex(name = 'V_516',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2894_2399,(0,2,1):C.UVGC_2894_2400,(0,2,3):C.UVGC_3445_3523,(0,2,8):C.UVGC_3445_3524,(0,2,4):C.UVGC_3445_3525,(0,2,5):C.UVGC_2048_1159,(0,1,0):C.UVGC_2895_2401,(0,1,1):C.UVGC_2895_2402,(0,1,4):C.UVGC_2048_1158,(0,1,5):C.UVGC_2048_1159,(0,5,3):C.UVGC_3155_3007,(0,5,8):C.UVGC_3155_3008,(0,5,2):C.UVGC_3155_3009,(0,5,5):C.UVGC_2124_1253,(0,5,6):C.UVGC_2346_1536,(0,5,7):C.UVGC_2346_1537,(0,4,2):C.UVGC_2124_1252,(0,4,5):C.UVGC_2124_1253,(0,4,6):C.UVGC_2348_1540,(0,4,7):C.UVGC_2348_1541,(0,0,0):C.UVGC_2895_2401,(0,0,1):C.UVGC_2895_2402,(0,0,3):C.UVGC_3444_3520,(0,0,8):C.UVGC_3444_3521,(0,0,4):C.UVGC_2048_1158,(0,0,5):C.UVGC_3444_3522,(0,3,3):C.UVGC_3154_3003,(0,3,8):C.UVGC_3154_3004,(0,3,2):C.UVGC_3154_3005,(0,3,5):C.UVGC_3154_3006,(0,3,6):C.UVGC_2347_1538,(0,3,7):C.UVGC_2347_1539})

V_517 = CTVertex(name = 'V_517',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.b, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,8,1):C.UVGC_2113_1242,(0,8,0):C.UVGC_2112_1240,(0,8,2):C.UVGC_2113_1243,(0,5,0):C.UVGC_2112_1240,(0,5,2):C.UVGC_2112_1241,(0,0,0):C.UVGC_1863_883,(0,1,0):C.UVGC_1863_883,(0,3,0):C.UVGC_1863_883,(0,7,2):C.UVGC_2114_1244,(0,4,2):C.UVGC_2114_1244,(0,9,2):C.UVGC_2114_1244,(0,6,2):C.UVGC_2114_1244,(0,2,1):C.UVGC_1359_238})

V_518 = CTVertex(name = 'V_518',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.b, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2132_1265,(0,2,0):C.UVGC_2132_1266,(0,2,2):C.UVGC_2132_1267,(0,1,0):C.UVGC_2132_1266,(0,1,2):C.UVGC_2133_1268,(0,0,1):C.UVGC_2134_1269,(0,0,0):C.UVGC_2134_1270,(0,0,2):C.UVGC_2134_1271})

V_519 = CTVertex(name = 'V_519',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.b, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.t] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.s, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,3):C.UVGC_2126_1256,(0,2,0):C.UVGC_2125_1254,(0,2,5):C.UVGC_2126_1257,(0,2,1):C.UVGC_2925_2498,(0,2,2):C.UVGC_2925_2499,(0,2,4):C.UVGC_2925_2500,(0,2,6):C.UVGC_2925_2501,(0,1,0):C.UVGC_2125_1254,(0,1,5):C.UVGC_2125_1255,(0,1,1):C.UVGC_2926_2502,(0,1,2):C.UVGC_2926_2503,(0,1,4):C.UVGC_2926_2504,(0,1,6):C.UVGC_2926_2505,(0,0,3):C.UVGC_2127_1258,(0,0,0):C.UVGC_2127_1259,(0,0,5):C.UVGC_2127_1260,(0,0,1):C.UVGC_2926_2502,(0,0,2):C.UVGC_2927_2506,(0,0,4):C.UVGC_2927_2507,(0,0,6):C.UVGC_2926_2505})

V_520 = CTVertex(name = 'V_520',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.c, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2006_1106,(0,2,6):C.UVGC_3373_3400,(0,2,2):C.UVGC_3376_3406,(0,2,7):C.UVGC_3376_3407,(0,13,1):C.UVGC_1927_979,(0,13,6):C.UVGC_3007_2687,(0,13,2):C.UVGC_3007_2688,(0,13,7):C.UVGC_3007_2689,(0,17,4):C.UVGC_2006_1103,(0,17,2):C.UVGC_2006_1104,(0,17,7):C.UVGC_2006_1105,(0,17,3):C.UVGC_2006_1106,(0,10,4):C.UVGC_1927_976,(0,10,2):C.UVGC_1927_977,(0,10,7):C.UVGC_1927_978,(0,10,3):C.UVGC_1927_979,(0,0,0):C.UVGC_2006_1106,(0,0,4):C.UVGC_3373_3399,(0,0,5):C.UVGC_3373_3400,(0,0,2):C.UVGC_3373_3401,(0,0,7):C.UVGC_3373_3402,(0,0,3):C.UVGC_3373_3403,(0,4,0):C.UVGC_2006_1106,(0,4,4):C.UVGC_3373_3399,(0,4,5):C.UVGC_3373_3400,(0,4,2):C.UVGC_3374_3404,(0,4,7):C.UVGC_3375_3405,(0,4,3):C.UVGC_3373_3403,(0,15,0):C.UVGC_2006_1106,(0,15,4):C.UVGC_3373_3399,(0,15,5):C.UVGC_3373_3400,(0,15,2):C.UVGC_3374_3404,(0,15,7):C.UVGC_3373_3402,(0,15,3):C.UVGC_3373_3403,(0,5,2):C.UVGC_1930_984,(0,5,7):C.UVGC_1929_983,(0,5,3):C.UVGC_1928_982,(0,6,2):C.UVGC_1928_980,(0,6,7):C.UVGC_1928_981,(0,6,3):C.UVGC_1928_982,(0,8,2):C.UVGC_1928_980,(0,8,7):C.UVGC_1929_983,(0,8,3):C.UVGC_1928_982,(0,1,2):C.UVGC_1544_450,(0,1,7):C.UVGC_1543_449,(0,12,2):C.UVGC_1419_284,(0,12,7):C.UVGC_1418_283,(0,16,2):C.UVGC_1550_456,(0,16,7):C.UVGC_1549_455,(0,9,2):C.UVGC_1413_278,(0,9,7):C.UVGC_1412_277,(0,3,7):C.UVGC_1543_449,(0,14,7):C.UVGC_1418_283,(0,18,7):C.UVGC_1549_455,(0,11,7):C.UVGC_1412_277,(0,7,0):C.UVGC_3002_2673,(0,7,4):C.UVGC_3002_2674,(0,7,5):C.UVGC_3002_2675,(0,7,2):C.UVGC_3002_2676,(0,7,7):C.UVGC_3002_2677})

V_521 = CTVertex(name = 'V_521',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.c, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2007_1110,(0,2,6):C.UVGC_3377_3408,(0,2,2):C.UVGC_3377_3409,(0,2,7):C.UVGC_3377_3410,(0,13,1):C.UVGC_1926_975,(0,13,6):C.UVGC_3003_2678,(0,13,2):C.UVGC_3003_2679,(0,13,7):C.UVGC_3003_2680,(0,18,4):C.UVGC_2007_1107,(0,18,2):C.UVGC_2007_1108,(0,18,7):C.UVGC_2007_1109,(0,18,3):C.UVGC_2007_1110,(0,10,4):C.UVGC_1926_972,(0,10,2):C.UVGC_1926_973,(0,10,7):C.UVGC_1926_974,(0,10,3):C.UVGC_1926_975,(0,16,7):C.UVGC_1555_461,(0,0,0):C.UVGC_2007_1110,(0,0,4):C.UVGC_3378_3411,(0,0,5):C.UVGC_3377_3408,(0,0,2):C.UVGC_3378_3412,(0,0,7):C.UVGC_3378_3413,(0,0,3):C.UVGC_3378_3414,(0,4,0):C.UVGC_2007_1110,(0,4,4):C.UVGC_3378_3411,(0,4,5):C.UVGC_3377_3408,(0,4,2):C.UVGC_3379_3415,(0,4,7):C.UVGC_3380_3416,(0,4,3):C.UVGC_3378_3414,(0,15,0):C.UVGC_2007_1110,(0,15,4):C.UVGC_3378_3411,(0,15,5):C.UVGC_3377_3408,(0,15,2):C.UVGC_3379_3415,(0,15,7):C.UVGC_3378_3413,(0,15,3):C.UVGC_3378_3414,(0,8,7):C.UVGC_1440_305,(0,5,0):C.UVGC_1926_975,(0,5,4):C.UVGC_3004_2681,(0,5,5):C.UVGC_3003_2678,(0,5,2):C.UVGC_3004_2682,(0,5,7):C.UVGC_3004_2683,(0,5,3):C.UVGC_3004_2684,(0,6,0):C.UVGC_1926_975,(0,6,4):C.UVGC_3004_2681,(0,6,5):C.UVGC_3003_2678,(0,6,2):C.UVGC_3005_2685,(0,6,7):C.UVGC_3006_2686,(0,6,3):C.UVGC_3004_2684,(0,7,0):C.UVGC_1926_975,(0,7,4):C.UVGC_3004_2681,(0,7,5):C.UVGC_3003_2678,(0,7,2):C.UVGC_3005_2685,(0,7,7):C.UVGC_3004_2683,(0,7,3):C.UVGC_3004_2684,(0,1,2):C.UVGC_1548_454,(0,1,7):C.UVGC_1547_453,(0,12,2):C.UVGC_1417_282,(0,12,7):C.UVGC_1416_281,(0,17,2):C.UVGC_1546_452,(0,17,7):C.UVGC_1545_451,(0,9,2):C.UVGC_1415_280,(0,9,7):C.UVGC_1414_279,(0,3,7):C.UVGC_1547_453,(0,14,7):C.UVGC_1416_281,(0,19,7):C.UVGC_1545_451,(0,11,7):C.UVGC_1414_279})

V_522 = CTVertex(name = 'V_522',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV22, L.FFV3, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.c, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,13,1):C.UVGC_2008_1114,(0,13,6):C.UVGC_3382_3418,(0,13,2):C.UVGC_3382_3419,(0,13,7):C.UVGC_3382_3420,(0,8,1):C.UVGC_1950_1019,(0,8,6):C.UVGC_3078_2851,(0,8,2):C.UVGC_3078_2852,(0,8,7):C.UVGC_3078_2853,(0,11,4):C.UVGC_2008_1111,(0,11,2):C.UVGC_2008_1112,(0,11,7):C.UVGC_2008_1113,(0,11,3):C.UVGC_2008_1114,(0,6,4):C.UVGC_1950_1016,(0,6,2):C.UVGC_1950_1017,(0,6,7):C.UVGC_1950_1018,(0,6,3):C.UVGC_1950_1019,(0,9,2):C.UVGC_3391_3447,(0,9,7):C.UVGC_3391_3448,(0,1,0):C.UVGC_2008_1114,(0,1,4):C.UVGC_3383_3421,(0,1,5):C.UVGC_3382_3418,(0,1,2):C.UVGC_3383_3422,(0,1,7):C.UVGC_3383_3423,(0,1,3):C.UVGC_3383_3424,(0,4,0):C.UVGC_2008_1114,(0,4,4):C.UVGC_3383_3421,(0,4,5):C.UVGC_3382_3418,(0,4,2):C.UVGC_3384_3425,(0,4,7):C.UVGC_3384_3426,(0,4,3):C.UVGC_3383_3424,(0,3,2):C.UVGC_3150_2997,(0,3,7):C.UVGC_3150_2998,(0,0,0):C.UVGC_1950_1019,(0,0,4):C.UVGC_3079_2854,(0,0,5):C.UVGC_3078_2851,(0,0,2):C.UVGC_3079_2855,(0,0,7):C.UVGC_3079_2856,(0,0,3):C.UVGC_3079_2857,(0,2,0):C.UVGC_1950_1019,(0,2,4):C.UVGC_3079_2854,(0,2,5):C.UVGC_3078_2851,(0,2,2):C.UVGC_3080_2858,(0,2,7):C.UVGC_3080_2859,(0,2,3):C.UVGC_3079_2857,(0,12,2):C.UVGC_1568_475,(0,12,7):C.UVGC_1568_476,(0,7,2):C.UVGC_1473_339,(0,7,7):C.UVGC_1473_340,(0,10,2):C.UVGC_1567_473,(0,10,7):C.UVGC_1567_474,(0,5,2):C.UVGC_1472_337,(0,5,7):C.UVGC_1472_338})

V_523 = CTVertex(name = 'V_523',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.g, P.g ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.c, P.g, P.t] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.UVGC_1572_483,(0,2,6):C.UVGC_1572_484,(2,2,0):C.UVGC_3387_3435,(2,2,4):C.UVGC_3387_3436,(2,2,5):C.UVGC_3387_3437,(2,2,2):C.UVGC_3387_3438,(2,2,6):C.UVGC_3387_3439,(2,2,3):C.UVGC_2010_1122,(1,2,0):C.UVGC_3385_3427,(1,2,4):C.UVGC_3385_3428,(1,2,5):C.UVGC_3385_3429,(1,2,2):C.UVGC_3385_3430,(1,2,6):C.UVGC_3385_3431,(1,2,3):C.UVGC_2009_1118,(0,1,2):C.UVGC_1572_483,(0,1,6):C.UVGC_1572_484,(2,1,4):C.UVGC_2010_1119,(2,1,2):C.UVGC_2010_1120,(2,1,6):C.UVGC_2010_1121,(2,1,3):C.UVGC_2010_1122,(1,1,4):C.UVGC_2009_1115,(1,1,2):C.UVGC_2009_1116,(1,1,6):C.UVGC_2009_1117,(1,1,3):C.UVGC_2009_1118,(2,0,1):C.UVGC_3385_3427,(2,0,4):C.UVGC_3386_3432,(2,0,5):C.UVGC_3385_3429,(2,0,2):C.UVGC_3386_3433,(2,0,6):C.UVGC_3386_3434,(1,0,1):C.UVGC_3387_3435,(1,0,4):C.UVGC_3388_3440,(1,0,5):C.UVGC_3387_3437,(1,0,2):C.UVGC_3388_3441,(1,0,6):C.UVGC_3388_3442,(0,0,2):C.UVGC_1573_485,(0,0,6):C.UVGC_1573_486,(0,5,2):C.UVGC_1482_357,(0,5,6):C.UVGC_1482_358,(2,5,0):C.UVGC_3092_2893,(2,5,4):C.UVGC_3092_2894,(2,5,5):C.UVGC_3092_2895,(2,5,2):C.UVGC_3092_2896,(2,5,6):C.UVGC_3092_2897,(2,5,3):C.UVGC_1953_1031,(1,5,0):C.UVGC_3090_2885,(1,5,4):C.UVGC_3090_2886,(1,5,5):C.UVGC_3090_2887,(1,5,2):C.UVGC_3090_2888,(1,5,6):C.UVGC_3090_2889,(1,5,3):C.UVGC_1952_1027,(0,4,2):C.UVGC_1482_357,(0,4,6):C.UVGC_1482_358,(2,4,4):C.UVGC_1953_1028,(2,4,2):C.UVGC_1953_1029,(2,4,6):C.UVGC_1953_1030,(2,4,3):C.UVGC_1953_1031,(1,4,4):C.UVGC_1952_1024,(1,4,2):C.UVGC_1952_1025,(1,4,6):C.UVGC_1952_1026,(1,4,3):C.UVGC_1952_1027,(2,3,1):C.UVGC_3090_2885,(2,3,4):C.UVGC_3091_2890,(2,3,5):C.UVGC_3090_2887,(2,3,2):C.UVGC_3091_2891,(2,3,6):C.UVGC_3091_2892,(1,3,1):C.UVGC_3092_2893,(1,3,4):C.UVGC_3093_2898,(1,3,5):C.UVGC_3092_2895,(1,3,2):C.UVGC_3093_2899,(1,3,6):C.UVGC_3093_2900,(0,3,2):C.UVGC_1483_359,(0,3,6):C.UVGC_1483_360})

V_524 = CTVertex(name = 'V_524',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.b, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS27, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.b, P.c, P.g] ], [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ] ],
                 couplings = {(0,8,1):C.UVGC_1858_871,(0,8,9):C.UVGC_1858_872,(0,8,3):C.UVGC_1858_873,(0,8,6):C.UVGC_1858_874,(0,5,7):C.UVGC_1859_875,(0,5,3):C.UVGC_1859_876,(0,5,6):C.UVGC_1859_877,(0,5,2):C.UVGC_1858_871,(0,10,5):C.UVGC_2167_1314,(0,0,3):C.UVGC_1861_881,(0,0,6):C.UVGC_1862_882,(0,0,2):C.UVGC_1860_880,(0,1,3):C.UVGC_1860_878,(0,1,6):C.UVGC_1860_879,(0,1,2):C.UVGC_1860_880,(0,3,3):C.UVGC_1861_881,(0,3,6):C.UVGC_1860_879,(0,3,2):C.UVGC_1860_880,(0,7,3):C.UVGC_1334_214,(0,7,6):C.UVGC_1352_228,(0,4,3):C.UVGC_1333_213,(0,4,6):C.UVGC_1354_233,(0,9,3):C.UVGC_1334_214,(0,6,3):C.UVGC_1333_213,(0,2,0):C.UVGC_1353_229,(0,2,7):C.UVGC_1353_230,(0,2,8):C.UVGC_1353_231,(0,2,4):C.UVGC_1353_232})

V_525 = CTVertex(name = 'V_525',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3518_3656,(0,2,3):C.UVGC_3518_3657,(0,2,1):C.UVGC_2505_1768,(0,2,2):C.UVGC_3519_3659,(0,6,1):C.UVGC_2505_1768,(0,6,2):C.UVGC_2505_1769,(0,0,0):C.UVGC_3518_3656,(0,0,3):C.UVGC_3518_3657,(0,0,1):C.UVGC_3518_3658,(0,3,0):C.UVGC_3518_3656,(0,3,3):C.UVGC_3518_3657,(0,3,1):C.UVGC_3518_3658,(0,4,0):C.UVGC_3518_3656,(0,4,3):C.UVGC_3518_3657,(0,4,1):C.UVGC_3518_3658,(0,1,2):C.UVGC_2503_1767,(0,5,2):C.UVGC_2503_1767})

V_526 = CTVertex(name = 'V_526',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_3435_3501,(0,2,3):C.UVGC_3435_3502,(0,2,0):C.UVGC_3435_3503,(0,2,2):C.UVGC_2376_1577,(0,6,0):C.UVGC_2376_1576,(0,6,2):C.UVGC_2376_1577,(0,0,1):C.UVGC_3435_3501,(0,0,3):C.UVGC_3435_3502,(0,0,2):C.UVGC_3436_3504,(0,3,1):C.UVGC_3435_3501,(0,3,3):C.UVGC_3435_3502,(0,3,2):C.UVGC_3436_3504,(0,4,1):C.UVGC_3435_3501,(0,4,3):C.UVGC_3435_3502,(0,4,2):C.UVGC_3436_3504,(0,1,0):C.UVGC_1913_958,(0,5,0):C.UVGC_1913_958})

V_527 = CTVertex(name = 'V_527',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3534_3696,(0,2,3):C.UVGC_3534_3697,(0,2,1):C.UVGC_2507_1772,(0,2,2):C.UVGC_3534_3698,(0,1,1):C.UVGC_2507_1772,(0,1,2):C.UVGC_2507_1773,(0,0,0):C.UVGC_3533_3692,(0,0,3):C.UVGC_3533_3693,(0,0,1):C.UVGC_3533_3694,(0,0,2):C.UVGC_3533_3695})

V_528 = CTVertex(name = 'V_528',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_3451_3541,(0,2,3):C.UVGC_3451_3542,(0,2,0):C.UVGC_3451_3543,(0,2,2):C.UVGC_2380_1583,(0,1,0):C.UVGC_2380_1582,(0,1,2):C.UVGC_2380_1583,(0,0,1):C.UVGC_3450_3537,(0,0,3):C.UVGC_3450_3538,(0,0,0):C.UVGC_3450_3539,(0,0,2):C.UVGC_3450_3540})

V_529 = CTVertex(name = 'V_529',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.d, P.g] ], [ [P.b, P.d, P.g, P.u] ], [ [P.c, P.d, P.g, P.t] ], [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ], [ [P.d, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,3):C.UVGC_3529_3681,(0,2,7):C.UVGC_3529_3682,(0,2,4):C.UVGC_2506_1770,(0,2,6):C.UVGC_3529_3683,(0,2,0):C.UVGC_2968_2603,(0,2,1):C.UVGC_2968_2604,(0,2,2):C.UVGC_2968_2605,(0,2,5):C.UVGC_2968_2606,(0,1,4):C.UVGC_2506_1770,(0,1,6):C.UVGC_2506_1771,(0,1,0):C.UVGC_2967_2599,(0,1,1):C.UVGC_2967_2600,(0,1,2):C.UVGC_2967_2601,(0,1,5):C.UVGC_2967_2602,(0,0,3):C.UVGC_3530_3684,(0,0,7):C.UVGC_3530_3685,(0,0,4):C.UVGC_3530_3686,(0,0,6):C.UVGC_3530_3687,(0,0,0):C.UVGC_2967_2599,(0,0,1):C.UVGC_2967_2600,(0,0,2):C.UVGC_2968_2605,(0,0,5):C.UVGC_2968_2606})

V_530 = CTVertex(name = 'V_530',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.s] ], [ [P.b, P.g, P.s, P.u] ], [ [P.c, P.g, P.s] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.s, P.t, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,4):C.UVGC_3446_3526,(0,2,7):C.UVGC_3446_3527,(0,2,2):C.UVGC_3446_3528,(0,2,5):C.UVGC_2378_1580,(0,2,0):C.UVGC_2980_2630,(0,2,1):C.UVGC_2980_2631,(0,2,3):C.UVGC_2980_2632,(0,2,6):C.UVGC_2980_2633,(0,1,2):C.UVGC_2378_1579,(0,1,5):C.UVGC_2378_1580,(0,1,0):C.UVGC_2979_2626,(0,1,1):C.UVGC_2979_2627,(0,1,3):C.UVGC_2979_2628,(0,1,6):C.UVGC_2979_2629,(0,0,4):C.UVGC_3447_3529,(0,0,7):C.UVGC_3447_3530,(0,0,2):C.UVGC_3447_3531,(0,0,5):C.UVGC_3447_3532,(0,0,0):C.UVGC_2979_2626,(0,0,1):C.UVGC_2979_2627,(0,0,3):C.UVGC_2980_2632,(0,0,6):C.UVGC_2980_2633})

V_531 = CTVertex(name = 'V_531',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.d, P.g, P.t] ], [ [P.d, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2239_1386,(0,2,6):C.UVGC_3452_3545,(0,2,2):C.UVGC_3455_3551,(0,2,7):C.UVGC_3455_3552,(0,8,4):C.UVGC_2239_1383,(0,8,2):C.UVGC_2239_1384,(0,8,7):C.UVGC_2239_1385,(0,8,3):C.UVGC_2239_1386,(0,6,7):C.UVGC_1595_515,(0,0,0):C.UVGC_2239_1386,(0,0,4):C.UVGC_3452_3544,(0,0,5):C.UVGC_3452_3545,(0,0,2):C.UVGC_3452_3546,(0,0,7):C.UVGC_3452_3547,(0,0,3):C.UVGC_3452_3548,(0,4,0):C.UVGC_2239_1386,(0,4,4):C.UVGC_3452_3544,(0,4,5):C.UVGC_3452_3545,(0,4,2):C.UVGC_3453_3549,(0,4,7):C.UVGC_3454_3550,(0,4,3):C.UVGC_3452_3548,(0,5,0):C.UVGC_2239_1386,(0,5,4):C.UVGC_3452_3544,(0,5,5):C.UVGC_3452_3545,(0,5,2):C.UVGC_3453_3549,(0,5,7):C.UVGC_3452_3547,(0,5,3):C.UVGC_3452_3548,(0,1,2):C.UVGC_1585_505,(0,1,7):C.UVGC_1584_504,(0,7,2):C.UVGC_1587_507,(0,7,7):C.UVGC_1586_506,(0,3,7):C.UVGC_1584_504,(0,9,7):C.UVGC_1586_506})

V_532 = CTVertex(name = 'V_532',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.g, P.s, P.t] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.s] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2369_1567,(0,2,4):C.UVGC_3369_3391,(0,2,5):C.UVGC_3372_3397,(0,2,7):C.UVGC_3372_3398,(0,8,2):C.UVGC_2369_1564,(0,8,5):C.UVGC_2369_1565,(0,8,7):C.UVGC_2369_1566,(0,8,6):C.UVGC_2369_1567,(0,6,7):C.UVGC_1554_460,(0,0,0):C.UVGC_2369_1567,(0,0,2):C.UVGC_3369_3390,(0,0,3):C.UVGC_3369_3391,(0,0,5):C.UVGC_3369_3392,(0,0,7):C.UVGC_3369_3393,(0,0,6):C.UVGC_3369_3394,(0,4,0):C.UVGC_2369_1567,(0,4,2):C.UVGC_3369_3390,(0,4,3):C.UVGC_3369_3391,(0,4,5):C.UVGC_3370_3395,(0,4,7):C.UVGC_3371_3396,(0,4,6):C.UVGC_3369_3394,(0,5,0):C.UVGC_2369_1567,(0,5,2):C.UVGC_3369_3390,(0,5,3):C.UVGC_3369_3391,(0,5,5):C.UVGC_3370_3395,(0,5,7):C.UVGC_3369_3393,(0,5,6):C.UVGC_3369_3394,(0,1,5):C.UVGC_1540_446,(0,1,7):C.UVGC_1539_445,(0,7,5):C.UVGC_1542_448,(0,7,7):C.UVGC_1541_447,(0,3,7):C.UVGC_1539_445,(0,9,7):C.UVGC_1541_447})

V_533 = CTVertex(name = 'V_533',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2483_1736,(0,2,1):C.UVGC_2483_1737,(0,2,2):C.UVGC_2483_1738,(0,6,1):C.UVGC_2484_1739,(0,6,2):C.UVGC_2483_1738,(0,0,0):C.UVGC_2483_1736,(0,0,2):C.UVGC_2485_1740,(0,3,0):C.UVGC_2483_1736,(0,3,2):C.UVGC_2485_1740,(0,4,0):C.UVGC_2483_1736,(0,4,2):C.UVGC_2485_1740,(0,1,1):C.UVGC_2173_1321,(0,5,1):C.UVGC_2173_1321})

V_534 = CTVertex(name = 'V_534',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2491_1750,(0,2,1):C.UVGC_2491_1751,(0,2,2):C.UVGC_2491_1752,(0,1,1):C.UVGC_2492_1753,(0,1,2):C.UVGC_2491_1752,(0,0,0):C.UVGC_2493_1754,(0,0,1):C.UVGC_2493_1755,(0,0,2):C.UVGC_2493_1756})

V_535 = CTVertex(name = 'V_535',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.t] ], [ [P.b, P.g], [P.g, P.u] ], [ [P.b, P.g, P.s, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,2):C.UVGC_2487_1742,(0,2,4):C.UVGC_2487_1743,(0,2,6):C.UVGC_2487_1744,(0,2,0):C.UVGC_2938_2544,(0,2,1):C.UVGC_2938_2545,(0,2,3):C.UVGC_2938_2546,(0,2,5):C.UVGC_2938_2547,(0,1,4):C.UVGC_2488_1745,(0,1,6):C.UVGC_2487_1744,(0,1,0):C.UVGC_2937_2540,(0,1,1):C.UVGC_2937_2541,(0,1,3):C.UVGC_2937_2542,(0,1,5):C.UVGC_2937_2543,(0,0,2):C.UVGC_2489_1746,(0,0,4):C.UVGC_2489_1747,(0,0,6):C.UVGC_2489_1748,(0,0,0):C.UVGC_2938_2544,(0,0,1):C.UVGC_2937_2541,(0,0,3):C.UVGC_2937_2542,(0,0,5):C.UVGC_2938_2547})

V_536 = CTVertex(name = 'V_536',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.b, P.g, P.u] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.u] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,1):C.UVGC_2472_1719,(0,2,7):C.UVGC_2472_1721,(0,2,2):C.UVGC_2476_1730,(0,2,8):C.UVGC_2472_1720,(0,8,5):C.UVGC_2475_1727,(0,8,2):C.UVGC_2475_1728,(0,8,8):C.UVGC_2475_1729,(0,8,4):C.UVGC_2472_1719,(0,6,3):C.UVGC_2107_1235,(0,0,0):C.UVGC_2472_1719,(0,0,5):C.UVGC_2472_1720,(0,0,6):C.UVGC_2472_1721,(0,0,2):C.UVGC_2472_1722,(0,0,8):C.UVGC_2472_1723,(0,0,4):C.UVGC_2472_1724,(0,4,0):C.UVGC_2472_1719,(0,4,5):C.UVGC_2472_1720,(0,4,6):C.UVGC_2472_1721,(0,4,2):C.UVGC_2473_1725,(0,4,8):C.UVGC_2474_1726,(0,4,4):C.UVGC_2472_1724,(0,5,0):C.UVGC_2472_1719,(0,5,5):C.UVGC_2472_1720,(0,5,6):C.UVGC_2472_1721,(0,5,2):C.UVGC_2473_1725,(0,5,8):C.UVGC_2472_1723,(0,5,4):C.UVGC_2472_1724,(0,1,2):C.UVGC_1793_752,(0,1,8):C.UVGC_1792_751,(0,7,2):C.UVGC_1795_754,(0,7,8):C.UVGC_1794_753,(0,3,8):C.UVGC_1792_751,(0,9,8):C.UVGC_1794_753})

V_537 = CTVertex(name = 'V_537',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_1898_939,(0,2,0):C.UVGC_2144_1281,(0,2,2):C.UVGC_2144_1282,(0,6,0):C.UVGC_2144_1281,(0,6,2):C.UVGC_2145_1283,(0,0,1):C.UVGC_1898_939,(0,0,0):C.UVGC_1898_940,(0,3,1):C.UVGC_1898_939,(0,3,0):C.UVGC_1898_940,(0,4,1):C.UVGC_1898_939,(0,4,0):C.UVGC_1898_940,(0,1,2):C.UVGC_2142_1280,(0,5,2):C.UVGC_2142_1280})

V_538 = CTVertex(name = 'V_538',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2160_1303,(0,2,0):C.UVGC_2160_1304,(0,2,2):C.UVGC_2160_1305,(0,1,0):C.UVGC_2160_1304,(0,1,2):C.UVGC_2162_1309,(0,0,1):C.UVGC_2161_1306,(0,0,0):C.UVGC_2161_1307,(0,0,2):C.UVGC_2161_1308})

V_539 = CTVertex(name = 'V_539',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.t] ], [ [P.b, P.g], [P.c, P.g] ], [ [P.b, P.g, P.s, P.t] ], [ [P.b, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ] ],
                 couplings = {(0,2,3):C.UVGC_2154_1295,(0,2,0):C.UVGC_2154_1296,(0,2,5):C.UVGC_2154_1297,(0,2,1):C.UVGC_2932_2522,(0,2,2):C.UVGC_2932_2523,(0,2,4):C.UVGC_2932_2524,(0,2,6):C.UVGC_2932_2525,(0,1,0):C.UVGC_2154_1296,(0,1,5):C.UVGC_2155_1298,(0,1,1):C.UVGC_2931_2518,(0,1,2):C.UVGC_2931_2519,(0,1,4):C.UVGC_2931_2520,(0,1,6):C.UVGC_2931_2521,(0,0,3):C.UVGC_2152_1290,(0,0,0):C.UVGC_2152_1291,(0,0,5):C.UVGC_2152_1292,(0,0,1):C.UVGC_2932_2522,(0,0,2):C.UVGC_2931_2519,(0,0,4):C.UVGC_2931_2520,(0,0,6):C.UVGC_2932_2525})

V_540 = CTVertex(name = 'V_540',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b], [P.c], [P.d], [P.s], [P.u] ], [ [P.b], [P.c], [P.d], [P.s], [P.u], [P.b, P.c, P.g] ], [ [P.b, P.c, P.g] ], [ [P.b, P.g] ], [ [P.b, P.g, P.t] ], [ [P.c, P.g] ], [ [P.g] ], [ [P.ghG] ], [ [P.g], [P.ghG] ] ],
                 couplings = {(0,2,1):C.UVGC_1887_921,(0,2,8):C.UVGC_1887_923,(0,2,3):C.UVGC_1891_932,(0,2,5):C.UVGC_1887_922,(0,8,6):C.UVGC_1890_929,(0,8,3):C.UVGC_1890_930,(0,8,5):C.UVGC_1890_931,(0,8,2):C.UVGC_1887_921,(0,6,4):C.UVGC_2106_1234,(0,0,0):C.UVGC_1887_921,(0,0,6):C.UVGC_1887_922,(0,0,7):C.UVGC_1887_923,(0,0,3):C.UVGC_1887_924,(0,0,5):C.UVGC_1887_925,(0,0,2):C.UVGC_1887_926,(0,4,0):C.UVGC_1887_921,(0,4,6):C.UVGC_1887_922,(0,4,7):C.UVGC_1887_923,(0,4,3):C.UVGC_1888_927,(0,4,5):C.UVGC_1889_928,(0,4,2):C.UVGC_1887_926,(0,5,0):C.UVGC_1887_921,(0,5,6):C.UVGC_1887_922,(0,5,7):C.UVGC_1887_923,(0,5,3):C.UVGC_1888_927,(0,5,5):C.UVGC_1887_925,(0,5,2):C.UVGC_1887_926,(0,1,3):C.UVGC_1368_244,(0,1,5):C.UVGC_1367_243,(0,7,3):C.UVGC_1370_246,(0,7,5):C.UVGC_1369_245,(0,3,5):C.UVGC_1367_243,(0,9,5):C.UVGC_1369_245})

V_541 = CTVertex(name = 'V_541',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5, L.FF6, L.FF7 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t], [P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1660_580,(0,3,0):C.UVGC_1611_530,(0,2,0):C.UVGC_3601_3819,(0,2,2):C.UVGC_3601_3820,(0,5,0):C.UVGC_3641_3944,(0,5,2):C.UVGC_3641_3945,(0,1,1):C.UVGC_1808_765,(0,4,1):C.UVGC_1719_633})

V_542 = CTVertex(name = 'V_542',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5, L.FF6, L.FF7 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g], [P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_1635_554,(0,3,2):C.UVGC_1610_529,(0,2,0):C.UVGC_3592_3801,(0,2,2):C.UVGC_3592_3802,(0,5,0):C.UVGC_3597_3811,(0,5,2):C.UVGC_3597_3812,(0,1,1):C.UVGC_1516_408,(0,4,1):C.UVGC_1465_326})

V_543 = CTVertex(name = 'V_543',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.b ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF4, L.FF7 ],
                 loop_particles = [ [ [P.b, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1330_211,(0,1,0):C.UVGC_1101_1,(0,2,0):C.UVGC_1101_1})

V_544 = CTVertex(name = 'V_544',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF4, L.FF7 ],
                 loop_particles = [ [ [P.c, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1330_211,(0,1,0):C.UVGC_1101_1,(0,2,0):C.UVGC_1101_1})

V_545 = CTVertex(name = 'V_545',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.d ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF4, L.FF7 ],
                 loop_particles = [ [ [P.d, P.g] ] ],
                 couplings = {(0,0,0):C.UVGC_1330_211,(0,1,0):C.UVGC_1101_1,(0,2,0):C.UVGC_1101_1})

V_546 = CTVertex(name = 'V_546',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.s ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF4, L.FF7 ],
                 loop_particles = [ [ [P.g, P.s] ] ],
                 couplings = {(0,0,0):C.UVGC_1330_211,(0,1,0):C.UVGC_1101_1,(0,2,0):C.UVGC_1101_1})

V_547 = CTVertex(name = 'V_547',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5, L.FF6, L.FF7 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g], [P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,2):C.UVGC_3392_3449,(0,3,2):C.UVGC_3152_3001,(0,2,0):C.UVGC_3389_3443,(0,2,2):C.UVGC_3389_3444,(0,5,0):C.UVGC_3146_2989,(0,5,2):C.UVGC_3146_2990,(0,1,1):C.UVGC_1563_467,(0,4,1):C.UVGC_1464_325})

V_548 = CTVertex(name = 'V_548',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF4, L.FF5, L.FF7 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3030_2755,(0,2,0):C.UVGC_3030_2755,(0,1,0):C.UVGC_2985_2642,(0,3,0):C.UVGC_2985_2642})

V_549 = CTVertex(name = 'V_549',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF2, L.FF3, L.FF4, L.FF5, L.FF6, L.FF7 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t], [P.g, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_3475_3603,(0,3,0):C.UVGC_3153_3002,(0,2,0):C.UVGC_3472_3597,(0,2,2):C.UVGC_3472_3598,(0,5,0):C.UVGC_3147_2991,(0,5,2):C.UVGC_3147_2992,(0,1,1):C.UVGC_1769_714,(0,4,1):C.UVGC_1720_634})

V_550 = CTVertex(name = 'V_550',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FF1, L.FF4, L.FF7 ],
                 loop_particles = [ [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1330_211,(0,1,0):C.UVGC_1101_1,(0,2,0):C.UVGC_1101_1})

V_551 = CTVertex(name = 'V_551',
                 type = 'UV',
                 particles = [ P.g, P.g ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VV1, L.VV3 ],
                 loop_particles = [ [ [P.g] ], [ [P.ghG] ], [ [P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_2983_2635,(0,0,1):C.UVGC_2983_2636,(0,0,2):C.UVGC_2983_2637,(0,1,2):C.UVGC_2982_2634})

V_552 = CTVertex(name = 'V_552',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV22, L.FFV3, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,1):C.UVGC_3582_3791,(0,4,1):C.UVGC_3582_3791,(0,0,1):C.UVGC_3298_3231,(0,2,1):C.UVGC_3298_3231,(0,13,0):C.UVGC_2735_2152,(0,13,2):C.UVGC_2735_2153,(0,13,1):C.UVGC_3582_3791,(0,8,0):C.UVGC_2616_1985,(0,8,2):C.UVGC_2616_1986,(0,8,1):C.UVGC_3298_3231,(0,9,1):C.UVGC_3414_3472,(0,3,1):C.UVGC_3309_3248,(0,12,0):C.UVGC_2736_2154,(0,12,2):C.UVGC_2736_2155,(0,7,0):C.UVGC_2617_1987,(0,7,2):C.UVGC_2617_1988,(0,10,0):C.UVGC_2736_2154,(0,10,2):C.UVGC_2736_2155,(0,5,0):C.UVGC_2617_1987,(0,5,2):C.UVGC_2617_1988,(0,11,0):C.UVGC_2734_2150,(0,11,2):C.UVGC_2734_2151,(0,6,0):C.UVGC_2615_1983,(0,6,2):C.UVGC_2615_1984})

V_553 = CTVertex(name = 'V_553',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV22, L.FFV3, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,1,1):C.UVGC_3512_3643,(0,4,1):C.UVGC_3512_3643,(0,0,1):C.UVGC_3303_3236,(0,2,1):C.UVGC_3303_3236,(0,13,0):C.UVGC_2703_2109,(0,13,2):C.UVGC_2703_2110,(0,13,1):C.UVGC_3512_3643,(0,8,0):C.UVGC_2651_2031,(0,8,2):C.UVGC_2651_2032,(0,8,1):C.UVGC_3303_3236,(0,9,1):C.UVGC_3496_3625,(0,3,1):C.UVGC_3310_3249,(0,12,0):C.UVGC_2704_2111,(0,12,2):C.UVGC_2704_2112,(0,7,0):C.UVGC_2652_2033,(0,7,2):C.UVGC_2652_2034,(0,10,0):C.UVGC_2704_2111,(0,10,2):C.UVGC_2704_2112,(0,5,0):C.UVGC_2652_2033,(0,5,2):C.UVGC_2652_2034,(0,11,0):C.UVGC_2702_2107,(0,11,2):C.UVGC_2702_2108,(0,6,0):C.UVGC_2650_2029,(0,6,2):C.UVGC_2650_2030})

V_554 = CTVertex(name = 'V_554',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV19, L.FFV20, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,6,1):C.UVGC_3412_3470,(0,4,0):C.UVGC_2207_1355,(0,2,0):C.UVGC_2207_1355,(0,5,0):C.UVGC_2206_1354,(0,5,1):C.UVGC_3277_3202,(0,3,0):C.UVGC_2205_1353,(0,0,1):C.UVGC_3277_3202,(0,1,1):C.UVGC_3277_3202})

V_555 = CTVertex(name = 'V_555',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV19, L.FFV20, L.FFV31, L.FFV32, L.FFV33, L.FFV34, L.FFV5 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,6,1):C.UVGC_3494_3623,(0,4,0):C.UVGC_2328_1517,(0,2,0):C.UVGC_2328_1517,(0,5,0):C.UVGC_2327_1516,(0,5,1):C.UVGC_3282_3207,(0,3,0):C.UVGC_2326_1515,(0,0,1):C.UVGC_3282_3207,(0,1,1):C.UVGC_3282_3207})

V_556 = CTVertex(name = 'V_556',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV2, L.FFV3, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_1632_551,(0,5,0):C.UVGC_2231_1376,(0,3,0):C.UVGC_2231_1376,(0,6,0):C.UVGC_2230_1375,(0,6,1):C.UVGC_1656_576,(0,4,0):C.UVGC_2229_1374,(0,0,1):C.UVGC_1656_576,(0,1,1):C.UVGC_1656_576})

V_557 = CTVertex(name = 'V_557',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV2, L.FFV3, L.FFV5, L.FFV6, L.FFV7, L.FFV8, L.FFV9 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_1623_542,(0,5,0):C.UVGC_2391_1592,(0,3,0):C.UVGC_2391_1592,(0,6,0):C.UVGC_2390_1591,(0,6,1):C.UVGC_1669_589,(0,4,0):C.UVGC_2389_1590,(0,0,1):C.UVGC_1669_589,(0,1,1):C.UVGC_1669_589})

V_558 = CTVertex(name = 'V_558',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS21, L.FFS22, L.FFS3, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,3,1):C.UVGC_3420_3478,(0,3,0):C.UVGC_3420_3479,(0,0,1):C.UVGC_3304_3237,(0,0,0):C.UVGC_3304_3238,(0,4,0):C.UVGC_2034_1141,(0,1,0):C.UVGC_1991_1085,(0,5,0):C.UVGC_2033_1140,(0,2,0):C.UVGC_1990_1084})

V_559 = CTVertex(name = 'V_559',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS21, L.FFS22, L.FFS3, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,3,1):C.UVGC_3561_3752,(0,3,0):C.UVGC_3561_3753,(0,3,2):C.UVGC_3561_3754,(0,0,1):C.UVGC_3305_3239,(0,0,0):C.UVGC_3305_3240,(0,0,2):C.UVGC_3305_3241,(0,4,0):C.UVGC_2035_1142,(0,4,2):C.UVGC_2731_2145,(0,1,0):C.UVGC_2057_1165,(0,1,2):C.UVGC_2837_2312,(0,5,0):C.UVGC_2035_1142,(0,2,0):C.UVGC_2057_1165})

V_560 = CTVertex(name = 'V_560',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS21, L.FFS22, L.FFS3, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,3,1):C.UVGC_3503_3632,(0,3,0):C.UVGC_3503_3633,(0,3,2):C.UVGC_3503_3634,(0,0,1):C.UVGC_3306_3242,(0,0,0):C.UVGC_3306_3243,(0,0,2):C.UVGC_3306_3244,(0,4,0):C.UVGC_2763_2203,(0,4,2):C.UVGC_2762_2202,(0,1,0):C.UVGC_2690_2090,(0,1,2):C.UVGC_2689_2089,(0,5,2):C.UVGC_2762_2202,(0,2,2):C.UVGC_2689_2089})

V_561 = CTVertex(name = 'V_561',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS18, L.FFS21, L.FFS22, L.FFS3, L.FFS5, L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,3,0):C.UVGC_3562_3755,(0,3,1):C.UVGC_3562_3756,(0,0,0):C.UVGC_3307_3245,(0,0,1):C.UVGC_3307_3246,(0,4,1):C.UVGC_2766_2205,(0,1,1):C.UVGC_2839_2315,(0,5,1):C.UVGC_2764_2204,(0,2,1):C.UVGC_2793_2253})

V_562 = CTVertex(name = 'V_562',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS35, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,0):C.UVGC_2036_1143,(0,6,0):C.UVGC_1992_1086,(0,0,1):C.UVGC_3422_3481,(0,2,1):C.UVGC_3422_3481,(0,8,1):C.UVGC_3422_3481,(0,3,1):C.UVGC_3107_2950,(0,4,1):C.UVGC_3107_2950,(0,5,1):C.UVGC_3107_2950,(0,1,1):C.UVGC_3422_3481,(0,7,1):C.UVGC_3107_2950})

V_563 = CTVertex(name = 'V_563',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS35, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,9,0):C.UVGC_2732_2146,(0,9,2):C.UVGC_2732_2147,(0,6,0):C.UVGC_2840_2316,(0,6,2):C.UVGC_2840_2317,(0,0,1):C.UVGC_3575_3784,(0,2,1):C.UVGC_3575_3784,(0,8,1):C.UVGC_3575_3784,(0,3,1):C.UVGC_3112_2955,(0,4,1):C.UVGC_3112_2955,(0,5,1):C.UVGC_3112_2955,(0,1,1):C.UVGC_3575_3784,(0,7,1):C.UVGC_3112_2955})

V_564 = CTVertex(name = 'V_564',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS35, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,9,0):C.UVGC_2767_2206,(0,9,2):C.UVGC_2767_2207,(0,6,0):C.UVGC_2691_2091,(0,6,2):C.UVGC_2691_2092,(0,0,1):C.UVGC_3505_3636,(0,2,1):C.UVGC_3505_3636,(0,8,1):C.UVGC_3505_3636,(0,3,1):C.UVGC_3117_2960,(0,4,1):C.UVGC_3117_2960,(0,5,1):C.UVGC_3117_2960,(0,1,1):C.UVGC_3505_3636,(0,7,1):C.UVGC_3117_2960})

V_565 = CTVertex(name = 'V_565',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS35, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,9,1):C.UVGC_2768_2208,(0,6,1):C.UVGC_2841_2318,(0,0,0):C.UVGC_3584_3793,(0,2,0):C.UVGC_3584_3793,(0,8,0):C.UVGC_3584_3793,(0,3,0):C.UVGC_3122_2965,(0,4,0):C.UVGC_3122_2965,(0,5,0):C.UVGC_3122_2965,(0,1,0):C.UVGC_3584_3793,(0,7,0):C.UVGC_3122_2965})

V_566 = CTVertex(name = 'V_566',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS35, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,9,1):C.UVGC_1992_1086,(0,9,3):C.UVGC_2841_2318,(0,6,1):C.UVGC_2036_1143,(0,6,3):C.UVGC_2768_2208,(0,0,0):C.UVGC_3623_3890,(0,0,2):C.UVGC_3623_3891,(0,0,4):C.UVGC_3623_3892,(0,2,0):C.UVGC_3623_3890,(0,2,2):C.UVGC_3623_3891,(0,2,4):C.UVGC_3623_3892,(0,8,0):C.UVGC_3623_3890,(0,8,2):C.UVGC_3623_3891,(0,8,4):C.UVGC_3623_3892,(0,3,0):C.UVGC_3632_3917,(0,3,2):C.UVGC_3632_3918,(0,3,4):C.UVGC_3632_3919,(0,4,0):C.UVGC_3632_3917,(0,4,2):C.UVGC_3632_3918,(0,4,4):C.UVGC_3632_3919,(0,5,0):C.UVGC_3632_3917,(0,5,2):C.UVGC_3632_3918,(0,5,4):C.UVGC_3632_3919,(0,1,0):C.UVGC_3623_3890,(0,1,2):C.UVGC_3623_3891,(0,1,4):C.UVGC_3623_3892,(0,7,0):C.UVGC_3632_3917,(0,7,2):C.UVGC_3632_3918,(0,7,4):C.UVGC_3632_3919})

V_567 = CTVertex(name = 'V_567',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,16,1):C.UVGC_3408_3466,(0,16,0):C.UVGC_2038_1145,(0,8,1):C.UVGC_3296_3229,(0,8,0):C.UVGC_1994_1088,(0,0,1):C.UVGC_3425_3484,(0,4,1):C.UVGC_3425_3484,(0,15,1):C.UVGC_3425_3484,(0,5,1):C.UVGC_3217_3132,(0,6,1):C.UVGC_3217_3132,(0,7,1):C.UVGC_3217_3132,(0,2,0):C.UVGC_2014_1125,(0,2,1):C.UVGC_3425_3484,(0,13,0):C.UVGC_1974_1054,(0,13,1):C.UVGC_3217_3132,(0,1,0):C.UVGC_2017_1127,(0,12,0):C.UVGC_1975_1055,(0,17,0):C.UVGC_2017_1127,(0,9,0):C.UVGC_1975_1055,(0,18,0):C.UVGC_2013_1124,(0,10,0):C.UVGC_1973_1053,(0,3,0):C.UVGC_2015_1126,(0,14,0):C.UVGC_1961_1046,(0,19,0):C.UVGC_2015_1126,(0,11,0):C.UVGC_1961_1046})

V_568 = CTVertex(name = 'V_568',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,16,1):C.UVGC_3410_3468,(0,16,0):C.UVGC_2738_2156,(0,16,2):C.UVGC_2738_2157,(0,8,0):C.UVGC_2846_2322,(0,8,2):C.UVGC_2846_2323,(0,8,1):C.UVGC_3297_3230,(0,0,1):C.UVGC_3578_3787,(0,4,1):C.UVGC_3578_3787,(0,15,1):C.UVGC_3578_3787,(0,5,1):C.UVGC_3213_3128,(0,6,1):C.UVGC_3213_3128,(0,7,1):C.UVGC_3213_3128,(0,2,0):C.UVGC_2720_2135,(0,2,2):C.UVGC_2720_2136,(0,2,1):C.UVGC_3578_3787,(0,13,0):C.UVGC_2596_1961,(0,13,2):C.UVGC_2596_1962,(0,13,1):C.UVGC_3213_3128,(0,1,0):C.UVGC_2723_2138,(0,1,2):C.UVGC_2721_2137,(0,12,0):C.UVGC_2599_1964,(0,12,2):C.UVGC_2597_1963,(0,17,0):C.UVGC_2723_2138,(0,17,2):C.UVGC_2721_2137,(0,9,0):C.UVGC_2599_1964,(0,9,2):C.UVGC_2597_1963,(0,18,0):C.UVGC_2719_2133,(0,18,2):C.UVGC_2719_2134,(0,10,0):C.UVGC_2595_1959,(0,10,2):C.UVGC_2595_1960,(0,3,2):C.UVGC_2721_2137,(0,14,2):C.UVGC_2597_1963,(0,19,2):C.UVGC_2721_2137,(0,11,2):C.UVGC_2597_1963})

V_569 = CTVertex(name = 'V_569',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,16,1):C.UVGC_3491_3620,(0,16,0):C.UVGC_2773_2212,(0,16,2):C.UVGC_2773_2213,(0,8,1):C.UVGC_3300_3233,(0,8,0):C.UVGC_2693_2095,(0,8,2):C.UVGC_2693_2096,(0,0,1):C.UVGC_3508_3639,(0,4,1):C.UVGC_3508_3639,(0,15,1):C.UVGC_3508_3639,(0,5,1):C.UVGC_3225_3140,(0,6,1):C.UVGC_3225_3140,(0,7,1):C.UVGC_3225_3140,(0,2,0):C.UVGC_2699_2104,(0,2,2):C.UVGC_2699_2105,(0,2,1):C.UVGC_3508_3639,(0,13,0):C.UVGC_2635_2015,(0,13,2):C.UVGC_2635_2016,(0,13,1):C.UVGC_3225_3140,(0,1,0):C.UVGC_2053_1163,(0,1,2):C.UVGC_2700_2106,(0,12,0):C.UVGC_1965_1048,(0,12,2):C.UVGC_2636_2017,(0,17,0):C.UVGC_2053_1163,(0,17,2):C.UVGC_2700_2106,(0,9,0):C.UVGC_1965_1048,(0,9,2):C.UVGC_2636_2017,(0,18,0):C.UVGC_2698_2102,(0,18,2):C.UVGC_2698_2103,(0,10,0):C.UVGC_2634_2013,(0,10,2):C.UVGC_2634_2014,(0,3,0):C.UVGC_2053_1163,(0,14,0):C.UVGC_1965_1048,(0,19,0):C.UVGC_2053_1163,(0,11,0):C.UVGC_1965_1048})

V_570 = CTVertex(name = 'V_570',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,16,0):C.UVGC_3493_3622,(0,16,1):C.UVGC_2775_2216,(0,8,0):C.UVGC_3301_3234,(0,8,1):C.UVGC_2847_2324,(0,0,0):C.UVGC_3587_3796,(0,4,0):C.UVGC_3587_3796,(0,15,0):C.UVGC_3587_3796,(0,5,0):C.UVGC_3221_3136,(0,6,0):C.UVGC_3221_3136,(0,7,0):C.UVGC_3221_3136,(0,2,1):C.UVGC_2826_2302,(0,2,0):C.UVGC_3587_3796,(0,13,1):C.UVGC_2647_2027,(0,13,0):C.UVGC_3221_3136,(0,1,1):C.UVGC_2827_2303,(0,12,1):C.UVGC_2648_2028,(0,17,1):C.UVGC_2827_2303,(0,9,1):C.UVGC_2648_2028,(0,18,1):C.UVGC_2825_2301,(0,10,1):C.UVGC_2646_2026,(0,3,1):C.UVGC_2757_2198,(0,14,1):C.UVGC_2611_1981,(0,19,1):C.UVGC_2757_2198,(0,11,1):C.UVGC_2611_1981})

V_571 = CTVertex(name = 'V_571',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,16,0):C.UVGC_3613_3854,(0,16,2):C.UVGC_3613_3855,(0,16,4):C.UVGC_3613_3856,(0,16,1):[ C.UVGC_3613_3857, C.UVGC_2848_2325 ],(0,16,3):[ C.UVGC_3613_3858, C.UVGC_2848_2326 ],(0,8,0):C.UVGC_3670_4037,(0,8,2):C.UVGC_3670_4038,(0,8,4):C.UVGC_3670_4039,(0,8,1):[ C.UVGC_3670_4040, C.UVGC_2774_2214 ],(0,8,3):[ C.UVGC_3670_4041, C.UVGC_2774_2215 ],(0,0,0):C.UVGC_3626_3899,(0,0,2):C.UVGC_3626_3900,(0,0,4):C.UVGC_3626_3901,(0,4,0):C.UVGC_3626_3899,(0,4,2):C.UVGC_3626_3900,(0,4,4):C.UVGC_3626_3901,(0,15,0):C.UVGC_3626_3899,(0,15,2):C.UVGC_3626_3900,(0,15,4):C.UVGC_3626_3901,(0,5,0):C.UVGC_3650_3968,(0,5,2):C.UVGC_3650_3969,(0,5,4):C.UVGC_3650_3970,(0,6,0):C.UVGC_3650_3968,(0,6,2):C.UVGC_3650_3969,(0,6,4):C.UVGC_3650_3970,(0,7,0):C.UVGC_3650_3968,(0,7,2):C.UVGC_3650_3969,(0,7,4):C.UVGC_3650_3970,(0,2,1):C.UVGC_2802_2268,(0,2,3):C.UVGC_2802_2269,(0,2,0):C.UVGC_3626_3899,(0,2,2):C.UVGC_3626_3900,(0,2,4):C.UVGC_3626_3901,(0,13,1):C.UVGC_2670_2059,(0,13,3):C.UVGC_2670_2060,(0,13,0):C.UVGC_3650_3968,(0,13,2):C.UVGC_3650_3969,(0,13,4):C.UVGC_3650_3970,(0,1,1):C.UVGC_2805_2272,(0,1,3):C.UVGC_2805_2273,(0,12,1):C.UVGC_2673_2063,(0,12,3):C.UVGC_2673_2064,(0,17,1):C.UVGC_2805_2272,(0,17,3):C.UVGC_2805_2273,(0,9,1):C.UVGC_2673_2063,(0,9,3):C.UVGC_2673_2064,(0,18,1):C.UVGC_2801_2266,(0,18,3):C.UVGC_2801_2267,(0,10,1):C.UVGC_2669_2057,(0,10,3):C.UVGC_2669_2058,(0,3,1):C.UVGC_2803_2270,(0,3,3):C.UVGC_2803_2271,(0,14,1):C.UVGC_2671_2061,(0,14,3):C.UVGC_2671_2062,(0,19,1):C.UVGC_2803_2270,(0,19,3):C.UVGC_2803_2271,(0,11,1):C.UVGC_2671_2061,(0,11,3):C.UVGC_2671_2062})

V_572 = CTVertex(name = 'V_572',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,1):C.UVGC_3405_3463,(0,6,0):C.UVGC_2312_1505,(0,3,0):C.UVGC_2312_1505,(0,7,0):C.UVGC_2311_1504,(0,7,1):C.UVGC_3185_3084,(0,4,0):C.UVGC_2310_1503,(0,8,0):C.UVGC_2312_1505,(0,5,0):C.UVGC_2312_1505,(0,0,1):C.UVGC_3185_3084,(0,1,1):C.UVGC_3185_3084,(0,2,1):C.UVGC_3185_3084})

V_573 = CTVertex(name = 'V_573',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,1):C.UVGC_3406_3464,(0,6,0):C.UVGC_2191_1343,(0,3,0):C.UVGC_2191_1343,(0,7,0):C.UVGC_2190_1342,(0,7,1):C.UVGC_3190_3089,(0,4,0):C.UVGC_2189_1341,(0,8,0):C.UVGC_2191_1343,(0,5,0):C.UVGC_2191_1343,(0,0,1):C.UVGC_3190_3089,(0,1,1):C.UVGC_3190_3089,(0,2,1):C.UVGC_3190_3089})

V_574 = CTVertex(name = 'V_574',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,1):C.UVGC_3488_3617,(0,6,0):C.UVGC_2320_1511,(0,3,0):C.UVGC_2320_1511,(0,7,0):C.UVGC_2319_1510,(0,7,1):C.UVGC_3195_3094,(0,4,0):C.UVGC_2318_1509,(0,8,0):C.UVGC_2320_1511,(0,5,0):C.UVGC_2320_1511,(0,0,1):C.UVGC_3195_3094,(0,1,1):C.UVGC_3195_3094,(0,2,1):C.UVGC_3195_3094})

V_575 = CTVertex(name = 'V_575',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,1):C.UVGC_3489_3618,(0,6,0):C.UVGC_2199_1349,(0,3,0):C.UVGC_2199_1349,(0,7,0):C.UVGC_2198_1348,(0,7,1):C.UVGC_3200_3099,(0,4,0):C.UVGC_2197_1347,(0,8,0):C.UVGC_2199_1349,(0,5,0):C.UVGC_2199_1349,(0,0,1):C.UVGC_3200_3099,(0,1,1):C.UVGC_3200_3099,(0,2,1):C.UVGC_3200_3099})

V_576 = CTVertex(name = 'V_576',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,6,0):C.UVGC_2439_1666,(0,6,1):C.UVGC_2439_1667,(0,3,0):C.UVGC_2439_1666,(0,3,1):C.UVGC_2439_1667,(0,7,0):C.UVGC_2438_1664,(0,7,1):C.UVGC_2438_1665,(0,7,2):C.UVGC_3647_3959,(0,7,3):C.UVGC_3647_3960,(0,7,4):C.UVGC_3647_3961,(0,4,0):C.UVGC_2437_1662,(0,4,1):C.UVGC_2437_1663,(0,8,0):C.UVGC_2439_1666,(0,8,1):C.UVGC_2439_1667,(0,5,0):C.UVGC_2439_1666,(0,5,1):C.UVGC_2439_1667,(0,9,2):C.UVGC_3611_3844,(0,9,3):C.UVGC_3611_3845,(0,9,4):C.UVGC_3611_3846,(0,0,2):C.UVGC_3647_3959,(0,0,3):C.UVGC_3647_3960,(0,0,4):C.UVGC_3647_3961,(0,1,2):C.UVGC_3647_3959,(0,1,3):C.UVGC_3647_3960,(0,1,4):C.UVGC_3647_3961,(0,2,2):C.UVGC_3647_3959,(0,2,3):C.UVGC_3647_3960,(0,2,4):C.UVGC_3647_3961})

V_577 = CTVertex(name = 'V_577',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,1):C.UVGC_1628_547,(0,1,0):C.UVGC_2227_1373,(0,6,0):C.UVGC_2227_1373,(0,2,0):C.UVGC_2226_1372,(0,2,1):C.UVGC_1651_571,(0,7,0):C.UVGC_2225_1371,(0,0,1):C.UVGC_1651_571,(0,3,1):C.UVGC_1651_571,(0,4,1):C.UVGC_1651_571})

V_578 = CTVertex(name = 'V_578',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,1):C.UVGC_1629_548,(0,1,0):C.UVGC_2250_1396,(0,6,0):C.UVGC_2250_1396,(0,2,0):C.UVGC_2249_1395,(0,2,1):C.UVGC_1675_595,(0,7,0):C.UVGC_2248_1394,(0,0,1):C.UVGC_1675_595,(0,3,1):C.UVGC_1675_595,(0,4,1):C.UVGC_1675_595})

V_579 = CTVertex(name = 'V_579',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,1):C.UVGC_1618_537,(0,1,0):C.UVGC_2358_1553,(0,6,0):C.UVGC_2358_1553,(0,2,0):C.UVGC_2357_1552,(0,2,1):C.UVGC_1640_560,(0,7,0):C.UVGC_2356_1551,(0,0,1):C.UVGC_1640_560,(0,3,1):C.UVGC_1640_560,(0,4,1):C.UVGC_1640_560})

V_580 = CTVertex(name = 'V_580',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,5,1):C.UVGC_1619_538,(0,1,0):C.UVGC_2387_1589,(0,6,0):C.UVGC_2387_1589,(0,2,0):C.UVGC_2386_1588,(0,2,1):C.UVGC_1664_584,(0,7,0):C.UVGC_2385_1587,(0,0,1):C.UVGC_1664_584,(0,3,1):C.UVGC_1664_584,(0,4,1):C.UVGC_1664_584})

V_581 = CTVertex(name = 'V_581',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,1,0):C.UVGC_2463_1703,(0,1,1):C.UVGC_2463_1704,(0,6,0):C.UVGC_2463_1703,(0,6,1):C.UVGC_2463_1704,(0,2,0):C.UVGC_2462_1701,(0,2,1):C.UVGC_2462_1702,(0,2,2):C.UVGC_1841_833,(0,2,3):C.UVGC_1841_834,(0,2,4):C.UVGC_1841_835,(0,7,0):C.UVGC_2461_1699,(0,7,1):C.UVGC_2461_1700,(0,5,2):C.UVGC_1836_818,(0,5,3):C.UVGC_1836_819,(0,5,4):C.UVGC_1836_820,(0,0,2):C.UVGC_1841_833,(0,0,3):C.UVGC_1841_834,(0,0,4):C.UVGC_1841_835,(0,3,2):C.UVGC_1841_833,(0,3,3):C.UVGC_1841_834,(0,3,4):C.UVGC_1841_835,(0,4,2):C.UVGC_1841_833,(0,4,3):C.UVGC_1841_834,(0,4,4):C.UVGC_1841_835})

V_582 = CTVertex(name = 'V_582',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS35, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,0):C.UVGC_2037_1144,(0,6,0):C.UVGC_1993_1087,(0,0,1):C.UVGC_3416_3474,(0,2,1):C.UVGC_3416_3474,(0,8,1):C.UVGC_3416_3474,(0,3,1):C.UVGC_3127_2970,(0,4,1):C.UVGC_3127_2970,(0,5,1):C.UVGC_3127_2970,(0,1,1):C.UVGC_3416_3474,(0,7,1):C.UVGC_3127_2970})

V_583 = CTVertex(name = 'V_583',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS35, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,9,0):C.UVGC_2733_2148,(0,9,2):C.UVGC_2733_2149,(0,6,0):C.UVGC_2843_2319,(0,6,2):C.UVGC_2843_2320,(0,0,1):C.UVGC_3551_3740,(0,2,1):C.UVGC_3551_3740,(0,8,1):C.UVGC_3551_3740,(0,3,1):C.UVGC_3132_2975,(0,4,1):C.UVGC_3132_2975,(0,5,1):C.UVGC_3132_2975,(0,1,1):C.UVGC_3551_3740,(0,7,1):C.UVGC_3132_2975})

V_584 = CTVertex(name = 'V_584',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS35, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,9,0):C.UVGC_2770_2209,(0,9,2):C.UVGC_2770_2210,(0,6,0):C.UVGC_2692_2093,(0,6,2):C.UVGC_2692_2094,(0,0,1):C.UVGC_3499_3628,(0,2,1):C.UVGC_3499_3628,(0,8,1):C.UVGC_3499_3628,(0,3,1):C.UVGC_3137_2980,(0,4,1):C.UVGC_3137_2980,(0,5,1):C.UVGC_3137_2980,(0,1,1):C.UVGC_3499_3628,(0,7,1):C.UVGC_3137_2980})

V_585 = CTVertex(name = 'V_585',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS35, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,9,1):C.UVGC_2771_2211,(0,6,1):C.UVGC_2844_2321,(0,0,0):C.UVGC_3556_3745,(0,2,0):C.UVGC_3556_3745,(0,8,0):C.UVGC_3556_3745,(0,3,0):C.UVGC_3142_2985,(0,4,0):C.UVGC_3142_2985,(0,5,0):C.UVGC_3142_2985,(0,1,0):C.UVGC_3556_3745,(0,7,0):C.UVGC_3142_2985})

V_586 = CTVertex(name = 'V_586',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.H ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS30, L.FFVS35, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,9,1):C.UVGC_1993_1087,(0,9,3):C.UVGC_2844_2321,(0,6,1):C.UVGC_2037_1144,(0,6,3):C.UVGC_2771_2211,(0,0,0):C.UVGC_3617_3870,(0,0,2):C.UVGC_3617_3871,(0,0,4):C.UVGC_3617_3872,(0,2,0):C.UVGC_3617_3870,(0,2,2):C.UVGC_3617_3871,(0,2,4):C.UVGC_3617_3872,(0,8,0):C.UVGC_3617_3870,(0,8,2):C.UVGC_3617_3871,(0,8,4):C.UVGC_3617_3872,(0,3,0):C.UVGC_3637_3932,(0,3,2):C.UVGC_3637_3933,(0,3,4):C.UVGC_3637_3934,(0,4,0):C.UVGC_3637_3932,(0,4,2):C.UVGC_3637_3933,(0,4,4):C.UVGC_3637_3934,(0,5,0):C.UVGC_3637_3932,(0,5,2):C.UVGC_3637_3933,(0,5,4):C.UVGC_3637_3934,(0,1,0):C.UVGC_3617_3870,(0,1,2):C.UVGC_3617_3871,(0,1,4):C.UVGC_3617_3872,(0,7,0):C.UVGC_3637_3932,(0,7,2):C.UVGC_3637_3933,(0,7,4):C.UVGC_3637_3934})

V_587 = CTVertex(name = 'V_587',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS27, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_3424_3483,(0,4,1):C.UVGC_3424_3483,(0,12,1):C.UVGC_3424_3483,(0,5,1):C.UVGC_3218_3133,(0,2,0):C.UVGC_2020_1129,(0,2,1):C.UVGC_3424_3483,(0,10,0):C.UVGC_1971_1051,(0,10,1):C.UVGC_3216_3131,(0,1,0):C.UVGC_2023_1131,(0,9,0):C.UVGC_1969_1050,(0,13,0):C.UVGC_2023_1131,(0,6,0):C.UVGC_1969_1050,(0,14,0):C.UVGC_2019_1128,(0,7,0):C.UVGC_1972_1052,(0,3,0):C.UVGC_2021_1130,(0,11,0):C.UVGC_1963_1047,(0,15,0):C.UVGC_2021_1130,(0,8,0):C.UVGC_1963_1047})

V_588 = CTVertex(name = 'V_588',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS27, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,0,1):C.UVGC_3577_3786,(0,4,1):C.UVGC_3577_3786,(0,12,1):C.UVGC_3577_3786,(0,5,1):C.UVGC_3214_3129,(0,2,0):C.UVGC_2726_2141,(0,2,2):C.UVGC_2726_2142,(0,2,1):C.UVGC_3577_3786,(0,10,0):C.UVGC_2602_1967,(0,10,2):C.UVGC_2602_1968,(0,10,1):C.UVGC_3212_3127,(0,1,0):C.UVGC_2729_2144,(0,1,2):C.UVGC_2727_2143,(0,9,0):C.UVGC_2605_1970,(0,9,2):C.UVGC_2603_1969,(0,13,0):C.UVGC_2729_2144,(0,13,2):C.UVGC_2727_2143,(0,6,0):C.UVGC_2605_1970,(0,6,2):C.UVGC_2603_1969,(0,14,0):C.UVGC_2725_2139,(0,14,2):C.UVGC_2725_2140,(0,7,0):C.UVGC_2601_1965,(0,7,2):C.UVGC_2601_1966,(0,3,2):C.UVGC_2727_2143,(0,11,2):C.UVGC_2603_1969,(0,15,2):C.UVGC_2727_2143,(0,8,2):C.UVGC_2603_1969})

V_589 = CTVertex(name = 'V_589',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS27, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,0,1):C.UVGC_3507_3638,(0,4,1):C.UVGC_3507_3638,(0,12,1):C.UVGC_3507_3638,(0,5,1):C.UVGC_3226_3141,(0,2,0):C.UVGC_2696_2098,(0,2,2):C.UVGC_2696_2099,(0,2,1):C.UVGC_3507_3638,(0,10,0):C.UVGC_2632_2009,(0,10,2):C.UVGC_2632_2010,(0,10,1):C.UVGC_3224_3139,(0,1,0):C.UVGC_2055_1164,(0,1,2):C.UVGC_2694_2097,(0,9,0):C.UVGC_1967_1049,(0,9,2):C.UVGC_2630_2008,(0,13,0):C.UVGC_2055_1164,(0,13,2):C.UVGC_2694_2097,(0,6,0):C.UVGC_1967_1049,(0,6,2):C.UVGC_2630_2008,(0,14,0):C.UVGC_2697_2100,(0,14,2):C.UVGC_2697_2101,(0,7,0):C.UVGC_2633_2011,(0,7,2):C.UVGC_2633_2012,(0,3,0):C.UVGC_2055_1164,(0,11,0):C.UVGC_1967_1049,(0,15,0):C.UVGC_2055_1164,(0,8,0):C.UVGC_1967_1049})

V_590 = CTVertex(name = 'V_590',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS27, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_3586_3795,(0,4,0):C.UVGC_3586_3795,(0,12,0):C.UVGC_3586_3795,(0,5,0):C.UVGC_3222_3137,(0,2,1):C.UVGC_2823_2299,(0,2,0):C.UVGC_3586_3795,(0,10,1):C.UVGC_2644_2024,(0,10,0):C.UVGC_3220_3135,(0,1,1):C.UVGC_2821_2298,(0,9,1):C.UVGC_2642_2023,(0,13,1):C.UVGC_2821_2298,(0,6,1):C.UVGC_2642_2023,(0,14,1):C.UVGC_2824_2300,(0,7,1):C.UVGC_2645_2025,(0,3,1):C.UVGC_2759_2199,(0,11,1):C.UVGC_2613_1982,(0,15,1):C.UVGC_2759_2199,(0,8,1):C.UVGC_2613_1982})

V_591 = CTVertex(name = 'V_591',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS2, L.FFVS27, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS4, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_3625_3896,(0,0,2):C.UVGC_3625_3897,(0,0,4):C.UVGC_3625_3898,(0,4,0):C.UVGC_3625_3896,(0,4,2):C.UVGC_3625_3897,(0,4,4):C.UVGC_3625_3898,(0,12,0):C.UVGC_3625_3896,(0,12,2):C.UVGC_3625_3897,(0,12,4):C.UVGC_3625_3898,(0,5,0):C.UVGC_3651_3971,(0,5,2):C.UVGC_3651_3972,(0,5,4):C.UVGC_3651_3973,(0,2,1):C.UVGC_2808_2276,(0,2,3):C.UVGC_2808_2277,(0,2,0):C.UVGC_3625_3896,(0,2,2):C.UVGC_3625_3897,(0,2,4):C.UVGC_3625_3898,(0,10,1):C.UVGC_2676_2067,(0,10,3):C.UVGC_2676_2068,(0,10,0):C.UVGC_3649_3965,(0,10,2):C.UVGC_3649_3966,(0,10,4):C.UVGC_3649_3967,(0,1,1):C.UVGC_2811_2280,(0,1,3):C.UVGC_2811_2281,(0,9,1):C.UVGC_2679_2071,(0,9,3):C.UVGC_2679_2072,(0,13,1):C.UVGC_2811_2280,(0,13,3):C.UVGC_2811_2281,(0,6,1):C.UVGC_2679_2071,(0,6,3):C.UVGC_2679_2072,(0,14,1):C.UVGC_2807_2274,(0,14,3):C.UVGC_2807_2275,(0,7,1):C.UVGC_2675_2065,(0,7,3):C.UVGC_2675_2066,(0,3,1):C.UVGC_2809_2278,(0,3,3):C.UVGC_2809_2279,(0,11,1):C.UVGC_2677_2069,(0,11,3):C.UVGC_2677_2070,(0,15,1):C.UVGC_2809_2278,(0,15,3):C.UVGC_2809_2279,(0,8,1):C.UVGC_2677_2069,(0,8,3):C.UVGC_2677_2070})

V_592 = CTVertex(name = 'V_592',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,0):C.UVGC_2316_1508,(0,1,0):C.UVGC_2316_1508,(0,5,0):C.UVGC_2315_1507,(0,5,1):C.UVGC_3186_3085,(0,2,0):C.UVGC_2314_1506,(0,6,0):C.UVGC_2316_1508,(0,3,0):C.UVGC_2316_1508,(0,0,1):C.UVGC_3184_3083})

V_593 = CTVertex(name = 'V_593',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,0):C.UVGC_2195_1346,(0,1,0):C.UVGC_2195_1346,(0,5,0):C.UVGC_2194_1345,(0,5,1):C.UVGC_3191_3090,(0,2,0):C.UVGC_2193_1344,(0,6,0):C.UVGC_2195_1346,(0,3,0):C.UVGC_2195_1346,(0,0,1):C.UVGC_3189_3088})

V_594 = CTVertex(name = 'V_594',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,0):C.UVGC_2324_1514,(0,1,0):C.UVGC_2324_1514,(0,5,0):C.UVGC_2323_1513,(0,5,1):C.UVGC_3196_3095,(0,2,0):C.UVGC_2322_1512,(0,6,0):C.UVGC_2324_1514,(0,3,0):C.UVGC_2324_1514,(0,0,1):C.UVGC_3194_3093})

V_595 = CTVertex(name = 'V_595',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,0):C.UVGC_2203_1352,(0,1,0):C.UVGC_2203_1352,(0,5,0):C.UVGC_2202_1351,(0,5,1):C.UVGC_3201_3100,(0,2,0):C.UVGC_2201_1350,(0,6,0):C.UVGC_2203_1352,(0,3,0):C.UVGC_2203_1352,(0,0,1):C.UVGC_3199_3098})

V_596 = CTVertex(name = 'V_596',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,0):C.UVGC_2443_1672,(0,4,1):C.UVGC_2443_1673,(0,1,0):C.UVGC_2443_1672,(0,1,1):C.UVGC_2443_1673,(0,5,0):C.UVGC_2442_1670,(0,5,1):C.UVGC_2442_1671,(0,5,2):C.UVGC_3648_3962,(0,5,3):C.UVGC_3648_3963,(0,5,4):C.UVGC_3648_3964,(0,2,0):C.UVGC_2441_1668,(0,2,1):C.UVGC_2441_1669,(0,6,0):C.UVGC_2443_1672,(0,6,1):C.UVGC_2443_1673,(0,3,0):C.UVGC_2443_1672,(0,3,1):C.UVGC_2443_1673,(0,0,2):C.UVGC_3646_3956,(0,0,3):C.UVGC_3646_3957,(0,0,4):C.UVGC_3646_3958})

V_597 = CTVertex(name = 'V_597',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2221_1368,(0,5,0):C.UVGC_2221_1368,(0,2,0):C.UVGC_2223_1369,(0,2,1):C.UVGC_1650_570,(0,6,0):C.UVGC_2224_1370,(0,0,1):C.UVGC_1650_570,(0,3,1):C.UVGC_1650_570,(0,4,1):C.UVGC_1650_570})

V_598 = CTVertex(name = 'V_598',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2244_1391,(0,5,0):C.UVGC_2244_1391,(0,2,0):C.UVGC_2246_1392,(0,2,1):C.UVGC_1674_594,(0,6,0):C.UVGC_2247_1393,(0,0,1):C.UVGC_1674_594,(0,3,1):C.UVGC_1674_594,(0,4,1):C.UVGC_1674_594})

V_599 = CTVertex(name = 'V_599',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2352_1548,(0,5,0):C.UVGC_2352_1548,(0,2,0):C.UVGC_2354_1549,(0,2,1):C.UVGC_1639_559,(0,6,0):C.UVGC_2355_1550,(0,0,1):C.UVGC_1639_559,(0,3,1):C.UVGC_1639_559,(0,4,1):C.UVGC_1639_559})

V_600 = CTVertex(name = 'V_600',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_2381_1584,(0,5,0):C.UVGC_2381_1584,(0,2,0):C.UVGC_2383_1585,(0,2,1):C.UVGC_1663_583,(0,6,0):C.UVGC_2384_1586,(0,0,1):C.UVGC_1663_583,(0,3,1):C.UVGC_1663_583,(0,4,1):C.UVGC_1663_583})

V_601 = CTVertex(name = 'V_601',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,1,0):C.UVGC_2457_1693,(0,1,1):C.UVGC_2457_1694,(0,5,0):C.UVGC_2457_1693,(0,5,1):C.UVGC_2457_1694,(0,2,0):C.UVGC_2459_1695,(0,2,1):C.UVGC_2459_1696,(0,2,2):C.UVGC_1840_830,(0,2,3):C.UVGC_1840_831,(0,2,4):C.UVGC_1840_832,(0,6,0):C.UVGC_2460_1697,(0,6,1):C.UVGC_2460_1698,(0,0,2):C.UVGC_1840_830,(0,0,3):C.UVGC_1840_831,(0,0,4):C.UVGC_1840_832,(0,3,2):C.UVGC_1840_830,(0,3,3):C.UVGC_1840_831,(0,3,4):C.UVGC_1840_832,(0,4,2):C.UVGC_1840_830,(0,4,3):C.UVGC_1840_831,(0,4,4):C.UVGC_1840_832})

V_602 = CTVertex(name = 'V_602',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.s] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,2):C.UVGC_3407_3465,(0,6,0):C.UVGC_2084_1206,(0,6,1):C.UVGC_2086_1207,(0,3,0):C.UVGC_2084_1206,(0,3,1):C.UVGC_2086_1207,(0,7,0):C.UVGC_2088_1208,(0,7,1):C.UVGC_2088_1209,(0,7,2):C.UVGC_3219_3134,(0,4,0):C.UVGC_2089_1210,(0,4,1):C.UVGC_2089_1211,(0,8,0):C.UVGC_2084_1206,(0,5,0):C.UVGC_2084_1206,(0,0,2):C.UVGC_3219_3134,(0,1,2):C.UVGC_3219_3134,(0,2,2):C.UVGC_3219_3134})

V_603 = CTVertex(name = 'V_603',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,9,2):C.UVGC_3409_3467,(0,6,0):C.UVGC_2062_1174,(0,6,1):C.UVGC_2064_1175,(0,3,0):C.UVGC_2062_1174,(0,3,1):C.UVGC_2064_1175,(0,7,0):C.UVGC_2066_1176,(0,7,1):C.UVGC_2066_1177,(0,7,2):C.UVGC_3215_3130,(0,4,0):C.UVGC_2067_1178,(0,4,1):C.UVGC_2067_1179,(0,8,0):C.UVGC_2062_1174,(0,5,0):C.UVGC_2062_1174,(0,0,2):C.UVGC_3215_3130,(0,1,2):C.UVGC_3215_3130,(0,2,2):C.UVGC_3215_3130})

V_604 = CTVertex(name = 'V_604',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.s] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,9,1):C.UVGC_3490_3619,(0,6,0):C.UVGC_2090_1212,(0,6,2):C.UVGC_2626_2003,(0,3,0):C.UVGC_2090_1212,(0,3,2):C.UVGC_2626_2003,(0,7,0):C.UVGC_2628_2004,(0,7,2):C.UVGC_2628_2005,(0,7,1):C.UVGC_3227_3142,(0,4,0):C.UVGC_2629_2006,(0,4,2):C.UVGC_2629_2007,(0,8,0):C.UVGC_2090_1212,(0,5,0):C.UVGC_2090_1212,(0,0,1):C.UVGC_3227_3142,(0,1,1):C.UVGC_3227_3142,(0,2,1):C.UVGC_3227_3142})

V_605 = CTVertex(name = 'V_605',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,9,1):C.UVGC_3492_3621,(0,6,0):C.UVGC_2068_1180,(0,6,2):C.UVGC_2638_2018,(0,3,0):C.UVGC_2068_1180,(0,3,2):C.UVGC_2638_2018,(0,7,0):C.UVGC_2640_2019,(0,7,2):C.UVGC_2640_2020,(0,7,1):C.UVGC_3223_3138,(0,4,0):C.UVGC_2641_2021,(0,4,2):C.UVGC_2641_2022,(0,8,0):C.UVGC_2068_1180,(0,5,0):C.UVGC_2068_1180,(0,0,1):C.UVGC_3223_3138,(0,1,1):C.UVGC_3223_3138,(0,2,1):C.UVGC_3223_3138})

V_606 = CTVertex(name = 'V_606',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.Z, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS25, L.FFVS26, L.FFVS28, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g] ], [ [P.b, P.g, P.s] ], [ [P.b, P.g, P.t, P.u] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,9,4):C.UVGC_3612_3847,(0,9,6):C.UVGC_3612_3848,(0,9,8):C.UVGC_3612_3849,(0,9,5):C.UVGC_3612_3850,(0,9,7):C.UVGC_3612_3851,(0,9,0):C.UVGC_3612_3852,(0,9,3):C.UVGC_3612_3853,(0,6,1):C.UVGC_2097_1223,(0,6,2):C.UVGC_2097_1224,(0,6,5):C.UVGC_2607_1971,(0,6,7):C.UVGC_2607_1972,(0,3,1):C.UVGC_2097_1223,(0,3,2):C.UVGC_2097_1224,(0,3,5):C.UVGC_2607_1971,(0,3,7):C.UVGC_2607_1972,(0,7,1):C.UVGC_2609_1973,(0,7,2):C.UVGC_2609_1974,(0,7,5):C.UVGC_2609_1975,(0,7,7):C.UVGC_2609_1976,(0,7,4):C.UVGC_3652_3974,(0,7,6):C.UVGC_3652_3975,(0,7,8):C.UVGC_3652_3976,(0,4,1):C.UVGC_2610_1977,(0,4,2):C.UVGC_2610_1978,(0,4,5):C.UVGC_2610_1979,(0,4,7):C.UVGC_2610_1980,(0,8,1):C.UVGC_2097_1223,(0,8,2):C.UVGC_2097_1224,(0,5,1):C.UVGC_2097_1223,(0,5,2):C.UVGC_2097_1224,(0,0,4):C.UVGC_3652_3974,(0,0,6):C.UVGC_3652_3975,(0,0,8):C.UVGC_3652_3976,(0,1,4):C.UVGC_3652_3974,(0,1,6):C.UVGC_3652_3975,(0,1,8):C.UVGC_3652_3976,(0,2,4):C.UVGC_3652_3974,(0,2,6):C.UVGC_3652_3975,(0,2,8):C.UVGC_3652_3976})

V_607 = CTVertex(name = 'V_607',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_3182_3081,(0,6,0):C.UVGC_1874_904,(0,6,1):C.UVGC_3183_3082,(0,8,1):C.UVGC_3401_3459,(0,5,0):C.UVGC_1875_905,(0,2,0):C.UVGC_1875_905,(0,3,0):C.UVGC_1873_903,(0,7,0):C.UVGC_1875_905,(0,4,0):C.UVGC_1875_905,(0,1,1):C.UVGC_3272_3197})

V_608 = CTVertex(name = 'V_608',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_3187_3086,(0,6,0):C.UVGC_2425_1647,(0,6,1):C.UVGC_3188_3087,(0,8,1):C.UVGC_3403_3461,(0,5,0):C.UVGC_2426_1648,(0,2,0):C.UVGC_2426_1648,(0,3,0):C.UVGC_2424_1646,(0,7,0):C.UVGC_2426_1648,(0,4,0):C.UVGC_2426_1648,(0,1,1):C.UVGC_3274_3199})

V_609 = CTVertex(name = 'V_609',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_3192_3091,(0,6,0):C.UVGC_1878_907,(0,6,1):C.UVGC_3193_3092,(0,8,1):C.UVGC_3484_3613,(0,5,0):C.UVGC_1879_908,(0,2,0):C.UVGC_1879_908,(0,3,0):C.UVGC_1877_906,(0,7,0):C.UVGC_1879_908,(0,4,0):C.UVGC_1879_908,(0,1,1):C.UVGC_3278_3203})

V_610 = CTVertex(name = 'V_610',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_3197_3096,(0,6,0):C.UVGC_2429_1650,(0,6,1):C.UVGC_3198_3097,(0,8,1):C.UVGC_3486_3615,(0,5,0):C.UVGC_2430_1651,(0,2,0):C.UVGC_2430_1651,(0,3,0):C.UVGC_2428_1649,(0,7,0):C.UVGC_2430_1651,(0,4,0):C.UVGC_2430_1651,(0,1,1):C.UVGC_3280_3205})

V_611 = CTVertex(name = 'V_611',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS30, L.FFVS31, L.FFVS32, L.FFVS33, L.FFVS34, L.FFVS35, L.FFVS36, L.FFVS6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.c, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_3644_3950,(0,0,5):C.UVGC_3644_3951,(0,0,6):C.UVGC_3644_3952,(0,6,3):C.UVGC_2343_1532,(0,6,4):C.UVGC_2343_1533,(0,6,2):C.UVGC_3645_3953,(0,6,5):C.UVGC_3645_3954,(0,6,6):C.UVGC_3645_3955,(0,1,2):C.UVGC_3668_4023,(0,1,5):C.UVGC_3668_4024,(0,1,6):C.UVGC_3668_4025,(0,1,3):C.UVGC_3668_4026,(0,1,4):C.UVGC_3668_4027,(0,1,0):C.UVGC_3668_4028,(0,1,1):C.UVGC_3668_4029,(0,5,3):C.UVGC_2344_1534,(0,5,4):C.UVGC_2344_1535,(0,2,3):C.UVGC_2344_1534,(0,2,4):C.UVGC_2344_1535,(0,3,3):C.UVGC_2342_1530,(0,3,4):C.UVGC_2342_1531,(0,7,3):C.UVGC_2344_1534,(0,7,4):C.UVGC_2344_1535,(0,4,3):C.UVGC_2344_1534,(0,4,4):C.UVGC_2344_1535,(0,8,2):C.UVGC_3606_3829,(0,8,5):C.UVGC_3606_3830,(0,8,6):C.UVGC_3606_3831})

V_612 = CTVertex(name = 'V_612',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS3, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,2):C.UVGC_1630_549,(0,0,0):C.UVGC_2078_1197,(0,0,1):C.UVGC_2051_1162,(0,5,0):C.UVGC_2078_1197,(0,5,1):C.UVGC_2051_1162,(0,1,0):C.UVGC_2077_1195,(0,1,1):C.UVGC_2077_1196,(0,1,2):C.UVGC_1653_573,(0,6,0):C.UVGC_2076_1193,(0,6,1):C.UVGC_2076_1194,(0,2,1):C.UVGC_2051_1162,(0,7,1):C.UVGC_2051_1162,(0,3,2):C.UVGC_1652_572})

V_613 = CTVertex(name = 'V_613',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS3, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.d, P.g] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,4,1):C.UVGC_1631_550,(0,0,0):C.UVGC_2755_2197,(0,0,2):C.UVGC_2753_2196,(0,5,0):C.UVGC_2755_2197,(0,5,2):C.UVGC_2753_2196,(0,1,0):C.UVGC_2752_2194,(0,1,2):C.UVGC_2752_2195,(0,1,1):C.UVGC_1677_597,(0,6,0):C.UVGC_2751_2192,(0,6,2):C.UVGC_2751_2193,(0,2,2):C.UVGC_2753_2196,(0,7,2):C.UVGC_2753_2196,(0,3,1):C.UVGC_1676_596})

V_614 = CTVertex(name = 'V_614',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS3, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.g, P.s] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,4,2):C.UVGC_1620_539,(0,0,0):C.UVGC_2102_1231,(0,0,1):C.UVGC_2011_1123,(0,5,0):C.UVGC_2102_1231,(0,5,1):C.UVGC_2011_1123,(0,1,0):C.UVGC_2101_1229,(0,1,1):C.UVGC_2101_1230,(0,1,2):C.UVGC_1642_562,(0,6,0):C.UVGC_2100_1227,(0,6,1):C.UVGC_2100_1228,(0,2,1):C.UVGC_2011_1123,(0,7,1):C.UVGC_2011_1123,(0,3,2):C.UVGC_1641_561})

V_615 = CTVertex(name = 'V_615',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS3, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.g, P.s] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,4,1):C.UVGC_1621_540,(0,0,0):C.UVGC_2717_2132,(0,0,2):C.UVGC_2715_2131,(0,5,0):C.UVGC_2717_2132,(0,5,2):C.UVGC_2715_2131,(0,1,0):C.UVGC_2714_2129,(0,1,2):C.UVGC_2714_2130,(0,1,1):C.UVGC_1666_586,(0,6,0):C.UVGC_2713_2127,(0,6,2):C.UVGC_2713_2128,(0,2,2):C.UVGC_2715_2131,(0,7,2):C.UVGC_2715_2131,(0,3,1):C.UVGC_1665_585})

V_616 = CTVertex(name = 'V_616',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.Z, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS10, L.FFVS11, L.FFVS12, L.FFVS3, L.FFVS6, L.FFVS7, L.FFVS8, L.FFVS9 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g] ], [ [P.b, P.g, P.s] ], [ [P.b, P.g, P.t, P.u] ], [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,4):C.UVGC_2921_2475,(0,4,6):C.UVGC_2921_2476,(0,4,8):C.UVGC_2921_2477,(0,4,5):C.UVGC_2921_2478,(0,4,7):C.UVGC_2921_2479,(0,4,0):C.UVGC_2921_2480,(0,4,3):C.UVGC_2921_2481,(0,0,1):C.UVGC_2799_2264,(0,0,2):C.UVGC_2799_2265,(0,0,5):C.UVGC_2797_2262,(0,0,7):C.UVGC_2797_2263,(0,5,1):C.UVGC_2799_2264,(0,5,2):C.UVGC_2799_2265,(0,5,5):C.UVGC_2797_2262,(0,5,7):C.UVGC_2797_2263,(0,1,1):C.UVGC_2796_2258,(0,1,2):C.UVGC_2796_2259,(0,1,5):C.UVGC_2796_2260,(0,1,7):C.UVGC_2796_2261,(0,1,4):C.UVGC_1843_839,(0,1,6):C.UVGC_1843_840,(0,1,8):C.UVGC_1843_841,(0,6,1):C.UVGC_2795_2254,(0,6,2):C.UVGC_2795_2255,(0,6,5):C.UVGC_2795_2256,(0,6,7):C.UVGC_2795_2257,(0,2,5):C.UVGC_2797_2262,(0,2,7):C.UVGC_2797_2263,(0,7,5):C.UVGC_2797_2262,(0,7,7):C.UVGC_2797_2263,(0,3,4):C.UVGC_1842_836,(0,3,6):C.UVGC_1842_837,(0,3,8):C.UVGC_1842_838})

V_617 = CTVertex(name = 'V_617',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS30, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_3423_3482,(0,3,1):C.UVGC_3423_3482,(0,5,1):C.UVGC_3423_3482,(0,2,0):C.UVGC_1894_934,(0,2,1):C.UVGC_3423_3482,(0,6,1):C.UVGC_3402_3460,(0,1,0):C.UVGC_1892_933,(0,7,0):C.UVGC_1892_933,(0,4,1):C.UVGC_3273_3198,(0,8,0):C.UVGC_1895_935})

V_618 = CTVertex(name = 'V_618',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS30, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_3576_3785,(0,3,1):C.UVGC_3576_3785,(0,5,1):C.UVGC_3576_3785,(0,2,0):C.UVGC_1905_950,(0,2,1):C.UVGC_3576_3785,(0,6,1):C.UVGC_3404_3462,(0,1,0):C.UVGC_1903_949,(0,7,0):C.UVGC_1903_949,(0,4,1):C.UVGC_3275_3200,(0,8,0):C.UVGC_1906_951})

V_619 = CTVertex(name = 'V_619',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS30, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_3506_3637,(0,3,1):C.UVGC_3506_3637,(0,5,1):C.UVGC_3506_3637,(0,2,0):C.UVGC_2455_1691,(0,2,1):C.UVGC_3506_3637,(0,6,1):C.UVGC_3485_3614,(0,1,0):C.UVGC_2453_1690,(0,7,0):C.UVGC_2453_1690,(0,4,1):C.UVGC_3279_3204,(0,8,0):C.UVGC_2456_1692})

V_620 = CTVertex(name = 'V_620',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS30, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.g, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,1):C.UVGC_3585_3794,(0,3,1):C.UVGC_3585_3794,(0,5,1):C.UVGC_3585_3794,(0,2,0):C.UVGC_2479_1732,(0,2,1):C.UVGC_3585_3794,(0,6,1):C.UVGC_3487_3616,(0,1,0):C.UVGC_2477_1731,(0,7,0):C.UVGC_2477_1731,(0,4,1):C.UVGC_3281_3206,(0,8,0):C.UVGC_2480_1733})

V_621 = CTVertex(name = 'V_621',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__plus__, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS10, L.FFVS11, L.FFVS2, L.FFVS30, L.FFVS4, L.FFVS6, L.FFVS7, L.FFVS8 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.g, P.t, P.u] ], [ [P.c, P.g] ], [ [P.d, P.g, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,2):C.UVGC_3624_3893,(0,0,5):C.UVGC_3624_3894,(0,0,6):C.UVGC_3624_3895,(0,3,2):C.UVGC_3624_3893,(0,3,5):C.UVGC_3624_3894,(0,3,6):C.UVGC_3624_3895,(0,5,2):C.UVGC_3624_3893,(0,5,5):C.UVGC_3624_3894,(0,5,6):C.UVGC_3624_3895,(0,2,3):C.UVGC_2372_1570,(0,2,4):C.UVGC_2372_1571,(0,2,2):C.UVGC_3624_3893,(0,2,5):C.UVGC_3624_3894,(0,2,6):C.UVGC_3624_3895,(0,4,2):C.UVGC_3669_4030,(0,4,5):C.UVGC_3669_4031,(0,4,6):C.UVGC_3669_4032,(0,4,3):C.UVGC_3669_4033,(0,4,4):C.UVGC_3669_4034,(0,4,0):C.UVGC_3669_4035,(0,4,1):C.UVGC_3669_4036,(0,1,3):C.UVGC_2370_1568,(0,1,4):C.UVGC_2370_1569,(0,7,3):C.UVGC_2370_1568,(0,7,4):C.UVGC_2370_1569,(0,6,2):C.UVGC_3607_3832,(0,6,5):C.UVGC_3607_3833,(0,6,6):C.UVGC_3607_3834,(0,8,3):C.UVGC_2373_1572,(0,8,4):C.UVGC_2373_1573})

V_622 = CTVertex(name = 'V_622',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2330_1518,(0,2,1):C.UVGC_3284_3209,(0,1,0):C.UVGC_2331_1519,(0,0,0):C.UVGC_2332_1520,(0,0,1):C.UVGC_3285_3210})

V_623 = CTVertex(name = 'V_623',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2209_1356,(0,2,1):C.UVGC_3286_3211,(0,1,0):C.UVGC_2210_1357,(0,0,0):C.UVGC_2211_1358,(0,0,1):C.UVGC_3287_3212})

V_624 = CTVertex(name = 'V_624',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2333_1521,(0,2,1):C.UVGC_3288_3213,(0,1,0):C.UVGC_2334_1522,(0,0,0):C.UVGC_2335_1523,(0,0,1):C.UVGC_3289_3214})

V_625 = CTVertex(name = 'V_625',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2212_1359,(0,2,1):C.UVGC_3290_3215,(0,1,0):C.UVGC_2213_1360,(0,0,0):C.UVGC_2214_1361,(0,0,1):C.UVGC_3291_3216})

V_626 = CTVertex(name = 'V_626',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2447_1678,(0,2,1):C.UVGC_2447_1679,(0,2,2):C.UVGC_3663_4006,(0,2,3):C.UVGC_3663_4007,(0,2,4):C.UVGC_3663_4008,(0,1,0):C.UVGC_2448_1680,(0,1,1):C.UVGC_2448_1681,(0,0,0):C.UVGC_2449_1682,(0,0,1):C.UVGC_2449_1683,(0,0,2):C.UVGC_3664_4009,(0,0,3):C.UVGC_3664_4010,(0,0,4):C.UVGC_3664_4011})

V_627 = CTVertex(name = 'V_627',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.s, P.t] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.UVGC_2874_2356,(0,2,0):C.UVGC_2874_2357,(0,2,1):C.UVGC_2874_2358,(0,2,3):C.UVGC_3263_3188,(0,1,2):C.UVGC_2873_2353,(0,1,0):C.UVGC_2873_2354,(0,1,1):C.UVGC_2873_2355,(0,0,2):C.UVGC_2875_2359,(0,0,0):C.UVGC_2875_2360,(0,0,1):C.UVGC_2873_2355,(0,0,3):C.UVGC_3262_3187})

V_628 = CTVertex(name = 'V_628',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2946_2554,(0,2,0):C.UVGC_2946_2555,(0,2,2):C.UVGC_2946_2556,(0,2,3):C.UVGC_3265_3190,(0,1,1):C.UVGC_2947_2557,(0,1,0):C.UVGC_2947_2558,(0,1,2):C.UVGC_2947_2559,(0,0,1):C.UVGC_2948_2560,(0,0,0):C.UVGC_2948_2561,(0,0,2):C.UVGC_2947_2559,(0,0,3):C.UVGC_3264_3189})

V_629 = CTVertex(name = 'V_629',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.s, P.t] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.UVGC_2877_2364,(0,2,0):C.UVGC_2877_2365,(0,2,1):C.UVGC_2877_2366,(0,2,3):C.UVGC_3267_3192,(0,1,2):C.UVGC_2876_2361,(0,1,0):C.UVGC_2876_2362,(0,1,1):C.UVGC_2876_2363,(0,0,2):C.UVGC_2878_2367,(0,0,0):C.UVGC_2878_2368,(0,0,1):C.UVGC_2876_2363,(0,0,3):C.UVGC_3266_3191})

V_630 = CTVertex(name = 'V_630',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2949_2562,(0,2,0):C.UVGC_2949_2563,(0,2,2):C.UVGC_2949_2564,(0,2,3):C.UVGC_3269_3194,(0,1,1):C.UVGC_2950_2565,(0,1,0):C.UVGC_2950_2566,(0,1,2):C.UVGC_2950_2567,(0,0,1):C.UVGC_2951_2568,(0,0,0):C.UVGC_2951_2569,(0,0,2):C.UVGC_2950_2567,(0,0,3):C.UVGC_3268_3193})

V_631 = CTVertex(name = 'V_631',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.W__plus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.s] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.u] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2922_2482,(0,2,5):C.UVGC_2922_2483,(0,2,1):C.UVGC_2922_2484,(0,2,2):C.UVGC_2922_2485,(0,2,3):C.UVGC_2922_2486,(0,2,4):C.UVGC_2922_2487,(0,2,6):C.UVGC_3661_4000,(0,2,7):C.UVGC_3661_4001,(0,2,8):C.UVGC_3661_4002,(0,1,0):C.UVGC_2923_2488,(0,1,5):C.UVGC_2923_2489,(0,1,1):C.UVGC_2923_2490,(0,1,2):C.UVGC_2923_2491,(0,1,3):C.UVGC_2923_2492,(0,1,4):C.UVGC_2923_2493,(0,0,0):C.UVGC_2924_2494,(0,0,5):C.UVGC_2924_2495,(0,0,1):C.UVGC_2924_2496,(0,0,2):C.UVGC_2923_2491,(0,0,3):C.UVGC_2924_2497,(0,0,4):C.UVGC_2923_2493,(0,0,6):C.UVGC_3660_3997,(0,0,7):C.UVGC_3660_3998,(0,0,8):C.UVGC_3660_3999})

V_632 = CTVertex(name = 'V_632',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2233_1377,(0,2,1):C.UVGC_1658_578,(0,1,0):C.UVGC_2235_1379,(0,0,0):C.UVGC_2234_1378,(0,0,1):C.UVGC_1657_577})

V_633 = CTVertex(name = 'V_633',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2252_1397,(0,2,1):C.UVGC_1682_602,(0,1,0):C.UVGC_2254_1399,(0,0,0):C.UVGC_2253_1398,(0,0,1):C.UVGC_1681_601})

V_634 = CTVertex(name = 'V_634',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2360_1554,(0,2,1):C.UVGC_1647_567,(0,1,0):C.UVGC_2362_1556,(0,0,0):C.UVGC_2361_1555,(0,0,1):C.UVGC_1646_566})

V_635 = CTVertex(name = 'V_635',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2393_1593,(0,2,1):C.UVGC_1671_591,(0,1,0):C.UVGC_2395_1595,(0,0,0):C.UVGC_2394_1594,(0,0,1):C.UVGC_1670_590})

V_636 = CTVertex(name = 'V_636',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.W__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2466_1707,(0,2,1):C.UVGC_2466_1708,(0,2,2):C.UVGC_1848_854,(0,2,3):C.UVGC_1848_855,(0,2,4):C.UVGC_1848_856,(0,1,0):C.UVGC_2468_1711,(0,1,1):C.UVGC_2468_1712,(0,0,0):C.UVGC_2467_1709,(0,0,1):C.UVGC_2467_1710,(0,0,2):C.UVGC_1847_851,(0,0,3):C.UVGC_1847_852,(0,0,4):C.UVGC_1847_853})

V_637 = CTVertex(name = 'V_637',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2962_2591,(0,2,0):C.UVGC_2962_2592,(0,2,2):C.UVGC_2962_2593,(0,2,3):C.UVGC_1654_574,(0,1,1):C.UVGC_2961_2588,(0,1,0):C.UVGC_2961_2589,(0,1,2):C.UVGC_2961_2590,(0,0,1):C.UVGC_2963_2594,(0,0,0):C.UVGC_2961_2589,(0,0,2):C.UVGC_2962_2593,(0,0,3):C.UVGC_1655_575})

V_638 = CTVertex(name = 'V_638',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.d, P.g, P.t] ], [ [P.d, P.g, P.t] ], [ [P.d, P.g, P.t, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2974_2612,(0,2,0):C.UVGC_2974_2613,(0,2,2):C.UVGC_2974_2614,(0,2,3):C.UVGC_1678_598,(0,1,1):C.UVGC_2973_2609,(0,1,0):C.UVGC_2973_2610,(0,1,2):C.UVGC_2973_2611,(0,0,1):C.UVGC_2975_2615,(0,0,0):C.UVGC_2973_2610,(0,0,2):C.UVGC_2974_2614,(0,0,3):C.UVGC_1679_599})

V_639 = CTVertex(name = 'V_639',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.g, P.s, P.t] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.UVGC_2879_2369,(0,2,0):C.UVGC_2879_2370,(0,2,1):C.UVGC_2879_2371,(0,2,3):C.UVGC_1643_563,(0,1,2):C.UVGC_2880_2372,(0,1,0):C.UVGC_2880_2373,(0,1,1):C.UVGC_2880_2374,(0,0,2):C.UVGC_2881_2375,(0,0,0):C.UVGC_2880_2373,(0,0,1):C.UVGC_2879_2371,(0,0,3):C.UVGC_1644_564})

V_640 = CTVertex(name = 'V_640',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.g, P.s, P.t] ], [ [P.c, P.g, P.s, P.t] ], [ [P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,2):C.UVGC_2882_2376,(0,2,0):C.UVGC_2882_2377,(0,2,1):C.UVGC_2882_2378,(0,2,3):C.UVGC_1667_587,(0,1,2):C.UVGC_2883_2379,(0,1,0):C.UVGC_2883_2380,(0,1,1):C.UVGC_2883_2381,(0,0,2):C.UVGC_2884_2382,(0,0,0):C.UVGC_2883_2380,(0,0,1):C.UVGC_2882_2378,(0,0,3):C.UVGC_1668_588})

V_641 = CTVertex(name = 'V_641',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.W__minus__, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.c, P.g, P.s] ], [ [P.b, P.c, P.g, P.t] ], [ [P.b, P.d, P.g, P.u] ], [ [P.b, P.g, P.t, P.u] ], [ [P.b, P.g, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2935_2532,(0,2,5):C.UVGC_2935_2533,(0,2,1):C.UVGC_2935_2534,(0,2,2):C.UVGC_2935_2535,(0,2,3):C.UVGC_2935_2536,(0,2,4):C.UVGC_2935_2537,(0,2,6):C.UVGC_1844_842,(0,2,7):C.UVGC_1844_843,(0,2,8):C.UVGC_1844_844,(0,1,0):C.UVGC_2934_2526,(0,1,5):C.UVGC_2934_2527,(0,1,1):C.UVGC_2934_2528,(0,1,2):C.UVGC_2934_2529,(0,1,3):C.UVGC_2934_2530,(0,1,4):C.UVGC_2934_2531,(0,0,0):C.UVGC_2936_2538,(0,0,5):C.UVGC_2936_2539,(0,0,1):C.UVGC_2934_2528,(0,0,2):C.UVGC_2935_2535,(0,0,3):C.UVGC_2934_2530,(0,0,4):C.UVGC_2935_2537,(0,0,6):C.UVGC_1845_845,(0,0,7):C.UVGC_1845_846,(0,0,8):C.UVGC_1845_847})

V_642 = CTVertex(name = 'V_642',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t], [P.c, P.g, P.s, P.t] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2867_2349,(0,2,1):C.UVGC_3428_3487,(0,0,0):C.UVGC_2868_2350,(0,0,1):C.UVGC_3427_3486,(0,5,0):C.UVGC_2861_2343,(0,5,1):C.UVGC_3255_3180,(0,3,0):C.UVGC_2862_2344,(0,3,1):C.UVGC_3254_3179,(0,1,0):C.UVGC_2868_2350,(0,4,0):C.UVGC_2863_2345})

V_643 = CTVertex(name = 'V_643',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t], [P.c, P.g, P.s, P.t] ], [ [P.b, P.g, P.t, P.u], [P.d, P.g, P.t, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2870_2351,(0,2,2):C.UVGC_3581_3790,(0,0,0):C.UVGC_2871_2352,(0,0,2):C.UVGC_3580_3789,(0,5,1):C.UVGC_2940_2548,(0,5,2):C.UVGC_3257_3182,(0,3,1):C.UVGC_2941_2549,(0,3,2):C.UVGC_3256_3181,(0,1,0):C.UVGC_2871_2352,(0,4,1):C.UVGC_2942_2550})

V_644 = CTVertex(name = 'V_644',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t], [P.c, P.g, P.s, P.t] ], [ [P.b, P.g, P.t, P.u], [P.d, P.g, P.t, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,1):C.UVGC_2958_2586,(0,2,2):C.UVGC_3511_3642,(0,0,1):C.UVGC_2959_2587,(0,0,2):C.UVGC_3510_3641,(0,5,0):C.UVGC_2864_2346,(0,5,2):C.UVGC_3259_3184,(0,3,0):C.UVGC_2865_2347,(0,3,2):C.UVGC_3258_3183,(0,1,1):C.UVGC_2959_2587,(0,4,0):C.UVGC_2866_2348})

V_645 = CTVertex(name = 'V_645',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.t, P.u], [P.d, P.g, P.t, P.u] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2970_2607,(0,2,1):C.UVGC_3590_3799,(0,0,0):C.UVGC_2971_2608,(0,0,1):C.UVGC_3589_3798,(0,5,0):C.UVGC_2943_2551,(0,5,1):C.UVGC_3261_3186,(0,3,0):C.UVGC_2944_2552,(0,3,1):C.UVGC_3260_3185,(0,1,0):C.UVGC_2971_2608,(0,4,0):C.UVGC_2945_2553})

V_646 = CTVertex(name = 'V_646',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.W__minus__, P.W__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g, P.t], [P.c, P.g, P.s, P.t] ], [ [P.b, P.g, P.t, P.u], [P.d, P.g, P.t, P.u] ], [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2964_2595,(0,2,1):C.UVGC_2964_2596,(0,2,2):C.UVGC_3629_3908,(0,2,3):C.UVGC_3629_3909,(0,2,4):C.UVGC_3629_3910,(0,0,0):C.UVGC_2965_2597,(0,0,1):C.UVGC_2965_2598,(0,0,2):C.UVGC_3628_3905,(0,0,3):C.UVGC_3628_3906,(0,0,4):C.UVGC_3628_3907,(0,5,0):C.UVGC_2952_2570,(0,5,1):C.UVGC_2952_2571,(0,5,2):C.UVGC_3659_3994,(0,5,3):C.UVGC_3659_3995,(0,5,4):C.UVGC_3659_3996,(0,3,0):C.UVGC_2953_2572,(0,3,1):C.UVGC_2953_2573,(0,3,2):C.UVGC_3658_3991,(0,3,3):C.UVGC_3658_3992,(0,3,4):C.UVGC_3658_3993,(0,1,0):C.UVGC_2965_2597,(0,1,1):C.UVGC_2965_2598,(0,4,0):C.UVGC_2954_2574,(0,4,1):C.UVGC_2954_2575})

V_647 = CTVertex(name = 'V_647',
                 type = 'UV',
                 particles = [ P.g, P.g, P.H ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.VVS1, L.VVS11, L.VVS2, L.VVS6, L.VVS8, L.VVS9 ],
                 loop_particles = [ [ [P.c, P.t] ], [ [P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_1853_863,(0,2,1):C.UVGC_1853_864,(0,0,0):C.UVGC_1852_861,(0,0,1):C.UVGC_1852_862,(0,3,0):C.UVGC_1853_863,(0,3,1):C.UVGC_1853_864,(0,4,0):C.UVGC_1851_859,(0,4,1):C.UVGC_1851_860,(0,5,0):C.UVGC_1849_857,(0,5,1):C.UVGC_1849_858,(0,1,0):C.UVGC_1849_857,(0,1,1):C.UVGC_1849_858})

V_648 = CTVertex(name = 'V_648',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_1810_767,(0,2,1):C.UVGC_1810_768,(0,1,0):C.UVGC_1810_767,(0,1,1):C.UVGC_1810_768,(0,5,0):C.UVGC_1723_637,(0,5,1):C.UVGC_1723_638,(0,4,0):C.UVGC_1723_637,(0,4,1):C.UVGC_1723_638,(0,0,0):C.UVGC_1811_769,(0,0,1):C.UVGC_1811_770,(0,3,0):C.UVGC_1724_639,(0,3,1):C.UVGC_1724_640})

V_649 = CTVertex(name = 'V_649',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_1771_716,(0,2,1):C.UVGC_1771_717,(0,1,0):C.UVGC_1771_716,(0,1,1):C.UVGC_1771_717,(0,5,0):C.UVGC_1725_641,(0,5,1):C.UVGC_1725_642,(0,4,0):C.UVGC_1725_641,(0,4,1):C.UVGC_1725_642,(0,0,0):C.UVGC_1772_718,(0,0,1):C.UVGC_1772_719,(0,3,0):C.UVGC_1726_643,(0,3,1):C.UVGC_1726_644})

V_650 = CTVertex(name = 'V_650',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_1518_410,(0,2,1):C.UVGC_1518_411,(0,1,0):C.UVGC_1518_410,(0,1,1):C.UVGC_1518_411,(0,5,0):C.UVGC_1470_333,(0,5,1):C.UVGC_1470_334,(0,4,0):C.UVGC_1470_333,(0,4,1):C.UVGC_1470_334,(0,0,0):C.UVGC_1519_412,(0,0,1):C.UVGC_1519_413,(0,3,0):C.UVGC_1471_335,(0,3,1):C.UVGC_1471_336})

V_651 = CTVertex(name = 'V_651',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.a, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_1565_469,(0,2,1):C.UVGC_1565_470,(0,1,0):C.UVGC_1565_469,(0,1,1):C.UVGC_1565_470,(0,5,0):C.UVGC_1468_329,(0,5,1):C.UVGC_1468_330,(0,4,0):C.UVGC_1468_329,(0,4,1):C.UVGC_1468_330,(0,0,0):C.UVGC_1566_471,(0,0,1):C.UVGC_1566_472,(0,3,0):C.UVGC_1469_331,(0,3,1):C.UVGC_1469_332})

V_652 = CTVertex(name = 'V_652',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1821_789,(0,0,1):C.UVGC_1821_790,(0,5,0):C.UVGC_1744_678,(0,5,1):C.UVGC_1744_679,(0,4,0):C.UVGC_1744_678,(0,4,1):C.UVGC_1744_679,(0,2,0):C.UVGC_1820_787,(0,2,1):C.UVGC_1820_788,(0,1,0):C.UVGC_1820_787,(0,1,1):C.UVGC_1820_788,(0,3,0):C.UVGC_1745_680,(0,3,1):C.UVGC_1745_681})

V_653 = CTVertex(name = 'V_653',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1782_738,(0,0,1):C.UVGC_1782_739,(0,5,0):C.UVGC_1746_682,(0,5,1):C.UVGC_1746_683,(0,4,0):C.UVGC_1746_682,(0,4,1):C.UVGC_1746_683,(0,2,0):C.UVGC_1781_736,(0,2,1):C.UVGC_1781_737,(0,1,0):C.UVGC_1781_736,(0,1,1):C.UVGC_1781_737,(0,3,0):C.UVGC_1747_684,(0,3,1):C.UVGC_1747_685})

V_654 = CTVertex(name = 'V_654',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1529_432,(0,0,1):C.UVGC_1529_433,(0,5,0):C.UVGC_1490_373,(0,5,1):C.UVGC_1490_374,(0,4,0):C.UVGC_1490_373,(0,4,1):C.UVGC_1490_374,(0,2,0):C.UVGC_1528_430,(0,2,1):C.UVGC_1528_431,(0,1,0):C.UVGC_1528_430,(0,1,1):C.UVGC_1528_431,(0,3,0):C.UVGC_1491_375,(0,3,1):C.UVGC_1491_376})

V_655 = CTVertex(name = 'V_655',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1576_491,(0,0,1):C.UVGC_1576_492,(0,5,0):C.UVGC_1488_369,(0,5,1):C.UVGC_1488_370,(0,4,0):C.UVGC_1488_369,(0,4,1):C.UVGC_1488_370,(0,2,0):C.UVGC_1575_489,(0,2,1):C.UVGC_1575_490,(0,1,0):C.UVGC_1575_489,(0,1,1):C.UVGC_1575_490,(0,3,0):C.UVGC_1489_371,(0,3,1):C.UVGC_1489_372})

V_656 = CTVertex(name = 'V_656',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2830_2305,(0,1,0):C.UVGC_2830_2305,(0,0,0):C.UVGC_2831_2306,(0,5,0):C.UVGC_2657_2040,(0,4,0):C.UVGC_2657_2040,(0,3,0):C.UVGC_2658_2041})

V_657 = CTVertex(name = 'V_657',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2739_2158,(0,2,1):C.UVGC_2739_2159,(0,1,0):C.UVGC_2739_2158,(0,1,1):C.UVGC_2739_2159,(0,0,0):C.UVGC_2740_2160,(0,0,1):C.UVGC_2740_2161,(0,5,0):C.UVGC_2619_1989,(0,5,1):C.UVGC_2619_1990,(0,4,0):C.UVGC_2619_1989,(0,4,1):C.UVGC_2619_1990,(0,3,0):C.UVGC_2620_1991,(0,3,1):C.UVGC_2620_1992})

V_658 = CTVertex(name = 'V_658',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2706_2113,(0,2,1):C.UVGC_2706_2114,(0,1,0):C.UVGC_2706_2113,(0,1,1):C.UVGC_2706_2114,(0,0,0):C.UVGC_2707_2115,(0,0,1):C.UVGC_2707_2116,(0,5,0):C.UVGC_2655_2036,(0,5,1):C.UVGC_2655_2037,(0,4,0):C.UVGC_2655_2036,(0,4,1):C.UVGC_2655_2037,(0,3,0):C.UVGC_2656_2038,(0,3,1):C.UVGC_2656_2039})

V_659 = CTVertex(name = 'V_659',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2814_2284,(0,2,1):C.UVGC_2814_2285,(0,1,0):C.UVGC_2814_2284,(0,1,1):C.UVGC_2814_2285,(0,0,0):C.UVGC_2815_2286,(0,0,1):C.UVGC_2815_2287,(0,5,0):C.UVGC_2682_2075,(0,5,1):C.UVGC_2682_2076,(0,4,0):C.UVGC_2682_2075,(0,4,1):C.UVGC_2682_2076,(0,3,0):C.UVGC_2683_2077,(0,3,1):C.UVGC_2683_2078})

V_660 = CTVertex(name = 'V_660',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.a, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2026_1133,(0,1,0):C.UVGC_2026_1133,(0,0,0):C.UVGC_2027_1134,(0,5,0):C.UVGC_1978_1057,(0,4,0):C.UVGC_1978_1057,(0,3,0):C.UVGC_1979_1058})

V_661 = CTVertex(name = 'V_661',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,2):C.UVGC_1825_797,(0,2,4):C.UVGC_1825_798,(0,2,0):C.UVGC_2907_2433,(0,2,3):C.UVGC_2907_2434,(0,2,1):C.UVGC_2907_2435,(0,1,2):C.UVGC_1825_797,(0,1,4):C.UVGC_1825_798,(0,1,0):C.UVGC_2907_2433,(0,1,3):C.UVGC_2907_2434,(0,1,1):C.UVGC_2907_2435,(0,0,2):C.UVGC_1826_799,(0,0,4):C.UVGC_1826_800,(0,0,0):C.UVGC_2906_2430,(0,0,3):C.UVGC_2906_2431,(0,0,1):C.UVGC_2906_2432,(0,5,2):C.UVGC_1755_700,(0,5,4):C.UVGC_1755_701,(0,5,0):C.UVGC_2902_2418,(0,5,3):C.UVGC_2902_2419,(0,5,1):C.UVGC_2902_2420,(0,4,2):C.UVGC_1755_700,(0,4,4):C.UVGC_1755_701,(0,4,0):C.UVGC_2902_2418,(0,4,3):C.UVGC_2902_2419,(0,4,1):C.UVGC_2902_2420,(0,3,2):C.UVGC_1754_698,(0,3,4):C.UVGC_1754_699,(0,3,0):C.UVGC_2901_2415,(0,3,3):C.UVGC_2901_2416,(0,3,1):C.UVGC_2901_2417})

V_662 = CTVertex(name = 'V_662',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,2):C.UVGC_1786_746,(0,2,4):C.UVGC_1786_747,(0,2,0):C.UVGC_2910_2442,(0,2,3):C.UVGC_2910_2443,(0,2,1):C.UVGC_2910_2444,(0,1,2):C.UVGC_1786_746,(0,1,4):C.UVGC_1786_747,(0,1,0):C.UVGC_2910_2442,(0,1,3):C.UVGC_2910_2443,(0,1,1):C.UVGC_2910_2444,(0,0,2):C.UVGC_1787_748,(0,0,4):C.UVGC_1787_749,(0,0,0):C.UVGC_2911_2445,(0,0,3):C.UVGC_2911_2446,(0,0,1):C.UVGC_2911_2447,(0,5,2):C.UVGC_1756_702,(0,5,4):C.UVGC_1756_703,(0,5,0):C.UVGC_2899_2409,(0,5,3):C.UVGC_2899_2410,(0,5,1):C.UVGC_2899_2411,(0,4,2):C.UVGC_1756_702,(0,4,4):C.UVGC_1756_703,(0,4,0):C.UVGC_2899_2409,(0,4,3):C.UVGC_2899_2410,(0,4,1):C.UVGC_2899_2411,(0,3,2):C.UVGC_1757_704,(0,3,4):C.UVGC_1757_705,(0,3,0):C.UVGC_2900_2412,(0,3,3):C.UVGC_2900_2413,(0,3,1):C.UVGC_2900_2414})

V_663 = CTVertex(name = 'V_663',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_1533_440,(0,2,3):C.UVGC_1533_441,(0,2,1):C.UVGC_2919_2469,(0,2,4):C.UVGC_2919_2470,(0,2,2):C.UVGC_2919_2471,(0,1,0):C.UVGC_1533_440,(0,1,3):C.UVGC_1533_441,(0,1,1):C.UVGC_2919_2469,(0,1,4):C.UVGC_2919_2470,(0,1,2):C.UVGC_2919_2471,(0,0,0):C.UVGC_1534_442,(0,0,3):C.UVGC_1534_443,(0,0,1):C.UVGC_2918_2466,(0,0,4):C.UVGC_2918_2467,(0,0,2):C.UVGC_2918_2468,(0,5,0):C.UVGC_1500_393,(0,5,3):C.UVGC_1500_394,(0,5,1):C.UVGC_2904_2424,(0,5,4):C.UVGC_2904_2425,(0,5,2):C.UVGC_2904_2426,(0,4,0):C.UVGC_1500_393,(0,4,3):C.UVGC_1500_394,(0,4,1):C.UVGC_2904_2424,(0,4,4):C.UVGC_2904_2425,(0,4,2):C.UVGC_2904_2426,(0,3,0):C.UVGC_1501_395,(0,3,3):C.UVGC_1501_396,(0,3,1):C.UVGC_2903_2421,(0,3,4):C.UVGC_2903_2422,(0,3,2):C.UVGC_2903_2423})

V_664 = CTVertex(name = 'V_664',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.c, P.g, P.t] ], [ [P.c, P.g, P.t, P.u] ], [ [P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_1580_499,(0,2,3):C.UVGC_1580_500,(0,2,1):C.UVGC_2914_2454,(0,2,4):C.UVGC_2914_2455,(0,2,2):C.UVGC_2914_2456,(0,1,0):C.UVGC_1580_499,(0,1,3):C.UVGC_1580_500,(0,1,1):C.UVGC_2914_2454,(0,1,4):C.UVGC_2914_2455,(0,1,2):C.UVGC_2914_2456,(0,0,0):C.UVGC_1581_501,(0,0,3):C.UVGC_1581_502,(0,0,1):C.UVGC_2915_2457,(0,0,4):C.UVGC_2915_2458,(0,0,2):C.UVGC_2915_2459,(0,5,0):C.UVGC_1499_391,(0,5,3):C.UVGC_1499_392,(0,5,1):C.UVGC_2897_2403,(0,5,4):C.UVGC_2897_2404,(0,5,2):C.UVGC_2897_2405,(0,4,0):C.UVGC_1499_391,(0,4,3):C.UVGC_1499_392,(0,4,1):C.UVGC_2897_2403,(0,4,4):C.UVGC_2897_2404,(0,4,2):C.UVGC_2897_2405,(0,3,0):C.UVGC_1498_389,(0,3,3):C.UVGC_1498_390,(0,3,1):C.UVGC_2898_2406,(0,3,4):C.UVGC_2898_2407,(0,3,2):C.UVGC_2898_2408})

V_665 = CTVertex(name = 'V_665',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2835_2310,(0,1,0):C.UVGC_2835_2310,(0,0,0):C.UVGC_2836_2311,(0,5,0):C.UVGC_2667_2055,(0,4,0):C.UVGC_2667_2055,(0,3,0):C.UVGC_2668_2056})

V_666 = CTVertex(name = 'V_666',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2744_2168,(0,2,1):C.UVGC_2744_2169,(0,1,0):C.UVGC_2744_2168,(0,1,1):C.UVGC_2744_2169,(0,0,0):C.UVGC_2745_2170,(0,0,1):C.UVGC_2745_2171,(0,5,0):C.UVGC_2624_1999,(0,5,1):C.UVGC_2624_2000,(0,4,0):C.UVGC_2624_1999,(0,4,1):C.UVGC_2624_2000,(0,3,0):C.UVGC_2625_2001,(0,3,1):C.UVGC_2625_2002})

V_667 = CTVertex(name = 'V_667',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2711_2123,(0,2,1):C.UVGC_2711_2124,(0,1,0):C.UVGC_2711_2123,(0,1,1):C.UVGC_2711_2124,(0,0,0):C.UVGC_2712_2125,(0,0,1):C.UVGC_2712_2126,(0,5,0):C.UVGC_2665_2051,(0,5,1):C.UVGC_2665_2052,(0,4,0):C.UVGC_2665_2051,(0,4,1):C.UVGC_2665_2052,(0,3,0):C.UVGC_2666_2053,(0,3,1):C.UVGC_2666_2054})

V_668 = CTVertex(name = 'V_668',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2819_2294,(0,2,1):C.UVGC_2819_2295,(0,1,0):C.UVGC_2819_2294,(0,1,1):C.UVGC_2819_2295,(0,0,0):C.UVGC_2820_2296,(0,0,1):C.UVGC_2820_2297,(0,5,0):C.UVGC_2687_2085,(0,5,1):C.UVGC_2687_2086,(0,4,0):C.UVGC_2687_2085,(0,4,1):C.UVGC_2687_2086,(0,3,0):C.UVGC_2688_2087,(0,3,1):C.UVGC_2688_2088})

V_669 = CTVertex(name = 'V_669',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.Z, P.Z ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2031_1138,(0,1,0):C.UVGC_2031_1138,(0,0,0):C.UVGC_2032_1139,(0,5,0):C.UVGC_1983_1062,(0,4,0):C.UVGC_1983_1062,(0,3,0):C.UVGC_1984_1063})

V_670 = CTVertex(name = 'V_670',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_1815_777,(0,2,1):C.UVGC_1815_778,(0,1,0):C.UVGC_1814_775,(0,1,1):C.UVGC_1814_776,(0,0,0):C.UVGC_1816_779,(0,0,1):C.UVGC_1816_780,(0,5,0):C.UVGC_1732_655,(0,5,1):C.UVGC_1732_656,(0,4,0):C.UVGC_1731_653,(0,4,1):C.UVGC_1731_654,(0,3,0):C.UVGC_1733_657,(0,3,1):C.UVGC_1733_658})

V_671 = CTVertex(name = 'V_671',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_1777_728,(0,2,1):C.UVGC_1777_729,(0,1,0):C.UVGC_1776_726,(0,1,1):C.UVGC_1776_727,(0,0,0):C.UVGC_1775_724,(0,0,1):C.UVGC_1775_725,(0,5,0):C.UVGC_1736_663,(0,5,1):C.UVGC_1736_664,(0,4,0):C.UVGC_1735_661,(0,4,1):C.UVGC_1735_662,(0,3,0):C.UVGC_1734_659,(0,3,1):C.UVGC_1734_660})

V_672 = CTVertex(name = 'V_672',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_1524_422,(0,2,1):C.UVGC_1524_423,(0,1,0):C.UVGC_1523_420,(0,1,1):C.UVGC_1523_421,(0,0,0):C.UVGC_1522_418,(0,0,1):C.UVGC_1522_419,(0,5,0):C.UVGC_1481_355,(0,5,1):C.UVGC_1481_356,(0,4,0):C.UVGC_1480_353,(0,4,1):C.UVGC_1480_354,(0,3,0):C.UVGC_1479_351,(0,3,1):C.UVGC_1479_352})

V_673 = CTVertex(name = 'V_673',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.a, P.g ],
                 color = [ 'T(4,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_1570_479,(0,2,1):C.UVGC_1570_480,(0,1,0):C.UVGC_1569_477,(0,1,1):C.UVGC_1569_478,(0,0,0):C.UVGC_1571_481,(0,0,1):C.UVGC_1571_482,(0,5,0):C.UVGC_1477_347,(0,5,1):C.UVGC_1477_348,(0,4,0):C.UVGC_1476_345,(0,4,1):C.UVGC_1476_346,(0,3,0):C.UVGC_1478_349,(0,3,1):C.UVGC_1478_350})

V_674 = CTVertex(name = 'V_674',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_1822_791,(0,2,1):C.UVGC_1822_792,(0,1,0):C.UVGC_1823_793,(0,1,1):C.UVGC_1823_794,(0,0,0):C.UVGC_1824_795,(0,0,1):C.UVGC_1824_796,(0,5,0):C.UVGC_1748_686,(0,5,1):C.UVGC_1748_687,(0,4,0):C.UVGC_1749_688,(0,4,1):C.UVGC_1749_689,(0,3,0):C.UVGC_1750_690,(0,3,1):C.UVGC_1750_691})

V_675 = CTVertex(name = 'V_675',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.t, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_1784_742,(0,2,1):C.UVGC_1784_743,(0,1,0):C.UVGC_1785_744,(0,1,1):C.UVGC_1785_745,(0,0,0):C.UVGC_1783_740,(0,0,1):C.UVGC_1783_741,(0,5,0):C.UVGC_1752_694,(0,5,1):C.UVGC_1752_695,(0,4,0):C.UVGC_1753_696,(0,4,1):C.UVGC_1753_697,(0,3,0):C.UVGC_1751_692,(0,3,1):C.UVGC_1751_693})

V_676 = CTVertex(name = 'V_676',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_1531_436,(0,2,1):C.UVGC_1531_437,(0,1,0):C.UVGC_1532_438,(0,1,1):C.UVGC_1532_439,(0,0,0):C.UVGC_1530_434,(0,0,1):C.UVGC_1530_435,(0,5,0):C.UVGC_1496_385,(0,5,1):C.UVGC_1496_386,(0,4,0):C.UVGC_1497_387,(0,4,1):C.UVGC_1497_388,(0,3,0):C.UVGC_1495_383,(0,3,1):C.UVGC_1495_384})

V_677 = CTVertex(name = 'V_677',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.t, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_1577_493,(0,2,1):C.UVGC_1577_494,(0,1,0):C.UVGC_1578_495,(0,1,1):C.UVGC_1578_496,(0,0,0):C.UVGC_1579_497,(0,0,1):C.UVGC_1579_498,(0,5,0):C.UVGC_1492_377,(0,5,1):C.UVGC_1492_378,(0,4,0):C.UVGC_1493_379,(0,4,1):C.UVGC_1493_380,(0,3,0):C.UVGC_1494_381,(0,3,1):C.UVGC_1494_382})

V_678 = CTVertex(name = 'V_678',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2833_2308,(0,1,0):C.UVGC_2834_2309,(0,0,0):C.UVGC_2832_2307,(0,5,0):C.UVGC_2663_2049,(0,4,0):C.UVGC_2664_2050,(0,3,0):C.UVGC_2662_2048})

V_679 = CTVertex(name = 'V_679',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2741_2162,(0,2,1):C.UVGC_2741_2163,(0,1,0):C.UVGC_2742_2164,(0,1,1):C.UVGC_2742_2165,(0,0,0):C.UVGC_2743_2166,(0,0,1):C.UVGC_2743_2167,(0,5,0):C.UVGC_2621_1993,(0,5,1):C.UVGC_2621_1994,(0,4,0):C.UVGC_2622_1995,(0,4,1):C.UVGC_2622_1996,(0,3,0):C.UVGC_2623_1997,(0,3,1):C.UVGC_2623_1998})

V_680 = CTVertex(name = 'V_680',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2709_2119,(0,2,1):C.UVGC_2709_2120,(0,1,0):C.UVGC_2710_2121,(0,1,1):C.UVGC_2710_2122,(0,0,0):C.UVGC_2708_2117,(0,0,1):C.UVGC_2708_2118,(0,5,0):C.UVGC_2660_2044,(0,5,1):C.UVGC_2660_2045,(0,4,0):C.UVGC_2661_2046,(0,4,1):C.UVGC_2661_2047,(0,3,0):C.UVGC_2659_2042,(0,3,1):C.UVGC_2659_2043})

V_681 = CTVertex(name = 'V_681',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ], [ [P.g, P.t, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2816_2288,(0,2,1):C.UVGC_2816_2289,(0,1,0):C.UVGC_2817_2290,(0,1,1):C.UVGC_2817_2291,(0,0,0):C.UVGC_2818_2292,(0,0,1):C.UVGC_2818_2293,(0,5,0):C.UVGC_2684_2079,(0,5,1):C.UVGC_2684_2080,(0,4,0):C.UVGC_2685_2081,(0,4,1):C.UVGC_2685_2082,(0,3,0):C.UVGC_2686_2083,(0,3,1):C.UVGC_2686_2084})

V_682 = CTVertex(name = 'V_682',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g, P.Z ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3, L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2028_1135,(0,1,0):C.UVGC_2029_1136,(0,0,0):C.UVGC_2030_1137,(0,5,0):C.UVGC_1981_1060,(0,4,0):C.UVGC_1982_1061,(0,3,0):C.UVGC_1980_1059})

V_683 = CTVertex(name = 'V_683',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.d, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2500_1764,(0,1,0):C.UVGC_2501_1765,(0,0,0):C.UVGC_2502_1766})

V_684 = CTVertex(name = 'V_684',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2450_1684,(0,2,1):C.UVGC_2450_1685,(0,1,0):C.UVGC_2451_1686,(0,1,1):C.UVGC_2451_1687,(0,0,0):C.UVGC_2452_1688,(0,0,1):C.UVGC_2452_1689})

V_685 = CTVertex(name = 'V_685',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.s, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,2,0):C.UVGC_1910_955,(0,1,0):C.UVGC_1911_956,(0,0,0):C.UVGC_1912_957})

V_686 = CTVertex(name = 'V_686',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2218_1365,(0,1,0):C.UVGC_2219_1366,(0,0,0):C.UVGC_2220_1367})

V_687 = CTVertex(name = 'V_687',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2215_1362,(0,1,0):C.UVGC_2216_1363,(0,0,0):C.UVGC_2217_1364})

V_688 = CTVertex(name = 'V_688',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2339_1527,(0,1,0):C.UVGC_2340_1528,(0,0,0):C.UVGC_2341_1529})

V_689 = CTVertex(name = 'V_689',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2336_1524,(0,1,0):C.UVGC_2337_1525,(0,0,0):C.UVGC_2338_1526})

V_690 = CTVertex(name = 'V_690',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.b, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2138_1275,(0,1,0):C.UVGC_2139_1276,(0,0,0):C.UVGC_2140_1277})

V_691 = CTVertex(name = 'V_691',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.b, P.g, P.W__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV4, L.FFVV5, L.FFVV6 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2135_1272,(0,1,0):C.UVGC_2136_1273,(0,0,0):C.UVGC_2137_1274})

V_692 = CTVertex(name = 'V_692',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2509_1775,(0,1,0):C.UVGC_2510_1776,(0,0,0):C.UVGC_2508_1774})

V_693 = CTVertex(name = 'V_693',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.c, P.g] ], [ [P.b, P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_2470_1715,(0,2,1):C.UVGC_2470_1716,(0,1,0):C.UVGC_2471_1717,(0,1,1):C.UVGC_2471_1718,(0,0,0):C.UVGC_2469_1713,(0,0,1):C.UVGC_2469_1714})

V_694 = CTVertex(name = 'V_694',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.c, P.g, P.s] ] ],
                 couplings = {(0,2,0):C.UVGC_1916_960,(0,1,0):C.UVGC_1917_961,(0,0,0):C.UVGC_1915_959})

V_695 = CTVertex(name = 'V_695',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2256_1401,(0,1,0):C.UVGC_2257_1402,(0,0,0):C.UVGC_2255_1400})

V_696 = CTVertex(name = 'V_696',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.d, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2237_1381,(0,1,0):C.UVGC_2238_1382,(0,0,0):C.UVGC_2236_1380})

V_697 = CTVertex(name = 'V_697',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2397_1597,(0,1,0):C.UVGC_2398_1598,(0,0,0):C.UVGC_2396_1596})

V_698 = CTVertex(name = 'V_698',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.g, P.s, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2364_1558,(0,1,0):C.UVGC_2365_1559,(0,0,0):C.UVGC_2363_1557})

V_699 = CTVertex(name = 'V_699',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2178_1325,(0,1,0):C.UVGC_2179_1326,(0,0,0):C.UVGC_2177_1324})

V_700 = CTVertex(name = 'V_700',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.g, P.W__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV2, L.FFVV3 ],
                 loop_particles = [ [ [P.b, P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_2164_1311,(0,1,0):C.UVGC_2165_1312,(0,0,0):C.UVGC_2163_1310})

V_701 = CTVertex(name = 'V_701',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.u__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF12, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF2, L.FFFF3, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
                 loop_particles = [ [ [P.g, P.H, P.t] ], [ [P.g, P.H, P.u] ] ],
                 couplings = {(1,0,0):C.UVGC_2550_1871,(1,0,1):C.UVGC_2550_1872,(0,0,0):C.UVGC_2549_1869,(0,0,1):C.UVGC_2549_1870,(1,11,0):C.UVGC_2532_1829,(1,11,1):C.UVGC_2532_1830,(0,11,0):C.UVGC_2531_1827,(0,11,1):C.UVGC_2531_1828,(1,12,0):C.UVGC_2542_1853,(1,12,1):C.UVGC_2542_1854,(0,12,0):C.UVGC_2541_1851,(0,12,1):C.UVGC_2541_1852,(1,13,0):C.UVGC_2520_1799,(1,13,1):C.UVGC_2520_1800,(0,13,0):C.UVGC_2519_1797,(0,13,1):C.UVGC_2519_1798,(1,10,0):C.UVGC_2548_1867,(1,10,1):C.UVGC_2548_1868,(0,10,0):C.UVGC_2547_1865,(0,10,1):C.UVGC_2547_1866,(1,8,0):C.UVGC_1304_183,(0,8,0):C.UVGC_1303_182,(1,9,1):C.UVGC_1328_207,(0,9,1):C.UVGC_1327_206,(1,2,0):C.UVGC_2528_1819,(1,2,1):C.UVGC_2528_1820,(0,2,0):C.UVGC_2527_1817,(0,2,1):C.UVGC_2527_1818,(1,5,0):C.UVGC_2540_1849,(1,5,1):C.UVGC_2540_1850,(0,5,0):C.UVGC_2539_1847,(0,5,1):C.UVGC_2539_1848,(1,7,0):C.UVGC_2516_1789,(1,7,1):C.UVGC_2516_1790,(0,7,0):C.UVGC_2515_1787,(0,7,1):C.UVGC_2515_1788,(1,15,0):C.UVGC_1300_179,(0,15,0):C.UVGC_1299_178,(1,14,1):C.UVGC_1322_201,(0,14,1):C.UVGC_1321_200,(1,1,0):C.UVGC_1294_173,(0,1,0):C.UVGC_1293_172,(1,3,0):C.UVGC_1280_159,(0,3,0):C.UVGC_1279_158,(1,4,1):C.UVGC_1326_205,(0,4,1):C.UVGC_1325_204,(1,6,1):C.UVGC_1316_195,(0,6,1):C.UVGC_1315_194})

V_702 = CTVertex(name = 'V_702',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.u__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF12, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF2, L.FFFF3, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
                 loop_particles = [ [ [P.c, P.g, P.H] ], [ [P.g, P.H, P.t] ], [ [P.g, P.H, P.u] ] ],
                 couplings = {(1,0,0):C.UVGC_2534_1834,(1,0,1):C.UVGC_2534_1835,(1,0,2):C.UVGC_2534_1836,(0,0,0):C.UVGC_2533_1831,(0,0,1):C.UVGC_2533_1832,(0,0,2):C.UVGC_2533_1833,(1,11,0):C.UVGC_2530_1824,(1,11,1):C.UVGC_2530_1825,(1,11,2):C.UVGC_2530_1826,(0,11,0):C.UVGC_2529_1821,(0,11,1):C.UVGC_2529_1822,(0,11,2):C.UVGC_2529_1823,(1,12,0):C.UVGC_2522_1804,(1,12,1):C.UVGC_2522_1805,(1,12,2):C.UVGC_2522_1806,(0,12,0):C.UVGC_2521_1801,(0,12,1):C.UVGC_2521_1802,(0,12,2):C.UVGC_2521_1803,(1,13,0):C.UVGC_2518_1794,(1,13,1):C.UVGC_2518_1795,(1,13,2):C.UVGC_2518_1796,(0,13,0):C.UVGC_2517_1791,(0,13,1):C.UVGC_2517_1792,(0,13,2):C.UVGC_2517_1793,(1,10,0):C.UVGC_2297_1481,(1,10,1):C.UVGC_2297_1482,(0,10,0):C.UVGC_2296_1479,(0,10,1):C.UVGC_2296_1480,(1,8,1):C.UVGC_1296_175,(0,8,1):C.UVGC_1295_174,(1,9,2):C.UVGC_1324_203,(0,9,2):C.UVGC_1323_202,(1,2,0):C.UVGC_2295_1477,(1,2,1):C.UVGC_2295_1478,(0,2,0):C.UVGC_2294_1475,(0,2,1):C.UVGC_2294_1476,(1,5,0):C.UVGC_2283_1453,(1,5,1):C.UVGC_2283_1454,(0,5,0):C.UVGC_2282_1451,(0,5,1):C.UVGC_2282_1452,(1,7,0):C.UVGC_2279_1445,(1,7,1):C.UVGC_2279_1446,(0,7,0):C.UVGC_2278_1443,(0,7,1):C.UVGC_2278_1444,(1,15,1):C.UVGC_1284_163,(0,15,1):C.UVGC_1283_162,(1,14,2):C.UVGC_1320_199,(0,14,2):C.UVGC_1319_198,(1,1,1):C.UVGC_1292_171,(0,1,1):C.UVGC_1291_170,(1,3,1):C.UVGC_1278_157,(0,3,1):C.UVGC_1277_156,(1,4,2):C.UVGC_1318_197,(0,4,2):C.UVGC_1317_196,(1,6,2):C.UVGC_1314_193,(0,6,2):C.UVGC_1313_192})

V_703 = CTVertex(name = 'V_703',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.u, P.c__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF12, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF2, L.FFFF3, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
                 loop_particles = [ [ [P.c, P.g, P.H] ], [ [P.g, P.H, P.t] ], [ [P.g, P.H, P.u] ] ],
                 couplings = {(1,0,0):C.UVGC_2546_1862,(1,0,1):C.UVGC_2546_1863,(1,0,2):C.UVGC_2546_1864,(0,0,0):C.UVGC_2545_1859,(0,0,1):C.UVGC_2545_1860,(0,0,2):C.UVGC_2545_1861,(1,11,0):C.UVGC_2526_1814,(1,11,1):C.UVGC_2526_1815,(1,11,2):C.UVGC_2526_1816,(0,11,0):C.UVGC_2525_1811,(0,11,1):C.UVGC_2525_1812,(0,11,2):C.UVGC_2525_1813,(1,12,0):C.UVGC_2538_1844,(1,12,1):C.UVGC_2538_1845,(1,12,2):C.UVGC_2538_1846,(0,12,0):C.UVGC_2537_1841,(0,12,1):C.UVGC_2537_1842,(0,12,2):C.UVGC_2537_1843,(1,13,0):C.UVGC_2514_1784,(1,13,1):C.UVGC_2514_1785,(1,13,2):C.UVGC_2514_1786,(0,13,0):C.UVGC_2513_1781,(0,13,1):C.UVGC_2513_1782,(0,13,2):C.UVGC_2513_1783,(1,10,1):C.UVGC_2544_1857,(1,10,2):C.UVGC_2544_1858,(0,10,1):C.UVGC_2543_1855,(0,10,2):C.UVGC_2543_1856,(1,8,1):C.UVGC_1302_181,(0,8,1):C.UVGC_1301_180,(1,9,0):C.UVGC_1236_115,(0,9,0):C.UVGC_1235_114,(1,2,1):C.UVGC_2524_1809,(1,2,2):C.UVGC_2524_1810,(0,2,1):C.UVGC_2523_1807,(0,2,2):C.UVGC_2523_1808,(1,5,1):C.UVGC_2536_1839,(1,5,2):C.UVGC_2536_1840,(0,5,1):C.UVGC_2535_1837,(0,5,2):C.UVGC_2535_1838,(1,7,1):C.UVGC_2512_1779,(1,7,2):C.UVGC_2512_1780,(0,7,1):C.UVGC_2511_1777,(0,7,2):C.UVGC_2511_1778,(1,15,1):C.UVGC_1298_177,(0,15,1):C.UVGC_1297_176,(1,14,0):C.UVGC_1230_109,(0,14,0):C.UVGC_1229_108,(1,1,1):C.UVGC_1288_167,(0,1,1):C.UVGC_1287_166,(1,3,1):C.UVGC_1276_155,(0,3,1):C.UVGC_1275_154,(1,4,0):C.UVGC_1234_113,(0,4,0):C.UVGC_1233_112,(1,6,0):C.UVGC_1224_103,(0,6,0):C.UVGC_1223_102})

V_704 = CTVertex(name = 'V_704',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.c, P.c__tilde__, P.t ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF1, L.FFFF10, L.FFFF11, L.FFFF12, L.FFFF13, L.FFFF14, L.FFFF15, L.FFFF16, L.FFFF2, L.FFFF3, L.FFFF4, L.FFFF5, L.FFFF6, L.FFFF7, L.FFFF8, L.FFFF9 ],
                 loop_particles = [ [ [P.c, P.g, P.H] ], [ [P.g, P.H, P.t] ] ],
                 couplings = {(1,0,0):C.UVGC_2293_1473,(1,0,1):C.UVGC_2293_1474,(0,0,0):C.UVGC_2292_1471,(0,0,1):C.UVGC_2292_1472,(1,11,0):C.UVGC_2289_1465,(1,11,1):C.UVGC_2289_1466,(0,11,0):C.UVGC_2288_1463,(0,11,1):C.UVGC_2288_1464,(1,12,0):C.UVGC_2285_1457,(1,12,1):C.UVGC_2285_1458,(0,12,0):C.UVGC_2284_1455,(0,12,1):C.UVGC_2284_1456,(1,13,0):C.UVGC_2277_1441,(1,13,1):C.UVGC_2277_1442,(0,13,0):C.UVGC_2276_1439,(0,13,1):C.UVGC_2276_1440,(1,10,0):C.UVGC_2291_1469,(1,10,1):C.UVGC_2291_1470,(0,10,0):C.UVGC_2290_1467,(0,10,1):C.UVGC_2290_1468,(1,8,1):C.UVGC_1290_169,(0,8,1):C.UVGC_1289_168,(1,9,0):C.UVGC_1232_111,(0,9,0):C.UVGC_1231_110,(1,2,0):C.UVGC_2287_1461,(1,2,1):C.UVGC_2287_1462,(0,2,0):C.UVGC_2286_1459,(0,2,1):C.UVGC_2286_1460,(1,5,0):C.UVGC_2281_1449,(1,5,1):C.UVGC_2281_1450,(0,5,0):C.UVGC_2280_1447,(0,5,1):C.UVGC_2280_1448,(1,7,0):C.UVGC_2275_1437,(1,7,1):C.UVGC_2275_1438,(0,7,0):C.UVGC_2274_1435,(0,7,1):C.UVGC_2274_1436,(1,15,1):C.UVGC_1282_161,(0,15,1):C.UVGC_1281_160,(1,14,0):C.UVGC_1228_107,(0,14,0):C.UVGC_1227_106,(1,1,1):C.UVGC_1286_165,(0,1,1):C.UVGC_1285_164,(1,3,1):C.UVGC_1274_153,(0,3,1):C.UVGC_1273_152,(1,4,0):C.UVGC_1226_105,(0,4,0):C.UVGC_1225_104,(1,6,0):C.UVGC_1222_101,(0,6,0):C.UVGC_1221_100})

V_705 = CTVertex(name = 'V_705',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.u, P.t__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF13, L.FFFF14, L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,2,0):C.UVGC_2273_1433,(1,2,1):C.UVGC_2273_1434,(0,2,0):C.UVGC_2272_1431,(0,2,1):C.UVGC_2272_1432,(1,1,0):C.UVGC_1220_99,(0,1,0):C.UVGC_1219_98,(1,0,1):C.UVGC_1272_151,(0,0,1):C.UVGC_1271_150})

V_706 = CTVertex(name = 'V_706',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.c, P.t__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF13, L.FFFF14, L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,2,0):C.UVGC_2267_1421,(1,2,1):C.UVGC_2267_1422,(0,2,0):C.UVGC_2266_1419,(0,2,1):C.UVGC_2266_1420,(1,1,0):C.UVGC_1214_93,(0,1,0):C.UVGC_1213_92,(1,0,1):C.UVGC_1266_145,(0,0,1):C.UVGC_1265_144})

V_707 = CTVertex(name = 'V_707',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.t, P.t__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF13, L.FFFF14, L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,2,0):C.UVGC_2271_1429,(1,2,1):C.UVGC_2271_1430,(0,2,0):C.UVGC_2270_1427,(0,2,1):C.UVGC_2270_1428,(1,1,0):C.UVGC_1218_97,(0,1,0):C.UVGC_1217_96,(1,0,1):C.UVGC_1270_149,(0,0,1):C.UVGC_1269_148})

V_708 = CTVertex(name = 'V_708',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.t, P.t__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF13, L.FFFF14, L.FFFF6 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,2,0):C.UVGC_2269_1425,(1,2,1):C.UVGC_2269_1426,(0,2,0):C.UVGC_2268_1423,(0,2,1):C.UVGC_2268_1424,(1,1,0):C.UVGC_1216_95,(0,1,0):C.UVGC_1215_94,(1,0,1):C.UVGC_1268_147,(0,0,1):C.UVGC_1267_146})

V_709 = CTVertex(name = 'V_709',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.t__tilde__, P.d ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF14, L.FFFF6, L.FFFF9 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_2263_1413,(1,1,1):C.UVGC_2263_1414,(0,1,0):C.UVGC_2262_1411,(0,1,1):C.UVGC_2262_1412,(1,0,1):C.UVGC_1262_141,(0,0,1):C.UVGC_1261_140,(1,2,0):C.UVGC_1210_89,(0,2,0):C.UVGC_1209_88})

V_710 = CTVertex(name = 'V_710',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.t__tilde__, P.s ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF14, L.FFFF6, L.FFFF9 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_2261_1409,(1,1,1):C.UVGC_2261_1410,(0,1,0):C.UVGC_2260_1407,(0,1,1):C.UVGC_2260_1408,(1,0,1):C.UVGC_1260_139,(0,0,1):C.UVGC_1259_138,(1,2,0):C.UVGC_1208_87,(0,2,0):C.UVGC_1207_86})

V_711 = CTVertex(name = 'V_711',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.u__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF14, L.FFFF6, L.FFFF9 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_2265_1417,(1,1,1):C.UVGC_2265_1418,(0,1,0):C.UVGC_2264_1415,(0,1,1):C.UVGC_2264_1416,(1,0,1):C.UVGC_1264_143,(0,0,1):C.UVGC_1263_142,(1,2,0):C.UVGC_1212_91,(0,2,0):C.UVGC_1211_90})

V_712 = CTVertex(name = 'V_712',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.c__tilde__, P.b ],
                 color = [ 'Identity(1,2)*Identity(3,4)', 'Identity(1,4)*Identity(2,3)' ],
                 lorentz = [ L.FFFF14, L.FFFF6, L.FFFF9 ],
                 loop_particles = [ [ [P.b, P.g, P.G__plus__] ], [ [P.g, P.G__plus__, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_2259_1405,(1,1,1):C.UVGC_2259_1406,(0,1,0):C.UVGC_2258_1403,(0,1,1):C.UVGC_2258_1404,(1,0,1):C.UVGC_1258_137,(0,0,1):C.UVGC_1257_136,(1,2,0):C.UVGC_1206_85,(0,2,0):C.UVGC_1205_84})

V_713 = CTVertex(name = 'V_713',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV3, L.FFV34, L.FFV9 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_3509_3640,(0,3,0):C.UVGC_3509_3640,(0,0,0):C.UVGC_3240_3165,(0,2,0):C.UVGC_3240_3165,(0,5,0):C.UVGC_3509_3640,(0,4,0):C.UVGC_3240_3165})

V_714 = CTVertex(name = 'V_714',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.a ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV3, L.FFV34, L.FFV9 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_3579_3788,(0,3,0):C.UVGC_3579_3788,(0,0,0):C.UVGC_3239_3164,(0,2,0):C.UVGC_3239_3164,(0,5,0):C.UVGC_3579_3788,(0,4,0):C.UVGC_3239_3164})

V_715 = CTVertex(name = 'V_715',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV3, L.FFV34, L.FFV9 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_3500_3629,(0,3,0):C.UVGC_3500_3629,(0,0,0):C.UVGC_3244_3169,(0,2,0):C.UVGC_3244_3169,(0,5,0):C.UVGC_3500_3629,(0,4,0):C.UVGC_3244_3169})

V_716 = CTVertex(name = 'V_716',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.g ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFV19, L.FFV2, L.FFV20, L.FFV3, L.FFV34, L.FFV9 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_3552_3741,(0,3,0):C.UVGC_3552_3741,(0,0,0):C.UVGC_3243_3168,(0,2,0):C.UVGC_3243_3168,(0,5,0):C.UVGC_3552_3741,(0,4,0):C.UVGC_3243_3168})

V_717 = CTVertex(name = 'V_717',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS22, L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_3396_3454,(0,0,0):C.UVGC_3234_3159})

V_718 = CTVertex(name = 'V_718',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS22, L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_3398_3456,(0,0,0):C.UVGC_3235_3160})

V_719 = CTVertex(name = 'V_719',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS22, L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_3479_3608,(0,0,0):C.UVGC_3236_3161})

V_720 = CTVertex(name = 'V_720',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS22, L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,1,0):C.UVGC_3481_3610,(0,0,0):C.UVGC_3237_3162})

V_721 = CTVertex(name = 'V_721',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3395_3453})

V_722 = CTVertex(name = 'V_722',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3397_3455})

V_723 = CTVertex(name = 'V_723',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3478_3607})

V_724 = CTVertex(name = 'V_724',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3480_3609})

V_725 = CTVertex(name = 'V_725',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1624_543})

V_726 = CTVertex(name = 'V_726',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1625_544})

V_727 = CTVertex(name = 'V_727',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1614_533})

V_728 = CTVertex(name = 'V_728',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1615_534})

V_729 = CTVertex(name = 'V_729',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS27, L.FFVS35, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3421_3480,(0,2,0):C.UVGC_3421_3480,(0,5,0):C.UVGC_3421_3480,(0,3,0):C.UVGC_3106_2949,(0,1,0):C.UVGC_3421_3480,(0,4,0):C.UVGC_3108_2951})

V_730 = CTVertex(name = 'V_730',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS27, L.FFVS35, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3574_3783,(0,2,0):C.UVGC_3574_3783,(0,5,0):C.UVGC_3574_3783,(0,3,0):C.UVGC_3111_2954,(0,1,0):C.UVGC_3574_3783,(0,4,0):C.UVGC_3113_2956})

V_731 = CTVertex(name = 'V_731',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS27, L.FFVS35, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3504_3635,(0,2,0):C.UVGC_3504_3635,(0,5,0):C.UVGC_3504_3635,(0,3,0):C.UVGC_3116_2959,(0,1,0):C.UVGC_3504_3635,(0,4,0):C.UVGC_3118_2961})

V_732 = CTVertex(name = 'V_732',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS27, L.FFVS35, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3583_3792,(0,2,0):C.UVGC_3583_3792,(0,5,0):C.UVGC_3583_3792,(0,3,0):C.UVGC_3121_2964,(0,1,0):C.UVGC_3583_3792,(0,4,0):C.UVGC_3123_2966})

V_733 = CTVertex(name = 'V_733',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.a, P.G0 ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS27, L.FFVS35, L.FFVS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_3622_3887,(0,0,1):C.UVGC_3622_3888,(0,0,2):C.UVGC_3622_3889,(0,2,0):C.UVGC_3622_3887,(0,2,1):C.UVGC_3622_3888,(0,2,2):C.UVGC_3622_3889,(0,5,0):C.UVGC_3622_3887,(0,5,1):C.UVGC_3622_3888,(0,5,2):C.UVGC_3622_3889,(0,3,0):C.UVGC_3631_3914,(0,3,1):C.UVGC_3631_3915,(0,3,2):C.UVGC_3631_3916,(0,1,0):C.UVGC_3622_3887,(0,1,1):C.UVGC_3622_3888,(0,1,2):C.UVGC_3622_3889,(0,4,0):C.UVGC_3633_3920,(0,4,1):C.UVGC_3633_3921,(0,4,2):C.UVGC_3633_3922})

V_734 = CTVertex(name = 'V_734',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS27, L.FFVS35, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3415_3473,(0,2,0):C.UVGC_3415_3473,(0,5,0):C.UVGC_3415_3473,(0,3,0):C.UVGC_3126_2969,(0,1,0):C.UVGC_3415_3473,(0,4,0):C.UVGC_3128_2971})

V_735 = CTVertex(name = 'V_735',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS27, L.FFVS35, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3550_3739,(0,2,0):C.UVGC_3550_3739,(0,5,0):C.UVGC_3550_3739,(0,3,0):C.UVGC_3131_2974,(0,1,0):C.UVGC_3550_3739,(0,4,0):C.UVGC_3133_2976})

V_736 = CTVertex(name = 'V_736',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS27, L.FFVS35, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3498_3627,(0,2,0):C.UVGC_3498_3627,(0,5,0):C.UVGC_3498_3627,(0,3,0):C.UVGC_3136_2979,(0,1,0):C.UVGC_3498_3627,(0,4,0):C.UVGC_3138_2981})

V_737 = CTVertex(name = 'V_737',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS27, L.FFVS35, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3555_3744,(0,2,0):C.UVGC_3555_3744,(0,5,0):C.UVGC_3555_3744,(0,3,0):C.UVGC_3141_2984,(0,1,0):C.UVGC_3555_3744,(0,4,0):C.UVGC_3143_2986})

V_738 = CTVertex(name = 'V_738',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.G0 ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS27, L.FFVS35, L.FFVS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_3616_3867,(0,0,1):C.UVGC_3616_3868,(0,0,2):C.UVGC_3616_3869,(0,2,0):C.UVGC_3616_3867,(0,2,1):C.UVGC_3616_3868,(0,2,2):C.UVGC_3616_3869,(0,5,0):C.UVGC_3616_3867,(0,5,1):C.UVGC_3616_3868,(0,5,2):C.UVGC_3616_3869,(0,3,0):C.UVGC_3636_3929,(0,3,1):C.UVGC_3636_3930,(0,3,2):C.UVGC_3636_3931,(0,1,0):C.UVGC_3616_3867,(0,1,1):C.UVGC_3616_3868,(0,1,2):C.UVGC_3616_3869,(0,4,0):C.UVGC_3638_3935,(0,4,1):C.UVGC_3638_3936,(0,4,2):C.UVGC_3638_3937})

V_739 = CTVertex(name = 'V_739',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS35, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3399_3457,(0,0,0):C.UVGC_3110_2953,(0,1,0):C.UVGC_3109_2952})

V_740 = CTVertex(name = 'V_740',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS35, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3400_3458,(0,0,0):C.UVGC_3115_2958,(0,1,0):C.UVGC_3114_2957})

V_741 = CTVertex(name = 'V_741',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS35, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3482_3611,(0,0,0):C.UVGC_3120_2963,(0,1,0):C.UVGC_3119_2962})

V_742 = CTVertex(name = 'V_742',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS35, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,2,0):C.UVGC_3483_3612,(0,0,0):C.UVGC_3125_2968,(0,1,0):C.UVGC_3124_2967})

V_743 = CTVertex(name = 'V_743',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.a, P.G__plus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS27, L.FFVS35, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,2,0):C.UVGC_3610_3841,(0,2,1):C.UVGC_3610_3842,(0,2,2):C.UVGC_3610_3843,(0,0,0):C.UVGC_3635_3926,(0,0,1):C.UVGC_3635_3927,(0,0,2):C.UVGC_3635_3928,(0,1,0):C.UVGC_3634_3923,(0,1,1):C.UVGC_3634_3924,(0,1,2):C.UVGC_3634_3925})

V_744 = CTVertex(name = 'V_744',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.s, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS27, L.FFVS35 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3129_2972,(0,1,0):C.UVGC_3130_2973})

V_745 = CTVertex(name = 'V_745',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.d, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS27, L.FFVS35 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3134_2977,(0,1,0):C.UVGC_3135_2978})

V_746 = CTVertex(name = 'V_746',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.s, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS27, L.FFVS35 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3139_2982,(0,1,0):C.UVGC_3140_2983})

V_747 = CTVertex(name = 'V_747',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.d, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS27, L.FFVS35 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_3144_2987,(0,1,0):C.UVGC_3145_2988})

V_748 = CTVertex(name = 'V_748',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.b, P.g, P.G__plus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS27, L.FFVS35 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_3639_3938,(0,0,1):C.UVGC_3639_3939,(0,0,2):C.UVGC_3639_3940,(0,1,0):C.UVGC_3640_3941,(0,1,1):C.UVGC_3640_3942,(0,1,2):C.UVGC_3640_3943})

V_749 = CTVertex(name = 'V_749',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,4,0):C.UVGC_1626_545,(0,0,0):C.UVGC_1649_569,(0,2,0):C.UVGC_1649_569,(0,3,0):C.UVGC_1649_569,(0,1,0):C.UVGC_1649_569})

V_750 = CTVertex(name = 'V_750',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,4,0):C.UVGC_1627_546,(0,0,0):C.UVGC_1673_593,(0,2,0):C.UVGC_1673_593,(0,3,0):C.UVGC_1673_593,(0,1,0):C.UVGC_1673_593})

V_751 = CTVertex(name = 'V_751',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,4,0):C.UVGC_1616_535,(0,0,0):C.UVGC_1638_558,(0,2,0):C.UVGC_1638_558,(0,3,0):C.UVGC_1638_558,(0,1,0):C.UVGC_1638_558})

V_752 = CTVertex(name = 'V_752',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,4,0):C.UVGC_1617_536,(0,0,0):C.UVGC_1662_582,(0,2,0):C.UVGC_1662_582,(0,3,0):C.UVGC_1662_582,(0,1,0):C.UVGC_1662_582})

V_753 = CTVertex(name = 'V_753',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.a, P.G__minus__ ],
                 color = [ 'Identity(1,2)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS4, L.FFVS6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,4,0):C.UVGC_1835_815,(0,4,1):C.UVGC_1835_816,(0,4,2):C.UVGC_1835_817,(0,0,0):C.UVGC_1839_827,(0,0,1):C.UVGC_1839_828,(0,0,2):C.UVGC_1839_829,(0,2,0):C.UVGC_1839_827,(0,2,1):C.UVGC_1839_828,(0,2,2):C.UVGC_1839_829,(0,3,0):C.UVGC_1839_827,(0,3,1):C.UVGC_1839_828,(0,3,2):C.UVGC_1839_829,(0,1,0):C.UVGC_1839_827,(0,1,1):C.UVGC_1839_828,(0,1,2):C.UVGC_1839_829})

V_754 = CTVertex(name = 'V_754',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.c, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1648_568,(0,2,0):C.UVGC_1648_568,(0,3,0):C.UVGC_1648_568,(0,1,0):C.UVGC_1648_568})

V_755 = CTVertex(name = 'V_755',
                 type = 'UV',
                 particles = [ P.d__tilde__, P.u, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1672_592,(0,2,0):C.UVGC_1672_592,(0,3,0):C.UVGC_1672_592,(0,1,0):C.UVGC_1672_592})

V_756 = CTVertex(name = 'V_756',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.c, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1637_557,(0,2,0):C.UVGC_1637_557,(0,3,0):C.UVGC_1637_557,(0,1,0):C.UVGC_1637_557})

V_757 = CTVertex(name = 'V_757',
                 type = 'UV',
                 particles = [ P.s__tilde__, P.u, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS4 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(0,0,0):C.UVGC_1661_581,(0,2,0):C.UVGC_1661_581,(0,3,0):C.UVGC_1661_581,(0,1,0):C.UVGC_1661_581})

V_758 = CTVertex(name = 'V_758',
                 type = 'UV',
                 particles = [ P.b__tilde__, P.t, P.g, P.G__minus__ ],
                 color = [ 'T(3,2,1)' ],
                 lorentz = [ L.FFVS1, L.FFVS11, L.FFVS2, L.FFVS4 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(0,0,0):C.UVGC_1838_824,(0,0,1):C.UVGC_1838_825,(0,0,2):C.UVGC_1838_826,(0,2,0):C.UVGC_1838_824,(0,2,1):C.UVGC_1838_825,(0,2,2):C.UVGC_1838_826,(0,3,0):C.UVGC_1838_824,(0,3,1):C.UVGC_1838_825,(0,3,2):C.UVGC_1838_826,(0,1,0):C.UVGC_1838_824,(0,1,1):C.UVGC_1838_825,(0,1,2):C.UVGC_1838_826})

V_759 = CTVertex(name = 'V_759',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.c, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV3, L.FFVV4, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_3419_3477,(0,1,0):C.UVGC_3418_3476,(1,0,0):C.UVGC_3418_3476,(0,0,0):C.UVGC_3419_3477,(1,3,0):C.UVGC_3247_3172,(0,3,0):C.UVGC_3246_3171,(1,2,0):C.UVGC_3246_3171,(0,2,0):C.UVGC_3247_3172})

V_760 = CTVertex(name = 'V_760',
                 type = 'UV',
                 particles = [ P.c__tilde__, P.u, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV3, L.FFVV4, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_3554_3743,(0,1,0):C.UVGC_3553_3742,(1,0,0):C.UVGC_3553_3742,(0,0,0):C.UVGC_3554_3743,(1,3,0):C.UVGC_3249_3174,(0,3,0):C.UVGC_3248_3173,(1,2,0):C.UVGC_3248_3173,(0,2,0):C.UVGC_3249_3174})

V_761 = CTVertex(name = 'V_761',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.c, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV3, L.FFVV4, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_3502_3631,(0,1,0):C.UVGC_3501_3630,(1,0,0):C.UVGC_3501_3630,(0,0,0):C.UVGC_3502_3631,(1,3,0):C.UVGC_3251_3176,(0,3,0):C.UVGC_3250_3175,(1,2,0):C.UVGC_3250_3175,(0,2,0):C.UVGC_3251_3176})

V_762 = CTVertex(name = 'V_762',
                 type = 'UV',
                 particles = [ P.u__tilde__, P.u, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV3, L.FFVV4, L.FFVV6 ],
                 loop_particles = [ [ [P.g, P.t] ] ],
                 couplings = {(1,1,0):C.UVGC_3559_3748,(0,1,0):C.UVGC_3558_3747,(1,0,0):C.UVGC_3558_3747,(0,0,0):C.UVGC_3559_3748,(1,3,0):C.UVGC_3253_3178,(0,3,0):C.UVGC_3252_3177,(1,2,0):C.UVGC_3252_3177,(0,2,0):C.UVGC_3253_3178})

V_763 = CTVertex(name = 'V_763',
                 type = 'UV',
                 particles = [ P.t__tilde__, P.t, P.g, P.g ],
                 color = [ 'T(3,-1,1)*T(4,2,-1)', 'T(3,2,-1)*T(4,-1,1)' ],
                 lorentz = [ L.FFVV1, L.FFVV3, L.FFVV4, L.FFVV6 ],
                 loop_particles = [ [ [P.c, P.g] ], [ [P.g, P.t] ], [ [P.g, P.u] ] ],
                 couplings = {(1,1,0):C.UVGC_3620_3879,(1,1,1):C.UVGC_3620_3880,(1,1,2):C.UVGC_3620_3881,(0,1,0):C.UVGC_3619_3876,(0,1,1):C.UVGC_3619_3877,(0,1,2):C.UVGC_3619_3878,(1,0,0):C.UVGC_3619_3876,(1,0,1):C.UVGC_3619_3877,(1,0,2):C.UVGC_3619_3878,(0,0,0):C.UVGC_3620_3879,(0,0,1):C.UVGC_3620_3880,(0,0,2):C.UVGC_3620_3881,(1,3,0):C.UVGC_3657_3988,(1,3,1):C.UVGC_3657_3989,(1,3,2):C.UVGC_3657_3990,(0,3,0):C.UVGC_3656_3985,(0,3,1):C.UVGC_3656_3986,(0,3,2):C.UVGC_3656_3987,(1,2,0):C.UVGC_3656_3985,(1,2,1):C.UVGC_3656_3986,(1,2,2):C.UVGC_3656_3987,(0,2,0):C.UVGC_3657_3988,(0,2,1):C.UVGC_3657_3989,(0,2,2):C.UVGC_3657_3990})


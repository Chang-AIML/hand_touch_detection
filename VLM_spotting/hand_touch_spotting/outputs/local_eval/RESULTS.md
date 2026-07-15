# Phase 2 (p2_slide_a70) — rigorous local eval (5090)
Config: finegym 640 / tennis·fs·tm 200 / finediving OOD full 1018. eval_max_tokens 64.

## s450  (in-domain _all mAP@2=20.7)
in-domain _all @(0,1,2,4) = 5.30, 14.74, 20.73, 28.97   (ndet 1540, ngt 1772)
  finegym      5.38, 11.57, 16.15, 23.08
  fs_perf      3.20, 17.37, 27.61, 42.70
  tennis       7.12, 27.73, 33.72, 41.18
  touchmoment  2.71, 19.76, 38.99, 56.25
OOD finediving @(0,1,2,4) = 0.23, 1.60, 5.77, 24.85     (ndet 1007, ngt 1063)

## Phase-1 ref: OOD @2 s450=2.27, peak 7.25@s600, @4 peak 21.06 -> Phase2 s450 already exceeds P1 peak @4

## s600  (in-domain _all @2=27.6, OOD flat vs s450)
in-domain _all @(0,1,2,4) = 6.84, 21.12, 27.59, 35.20   (ndet 1666, ngt 1772)
  finegym      7.19, 20.11, 25.75, 32.38
  fs_perf      6.87, 20.96, 27.96, 42.06
  tennis       5.60, 22.40, 29.82, 35.32
  touchmoment  5.18, 33.19, 48.59, 64.82
OOD finediving @(0,1,2,4) = 0.53, 2.01, 5.23, 24.08     (ndet 964, ngt 1063)

## OOD trajectory: s450 @2=5.77/@4=24.85 -> s600 @2=5.23/@4=24.08  (PLATEAU, not collapse; vs P1 peaked s600 then crashed)

## s750 (in-domain @2=39.9 accelerating; OOD @2=7.68 also up -> DOUBLE RISE, no tradeoff)
in-domain _all @(0,1,2,4,8,16) = 8.11, 28.38, 39.93, 48.82, 55.50, 59.61  (ndet 1756, ngt 1772)
  finegym      5.20, 22.67, 33.03, 40.99, 48.80, 53.42   [finegym @8=48.8 ~= E2E FG-Full @1=47.9]
  fs_perf     12.75, 39.82, 54.33, 66.18, 66.45, 70.73
  tennis      20.70, 48.64, 62.86, 70.87, 75.16, 76.90
  touchmoment  6.25, 33.22, 49.14, 69.35, 78.42, 81.52
OOD finediving @(0,1,2,4,8,16) = 0.24, 3.01, 7.68, 26.45, 48.17, 65.81

## in-domain _all @2 trajectory: s450=20.7 -> s600=27.6 -> s750=39.9 (accelerating)
## OOD @2 trajectory:            s450=5.77 -> s600=5.23 -> s750=7.68 (rising, NOT collapsing; P1 collapsed by s920)

## s900 (OOD DECLINING -> emerge-then-collapse present but MITIGATED vs P1)
OOD finediving @(0,1,2,4,8,16) = 0.41, 2.49, 5.58, 20.40, 38.62, 46.94  (ndet 985, ngt 1063)

## OOD @2 full trajectory: s450=5.77 -> s600=5.23 -> s750=7.68(PEAK) -> s900=5.58(declining)
## OOD @16 trajectory:      s450=60.9 -> s750=65.8(PEAK) -> s900=46.9(declining)
## vs Phase1: peak later(s750 vs s600) + higher(@2 7.68 vs 7.25, @16 65.8 vs 53.6) + gentler decline
##            (s900 @2=5.58 vs P1 s920 @2=0.55 ; s900 @16=46.9 vs P1 s920 @15=14.6)
## => MITIGATED collapse, NOT sustained. OOD-optimal ckpt = s750.

## s900 in-domain (still rising but DECELERATING; OOD declining -> divergence/tradeoff re-emerges)
in-domain _all @(0,1,2,4,8,16) = 9.29, 30.80, 42.50, 54.67, 60.56, 64.93  (ndet 1840, ngt 1772)
## in-domain _all @2: 20.7 -> 27.6 -> 39.9 -> 42.5 (deltas +6.9,+12.3,+2.6 -> decelerating)
## OOD _all @2:       5.77 -> 5.23 -> 7.68 -> 5.58 (peak s750, declining)
## => s900: in-domain UP (42.5, near plateau) + OOD DOWN (5.58) = specialization>generalization tradeoff

## s1050 in-domain (rising) vs OOD (declining) — divergence widening
in-domain _all @(0,1,2,4,8,16) = 10.61, 34.06, 45.94, 57.00, 63.07, 66.64  (ndet 1761, ngt 1772)
## in-domain @2: 20.7 -> 27.6 -> 39.9 -> 42.5 -> 45.9 (rising)
## OOD @2:       5.77 -> 5.23 -> 7.68 -> 5.58 -> 3.51 (peak s750, declining)

## s1200 full — OOD STABILIZES (decline halted, slight recovery) => HIGH FLOOR not full crash
in-domain _all @(0,1,2,4,8,16) = 11.24, 37.59, 50.18, 59.13, 66.40, 70.16  (ndet 1746, ngt 1772)
OOD finediving  @(0,1,2,4,8,16) =  0.20,  1.73,  4.43, 15.44, 31.36, 38.62  (ndet 926, ngt 1063)
## OOD @2 full: 5.77 5.23 7.68(PK) 5.58 3.51 4.43  -> floor ~4 (vs P1 crashed to 0.55)
## OOD @16:     60.9  -  65.8(PK) 46.9 37.7 38.6   -> floor ~38 (vs P1 crashed to 14.6)
## in-domain @2: 20.7 27.6 39.9 42.5 45.9 50.2 (monotone rising)

## s1350 (in-domain PLATEAU + OOD FLOOR confirmed)
in-domain _all @(0,1,2,4,8,16) = 13.07, 39.13, 50.72, 61.03, 67.83, 71.41
OOD finediving  @(0,1,2,4,8,16) =  0.29,  1.79,  4.65, 15.11, 30.77, 36.51
## OOD @2 tail: s1050=3.51 s1200=4.43 s1350=4.65  -> STABLE floor ~4.5 (3 pts), NOT crash
## OOD @16 tail: 37.7 38.6 36.5 -> stable ~37 (vs P1 crashed to 14.6)
## in-domain @2: s1050=45.9 s1200=50.2 s1350=50.7 -> PLATEAU ~50

"""
Generate a realistic synthetic document corpus for the RAG knowledge base.

Produces 40+ documents across four categories:
  - well_reports       (15 documents)
  - maintenance_logs   (10 documents)
  - hse_procedures     (8 documents)
  - equipment_manuals  (8 documents)
"""

from __future__ import annotations

import textwrap
from pathlib import Path


# ─── Well Reports ─────────────────────────────────────────────────────────────

WELL_REPORTS: list[dict[str, str]] = [
    {
        "filename": "WR-001_Draugen_D1H_Q1_2024.md",
        "title": "Well Performance Report – Draugen D-1H – Q1 2024",
        "content": textwrap.dedent("""\
            # Well Performance Report – Draugen D-1H – Q1 2024

            **Well ID:** D-1H
            **Field:** Draugen
            **Reservoir:** Rogn Formation (Jurassic)
            **Report Period:** January 1 – March 31, 2024
            **Prepared by:** Reservoir Engineering Team

            ## Executive Summary

            D-1H maintained stable production throughout Q1 2024, averaging 2,450 BOPD with a
            water-cut of 12.3%. No significant anomalies were observed. The well is producing
            from the Rogn sandstone at approximately 2,840 m TVD.

            ## Production Performance

            | Month   | Oil (BOPD) | Water Cut (%) | GOR (scf/bbl) | BHP (psi) |
            |---------|-----------|---------------|---------------|-----------|
            | January | 2,480     | 11.8          | 632           | 3,210     |
            | February| 2,430     | 12.1          | 641           | 3,205     |
            | March   | 2,440     | 12.9          | 638           | 3,198     |

            ## Observations

            - Water cut increased marginally (+1.1%) over the quarter, consistent with
              aquifer influx modelling predictions.
            - GOR remains within normal operating range (600–700 scf/bbl).
            - Bottomhole pressure declined ~12 psi, tracking reservoir depletion model.
            - No interventions performed during the period.

            ## Reservoir Status

            Current reservoir pressure estimated at 3,850 psi (above bubble point of 3,100 psi).
            Sweep efficiency in the northern fault block remains high (~78%).

            ## Recommendations

            1. Continue routine production monitoring at 6-hour intervals.
            2. Schedule a PLT survey in Q2 to confirm contribution intervals.
            3. Monitor water cut trend — if >20% within 90 days, initiate conformance review.

            ## Risk Assessment

            - **Production risk:** LOW
            - **Well integrity risk:** LOW
            - **Intervention trigger:** Water cut > 25% or BHP decline > 100 psi/month
        """),
    },
    {
        "filename": "WR-002_Volve_F4_WaterBreakthrough_2024.md",
        "title": "Incident Report – Volve 15/9-F-4 – Rapid Water Breakthrough – March 2024",
        "content": textwrap.dedent("""\
            # Incident Report – Volve 15/9-F-4 – Rapid Water Breakthrough

            **Well ID:** 15/9-F-4
            **Field:** Volve (Block 15/9, Norwegian North Sea)
            **Date of Incident:** March 15, 2024
            **Reported by:** Production Operations

            ## Incident Summary

            Well 15/9-F-4 exhibited a rapid increase in water-cut from 18% to 61% over a
            72-hour period beginning March 15, 2024. Concurrent oil rate declined from 2,200
            BOPD to approximately 870 BOPD. The anomaly was detected by the production
            surveillance system at hour 8.

            ## Timeline

            | Time       | Event                                            |
            |------------|--------------------------------------------------|
            | 06:00 Mar15| Water cut at 18% (baseline)                     |
            | 14:00 Mar15| Water cut 28% — surveillance alert triggered    |
            | 22:00 Mar15| Water cut 42% — ops engineer notified           |
            | 10:00 Mar16| Water cut 55% — production optimiser called     |
            | 18:00 Mar16| Choke reduced to 32/64" to reduce drawdown      |
            | 06:00 Mar17| Water cut stabilised at 61% — investigation open|

            ## Root Cause Analysis

            Coning analysis indicates preferential aquifer breakthrough along the eastern
            fault plane. Water saturation logs from the nearest injector (F-11H) suggest
            injected water has arrived ahead of schedule, approximately 14 months earlier
            than the dynamic model predicted.

            **Primary cause:** Injector-producer connectivity through fracture network not
            captured in the static model.

            **Secondary cause:** Higher-than-expected aquifer drive from the underlying
            Hugin formation.

            ## Corrective Actions

            1. Reduced choke size from 48 to 32 (64ths inch) to reduce drawdown.
            2. Requested revised reservoir simulation with updated aquifer model.
            3. Placed injector 15/9-F-11H on monitoring-only for 30 days.
            4. Initiated conformance chemical treatment study.

            ## Production Impact

            Estimated deferred production: 1,330 BBL/day × 5 days = 6,650 barrels.
            At $80/bbl: approximately $532,000 revenue impact.

            ## Lessons Learned

            - Automated water-cut monitoring thresholds (>25% increase/24h) correctly
              triggered early alert. Response time could be improved to <4 hours.
            - Fracture connectivity uncertainty should be flagged in well models.

            ## References

            - Volve Field Development Plan, Section 4.3 (Aquifer Modelling)
            - Norwegian Offshore Directorate Well Report 15/9-F-4 (2008)
        """),
    },
    {
        "filename": "WR-003_Gullfaks_GFA2H_LiquidLoading_2024.md",
        "title": "Well Performance Report – Gullfaks GF-A-2H – Liquid Loading Investigation",
        "content": textwrap.dedent("""\
            # Well Performance Report – Gullfaks GF-A-2H – Liquid Loading Investigation

            **Well ID:** GF-A-2H
            **Field:** Gullfaks
            **Date:** April 2024
            **Reservoir:** Cook Formation

            ## Problem Statement

            GF-A-2H has shown a progressive decline in oil rate over the past 6 weeks,
            consistent with liquid loading behaviour. Current production: 890 BOPD
            (down from 1,450 BOPD baseline). Wellhead pressure has increased from 180 to
            310 psi despite unchanged separator pressure.

            ## Liquid Loading Diagnostics

            Liquid loading occurs when gas velocity in the tubing falls below the critical
            velocity required to lift liquids to surface. Turner's critical velocity
            correlation gives:

                Vcrit = 5.62 × [(σ(ρL - ρG))^0.25] / ρG^0.5

            For current conditions (WHP=310 psi, T=145°F), Vcrit ≈ 14.2 ft/s.
            Current calculated velocity ≈ 11.8 ft/s → below critical → confirmed loading.

            ## Symptoms Observed

            - Erratic wellhead pressure oscillations (slug flow pattern)
            - Declining average oil rate with increasing variance
            - Increasing surface liquid-gas ratio in produced stream
            - Temperature gradient anomaly at 1,840 m depth (liquid accumulation point)

            ## Recommended Interventions

            ### Immediate (0–7 days)
            1. Increase gas-lift injection rate from 0.8 to 1.4 MMscf/day.
            2. Adjust choke from 40 to 48 (64ths) to reduce back-pressure.

            ### Short-term (7–30 days)
            3. Install velocity string to reduce tubing diameter below liquid accumulation point.
            4. Consider intermittent production cycling to clear accumulated liquids.

            ### Medium-term (30–90 days)
            5. Evaluate ESP installation for artificial lift optimisation.

            ## Risk Assessment

            - **Production risk:** HIGH — continued loading may cause well kill
            - **HSE risk:** MEDIUM — gas-lift pressure changes require safety review
            - **Escalation required:** Yes — changes to gas-lift rates require production
              engineer approval per MOC-GL-002.

            ## References

            - Turner, R.G., Hubbard, M.G., Dukler, A.E. (1969): "Analysis and Prediction
              of Minimum Flow Rate for the Continuous Removal of Liquids from Gas Wells"
            - Gullfaks Operations Manual, Section 7.4 (Gas Lift Operations)
        """),
    },
    {
        "filename": "WR-004_Oseberg_OS1H_Annual_2023.md",
        "title": "Annual Well Review – Oseberg OS-1H – 2023",
        "content": textwrap.dedent("""\
            # Annual Well Review – Oseberg OS-1H – 2023

            **Well ID:** OS-1H
            **Field:** Oseberg
            **Review Period:** January – December 2023

            ## Production Summary

            OS-1H produced 1.12 MMbbls in 2023, 3.2% above forecast.
            Average daily rate: 3,070 BOPD. Uptime: 97.4%.

            ## GOR Trend Analysis

            Gas-Oil Ratio increased from 520 to 680 scf/bbl over the year, indicating
            progressive reservoir pressure depletion approaching the bubble point.
            Current reservoir pressure: 3,240 psi. Bubble point: 3,100 psi.

            **Action:** If GOR exceeds 750 scf/bbl, initiate gas reinjection study.

            ## Water Cut Stability

            Water cut remained stable at 8.2% ± 1.1% throughout 2023.
            No evidence of channelling or coning detected.

            ## Intervention History

            | Date       | Intervention Type         | Result              |
            |------------|--------------------------|---------------------|
            | Mar 12     | Coiled tubing cleanout    | +180 BOPD uplift    |
            | Aug 5      | Choke replacement         | Resolved erosion    |
            | Nov 20     | PLT survey                | Confirmed single zone|

            ## 2024 Outlook

            Forecast production: 2.95 MMbbls (target: 3.10 MMbbls).
            Risk factor: reservoir pressure approaching bubble point in Q3 2024.
        """),
    },
    {
        "filename": "WR-005_Draugen_D3H_SeparatorUpset_2024.md",
        "title": "Production Loss Investigation – Draugen D-3H – Separator Upset",
        "content": textwrap.dedent("""\
            # Production Loss Investigation – Draugen D-3H – Separator Upset

            **Well ID:** D-3H
            **Field:** Draugen
            **Date:** February 8, 2024

            ## Incident Description

            At 03:45 on February 8, well D-3H registered a 47% oil rate decline
            (from 1,980 to 1,050 BOPD) accompanied by a 220% GOR spike
            (from 590 to 1,890 scf/bbl). Wellhead pressure and BHP remained stable,
            ruling out downhole causes.

            ## Investigation Findings

            Surface facility inspection revealed:
            - Train A separator level controller (LIC-101) failed in the open position
            - Liquid carryover to the gas line caused slug flow to flare KO drum
            - High gas presence in oil outlet degraded metering accuracy

            The apparent GOR spike was partially a metering artefact due to gas
            phase in the liquid metering train. True well GOR was unaffected.

            ## Root Cause

            Separator level controller LIC-101 experienced a solenoid valve failure
            causing uncontrolled liquid level drop. Oil carryover through gas outlet.

            ## Corrective Actions

            1. Replaced LIC-101 solenoid valve (completed 09:30 Feb 8).
            2. Recalibrated oil metering train after stabilisation.
            3. Production recovered to 1,940 BOPD by 14:00 Feb 8.

            ## HSE Notes

            Liquid hydrocarbon carryover to flare system — flare DMPV-201 operated
            within design parameters. No safety device actuations. Reported to
            Norwegian Petroleum Safety Authority per NOG-070 requirements.

            ## Deferred Production

            Estimated 930 BOPD × 10.25 hours = 399 bbls ≈ $31,920 at $80/bbl.
        """),
    },
    {
        "filename": "WR-006_Volve_F5_GOR_Investigation.md",
        "title": "GOR Increase Investigation – Volve 15/9-F-5",
        "content": textwrap.dedent("""\
            # GOR Increase Investigation – Volve 15/9-F-5

            **Date:** January 2024

            ## Observation

            GOR increased from 620 to 1,340 scf/bbl over 30 days.
            Oil rate relatively stable (2,100 → 1,920 BOPD).

            ## Possible Causes

            1. **Gas coning**: Gas cap approaching perforations due to drawdown.
               Probability: HIGH. Consistent with reservoir simulation.

            2. **Tubing leak above gas-lift mandrel**: Would allow gas injection
               to appear in produced stream. Probability: MEDIUM.

            3. **Changes in separator operating pressure**: Lower separator
               pressure increases GOR measurement. Pressure log reviewed —
               UNCHANGED. Probability: LOW.

            ## Recommended Actions

            1. Reduce drawdown: choke from 56 to 40 (64ths).
            2. Perform spinner survey to check for tubing integrity.
            3. Update gas-coning model with current GOR data.

            ## Risk Level: MEDIUM
        """),
    },
    {
        "filename": "WR-007_Draugen_D4AH_ChokeFailed.md",
        "title": "Choke Valve Failure Report – Draugen D-4AH",
        "content": textwrap.dedent("""\
            # Choke Valve Failure Report – Draugen D-4AH

            **Date:** March 22, 2024

            ## Failure Description

            Production choke on D-4AH failed in partially closed position (equivalent
            to 20/64" opening). Well producing at 640 BOPD vs. expected 2,200 BOPD.
            Wellhead pressure increased from 320 to 890 psi.

            ## Diagnostic Steps

            1. Confirmed choke actuator position feedback mismatch.
            2. Ruled out wellbore obstruction (stable BHP at expected level).
            3. Identified hydraulic actuator seal failure from maintenance history.

            ## Action Taken

            Emergency choke replacement executed within 6 hours using pre-positioned
            spare. Production restored to 2,150 BOPD after flushing.

            ## Parts Required for Prevention

            - Hydraulic choke actuator seal kit (P/N: DRG-CH-4420)
            - Minimum 2 spares on-platform per MOC-022 requirements

            ## Total Deferred Production: ~9,600 bbls
        """),
    },
    {
        "filename": "WR-008_Gullfaks_GFB1H_PressureDecline.md",
        "title": "BHP Decline Analysis – Gullfaks GF-B-1H",
        "content": textwrap.dedent("""\
            # BHP Decline Analysis – Gullfaks GF-B-1H

            **Date:** Q1 2024

            ## Observation

            Bottomhole pressure has declined 340 psi over 90 days (from 3,820 to 3,480 psi),
            approximately 2.3× the material balance decline prediction.

            ## Analysis

            Pressure transient analysis indicates a possible communication path with a
            depleted fault block to the northwest. Static reservoir pressure in that block
            was measured at 2,900 psi in 2022 — significantly below GF-B-1H's current pressure.

            ## Implications

            1. Accelerated depletion — reduce production rate to preserve reservoir energy.
            2. Unexpected connectivity may require updating the full-field model.

            ## Recommended Actions

            1. Shut-in for 48-hour pressure build-up test.
            2. Re-run material balance with updated connectivity model.
            3. Consider water injection conversion for pressure support.

            ## Risk Level: HIGH — requires production engineer sign-off for rate change.
        """),
    },
    {
        "filename": "WR-009_MultiWell_ProductionOptimisation_2024.md",
        "title": "Multi-Well Production Optimisation Study – Draugen Field – 2024",
        "content": textwrap.dedent("""\
            # Multi-Well Production Optimisation Study – Draugen Field – 2024

            ## Objective

            Maximise total field oil production while respecting facility and pipeline
            constraints (max 28,000 BOPD total liquid handling capacity, max gas
            export 120 MMscf/day).

            ## Current Production (March 2024)

            | Well   | Oil (BOPD) | Water (BWPD) | Gas (MMscf/day) | Choke |
            |--------|-----------|--------------|-----------------|-------|
            | D-1H   | 2,450     | 340          | 1.55            | 48/64 |
            | D-2H   | 1,870     | 920          | 1.21            | 40/64 |
            | D-3H   | 1,980     | 1,240        | 1.17            | 44/64 |
            | D-4AH  | 2,200     | 460          | 1.43            | 48/64 |
            | D-5H   | 1,650     | 2,100        | 1.07            | 36/64 |
            | Total  | 10,150    | 5,060        | 6.43            |       |

            ## Optimisation Recommendations

            1. D-5H: Reduce choke to 28/64" — high water cut (56%) limits value.
               Deferred oil minimal; significant liquid handling capacity freed.
            2. D-1H: Increase choke to 56/64" — low water cut, high productivity.
               Expected uplift: +180 BOPD.
            3. D-4AH: Hold at current rate — recently replaced choke, allow stabilisation.

            ## Expected Net Impact: +120 BOPD with 8% reduction in water handling load.
        """),
    },
    {
        "filename": "WR-010_Volve_F11H_InjectorPerformance.md",
        "title": "Injector Performance Review – Volve 15/9-F-11H",
        "content": textwrap.dedent("""\
            # Injector Performance Review – Volve 15/9-F-11H

            ## Summary

            Water injector 15/9-F-11H shows increased injectivity index (from 8.2 to
            11.4 BWPD/psi) over the past 6 months, indicating possible fracture stimulation
            or formation breakdown near wellbore.

            ## Concern

            Higher-than-expected injectivity correlates with earlier-than-predicted water
            breakthrough in offsetting producers (particularly 15/9-F-4, see WR-002).
            Fractured flow paths provide preferential water movement.

            ## Recommended Action

            1. Reduce injection rate by 15% to restore injectivity to design window.
            2. Perform injection profile survey (spinner + temperature) to identify
               thief zones.
            3. Consider conformance gel treatment in high-permeability streaks.

            ## Risk Level: MEDIUM
        """),
    },
    {
        "filename": "WR-011_Draugen_D2H_ESPFailure.md",
        "title": "ESP Failure Investigation – Draugen D-2H",
        "content": textwrap.dedent("""\
            # ESP Failure Investigation – Draugen D-2H

            **Date:** January 29, 2024

            ## Event

            D-2H ESP tripped at 14:23 due to motor overtemperature (motor temperature
            reached 185°C vs. 165°C operating limit). Well shut-in automatically.

            ## Root Cause

            1. Scale deposition on pump stages reduced hydraulic efficiency,
               increasing motor load by ~22%.
            2. Downhole cooling flow restricted by partial formation damage.

            ## Actions Taken

            1. Immediate: Well shut-in, ESP powered down.
            2. Short-term: Scale inhibitor squeeze treatment (48 hours).
            3. Medium-term: Pump replacement with enhanced-stage design (60-day lead time).

            ## Deferred Production

            Estimated 1,870 BOPD × 18 days = 33,660 bbls ≈ $2.7M revenue impact.

            ## Prevention

            Increase scale inhibitor injection rate from 15 to 25 ppm.
            Quarterly downhole temperature monitoring via DTS.
        """),
    },
    {
        "filename": "WR-012_Oseberg_OS3H_IntermittentProduction.md",
        "title": "Intermittent Production Analysis – Oseberg OS-3H",
        "content": textwrap.dedent("""\
            # Intermittent Production Analysis – Oseberg OS-3H

            ## Issue

            OS-3H exhibits oscillating production every 4–6 hours between 0 and 2,800 BOPD
            due to slugging in the 12 km subsea flowline.

            ## Cause

            Low flow velocity in 8" flowline at current production rates allows liquid
            accumulation in a 340 m inclined section (12° inclination).

            ## Solutions Evaluated

            1. **Riser-base gas injection** (recommended): Inject 0.4 MMscf/day lift gas
               at flowline riser base. Capital: $2.1M. Eliminates slugging.
            2. **Production rate increase**: Requires choking D-3H down — not feasible
               given field constraints.
            3. **Slug catcher upgrade**: Existing KO drum has sufficient capacity if
               maximum slug volume < 150 bbls (current max: 210 bbls). Not viable.

            ## Recommendation: Proceed with riser-base gas injection — ROI < 4 months.
        """),
    },
    {
        "filename": "WR-013_Gullfaks_GFA1H_PeriodicReview.md",
        "title": "Routine Well Review – Gullfaks GF-A-1H – February 2024",
        "content": textwrap.dedent("""\
            # Routine Well Review – Gullfaks GF-A-1H – February 2024

            ## Status: STABLE

            - Oil rate: 2,820 BOPD (within ±5% of plan)
            - Water cut: 22.4% (expected: 20–25%)
            - GOR: 580 scf/bbl (normal range: 550–650)
            - BHP: 3,680 psi (above bubble point: 3,100 psi)

            ## Action Items

            None. Continue routine monitoring.

            ## Next Review: March 2024
        """),
    },
    {
        "filename": "WR-014_Volve_F12H_WellIntegrity.md",
        "title": "Well Integrity Assessment – Volve 15/9-F-12H",
        "content": textwrap.dedent("""\
            # Well Integrity Assessment – Volve 15/9-F-12H

            **Assessment Date:** February 2024

            ## Integrity Status: ACCEPTABLE (Category 2)

            ## Findings

            1. Sustained casing pressure (SCP) detected on B-annulus: 45 psi.
               Below alert threshold (100 psi). Cause: micro-annular flow from
               P&A cement job (2011). Monitored quarterly.

            2. Christmas tree SCSSV tested February 12 — passed with 8,500 psi
               function test pressure. No leaks detected.

            3. Wellhead seal ring replacement scheduled for Q2 maintenance window.

            ## Risk Assessment

            - **HSE risk:** LOW — SCP well below kick threshold.
            - **Remediation required:** NO — monitoring programme sufficient.
            - **Next full assessment:** August 2024.

            ## Regulatory Reference

            Norwegian Petroleum Safety Authority Regulation: Facilities Regulations
            §48 Well Barriers.
        """),
    },
    {
        "filename": "WR-015_NorthSea_WaterCut_BestPractices.md",
        "title": "Technical Guidance – Water Cut Management in Mature Fields",
        "content": textwrap.dedent("""\
            # Technical Guidance – Water Cut Management in Mature Fields

            ## Overview

            This document provides guidance for managing high water-cut wells in
            mature North Sea assets. As fields age, water cut typically increases
            from <20% to >80%, fundamentally changing operational economics.

            ## Economic Limit

            Operating cost limit for North Sea offshore wells is typically
            $25–35/bbl (OPEX + tariff). At 80% water cut and 5,000 BLPD:
            - Oil produced: 1,000 BOPD
            - Break-even oil price: ~$60/bbl (at $35/bbl OPEX)
            - Below break-even → candidate for workover or abandonment.

            ## Intervention Options by Water Cut Range

            ### 20–40% Water Cut
            - Profile conformance squeeze
            - Mechanical zone isolation (packers)
            - Polymer flooding

            ### 40–70% Water Cut
            - In-depth water diversion
            - Infill drilling to access bypassed oil
            - Rate reduction to minimise coning

            ### >70% Water Cut
            - Economic limit analysis required
            - Consider water disposal optimisation
            - P&A candidate if production uneconomic

            ## Early Warning Indicators

            1. Water cut increase >5%/week
            2. Brine salinity change (indicating change in source formation)
            3. Tracer arrival from injector
            4. Pressure interference test anomaly

            ## HSE Considerations

            High water production increases NORM (Naturally Occurring Radioactive
            Material) risk. Assess scale deposits in production tubing per
            NORSOK S-003.
        """),
    },
]

MAINTENANCE_LOGS: list[dict[str, str]] = [
    {
        "filename": "ML-001_Draugen_ChristmasTree_Inspection.md",
        "title": "Maintenance Log – Christmas Tree Inspection – Draugen Platform – Feb 2024",
        "content": textwrap.dedent("""\
            # Maintenance Log – Christmas Tree Inspection – Draugen Platform

            **Date:** February 14–16, 2024
            **Work Order:** WO-2024-0142
            **Technician:** R. Hansen, A. Berg
            **Asset:** Production Christmas Trees, Wells D-1H through D-5H

            ## Work Performed

            ### D-1H Christmas Tree
            - Wing valve (WV) function test: PASS (3,200 psi test pressure)
            - Master valve (PMV, AMV) function test: PASS
            - SCSSV function test: PASS
            - Seal inspection: No visible leaks. One packing gland shows minor
              weepage — within acceptable limits. Flagged for next PM cycle.
            - Replaced: Hydraulic control hose bundle connector (preventive)

            ### D-2H Christmas Tree
            - Wing valve function test: PASS
            - SCSSV function test: PASS (8,500 psi test)
            - Choke actuator torque test: FAIL — actuator stiff at low temperatures.
              Root cause: hydraulic fluid degradation. Fluid replaced with cold-rated
              Mobil DTE 26 Ultra grade. Retest: PASS.
            - Replaced: Hydraulic actuator fluid charge

            ### D-3H Christmas Tree
            - All function tests: PASS
            - Corrosion inspection: Mild surface corrosion on topside valve body.
              Applied zinc-rich coating. Scheduled epoxy coating for next shutdown.

            ### D-4AH Christmas Tree
            - Wing valve function test: PASS
            - Noted: Actuator feedback position sensor reading ±3% from reference.
              Recalibrated. Position error was the precursor to the choke failure on
              March 22, 2024 (see WR-007).
            - Recommendation: Replace sensor before next production cycle.

            ### D-5H Christmas Tree
            - All tests PASS. No issues noted.

            ## Parts Consumed

            | Part                          | Quantity | Part Number  |
            |-------------------------------|----------|--------------|
            | Hydraulic control hose conn.  | 1        | DRG-HC-1102  |
            | Hydraulic fluid (DTE 26 Ultra)| 20L      | DRG-FL-0055  |
            | Zinc-rich coating spray       | 2 cans   | DRG-CT-0012  |

            ## Outstanding Items

            1. D-1H packing gland weepage — next PM cycle (Q2 2024)
            2. D-4AH actuator position sensor replacement — PRIORITY by April 1, 2024
            3. D-3H full epoxy coating — next planned shutdown
        """),
    },
    {
        "filename": "ML-002_Volve_Separator_Overhaul.md",
        "title": "Maintenance Log – Train A Separator Overhaul – Volve – January 2024",
        "content": textwrap.dedent("""\
            # Maintenance Log – Train A Separator Overhaul – Volve

            **Date:** January 8–12, 2024
            **Work Order:** WO-2024-0018
            **Duration:** 5 days planned shutdown

            ## Scope of Work

            Full overhaul of Train A three-phase separator (SEP-101):
            - Internal inspection and cleaning
            - Internals replacement (mist eliminator, weir plates, vortex breaker)
            - Pressure Safety Valve (PSV-101) replacement
            - Level controller (LIC-101) full overhaul

            ## Level Controller (LIC-101) Findings

            The separator level controller solenoid valve showed evidence of
            sulphide stress cracking in the valve body — likely caused by H₂S in
            produced water (measured 45 ppm H₂S at this separator inlet).

            **Action:** Replaced solenoid valve with H₂S-resistant grade
            (Inconel 625 valve body, NACE MR0175 compliant).

            **Note:** Despite this overhaul, LIC-101 failed again on February 8, 2024
            (see WR-005). Root cause was a different failure mode — hydraulic
            actuator seal, not the solenoid. The January overhaul did not address
            the actuator seal condition. Added actuator seal to next PM checklist.

            ## Internal Condition

            - Mist eliminator: Heavy wax deposits on upstream face. Recommend
              wax inhibitor injection increase at wellheads.
            - Weir plates: Moderate scale (calcium carbonate). Descaled with
              5% HCl solution.
            - Vortex breaker: Good condition. Retained.

            ## PSV Testing Results

            PSV-101: Set pressure 120 psi. Tested at 132 psi (10% above set).
            Opened at 118 psi (within -2% tolerance). PASS.

            ## Lessons Learned

            - H₂S corrosion requires material upgrade programme for all production
              valves handling sour fluids (>10 ppm H₂S).
            - Level controller overhaul should include actuator seal as standard.
        """),
    },
    {
        "filename": "ML-003_Gullfaks_ESP_Replacement.md",
        "title": "Maintenance Log – ESP Replacement – Gullfaks GF-A-2H",
        "content": textwrap.dedent("""\
            # Maintenance Log – ESP Replacement – Gullfaks GF-A-2H

            **Date:** March 10–18, 2024
            **Work Order:** WO-2024-0287
            **Contractor:** Weatherford Well Services

            ## Background

            Original ESP (Centrilift GN2000, installed 2021) failed due to scale
            deposition on pump stages. See WR-011 for root cause analysis.

            ## New ESP Specifications

            - Model: Baker Hughes CENesis HPS (High Performance Stage)
            - Motor: 150 HP, 540V, 3-phase induction
            - Pump stages: 72 stages (vs. 62 original) for higher head
            - Motor protector: Tandem protector for enhanced shaft seal protection
            - Cable: REDA HOTLINE® premium cable with EPDM jacket

            ## Installation Notes

            1. Scale inhibitor squeeze completed prior to ESP installation.
               300 gallons INHIBITOR-X at 0.5 bbl/min. 48-hour soaking time.

            2. ESP run to 2,240 m MD (10 m above perforations).

            3. Motor megger test: 800 MΩ (excellent — >100 MΩ is acceptable).

            4. Startup sequence:
               - 06:00: ESP energised at 40 Hz
               - 06:15: Production at surface confirmed
               - 08:00: Frequency raised to 50 Hz (design point)
               - 12:00: Stable production at 1,890 BOPD (103% of target)

            ## Post-Installation Performance

            | Day | Oil (BOPD) | WC (%) | Motor Temp (°C) |
            |-----|-----------|--------|-----------------|
            | 1   | 1,890     | 15.2   | 142             |
            | 3   | 1,920     | 15.8   | 144             |
            | 7   | 1,940     | 16.1   | 143             |

            Motor temperature well within limits (165°C max). Scale inhibitor
            injection maintained at 25 ppm per WR-011 recommendation.
        """),
    },
    {
        "filename": "ML-004_Draugen_InstrumentCalibration.md",
        "title": "Instrument Calibration Record – Draugen Production Meters – Q1 2024",
        "content": textwrap.dedent("""\
            # Instrument Calibration Record – Draugen Production Meters – Q1 2024

            **Date:** February 5, 2024
            **Standard:** NORSOK I-104 / ISO 17089

            ## Calibration Results

            | Instrument       | Tag      | Measured | Reference | Error  | Status |
            |-----------------|----------|----------|-----------|--------|--------|
            | Oil flow meter   | FIT-D1-01| 2,452    | 2,450     | +0.08% | PASS   |
            | Oil flow meter   | FIT-D2-01| 1,865    | 1,870     | -0.27% | PASS   |
            | Oil flow meter   | FIT-D3-01| 1,985    | 1,980     | +0.25% | PASS   |
            | Water flow meter | FIT-D5-WC| 2,094    | 2,100     | -0.29% | PASS   |
            | Pressure xmtr   | PIT-D1-BH| 3,208    | 3,210     | -0.06% | PASS   |
            | Temp transmitter | TIT-D3-WH| 143.2    | 145       | -1.24% | PASS   |

            All instruments within ±0.5% accuracy requirement (NORSOK I-104 Class B).

            ## Next Calibration Due: May 2024
        """),
    },
    {
        "filename": "ML-005_Oseberg_FlowlineInspection.md",
        "title": "Subsea Flowline Inspection – Oseberg OS-3H – February 2024",
        "content": textwrap.dedent("""\
            # Subsea Flowline Inspection – Oseberg OS-3H

            **Date:** February 20, 2024
            **Method:** ROV-based external inspection + intelligent pigging (MFL)

            ## Key Findings

            ### External Inspection
            - No evidence of free-span greater than 4 m (DNV-ST-F101 limit: 8 m)
            - Anode depletion: Section 4 (km 8.2–9.1): 68% remaining (acceptable,
              design life 25 years, installed 2010)
            - No coating disbondment detected

            ### MFL (Magnetic Flux Leakage) Pigging Results
            - 3 metal loss indications detected:
              - km 4.2: 18% wall loss (threshold: 30%) — MONITOR
              - km 7.8: 22% wall loss — MONITOR, re-inspect in 12 months
              - km 11.1: 8% wall loss — ACCEPTABLE
            - No through-wall defects detected

            ## Recommendations

            1. Increase cathodic protection survey frequency to annual for Section 4.
            2. Re-pig in 12 months with UT tool at km 7.8 indication.
            3. No immediate integrity concern identified.
        """),
    },
    {
        "filename": "ML-006_AllFields_ChemicalInjection_Review.md",
        "title": "Chemical Injection Programme Review – North Sea Assets – 2024",
        "content": textwrap.dedent("""\
            # Chemical Injection Programme Review – North Sea Assets – 2024

            ## Scale Inhibitor

            Current dose rates:
            - Draugen (D-1H to D-5H): 15–25 ppm (increased from 15 ppm per ML-003)
            - Volve (all wells): 20 ppm
            - Oseberg: 18 ppm
            - Gullfaks: 22 ppm

            Performance: Scale buildup in ESP stages reported at Draugen (see WR-011)
            suggests scale inhibitor below minimum effective concentration in D-2H.
            Increase to 30 ppm recommended.

            ## Corrosion Inhibitor

            H₂S scavenger rate: 50 ppm (all fields).
            General corrosion inhibitor: Continuous injection at 15 ppm.

            Inspection data (ML-002) shows ongoing H₂S attack on sour-service
            components. Current scavenger programme is sufficient for bulk
            produced fluid but cannot protect metal components from localised
            H₂S exposure in dead-leg sections.

            Recommendation: Identify and eliminate all dead-legs >30 cm in
            produced water handling systems.

            ## Wax Inhibitor

            Volve separator inspection (ML-002) identified heavy wax deposits.
            Increase wax inhibitor injection at Volve wellheads from 50 to 80 ppm.
            Consider pour point depressant additive for winter operations.
        """),
    },
    {
        "filename": "ML-007_Draugen_AnnualShutdown_2024.md",
        "title": "Planned Shutdown Report – Draugen Platform – Annual Maintenance 2024",
        "content": textwrap.dedent("""\
            # Planned Shutdown Report – Draugen Platform – Annual Maintenance 2024

            **Shutdown Period:** April 15–28, 2024 (14 days)
            **Shutdown Manager:** K. Johnsen

            ## Scope Summary

            Critical path activities:
            1. High-pressure test of export riser (day 1–2)
            2. Turbine overhaul – GT-01 (days 3–7)
            3. Safety valve function testing (days 1–14, parallel)
            4. Pigging of oil export line (day 8)
            5. Inspection of gas compression train (days 9–12)

            ## Production Impact

            Total deferred production: ~142,000 bbls (10,150 BOPD × 14 days).
            Revenue impact at $80/bbl: ~$11.4M.
            Shutdown cost: $8.2M.

            Net result: Annual maintenance window is justified by safety and
            integrity compliance requirements.

            ## HSE Performance

            Zero LTIs. One first-aid case (minor laceration, day 4).
            Permit-to-work system performance: 98.2% compliance.
        """),
    },
    {
        "filename": "ML-008_Volve_SubseaControl_Maintenance.md",
        "title": "Subsea Control System Maintenance – Volve – Q1 2024",
        "content": textwrap.dedent("""\
            # Subsea Control System Maintenance – Volve – Q1 2024

            **System:** Subsea Electronic Module (SEM) replacement on 15/9-F manifold

            ## Work Performed

            SEM-B on the Volve subsea manifold showed communication dropouts
            (3 dropouts >30 minutes in January 2024). Root cause: firmware version
            incompatibility after MCS upgrade in December 2023.

            Replacement SEM-B installed via ROV on February 3, 2024.
            Firmware updated to version 4.2.1. Zero dropouts since installation.

            ## Operational Impact

            During SEM-B dropouts, wells 15/9-F-4, F-5, F-12H operated in
            degraded mode (local fail-safe position). Production reduced by ~15%.

            ## Preventive Actions

            - Establish MCS/SEM firmware compatibility matrix before upgrades.
            - Add automated SEM health monitoring to control room dashboard.
        """),
    },
    {
        "filename": "ML-009_Draugen_CorrosionMonitoring.md",
        "title": "Corrosion Monitoring Report – Draugen – Q1 2024",
        "content": textwrap.dedent("""\
            # Corrosion Monitoring Report – Draugen – Q1 2024

            ## Corrosion Coupon Results

            | Location              | Coupon ID | Exposure (days) | Rate (mpy) | Threshold |
            |-----------------------|-----------|----------------|------------|-----------|
            | Wellhead D-1H         | C-D1-01   | 90             | 0.8        | <3.0 PASS |
            | Separator inlet       | C-SEP-01  | 90             | 2.1        | <3.0 PASS |
            | Produced water header | C-PW-01   | 90             | 4.8        | <3.0 FAIL |
            | Export pipeline       | C-EXP-01  | 90             | 1.2        | <3.0 PASS |

            ## Action Required

            Produced water header corrosion rate 4.8 mpy exceeds 3.0 mpy threshold.
            Actions:
            1. Increase corrosion inhibitor to 25 ppm in produced water stream.
            2. UT thickness survey of produced water header — schedule within 30 days.
            3. Evaluate piping replacement in next shutdown if UT shows >15% wall loss.
        """),
    },
    {
        "filename": "ML-010_Gullfaks_GasliftValve_Replacement.md",
        "title": "Gas Lift Valve Replacement – Gullfaks GF-A-2H – April 2024",
        "content": textwrap.dedent("""\
            # Gas Lift Valve Replacement – Gullfaks GF-A-2H – April 2024

            ## Background

            Following the liquid loading diagnosis (WR-003), a gas-lift optimisation
            programme was implemented. During coiled tubing intervention, two of the
            three gas-lift mandrels showed worn valve seats.

            ## Valves Replaced

            - Mandrel 1 (1,450 m MD): Replaced with Camco BK-series (3/4" port)
            - Mandrel 2 (1,820 m MD): Replaced with Camco BK-series (1/2" port)
            - Mandrel 3 (2,100 m MD): Retained (valve seat in good condition)

            ## Post-Replacement Performance

            Gas lift injection rate: 1.4 MMscf/day (as per WR-003 recommendation).
            Oil production recovered to 1,340 BOPD (from 890 BOPD pre-intervention).
            Wellhead pressure reduced from 310 to 195 psi — liquid loading resolved.

            ## Lift Efficiency

            Post-optimisation lift efficiency: 0.89 (excellent, >0.80 is target).
        """),
    },
]

HSE_PROCEDURES: list[dict[str, str]] = [
    {
        "filename": "HSE-001_Production_Anomaly_Response_Procedure.md",
        "title": "HSE Procedure – Production Anomaly Response – NorthSea Assets",
        "content": textwrap.dedent("""\
            # HSE Procedure – Production Anomaly Response

            **Document Number:** HSE-OPS-001
            **Revision:** 4
            **Effective Date:** January 1, 2024

            ## Purpose

            This procedure defines the response steps when a production anomaly is
            detected on any offshore asset. It ensures safe, timely, and documented
            investigation of anomalies that may indicate equipment failure, wellbore
            integrity issues, or process upsets.

            ## Scope

            Applies to all production wells, process facilities, and subsea equipment
            on Draugen, Volve, Gullfaks, and Oseberg platforms.

            ## Risk Classification

            | Class | Criteria                                        | Response Time |
            |-------|-------------------------------------------------|---------------|
            | RED   | HSE risk / potential well control event          | IMMEDIATE     |
            | AMBER | Significant production loss (>20%) or equipment  | < 2 hours     |
            | GREEN | Minor anomaly (<10% production variance)         | < 24 hours    |

            ## Response Steps

            ### RED Class (Immediate Response)

            1. **Alert the Control Room Operator immediately.**
            2. Initiate Emergency Response Plan if well control event suspected.
            3. Shut in affected well via ESD if required.
            4. Notify Offshore Installation Manager (OIM) within 15 minutes.
            5. Contact Petroleum Safety Authority if well barrier failure suspected.

            ### AMBER Class

            1. Production engineer to be notified within 30 minutes.
            2. Conduct initial data review: review last 72h telemetry trends.
            3. Check equipment alarms and maintenance history.
            4. Escalate to senior reservoir engineer if cause not identified in 2 hours.
            5. Document findings in the Production Event Log.

            ### GREEN Class

            1. Log anomaly in Production Event Log.
            2. Monitor for recurrence over next 24 hours.
            3. Investigate root cause within 48 hours.

            ## Human Escalation Triggers

            The following conditions REQUIRE mandatory escalation to a senior engineer
            regardless of AI-assisted analysis output:

            - Any anomaly classified RED
            - BHP decline >150 psi/week
            - Water cut increase >15% in 24 hours
            - Sustained casing pressure (B-annulus) increase >20 psi/hour
            - GOR increase >300 scf/bbl in 48 hours
            - Any anomaly where root cause cannot be confirmed

            ## Automated Monitoring Thresholds

            | Parameter     | Alert Threshold    | Shutdown Threshold |
            |--------------|-------------------|-------------------|
            | Water cut    | Δ+10%/24h          | Δ+30%/6h          |
            | Oil rate drop| <70% of 7-day avg  | <40% of 7-day avg |
            | BHP           | <2,500 psi OR -100/week | <2,000 psi   |
            | GOR           | >900 scf/bbl       | >1,500 scf/bbl    |
            | WHP           | >800 psi           | >1,100 psi        |

            ## Documentation Requirements

            All anomaly investigations must produce a written Production Event Report
            within 5 business days, including: timeline, root cause, corrective actions,
            HSE impact assessment, and lessons learned.

            ## References

            - Norwegian PSA: Facilities Regulations §§ 20, 47, 48
            - NORSOK D-010: Well Integrity in Drilling and Well Operations
            - Company ERP: Emergency Response Plan Document ERP-001
        """),
    },
    {
        "filename": "HSE-002_H2S_Management.md",
        "title": "HSE Procedure – H₂S Management on Production Facilities",
        "content": textwrap.dedent("""\
            # HSE Procedure – H₂S Management on Production Facilities

            **Document Number:** HSE-OPS-002
            **Revision:** 3

            ## Introduction

            Hydrogen sulphide (H₂S) is present in produced fluids at all North Sea
            assets (typical range: 5–200 ppm). H₂S is highly toxic (IDLH: 50 ppm)
            and causes sulphide stress cracking (SSC) in susceptible materials.

            ## Alarm Levels

            | Level | Concentration | Action Required              |
            |-------|--------------|------------------------------|
            | A1    | 5 ppm        | Warning — check equipment    |
            | A2    | 10 ppm       | Don SCBA — investigate source|
            | A3    | 20 ppm       | Muster — non-essential leave |
            | A4    | 50 ppm       | Full evacuation              |

            ## Material Requirements

            All equipment exposed to produced fluids with >10 ppm H₂S shall meet
            NACE MR0175 / ISO 15156 requirements for sour service.

            Materials NOT acceptable in sour service:
            - High-strength steel >110 ksi yield (unless specifically qualified)
            - Martensitic stainless steels (without qualification testing)
            - Standard carbon steel with hardness >HRC 22

            ## Monitoring Requirements

            - Fixed H₂S detectors in all production modules, tested monthly.
            - Personal H₂S monitors worn by all personnel in production areas.
            - Annual calibration of all H₂S detection equipment.

            ## Production Anomaly Links

            H₂S concentration increases in produced fluids may indicate:
            1. Reservoir souring (bacterial sulphate reduction in injected water)
            2. Cross-flow between zones with different H₂S concentrations
            3. Thermal cracking of heavier hydrocarbons at high temperatures

            **If H₂S in produced water increases by >20 ppm within 30 days:**
            → Escalate to HSE Manager and Reservoir Engineer.
            → Sample water for sulphate-reducing bacteria (SRB).
            → Review biocide treatment programme.
        """),
    },
    {
        "filename": "HSE-003_Well_Control_Procedures.md",
        "title": "HSE Procedure – Well Control and Kick Response",
        "content": textwrap.dedent("""\
            # HSE Procedure – Well Control and Kick Response

            **Document Number:** HSE-OPS-003
            **Revision:** 6

            ## Purpose

            Prevent uncontrolled flow of formation fluids to surface (blowout) by
            providing clear procedures for detecting and responding to well kicks.

            ## Early Kick Indicators (Production Wells)

            - Flowing wellhead pressure increase with constant choke setting
            - Tubing pressure increase without corresponding surface flow rate change
            - Unexplained BHP decrease (reservoir influx)
            - Flow-check positive when expected to be static

            ## Response to Suspected Kick

            1. Notify OIM and well control team immediately.
            2. Do NOT shut in well without OIM authorisation (risk of pressure surge).
            3. Record wellhead and casing pressures every 5 minutes.
            4. Prepare choke and kill manifold for shut-in operations.
            5. Increase well monitoring frequency to continuous.

            ## Shut-In Procedure (Production)

            1. Close surface safety valve (SSV) — remote operated from control room.
            2. Close wellhead master valve (PMV) if SSV fails.
            3. Monitor shut-in tubing pressure (SITP) and casing pressure (SICP).
            4. Contact Well Control Team / Emergency Well Intervention (EWI) contractor.
            5. Do NOT re-open well without formal well control plan approval.

            ## Mandatory Escalation

            Well control events are ALWAYS Class RED under HSE-OPS-001. No AI system
            or automated tool is authorised to recommend well shut-in or re-opening.
            All well control decisions require human engineer authorisation.
        """),
    },
    {
        "filename": "HSE-004_NORM_Management.md",
        "title": "HSE Procedure – NORM (Naturally Occurring Radioactive Material) Management",
        "content": textwrap.dedent("""\
            # HSE Procedure – NORM Management

            **Document Number:** HSE-OPS-004

            ## Background

            North Sea production equipment exposed to formation water can accumulate
            radioactive scale (primarily Ra-226, Ra-228 from radium dissolved in
            produced brine). NORM represents a radiation exposure hazard during
            maintenance and a regulated waste disposal challenge.

            ## Monitoring

            - Annual radiation survey of all production equipment likely to accumulate scale.
            - Dose rate limit for unrestricted areas: 7.5 μSv/hour.
            - Areas with measured dose rate >7.5 μSv/hour require radiation work permit.

            ## High-Risk Equipment

            NORM accumulates preferentially in:
            - Production choke bodies (turbulent flow + scale deposition)
            - Wellhead outlets (water condensation zone)
            - Separator water outlets
            - ESP pump housings (scale + high water cut wells)

            ## Link to Production Anomalies

            High NORM readings may indicate:
            1. Increased water production (more Ra-laden brine flowing through equipment)
            2. Scale precipitation (conditions favouring BaSO₄/SrSO₄ co-precipitation with Ra)

            If water cut increases significantly (>10%/week), increase NORM monitoring
            frequency for affected wellhead and separator.

            ## Waste Disposal

            NORM waste classified as >4 Bq/g (radium) must be disposed of at licensed
            facility per NORSOK S-003 and Norwegian Radiation Protection Act §§ 29–31.
        """),
    },
    {
        "filename": "HSE-005_Emergency_Shutdown_Procedure.md",
        "title": "HSE Procedure – Emergency Shutdown (ESD) System Operations",
        "content": textwrap.dedent("""\
            # HSE Procedure – Emergency Shutdown (ESD) System Operations

            **Document Number:** HSE-OPS-005

            ## ESD Philosophy

            The Emergency Shutdown system provides automated and manual protection
            against hazardous events by isolating hydrocarbon inventories.

            ## ESD Levels

            | Level | Designation | Action                              |
            |-------|-------------|-------------------------------------|
            | ESD-0 | Platform ESD| Full platform shutdown + muster     |
            | ESD-1 | Process ESD | Isolate process, keep utilities     |
            | ESD-2 | Well ESD    | Close all wellhead valves           |
            | WHCP  | Wellhead ESD| Close individual wellhead valves    |

            ## Automatic ESD Triggers

            - Fire or gas detection (HC gas >20% LEL)
            - High-high pressure on process vessels
            - Well control event (kick indicator)
            - Riser emergency disconnection
            - Loss of power (fail-safe closure of all SDVs)

            ## Manual ESD Triggers

            OIM can initiate any ESD level via control room panel or field pushbuttons.
            Production engineers CAN manually actuate WHCP for individual wells during:
            - Emergency maintenance
            - Well integrity concerns
            - Unusual production behaviour requiring immediate investigation

            ## Post-ESD Actions

            Do NOT restart after ESD without:
            1. Root cause investigation complete
            2. OIM written authorisation
            3. All affected PSVs and SDVs confirmed reset and tested

            ## AI System Limitations

            Automated AI systems and digital assistants are NOT authorised to:
            - Recommend ESD activation
            - Recommend restarting after ESD
            - Override any safety instrumented function
            All such decisions require qualified human engineer authorisation.
        """),
    },
    {
        "filename": "HSE-006_Confined_Space_Entry.md",
        "title": "HSE Procedure – Confined Space Entry in Production Equipment",
        "content": textwrap.dedent("""\
            # HSE Procedure – Confined Space Entry in Production Equipment

            **Document Number:** HSE-OPS-006

            ## Confined Spaces on Production Platform

            Vessels requiring confined space permit:
            - Three-phase separators (SEP-101, SEP-102)
            - Surge drums and knock-out drums
            - Storage tanks (crude, methanol, diesel)
            - Suction vessels and compressor scrubbers

            ## Prerequisites for Entry

            1. Vessel fully isolated (all inlet/outlet valves closed and locked)
            2. Vessel depressurised and vented to safe atmospheric pressure
            3. Gas test: O₂ 19.5–23.5%, LEL <1%, H₂S <1 ppm, CO <25 ppm
            4. Standby person stationed outside vessel
            5. Rescue equipment in place (lifeline, SCBA)
            6. Continuous gas monitoring during entry

            ## Requirements for Maintenance After Production Anomaly

            If entering a vessel following a production anomaly (e.g., separator upset):
            - Assume H₂S present until proven otherwise.
            - Increase gas test frequency to every 15 minutes during entry.
            - Consider elevated NORM risk if recent high water production.
            - OIM must sign confined space entry permit.
        """),
    },
    {
        "filename": "HSE-007_MOC_Management_of_Change.md",
        "title": "HSE Procedure – Management of Change (MOC) for Production Operations",
        "content": textwrap.dedent("""\
            # HSE Procedure – Management of Change (MOC) for Production Operations

            **Document Number:** HSE-OPS-007

            ## Purpose

            Ensure that changes to production rates, chemical injection rates, well
            configurations, and process settings are evaluated for safety and integrity
            impacts before implementation.

            ## MOC Triggers

            Changes requiring MOC:
            - Choke setting change >20% in a single step
            - Gas-lift injection rate change >0.5 MMscf/day
            - Chemical injection rate change >50%
            - Any well intervention (perforating, stimulation, workover)
            - Process operating set-point changes beyond ±10%

            ## Approval Levels

            | Change Type                  | Approver                      |
            |------------------------------|-------------------------------|
            | Minor rate adjustment (<10%) | Operations Foreman            |
            | Moderate change (10–20%)     | Production Engineer           |
            | Major change (>20%)          | Senior Reservoir Engineer     |
            | Well intervention            | Drilling & Well Engineering   |
            | Emergency change             | OIM (within 24h, formal MOC)  |

            ## AI-Assisted Recommendations

            Any recommendation generated by an AI or automated system that triggers
            an MOC threshold MUST be reviewed and approved by the appropriate
            human authority before implementation. AI recommendations are advisory
            only and carry no authority to initiate physical changes.
        """),
    },
    {
        "filename": "HSE-008_Incident_Reporting.md",
        "title": "HSE Procedure – Production Incident Reporting Requirements",
        "content": textwrap.dedent("""\
            # HSE Procedure – Production Incident Reporting Requirements

            **Document Number:** HSE-OPS-008

            ## Reportable Events

            ### Internal Reporting (Production Event Log)
            - Any unplanned production loss >5%
            - Equipment failure causing >2 hour outage
            - Instrument malfunction affecting measurement accuracy
            - Near-miss events (actual or potential safety consequence)

            ### Regulatory Reporting (Norwegian PSA)
            - Uncontrolled hydrocarbon releases (any quantity)
            - Well control events
            - Personnel injuries (lost time or medical treatment)
            - Significant structural damage
            - Any event requiring ESD-0 or ESD-1 activation

            ## Reporting Timeline

            | Event Type              | Internal Report | Regulatory Report |
            |-------------------------|----------------|------------------|
            | Minor production loss   | 24 hours       | Not required     |
            | Equipment failure       | 4 hours        | Not required     |
            | HC release (unignited)  | 1 hour         | Same day         |
            | Injury (lost time)      | 1 hour         | Within 24 hours  |
            | Well control event      | Immediate      | Immediate        |

            ## Root Cause Investigation

            All events rated Class RED or AMBER (per HSE-OPS-001) require:
            - Preliminary report within 24 hours
            - Root cause investigation completed within 14 days
            - Lessons learned distributed within 30 days
        """),
    },
]

EQUIPMENT_MANUALS: list[dict[str, str]] = [
    {
        "filename": "EQ-001_Three_Phase_Separator_Operations.md",
        "title": "Equipment Manual – Three-Phase Separator Operation and Troubleshooting",
        "content": textwrap.dedent("""\
            # Equipment Manual – Three-Phase Separator Operation and Troubleshooting

            **Equipment:** Horizontal Three-Phase Separator (SEP-101, SEP-102)
            **Manufacturer:** National Tank Company
            **Design Pressure:** 150 psig
            **Operating Pressure:** 80–120 psig
            **Temperature Range:** 60–180°F

            ## Operating Principles

            The three-phase separator uses gravity separation to split the production
            stream into oil, water, and gas phases. Key internals:

            - **Inlet diverter:** Reduces turbulence at inlet, promotes initial separation
            - **Mist eliminator:** Removes liquid droplets from gas phase
            - **Weir plate:** Controls oil level and prevents water carryover to oil outlet
            - **Vortex breaker:** Prevents turbulence at liquid outlets

            ## Normal Operating Parameters

            | Parameter              | Target   | Alert Low | Alert High |
            |------------------------|---------|-----------|-----------|
            | Operating pressure (psi)| 100     | 80        | 120       |
            | Oil level (%)          | 50–60   | 35        | 75        |
            | Water level (%)        | 30–40   | 20        | 55        |
            | Inlet temperature (°F) | 120–150 | 90        | 175       |

            ## Troubleshooting Guide

            ### Symptom: Oil carryover to gas outlet

            **Cause:** Oil level too high, damaged mist eliminator, excessive inlet flow rate.

            **Actions:**
            1. Check and adjust oil level controller (LIC).
            2. Reduce production rate if inlet flow exceeds design capacity.
            3. Inspect mist eliminator pads (normally done in annual shutdown).
            4. If oil carry-over to flare continues > 30 minutes, reduce rate by 20%.

            ### Symptom: Water carryover to oil outlet (off-spec crude)

            **Cause:** Water level too high, damaged weir plate, emulsion formation.

            **Actions:**
            1. Check and adjust water level controller.
            2. Verify demulsifier injection rate (minimum 15 ppm).
            3. Increase demulsifier dose if emulsion layer >15 cm in sight glass.
            4. If BS&W >1% in export crude, notify oil export terminal and reduce rate.

            ### Symptom: Pressure spike or rapid pressure increase

            **Cause:** Downstream blockage (PSV failed closed), gas-blocked outlet, instrument error.

            **Actions:**
            1. Check PSV position and test — if failed shut, initiate ESD-1.
            2. Check gas export control valve — ensure not in manual-closed position.
            3. If pressure > 130 psi and rising, manually open bypass.
            4. If PSV lifts (pressure > 150 psi), initiate full ESD-1.

            ### Symptom: Apparent GOR spike from well (as in WR-005 / D-3H incident)

            **Cause:** Level controller failure causing gas carryover to oil metering train.

            **Actions:**
            1. Check level controller (LIC) valve position — verify fully functional.
            2. Review oil meter readings for consistency with well test data.
            3. Compare GOR with wellhead measured values (wireless transmitter data).
            4. If GOR spike is metering artefact: correct measurement, note in event log.
            5. If GOR spike is confirmed downhole: treat as real anomaly per HSE-OPS-001.

            ## Maintenance Schedule

            | Interval  | Activity                                          |
            |-----------|--------------------------------------------------|
            | Monthly   | Function test level controllers, check PSV seats |
            | Quarterly | Calibrate pressure transmitters                  |
            | Annually  | Internal inspection, mist eliminator replacement |
            | 5-yearly  | Pressure vessel ASME/PED re-certification        |
        """),
    },
    {
        "filename": "EQ-002_ESP_Operation_Manual.md",
        "title": "Equipment Manual – Electric Submersible Pump (ESP) Operation",
        "content": textwrap.dedent("""\
            # Equipment Manual – Electric Submersible Pump (ESP) Operation

            **Manufacturer:** Baker Hughes / SLB Reda
            **Application:** Artificial lift for wells below critical velocity

            ## System Components

            1. **Motor:** Three-phase induction motor (100–400 HP typical)
            2. **Motor protector:** Equalises pressure, prevents fluid ingress
            3. **Pump stages:** Multi-stage centrifugal pump (30–120 stages typical)
            4. **Gas separator (optional):** Removes free gas before pump
            5. **Surface variable speed drive (VSD):** Controls motor frequency/speed

            ## Performance Parameters

            | Parameter         | Normal Range      | Alert Threshold    |
            |-------------------|-------------------|--------------------|
            | Motor temperature | 120–155°C         | >165°C → ALARM     |
            | Motor current     | Within ±10% design| >115% design → TRIP|
            | Motor resistance  | >100 MΩ (megger)  | <5 MΩ → FAULT      |
            | Intake pressure   | >500 psi          | <300 psi → ALERT   |

            ## Fault Diagnosis

            ### Motor Overtemperature (>165°C)

            Primary causes (in order of likelihood):
            1. **Scale on pump stages** → reduced hydraulic efficiency, increased motor load.
               Diagnostic: Compare motor current vs. design curve — high current = scale.
            2. **Underloaded pump** (insufficient fluid above motor) → reduced cooling flow.
               Diagnostic: Low intake pressure (<400 psi) + high temperature.
            3. **Cooling restriction** → formation damage near wellbore.
               Diagnostic: Injectivity or PI test shows reduced index.
            4. **Motor winding fault** → partial short circuit.
               Diagnostic: Megger test shows low resistance on one phase.

            ### ESP Trip on Overtemperature

            1. Do NOT restart ESP without root cause investigation.
            2. Run downhole memory gauge to record BHP and temperature trends.
            3. If scale suspected: squeeze scale inhibitor before restart.
            4. If underload suspected: increase flowrate or reduce motor speed.
            5. Full restart procedure requires production engineer sign-off.

            ### Motor Low Insulation Resistance (<5 MΩ)

            Indicates cable or motor winding moisture ingress. Pull and replace
            motor — do not attempt restart.

            ## Startup Procedure

            1. Pre-start megger test: confirm >100 MΩ.
            2. Start at minimum frequency (35 Hz).
            3. Confirm flow to surface within 15 minutes.
            4. Ramp to operating frequency over 30-minute period.
            5. Monitor motor temperature during first 4 hours continuously.

            ## Scale Prevention

            Minimum scale inhibitor concentration: 20 ppm (increase to 30 ppm for
            high water cut wells). Quarterly inhibitor squeeze recommended for ESPs
            with WC >50%.
        """),
    },
    {
        "filename": "EQ-003_Christmas_Tree_Manual.md",
        "title": "Equipment Manual – Production Christmas Tree Operation",
        "content": textwrap.dedent("""\
            # Equipment Manual – Production Christmas Tree Operation

            ## Overview

            The Christmas tree (XT) is the surface assembly of valves and fittings
            above the wellhead that controls production flow and provides well barriers.

            ## Primary Components

            - **Master valve (PMV/AMV):** Primary well barrier (fail-safe closed)
            - **Wing valve (WV):** Isolation valve in production wing
            - **Choke valve:** Flow restriction controlling production rate
            - **SCSSV:** Surface-controlled subsurface safety valve (1st barrier)
            - **Actuators:** Hydraulic actuators operated from WHCP

            ## Choke Valve Operation

            Choke size is specified in 64ths of an inch.
            - Full open: 64/64" (rarely used)
            - Typical operating range: 24/64" – 56/64"
            - Minimum recommended: 16/64" (below this, erosion risk increases)

            **Erosion risk:** High pressure drop across choke + sand production can
            cause rapid choke body erosion. Maximum allowable pressure differential
            per NORSOK D-010: 1,500 psi per choke stage.

            ## Choke Failure Modes

            | Mode          | Symptom                          | Action                      |
            |---------------|----------------------------------|-----------------------------|
            | Stuck closed  | Low/zero production rate         | Manual override → WO        |
            | Stuck open    | Higher-than-expected rate, low WHP| Close PMV, replace choke   |
            | Erosion       | Increasing noise + vibration     | Reduce rate, plan replacement|
            | Actuator fault| Feedback mismatch (ML-001 D-4AH) | Recalibrate or replace      |

            ## Safety Valve (SCSSV) Testing

            Per Norwegian PSA Facilities Regulations §48, all SSCVs shall be tested
            at minimum every 6 months. Test procedure:
            1. Close SCSSV using WHCP surface control panel.
            2. Verify pressure build-up above SCSSV (confirms closure).
            3. Function test at 8,500 psi (or MAWP if lower).
            4. Open SCSSV and confirm production resumes.
            5. Record test result in maintenance log with OIM sign-off.
        """),
    },
    {
        "filename": "EQ-004_Gas_Lift_System_Manual.md",
        "title": "Equipment Manual – Gas Lift System Operation",
        "content": textwrap.dedent("""\
            # Equipment Manual – Gas Lift System Operation

            ## Principle

            Gas lift artificially reduces hydrostatic head in the tubing string by
            injecting high-pressure gas, allowing reservoir pressure to lift fluids
            to surface that would otherwise be trapped by backpressure.

            ## System Components

            - **Injection compressor:** Compresses gas to injection pressure (1,200–2,500 psi)
            - **Distribution manifold:** Distributes gas to individual well injection headers
            - **Downhole mandrels:** Pockets in the tubing string that hold gas-lift valves
            - **Gas-lift valves (GLVs):** Regulate gas entry into the tubing from annulus

            ## Critical Operating Parameters

            | Parameter              | Target         | Alert                         |
            |------------------------|---------------|-------------------------------|
            | Injection pressure (psi)| 1,800–2,200   | <1,500 = insufficient lift    |
            | Injection rate (MMscf/d)| 0.8–2.0/well  | Per well design basis         |
            | GLV differential       | 100–300 psi   | >400 psi = valve chattering   |
            | Lift efficiency        | >0.80         | <0.70 = intervention needed   |

            ## Liquid Loading Diagnosis

            Liquid loading occurs when gas injection rate is insufficient to lift
            liquids continuously. Early symptoms:
            1. Declining oil rate with increasing variance (slugging)
            2. Rising wellhead pressure with constant choke and separator pressure
            3. Temperature anomaly at liquid accumulation depth (DTS survey)
            4. Calculated velocity below Turner's critical velocity

            ## Response to Suspected Liquid Loading

            1. Increase gas injection rate to maximum design rate.
            2. If WHP continues to rise: consider intermittent lift (cycling).
            3. Perform DTS survey to confirm accumulation depth.
            4. Re-design GLV spacing if loading recurs at normal injection rates.
            5. For severe loading: coiled tubing cleanout + GLV replacement (ML-010).

            ## Gas Lift Optimisation

            Optimal injection rate is where incremental oil production per unit of
            injected gas (lift efficiency) is maximised. Above optimum, additional
            gas injection yields diminishing oil returns and wastes compression energy.
            Use nodal analysis model for each well's optimal injection rate.
        """),
    },
    {
        "filename": "EQ-005_SCADA_System_Manual.md",
        "title": "Equipment Manual – SCADA System and Production Monitoring",
        "content": textwrap.dedent("""\
            # Equipment Manual – SCADA System and Production Monitoring

            ## Overview

            The SCADA (Supervisory Control and Data Acquisition) system provides
            real-time monitoring and control of all production equipment. Data is
            transmitted from field instruments to the control room and onshore
            operations centre.

            ## Data Architecture

            - **Field instruments:** Pressure, temperature, flow transmitters (HART/FF)
            - **RTUs (Remote Terminal Units):** Collect and transmit field signals
            - **Control room HMI:** Real-time displays and operator controls
            - **Historian (PI Server):** Archives all tag data at 1-minute intervals

            ## Key Production Tags

            | Tag Type        | Examples              | Update Rate |
            |-----------------|-----------------------|-------------|
            | Oil flow rate   | FIT-D1-01, FIT-D2-01 | 1 minute    |
            | Water cut       | AT-D1-WC             | 5 minutes   |
            | Wellhead pressure| PIT-D1-WH            | 1 minute    |
            | BHP (downhole)  | PIT-D1-BH (via DHPG) | 15 minutes  |
            | Temperature     | TIT-D1-WH            | 1 minute    |

            ## Data Quality Issues

            Common data quality problems and their meaning:

            | Symptom                  | Likely Cause                        |
            |--------------------------|-------------------------------------|
            | Flatlined tag value      | Instrument failure or RTU freeze    |
            | Unrealistic spike        | Electromagnetic interference        |
            | Constant 0 or -9999      | Communication loss (bad quality flag)|
            | Gradual drift            | Calibration drift (see ML-004)      |
            | Oscillation              | Control loop instability or slugging |

            **Important:** Always check data quality flags before using SCADA data
            for decisions. Bad-quality data (flag ≠ 0) should not be used for
            anomaly detection without engineer review.

            ## Alarm Management

            Total alarm count should be managed per EEMUA-191 guidelines:
            - Acceptable average alarm rate: <1 alarm per 10 minutes
            - High priority alarms: <5% of total
            - Stale alarms (active >24h): Investigate for root cause

            ## Integration with Production Monitoring

            The PI historian integrates with production allocation systems, well
            test databases, and the production planning system. When telemetry
            data is used for automated analysis:
            1. Verify data is within quality window (last 15 minutes updated)
            2. Cross-check with adjacent well trends for consistency
            3. Flag data gaps >30 minutes in anomaly investigations
        """),
    },
    {
        "filename": "EQ-006_Wellhead_Pressure_Systems.md",
        "title": "Equipment Manual – Wellhead Pressure Safety Systems",
        "content": textwrap.dedent("""\
            # Equipment Manual – Wellhead Pressure Safety Systems

            ## Overview

            Wellhead pressure safety systems provide automatic protection against
            overpressure in the production tubing and surface equipment.

            ## Pressure Safety Valves (PSVs)

            PSVs are the last line of defense against vessel overpressure.
            They open automatically when system pressure exceeds set pressure.

            Set pressure should be 110% of normal operating pressure (ASME VIII).

            **After a PSV lifts:**
            - Do NOT reset without identifying root cause of pressure excursion.
            - Replace PSV seat if PSV has lifted (seats may not reseal reliably).
            - Notify OIM and document in event log.

            ## High-Pressure Alarm Interpretation

            | Situation                           | Likely Cause               |
            |-------------------------------------|---------------------------|
            | WHP high + oil rate normal          | Downstream restriction     |
            | WHP high + oil rate declining       | Choke restriction (WR-007) |
            | WHP high + BHP normal               | Flowline blockage          |
            | WHP high + BHP also high            | Reservoir pressure surge   |
            | All wells WHP high simultaneously   | Separator pressure high    |

            ## Low-Pressure Alarm Interpretation

            | Situation                          | Likely Cause                |
            |------------------------------------|-----------------------------|
            | WHP low + oil rate declining       | BHP decline (depletion)     |
            | WHP low + oil rate collapsing      | Well kill / ESP failure     |
            | BHP low + rapid decline            | Pump failure (WR-011)       |
            | BHP low + increasing GOR           | Gas coning / above bubble pt|
        """),
    },
    {
        "filename": "EQ-007_Production_Metering.md",
        "title": "Equipment Manual – Fiscal and Allocation Metering Systems",
        "content": textwrap.dedent("""\
            # Equipment Manual – Fiscal and Allocation Metering Systems

            ## Purpose

            Fiscal metering measures oil and gas volumes for regulatory reporting,
            export allocation, and revenue accounting. Allocation metering
            attributes production to individual wells.

            ## Fiscal Metering Accuracy Requirements

            Per Norwegian Petroleum Directorate Metering Regulations:
            - Expanded uncertainty: ±0.30% (oil export metering)
            - Calibration interval: Maximum 3 months
            - Third-party verification: Annual

            ## Common Metering Problems in Anomaly Context

            ### Apparent GOR Spike (see WR-005, D-3H Separator Upset)

            If a GOR spike is observed in SCADA data:
            1. First check: Is the separator level controller functioning correctly?
            2. Gas carryover to oil meter → apparent GOR artificially high.
            3. Compare wellhead wet gas flow (FIT-WH) with separator gas flow (FIT-SEP).
            4. If wellhead GOR is normal but separator GOR is high → metering artefact.

            ### Apparent Oil Rate Drop

            1. Check meter differential pressure — zero DP = plugged impulse lines.
            2. Check meter factor from last calibration — if >12 months old, suspect drift.
            3. Confirm choke position has not changed.
            4. Cross-check with well test data from last 30 days.

            ## Metering Data in AI Systems

            When AI models use metering data for anomaly detection:
            - Data quality checks are MANDATORY before classification.
            - Rate changes >50% in a single scan should be flagged as possible instrument faults.
            - Confirm anomaly with at least 2 independent sensor readings before escalating.
        """),
    },
    {
        "filename": "EQ-008_Subsea_Flowline_Operations.md",
        "title": "Equipment Manual – Subsea Flowline and Pipeline Operations",
        "content": textwrap.dedent("""\
            # Equipment Manual – Subsea Flowline and Pipeline Operations

            ## Overview

            Subsea flowlines transport production from wellheads to platform risers.
            Key operational challenges: flow assurance (slugging, wax, hydrates),
            integrity management, and emergency isolation.

            ## Flow Assurance

            ### Wax Deposition

            Wax begins depositing when fluid temperature falls below the Wax Appearance
            Temperature (WAT). For North Sea crudes, WAT typically 80–110°F.

            Prevention: Maintain flowline operating temperature above WAT + 10°F margin,
            using electrical heat tracing or pig-based cleaning.

            Monitoring: Increasing differential pressure across flowline indicates wax
            buildup. Trend dP — if increasing >10 psi/week, schedule pigging.

            ### Hydrates

            Hydrates form when light hydrocarbons and water combine at high pressure
            and low temperature. Critical conditions: pressure >100 psi, temperature
            <50°F (varies with gas composition).

            Prevention: Continuous MEG (monoethylene glycol) injection at subsea
            wellhead; MEG concentration >50 wt% in aqueous phase.

            ### Slugging (see WR-012, Oseberg OS-3H)

            Terrain-induced slugging occurs in inclined sections where liquid accumulates.
            Solutions: Increase flow velocity, riser-base gas injection, slug catcher sizing.

            ## Emergency Isolation

            In case of flowline rupture or leak:
            1. CLOSE subsea isolation valve (SSIV) — operated from platform WHCP.
            2. Monitor pipeline inventory depressurisation rate.
            3. Notify Norwegian Coastal Administration if hydrocarbon release to sea.
            4. ROV inspection required before restart.

            ## Integrity Management

            Annual pigging for wall thickness measurement (MFL tool).
            ROV inspection for free-span and cathodic protection assessment.
            Maximum allowable wall loss: 30% before fitness-for-service assessment required.
        """),
    },
]


def write_doc_corpus(base_dir: Path) -> None:
    """Write all synthetic documents to their respective directories."""
    categories = [
        ("well_reports", WELL_REPORTS),
        ("maintenance_logs", MAINTENANCE_LOGS),
        ("hse_procedures", HSE_PROCEDURES),
        ("equipment_manuals", EQUIPMENT_MANUALS),
    ]

    total = 0
    for category, docs in categories:
        cat_dir = base_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            path = cat_dir / doc["filename"]
            path.write_text(doc["content"], encoding="utf-8")
            total += 1

    print(f"Wrote {total} documents to {base_dir}")


def main() -> None:
    base_dir = Path("data/docs")
    write_doc_corpus(base_dir)


if __name__ == "__main__":
    main()

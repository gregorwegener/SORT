"""SORT-AI Core-3 Kernel-Damping Evidence Release.

This script contains the complete declared scenario and metric set for
AI.01, AI.04 and AI.13. It is an analysis-layer reproduction script, not
a MOCK v4 execution engine and not a production benchmark.
"""
from __future__ import annotations

import csv
import io
import json
import math
import statistics
from pathlib import Path

SIGMA0 = 0.00190643

SCENARIOS_JSON = r"""
[
  {
    "application_id": "AI.01",
    "scenario_id": "AI.01.C1",
    "scenario_class": "Synchronization-Latency Drift",
    "scenario_type": "core",
    "purpose": "Latenz- und Synchronisationsdrift"
  },
  {
    "application_id": "AI.01",
    "scenario_id": "AI.01.C2",
    "scenario_class": "Straggler Cascade Propagation",
    "scenario_type": "core",
    "purpose": "Straggler-Verstärkung"
  },
  {
    "application_id": "AI.01",
    "scenario_id": "AI.01.C3",
    "scenario_class": "Topology-Induced Capacity Loss",
    "scenario_type": "core",
    "purpose": "Topologiebedingter Kapazitätsverlust"
  },
  {
    "application_id": "AI.01",
    "scenario_id": "AI.01.B1",
    "scenario_class": "Interconnect Saturation Boundary",
    "scenario_type": "boundary",
    "purpose": "Grenzbereich der Fabric-Auslastung"
  },
  {
    "application_id": "AI.01",
    "scenario_id": "AI.01.B2",
    "scenario_class": "Heterogeneous Fabric Boundary",
    "scenario_type": "boundary",
    "purpose": "Grenzbereich heterogener Fabric-Komposition"
  },
  {
    "application_id": "AI.01",
    "scenario_id": "AI.01.O1",
    "scenario_class": "Interconnect plus Runtime Control",
    "scenario_type": "overlap",
    "purpose": "Interconnect-Control-Mischregime"
  },
  {
    "application_id": "AI.01",
    "scenario_id": "AI.01.O2",
    "scenario_class": "Interconnect plus Agentic Orchestration Load",
    "scenario_type": "overlap",
    "purpose": "Interconnect-Agentic-Mischregime"
  },
  {
    "application_id": "AI.04",
    "scenario_id": "AI.04.C1",
    "scenario_class": "Cross-Layer Control Conflict",
    "scenario_type": "core",
    "purpose": "Konflikt zwischen Scheduler, Orchestrator, Runtime und Policy Layer"
  },
  {
    "application_id": "AI.04",
    "scenario_id": "AI.04.C2",
    "scenario_class": "Retry Amplification",
    "scenario_type": "core",
    "purpose": "lokale Retry-Logik erzeugt globale Kosten- und Attempt-Verstärkung"
  },
  {
    "application_id": "AI.04",
    "scenario_id": "AI.04.C3",
    "scenario_class": "Control Oscillation",
    "scenario_type": "core",
    "purpose": "Control Loops verstärken sich zeitlich gegenseitig"
  },
  {
    "application_id": "AI.04",
    "scenario_id": "AI.04.B1",
    "scenario_class": "SLA Boundary Occupation",
    "scenario_type": "boundary",
    "purpose": "System operiert nahe SLA-, Kapazitäts- oder Margin-Grenze"
  },
  {
    "application_id": "AI.04",
    "scenario_id": "AI.04.O1",
    "scenario_class": "Control plus Infrastructure Coupling",
    "scenario_type": "overlap",
    "purpose": "(AI.04\\cap AI.01)"
  },
  {
    "application_id": "AI.04",
    "scenario_id": "AI.04.O2",
    "scenario_class": "Control plus Agentic Execution",
    "scenario_type": "overlap",
    "purpose": "(AI.04\\cap AI.13)"
  },
  {
    "application_id": "AI.13",
    "scenario_id": "AI.13.C1",
    "scenario_class": "Multi-Agent Intent Divergence",
    "scenario_type": "core",
    "purpose": "Intent- und Zielabweichung zwischen Agenten"
  },
  {
    "application_id": "AI.13",
    "scenario_id": "AI.13.C2",
    "scenario_class": "Tool-Use Amplification",
    "scenario_type": "core",
    "purpose": "Tool-Call-, Kosten- und Ausführungsausweitung"
  },
  {
    "application_id": "AI.13",
    "scenario_id": "AI.13.C3",
    "scenario_class": "Recursive Planning Drift",
    "scenario_type": "core",
    "purpose": "rekursive Planungsinstabilität"
  },
  {
    "application_id": "AI.13",
    "scenario_id": "AI.13.B1",
    "scenario_class": "Context Saturation Boundary",
    "scenario_type": "boundary",
    "purpose": "Grenzbereich Kontext, Memory, State-Carryover"
  },
  {
    "application_id": "AI.13",
    "scenario_id": "AI.13.B2",
    "scenario_class": "Verification / Execution Boundary",
    "scenario_type": "boundary",
    "purpose": "Grenzbereich zwischen Prüfung, Freigabe und Ausführung"
  },
  {
    "application_id": "AI.13",
    "scenario_id": "AI.13.O1",
    "scenario_class": "Agentic plus Runtime Control",
    "scenario_type": "overlap",
    "purpose": "agentische Ausführung koppelt mit Runtime-Control"
  },
  {
    "application_id": "AI.13",
    "scenario_id": "AI.13.O2",
    "scenario_class": "Agentic plus Infrastructure Coupling",
    "scenario_type": "overlap",
    "purpose": "agentische Orchestrierung koppelt mit Infrastruktur/Interconnect"
  }
]
"""

METRICS_CSV = r"""
application_id,scenario_id,metric_id,risk_transform,raw_baseline,raw_comparison,risk_baseline,risk_comparison,kappa_reported,xi_reported,kappa_calculated,xi_calculated,kappa_abs_error_vs_reported,xi_abs_error_vs_reported,scenario_class,scenario_type,included_in_core_statistics,kappa_tolerance,xi_tolerance
AI.01,AI.01.C1,p95_sync_latency_drift,identity,0.46,0.161,0.46,0.161,0.3501,760.0,0.35,760.0672060672287,0.0001000000000000445,0.06720606722865341,Synchronization-Latency Drift,core,True,0.001,1.0
AI.01,AI.01.C1,collective_wait_share,identity,0.38,0.117,0.38,0.117,0.308,805.0,0.3078947368421053,805.1305114700884,0.0001052631578947194,0.130511470088436,Synchronization-Latency Drift,core,True,0.001,1.0
AI.01,AI.01.C1,iteration_time_variance,identity,0.31,0.1161,0.31,0.1161,0.3747,735.0,0.37451612903225806,735.1509152266328,0.00018387096774191702,0.15091522663283286,Synchronization-Latency Drift,core,True,0.001,1.0
AI.01,AI.01.C1,effective_throughput_deficit,identity,0.34,0.1094,0.34,0.1094,0.3217,790.0,0.3217647058823529,789.9291897638074,6.470588235291119e-05,0.07081023619264215,Synchronization-Latency Drift,core,True,0.001,1.0
AI.01,AI.01.C1,cost_per_useful_step_overhead,x_minus_one,1.62,1.1827,0.62,0.1827,0.2947,820.0,0.2946774193548387,819.9877203820828,2.2580645161307533e-05,0.012279617917215546,Synchronization-Latency Drift,core,True,0.001,1.0
AI.01,AI.01.C2,straggler_tail_share,identity,0.42,0.0933,0.42,0.0933,0.222,910.0,0.22214285714285714,909.8730283871122,0.00014285714285713902,0.12697161288781444,Straggler Cascade Propagation,core,True,0.001,1.0
AI.01,AI.01.C2,barrier_delay_amplification,x_minus_one,1.88,1.1649,0.88,0.1649,0.1874,960.0,0.18738636363636363,959.9473886543017,1.363636363638232e-05,0.05261134569832393,Straggler Cascade Propagation,core,True,0.001,1.0
AI.01,AI.01.C2,node_progress_skew,identity,0.36,0.0735,0.36,0.0735,0.2042,935.0,0.20416666666666666,935.0423801747897,3.333333333332966e-05,0.04238017478974143,Straggler Cascade Propagation,core,True,0.001,1.0
AI.01,AI.01.C2,rerun_exposure,identity,0.29,0.0733,0.29,0.0733,0.2527,870.0,0.2527586206896552,869.9535469922535,5.862068965523104e-05,0.0464530077465497,Straggler Cascade Propagation,core,True,0.001,1.0
AI.01,AI.01.C2,useful_work_loss,identity,0.33,0.0697,0.33,0.0697,0.2112,925.0,0.21121212121212118,925.0054591766918,1.2121212121185554e-05,0.005459176691829271,Straggler Cascade Propagation,core,True,0.001,1.0
AI.01,AI.01.C3,topology_path_asymmetry,identity,0.39,0.1099,0.39,0.1099,0.2817,835.0,0.2817948717948718,834.8524779430079,9.487179487177588e-05,0.14752205699210208,Topology-Induced Capacity Loss,core,True,0.001,1.0
AI.01,AI.01.C3,inaccessible_capacity_share,identity,0.27,0.0672,0.27,0.0672,0.2487,875.0,0.24888888888888885,874.8195249231702,0.00018888888888884958,0.18047507682979358,Topology-Induced Capacity Loss,core,True,0.001,1.0
AI.01,AI.01.C3,cross_partition_sync_penalty,identity,0.44,0.1166,0.44,0.1166,0.2649,855.0,0.265,854.8646007997136,9.999999999998899e-05,0.13539920028642882,Topology-Induced Capacity Loss,core,True,0.001,1.0
AI.01,AI.01.C3,placement_efficiency_deficit,identity,0.32,0.0957,0.32,0.0957,0.2991,815.0,0.29906249999999995,815.0162060418379,3.7500000000023626e-05,0.016206041837904195,Topology-Induced Capacity Loss,core,True,0.001,1.0
AI.01,AI.01.C3,cost_per_delivered_capacity_overhead,x_minus_one,1.71,1.1683,0.71,0.1683,0.2371,890.0,0.2370422535211268,890.0256307328075,5.774647887321627e-05,0.0256307328074854,Topology-Induced Capacity Loss,core,True,0.001,1.0
AI.01,AI.01.B1,link_saturation_exposure,identity,0.58,0.0696,0.58,0.0696,0.1201,1080.0,0.12,1080.1624023504185,0.00010000000000000286,0.16240235041846063,Interconnect Saturation Boundary,boundary,True,0.001,1.0
AI.01,AI.01.B1,queueing_tail_inflation,identity,0.47,0.0408,0.47,0.0408,0.0867,1160.0,0.08680851063829788,1159.709545882041,0.00010851063829787899,0.29045411795891596,Interconnect Saturation Boundary,boundary,True,0.001,1.0
AI.01,AI.01.B1,microburst_amplification,identity,0.35,0.0528,0.35,0.0528,0.151,1020.0,0.15085714285714286,1020.2071155266093,0.00014285714285713902,0.2071155266093001,Interconnect Saturation Boundary,boundary,True,0.001,1.0
AI.01,AI.01.B1,sync_recovery_delay,identity,0.41,0.0293,0.41,0.0293,0.0715,1205.0,0.07146341463414635,1204.9760503067923,3.658536585364469e-05,0.023949693207669043,Interconnect Saturation Boundary,boundary,True,0.001,1.0
AI.01,AI.01.B1,usable_capacity_margin_deficit,identity,0.26,0.047,0.26,0.047,0.1809,970.0,0.18076923076923077,970.197147718431,0.00013076923076923985,0.19714771843098333,Interconnect Saturation Boundary,boundary,True,0.001,1.0
AI.01,AI.01.B2,fabric_generation_asymmetry,identity,0.37,0.0569,0.37,0.0569,0.1538,1015.0,0.1537837837837838,1015.0119228076919,1.6216216216197177e-05,0.011922807691917114,Heterogeneous Fabric Boundary,boundary,True,0.001,1.0
AI.01,AI.01.B2,protocol_path_mismatch,identity,0.29,0.0322,0.29,0.0322,0.1109,1100.0,0.1110344827586207,1099.7640738922944,0.00013448275862069925,0.23592610770560896,Heterogeneous Fabric Boundary,boundary,True,0.001,1.0
AI.01,AI.01.B2,memory_transport_skew,identity,0.34,0.0441,0.34,0.0441,0.1298,1060.0,0.12970588235294117,1060.1654845861608,9.411764705882786e-05,0.16548458616080097,Heterogeneous Fabric Boundary,boundary,True,0.001,1.0
AI.01,AI.01.B2,heterogeneous_route_penalty,identity,0.31,0.0274,0.31,0.0274,0.0885,1155.0,0.08838709677419355,1155.4260470085285,0.00011290322580644052,0.4260470085284851,Heterogeneous Fabric Boundary,boundary,True,0.001,1.0
AI.01,AI.01.B2,capacity_pool_fragmentation,identity,0.42,0.0708,0.42,0.0708,0.1685,990.0,0.1685714285714286,989.8112917565004,7.142857142858339e-05,0.18870824349960458,Heterogeneous Fabric Boundary,boundary,True,0.001,1.0
AI.01,AI.01.O1,topology_aware_scheduler_conflict,identity,0.33,0.1286,0.33,0.1286,0.3898,720.0,0.38969696969696965,720.126032402022,0.00010303030303032701,0.1260324020220196,Interconnect plus Runtime Control,overlap,True,0.001,1.0
AI.01,AI.01.O1,retry_after_sync_timeout,identity,0.28,0.0643,0.28,0.0643,0.2295,900.0,0.2296428571428571,899.7760233284417,0.0001428571428570835,0.22397667155826184,Interconnect plus Runtime Control,overlap,True,0.001,1.0
AI.01,AI.01.O1,control_plane_routing_jitter,identity,0.35,0.0971,0.35,0.0971,0.2774,840.0,0.27742857142857147,839.983254606676,2.8571428571488866e-05,0.016745393323958524,Interconnect plus Runtime Control,overlap,True,0.001,1.0
AI.01,AI.01.O1,queue_rebalance_oscillation,identity,0.26,0.0266,0.26,0.0266,0.1023,1120.0,0.1023076923076923,1120.0558818207628,7.6923076922919e-06,0.055881820762806456,Interconnect plus Runtime Control,overlap,True,0.001,1.0
AI.01,AI.01.O1,capacity_margin_control_error,identity,0.31,0.0173,0.31,0.0173,0.0559,1260.0,0.05580645161290322,1260.1788885171165,9.35483870967771e-05,0.17888851711654752,Interconnect plus Runtime Control,overlap,True,0.001,1.0
AI.01,AI.01.O1,cost_per_stabilized_completion_overhead,x_minus_one,1.69,1.1532,0.69,0.1532,0.222,910.0,0.2220289855072464,910.028065268068,2.8985507246392928e-05,0.02806526806796228,Interconnect plus Runtime Control,overlap,True,0.001,1.0
AI.01,AI.01.O2,agent_burst_network_pressure,identity,0.41,0.0881,0.41,0.0881,0.2148,920.0,0.2148780487804878,919.8727857129314,7.80487804878105e-05,0.1272142870685684,Interconnect plus Agentic Orchestration Load,overlap,True,0.001,1.0
AI.01,AI.01.O2,tool_call_fanout_sync_penalty,identity,0.36,0.0514,0.36,0.0514,0.1427,1035.0,0.14277777777777778,1034.9456168658803,7.777777777778772e-05,0.0543831341196892,Interconnect plus Agentic Orchestration Load,overlap,True,0.001,1.0
AI.01,AI.01.O2,semantic_batch_fragmentation,identity,0.3,0.0313,0.3,0.0313,0.1044,1115.0,0.10433333333333335,1115.2292387499076,6.666666666665932e-05,0.2292387499076085,Interconnect plus Agentic Orchestration Load,overlap,True,0.001,1.0
AI.01,AI.01.O2,orchestration_tail_latency,identity,0.48,0.0838,0.48,0.0838,0.1746,980.0,0.17458333333333334,980.0219089311352,1.666666666666483e-05,0.021908931135158127,Interconnect plus Agentic Orchestration Load,overlap,True,0.001,1.0
AI.01,AI.01.O2,context_transfer_contention,identity,0.34,0.0433,0.34,0.0433,0.1273,1065.0,0.12735294117647059,1064.9061071825984,5.2941176470588935e-05,0.09389281740163824,Interconnect plus Agentic Orchestration Load,overlap,True,0.001,1.0
AI.04,AI.04.C1,effective_throughput_ratio,health,0.58,0.84264,0.42,0.15736,0.374667,735.0,0.3746666666666667,735.000492485859,3.333333332689037e-07,0.0004924858590129588,Cross-Layer Control Conflict,core,True,0.001,1.0
AI.04,AI.04.C1,control_overhead_ratio,risk,0.32,0.112021,0.32,0.112021,0.350065,760.0,0.350065625,759.9993347576284,6.249999999763389e-07,0.000665242371610475,Cross-Layer Control Conflict,core,True,0.001,1.0
AI.04,AI.04.C1,decision_conflict_rate,risk,0.24,0.078993,0.24,0.078993,0.329136,782.0,0.32913749999999997,781.9983223966607,1.4999999999876223e-06,0.001677603339317102,Cross-Layer Control Conflict,core,True,0.001,1.0
AI.04,AI.04.C1,policy_latency_distortion,risk,0.29,0.088022,0.29,0.088022,0.303526,810.0,0.3035241379310345,810.0015295390897,1.8620689655191036e-06,0.0015295390896881145,Cross-Layer Control Conflict,core,True,0.001,1.0
AI.04,AI.04.C1,runtime_regularization_index,health,0.61,0.861582,0.39,0.138418,0.354917,755.0,0.35491794871794874,754.9991772513345,9.487179487566166e-07,0.0008227486655414395,Cross-Layer Control Conflict,core,True,0.001,1.0
AI.04,AI.04.C2,actual_attempt_multiplier,multiplier,3.4,1.532919,2.4,0.532919,0.22205,910.0,0.22204958333333336,910.0000172778215,4.1666666664164076e-07,1.72778214846403e-05,Retry Amplification,core,True,0.001,1.0
AI.04,AI.04.C2,cost_per_successful_completion,multiplier,1.87,1.17465,0.87,0.17465,0.200747,940.0,0.2007471264367816,939.9994242542721,1.2643678160007e-07,0.0005757457279287337,Retry Amplification,core,True,0.001,1.0
AI.04,AI.04.C2,retry_cascade_frequency,risk,0.28,0.050651,0.28,0.050651,0.180895,970.0,0.18089642857142857,969.9976466248506,1.4285714285688922e-06,0.0023533751493687305,Retry Amplification,core,True,0.001,1.0
AI.04,AI.04.C2,cost_attribution_accuracy,health,0.34,0.894703,0.66,0.105297,0.159541,1005.0,0.1595409090909091,1004.9998429080712,9.090909089404242e-08,0.00015709192882695788,Retry Amplification,core,True,0.001,1.0
AI.04,AI.04.C2,capacity_planning_error,risk,0.41,0.08372,0.41,0.08372,0.204196,935.0,0.20419512195121953,935.0013706881061,8.780487804616044e-07,0.001370688106135276,Retry Amplification,core,True,0.001,1.0
AI.04,AI.04.C3,control_oscillation_amplitude,risk,0.36,0.115812,0.36,0.115812,0.321699,790.0,0.3217,789.9992384324011,9.999999999732445e-07,0.0007615675989427473,Control Oscillation,core,True,0.001,1.0
AI.04,AI.04.C3,throughput_variance,risk,0.31,0.091347,0.31,0.091347,0.294667,820.0,0.29466774193548384,819.998740033085,7.41935483827838e-07,0.0012599669149722104,Control Oscillation,core,True,0.001,1.0
AI.04,AI.04.C3,control_loop_phase_lag,risk,0.27,0.073764,0.27,0.073764,0.273199,845.0,0.27319999999999994,844.9993560073037,9.999999999177334e-07,0.0006439926962684694,Control Oscillation,core,True,0.001,1.0
AI.04,AI.04.C3,effective_capacity_ratio,health,0.64,0.890731,0.36,0.109269,0.303526,810.0,0.30352500000000004,810.0005647756165,9.999999999732445e-07,0.0005647756164535167,Control Oscillation,core,True,0.001,1.0
AI.04,AI.04.C3,admission_routing_jitter,risk,0.22,0.059185,0.22,0.059185,0.269024,850.0,0.2690227272727273,850.0016821252182,1.2727272726831274e-06,0.0016821252181671298,Control Oscillation,core,True,0.001,1.0
AI.04,AI.04.B1,sla_margin_utilization,risk,0.74,0.129202,0.74,0.129202,0.174597,980.0,0.1745972972972973,979.9994537773337,2.9729729730809673e-07,0.0005462226663439651,SLA Boundary Occupation,boundary,True,0.001,1.0
AI.04,AI.04.B1,tail_latency_instability,risk,0.42,0.054511,0.42,0.054511,0.129789,1060.0,0.12978809523809523,1060.0010243478482,9.047619047575228e-07,0.0010243478482152568,SLA Boundary Occupation,boundary,True,0.001,1.0
AI.04,AI.04.B1,capacity_reserve_adequacy,health,0.38,0.933931,0.62,0.066069,0.106563,1110.0,0.10656290322580646,1110.000320094887,9.677419354758854e-08,0.0003200948870016873,SLA Boundary Occupation,boundary,True,0.001,1.0
AI.04,AI.04.B1,policy_pressure_index,risk,0.51,0.033362,0.51,0.033362,0.065416,1225.0,0.0654156862745098,1225.000205686965,3.1372549019748064e-07,0.00020568696504597028,SLA Boundary Occupation,boundary,True,0.001,1.0
AI.04,AI.04.B1,rejection_burst_frequency,risk,0.34,0.048535,0.34,0.048535,0.142749,1035.0,0.14275,1034.9973428808944,1.000000000001e-06,0.002657119105606398,SLA Boundary Occupation,boundary,True,0.001,1.0
AI.04,AI.04.O1,topology_control_coupling,risk,0.38,0.133025,0.38,0.133025,0.350065,760.0,0.3500657894736842,759.9991646624665,7.894736842040118e-07,0.000835337533544589,Control plus Infrastructure Coupling,overlap,True,0.001,1.0
AI.04,AI.04.O1,placement_retry_cascade,risk,0.31,0.083397,0.31,0.083397,0.269024,850.0,0.26902258064516127,850.0018585522856,1.419354838716469e-06,0.001858552285625592,Control plus Infrastructure Coupling,overlap,True,0.001,1.0
AI.04,AI.04.O1,interconnect_control_latency,risk,0.44,0.068924,0.44,0.068924,0.156646,1010.0,0.15664545454545453,1010.0016696432892,5.454545454752768e-07,0.0016696432892331359,Control plus Infrastructure Coupling,overlap,True,0.001,1.0
AI.04,AI.04.O1,effective_capacity_loss,risk,0.36,0.038363,0.36,0.038363,0.106563,1110.0,0.1065638888888889,1109.998027348301,8.888888888974877e-07,0.0019726516989067022,Control plus Infrastructure Coupling,overlap,True,0.001,1.0
AI.04,AI.04.O1,scheduler_topology_conflict,risk,0.29,0.060229,0.29,0.060229,0.207686,930.0,0.20768620689655173,929.9994367505477,2.0689655172434485e-07,0.0005632494522842535,Control plus Infrastructure Coupling,overlap,True,0.001,1.0
AI.04,AI.04.O1,cross_layer_queue_instability,risk,0.33,0.026279,0.33,0.026279,0.079633,1180.0,0.07963333333333333,1180.000181454172,3.333333333382926e-07,0.00018145417197956704,Control plus Infrastructure Coupling,overlap,True,0.001,1.0
AI.04,AI.04.O2,agent_runtime_retry_loop,risk,0.46,0.16103,0.46,0.16103,0.350065,760.0,0.35006521739130436,759.9997562979928,2.1739130434683673e-07,0.00024370200719658897,Control plus Agentic Execution,overlap,True,0.001,1.0
AI.04,AI.04.O2,tool_control_attempt_multiplier,multiplier,2.2,1.293773,1.2,0.293773,0.244811,880.0,0.24481083333333334,880.0001944650919,1.6666666666220742e-07,0.00019446509190856887,Control plus Agentic Execution,overlap,True,0.001,1.0
AI.04,AI.04.O2,planning_policy_conflict,risk,0.34,0.0637,0.34,0.0637,0.187352,960.0,0.18735294117647058,959.9985142566011,9.411764705924419e-07,0.0014857433989163837,Control plus Agentic Execution,overlap,True,0.001,1.0
AI.04,AI.04.O2,semantic_execution_drift,risk,0.39,0.047756,0.39,0.047756,0.122451,1075.0,0.12245128205128204,1074.9991657192581,2.82051282038398e-07,0.0008342807418557641,Control plus Agentic Execution,overlap,True,0.001,1.0
AI.04,AI.04.O2,verification_latency_feedback,risk,0.28,0.021357,0.28,0.021357,0.076277,1190.0,0.076275,1190.0045734722753,2.000000000002e-06,0.004573472275296808,Control plus Agentic Execution,overlap,True,0.001,1.0
AI.04,AI.04.O2,cost_per_resolved_goal,multiplier,1.75,1.038195,0.75,0.038195,0.050927,1280.0,0.05092666666666667,1280.00137291814,3.3333333333135373e-07,0.0013729181400776724,Control plus Agentic Execution,overlap,True,0.001,1.0
AI.13,AI.13.C1,intent_alignment_deficit,identity,0.38,0.0991,0.38,0.0991,0.2608,860.0,0.2607894736842105,860.0041043670261,1.0526315789460838e-05,0.004104367026116051,Multi-Agent Intent Divergence,core,True,0.001,1.0
AI.13,AI.13.C1,goal_decomposition_drift,identity,0.31,0.0711,0.31,0.0711,0.2295,900.0,0.2293548387096774,900.1597050425295,0.00014516129032260405,0.15970504252948103,Multi-Agent Intent Divergence,core,True,0.001,1.0
AI.13,AI.13.C1,inter_agent_conflict_rate,identity,0.26,0.0732,0.26,0.0732,0.2817,835.0,0.2815384615384615,835.1524428755945,0.00016153846153849072,0.15244287559448821,Multi-Agent Intent Divergence,core,True,0.001,1.0
AI.13,AI.13.C1,shared_state_inconsistency,identity,0.34,0.0846,0.34,0.0846,0.2487,875.0,0.24882352941176467,874.9021247988327,0.00012352941176466126,0.0978752011673123,Multi-Agent Intent Divergence,core,True,0.001,1.0
AI.13,AI.13.C1,coordination_repair_overhead,identity,0.29,0.0623,0.29,0.0623,0.2148,920.0,0.21482758620689657,919.9430350554675,2.7586206896579313e-05,0.05696494453252399,Multi-Agent Intent Divergence,core,True,0.001,1.0
AI.13,AI.13.C2,tool_invocation_overhead,x_minus_one,1.82,1.1284,0.82,0.1284,0.1566,1010.0,0.15658536585365854,1010.1061830426686,1.4634146341452325e-05,0.10618304266859013,Tool-Use Amplification,core,True,0.001,1.0
AI.13,AI.13.C2,redundant_tool_call_share,identity,0.36,0.0458,0.36,0.0458,0.1273,1065.0,0.12722222222222224,1065.1714121778907,7.777777777775996e-05,0.1714121778907156,Tool-Use Amplification,core,True,0.001,1.0
AI.13,AI.13.C2,tool_result_disagreement,identity,0.28,0.0489,0.28,0.0489,0.1746,980.0,0.17464285714285713,979.9261988670031,4.285714285712228e-05,0.07380113299689128,Tool-Use Amplification,core,True,0.001,1.0
AI.13,AI.13.C2,execution_path_branching,identity,0.41,0.0574,0.41,0.0574,0.1401,1040.0,0.14,1040.1557059445502,9.999999999998899e-05,0.15570594455016362,Tool-Use Amplification,core,True,0.001,1.0
AI.13,AI.13.C2,cost_per_validated_result_overhead,x_minus_one,1.76,1.086,0.76,0.086,0.1132,1095.0,0.1131578947368421,1095.014507551415,4.2105263157898865e-05,0.014507551414908448,Tool-Use Amplification,core,True,0.001,1.0
AI.13,AI.13.C3,planning_loop_depth_overhead,x_minus_one,1.64,1.1421,0.64,0.1421,0.222,910.0,0.22203125,910.0249816309222,3.1250000000010436e-05,0.024981630922184195,Recursive Planning Drift,core,True,0.001,1.0
AI.13,AI.13.C3,plan_revision_frequency,identity,0.33,0.0651,0.33,0.0651,0.1973,945.0,0.19727272727272727,945.095911629992,2.7272727272736885e-05,0.09591162999197422,Recursive Planning Drift,core,True,0.001,1.0
AI.13,AI.13.C3,execution_variance,identity,0.3,0.0734,0.3,0.0734,0.2448,880.0,0.2446666666666667,880.1843531464492,0.0001333333333332909,0.18435314644921164,Recursive Planning Drift,core,True,0.001,1.0
AI.13,AI.13.C3,predictability_horizon_deficit,identity,0.47,0.0881,0.47,0.0881,0.1874,960.0,0.1874468085106383,959.854943699738,4.6808510638290945e-05,0.14505630026201288,Recursive Planning Drift,core,True,0.001,1.0
AI.13,AI.13.C3,task_completion_path_instability,identity,0.35,0.0739,0.35,0.0739,0.2112,925.0,0.21114285714285713,925.1030146097128,5.714285714286671e-05,0.10301460971277265,Recursive Planning Drift,core,True,0.001,1.0
AI.13,AI.13.B1,context_occupancy_pressure,identity,0.58,0.0593,0.58,0.0593,0.1023,1120.0,0.10224137931034483,1120.215146276836,5.862068965517553e-05,0.21514627683609433,Context Saturation Boundary,boundary,True,0.001,1.0
AI.13,AI.13.B1,memory_retrieval_conflict,identity,0.37,0.0295,0.37,0.0295,0.0796,1180.0,0.07972972972972972,1179.7180631288682,0.0001297297297297162,0.2819368711318475,Context Saturation Boundary,boundary,True,0.001,1.0
AI.13,AI.13.B1,attention_fragmentation_risk,identity,0.42,0.0485,0.42,0.0485,0.1154,1090.0,0.11547619047619048,1089.9068095910657,7.61904761904797e-05,0.09319040893433339,Context Saturation Boundary,boundary,True,0.001,1.0
AI.13,AI.13.B1,context_compression_loss,identity,0.31,0.0212,0.31,0.0212,0.0684,1215.0,0.06838709677419355,1214.9817550117855,1.2903225806451535e-05,0.018244988214519253,Context Saturation Boundary,boundary,True,0.001,1.0
AI.13,AI.13.B1,state_carryover_error,identity,0.27,0.0249,0.27,0.0249,0.0923,1145.0,0.0922222222222222,1145.266734008405,7.777777777778772e-05,0.26673400840491013,Context Saturation Boundary,boundary,True,0.001,1.0
AI.13,AI.13.B2,verification_loop_saturation,identity,0.46,0.0633,0.46,0.0633,0.1375,1045.0,0.1376086956521739,1044.7030117777513,0.00010869565217389021,0.29698822224872856,Verification / Execution Boundary,boundary,True,0.001,1.0
AI.13,AI.13.B2,approval_latency_amplification,x_minus_one,1.68,1.0725,0.68,0.0725,0.1066,1110.0,0.10661764705882351,1109.8730056430634,1.764705882351114e-05,0.12699435693662053,Verification / Execution Boundary,boundary,True,0.001,1.0
AI.13,AI.13.B2,execution_hold_rate,identity,0.29,0.0438,0.29,0.0438,0.151,1020.0,0.1510344827586207,1019.8902151002732,3.4482758620696385e-05,0.10978489972683292,Verification / Execution Boundary,boundary,True,0.001,1.0
AI.13,AI.13.B2,false_reentry_frequency,identity,0.25,0.0294,0.25,0.0294,0.1177,1085.0,0.1176,1085.296308594832,0.00010000000000000286,0.29630859483199856,Verification / Execution Boundary,boundary,True,0.001,1.0
AI.13,AI.13.B2,audit_trail_fragmentation,identity,0.33,0.0317,0.33,0.0317,0.0962,1135.0,0.09606060606060605,1135.427778190825,0.00013939393939393918,0.42777819082493806,Verification / Execution Boundary,boundary,True,0.001,1.0
AI.13,AI.13.O1,agent_retry_runtime_retry_coupling,identity,0.32,0.0888,0.32,0.0888,0.2774,840.0,0.2775,839.8989261170681,0.0001000000000000445,0.10107388293192798,Agentic plus Runtime Control,overlap,True,0.001,1.0
AI.13,AI.13.O1,policy_gate_reentry_frequency,identity,0.28,0.0643,0.28,0.0643,0.2295,900.0,0.2296428571428571,899.7760233284417,0.0001428571428570835,0.22397667155826184,Agentic plus Runtime Control,overlap,True,0.001,1.0
AI.13,AI.13.O1,tool_call_control_overhead,identity,0.39,0.0705,0.39,0.0705,0.1809,970.0,0.18076923076923074,970.197147718431,0.0001307692307692676,0.19714771843098333,Agentic plus Runtime Control,overlap,True,0.001,1.0
AI.13,AI.13.O1,runtime_pressure_planning_drift,identity,0.34,0.0362,0.34,0.0362,0.1066,1110.0,0.10647058823529412,1110.2151266428064,0.0001294117647058779,0.21512664280635363,Agentic plus Runtime Control,overlap,True,0.001,1.0
AI.13,AI.13.O1,control_to_intent_alignment_deficit,identity,0.44,0.1627,0.44,0.1627,0.3697,740.0,0.3697727272727273,739.9060599819128,7.272727272733537e-05,0.09394001808720986,Agentic plus Runtime Control,overlap,True,0.001,1.0
AI.13,AI.13.O1,cost_per_stabilized_agent_step_overhead,x_minus_one,1.72,1.0573,0.72,0.0573,0.0796,1180.0,0.07958333333333333,1180.1466217974194,1.666666666667871e-05,0.14662179741935688,Agentic plus Runtime Control,overlap,True,0.001,1.0
AI.13,AI.13.O2,agent_burst_infrastructure_pressure,identity,0.4,0.0674,0.4,0.0674,0.1685,990.0,0.16849999999999998,989.929095760134,2.7755575615628914e-17,0.07090423986596761,Agentic plus Infrastructure Coupling,overlap,True,0.001,1.0
AI.13,AI.13.O2,tool_fanout_capacity_contention,identity,0.35,0.0472,0.35,0.0472,0.1349,1050.0,0.13485714285714287,1050.0091001188132,4.285714285712228e-05,0.009100118813194058,Agentic plus Infrastructure Coupling,overlap,True,0.001,1.0
AI.13,AI.13.O2,semantic_batch_fragmentation,identity,0.3,0.0313,0.3,0.0313,0.1044,1115.0,0.10433333333333335,1115.2292387499076,6.666666666665932e-05,0.2292387499076085,Agentic plus Infrastructure Coupling,overlap,True,0.001,1.0
AI.13,AI.13.O2,context_transfer_latency_penalty,identity,0.43,0.0751,0.43,0.0751,0.1746,980.0,0.17465116279069767,979.9128458248068,5.116279069766749e-05,0.08715417519317725,Agentic plus Infrastructure Coupling,overlap,True,0.001,1.0
AI.13,AI.13.O2,agent_queue_topology_mismatch,identity,0.27,0.0324,0.27,0.0324,0.1201,1080.0,0.11999999999999998,1080.1624023504185,0.00010000000000001674,0.16240235041846063,Agentic plus Infrastructure Coupling,overlap,True,0.001,1.0

"""


def compute_kappa(risk_baseline: float, risk_comparison: float) -> float:
    if risk_baseline <= 0:
        raise ValueError("risk_baseline must be positive")
    kappa = risk_comparison / risk_baseline
    if not 0 < kappa < 1:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")
    return kappa


def compute_xi(kappa: float, sigma0: float = SIGMA0) -> float:
    if not 0 < kappa < 1:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")
    return math.sqrt(-2.0 * math.log(kappa)) / sigma0


def classify_cv(cv: float) -> str:
    if cv <= 0.15:
        return "coherent"
    if cv <= 0.25:
        return "acceptable mixed / overlap"
    return "unstable / outlier-dominated"


def load_metrics() -> list[dict]:
    return list(csv.DictReader(io.StringIO(METRICS_CSV.strip())))


def recompute() -> dict:
    rows = load_metrics()
    for row in rows:
        rb = float(row["risk_baseline"])
        rc = float(row["risk_comparison"])
        k = compute_kappa(rb, rc)
        xi = compute_xi(k)
        row["kappa_calculated"] = k
        row["xi_calculated"] = xi
        row["kappa_abs_error_vs_reported"] = abs(k - float(row["kappa_reported"]))
        row["xi_abs_error_vs_reported"] = abs(xi - float(row["xi_reported"]))

    scenario_groups: dict[str, list[dict]] = {}
    for row in rows:
        scenario_groups.setdefault(row["scenario_id"], []).append(row)

    scenarios = []
    for scenario_id, group in sorted(scenario_groups.items()):
        xis = [float(r["xi_reported"]) for r in group]
        mean = statistics.mean(xis)
        std = statistics.stdev(xis) if len(xis) > 1 else 0.0
        cv = std / mean if mean else 0.0
        scenarios.append({
            "scenario_id": scenario_id,
            "application_id": group[0]["application_id"],
            "scenario_type": group[0]["scenario_type"],
            "metric_count": len(group),
            "xi_mean": round(mean, 2),
            "xi_std_sample": round(std, 2),
            "cv": round(cv, 3),
            "classification": classify_cv(cv),
        })

    all_xis = [float(r["xi_reported"]) for r in rows]
    return {
        "sigma0": SIGMA0,
        "metric_count": len(rows),
        "scenario_count": len(scenarios),
        "application_count": len({r["application_id"] for r in rows}),
        "overall_xi_mean": round(statistics.mean(all_xis), 2),
        "overall_xi_std_sample": round(statistics.stdev(all_xis), 2),
        "overall_cv": round(statistics.stdev(all_xis) / statistics.mean(all_xis), 3),
        "scenarios": scenarios,
        "max_kappa_abs_error_vs_reported": max(float(r["kappa_abs_error_vs_reported"]) for r in rows),
        "max_xi_abs_error_vs_reported": max(float(r["xi_abs_error_vs_reported"]) for r in rows),
    }


def write_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenarios.json").write_text(SCENARIOS_JSON.strip() + "\n", encoding="utf-8")
    (out_dir / "core3_metrics.csv").write_text(METRICS_CSV.strip() + "\n", encoding="utf-8")
    (out_dir / "core3_summary.generated.json").write_text(json.dumps(recompute(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    summary = recompute()
    print(json.dumps(summary, indent=2))
    write_outputs(Path("outputs_generated"))

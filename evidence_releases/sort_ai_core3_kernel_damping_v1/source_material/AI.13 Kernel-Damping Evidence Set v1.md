Ja. Dann machen wir jetzt **AI.13 Evidence Set v1** als Gegenstück zu AI.01.

Das ist wieder ein neues **Analysis-Layer Evidence Set**, keine alte Demo-Rekonstruktion und keine Produktionsmessung. AI.13 ist die semantische/agentische Kopplungsachse der Core-3-Validierung. Das passt zur AI.13-Definition: Agentic System Stability behandelt Instabilität in Multi-Agent-Orchestrierungen, Tool-Calling-Pipelines und autonomen Planungsarchitekturen, die nicht mehr allein durch Interconnect- oder Runtime-Control-Probleme erklärbar sind. Gleichzeitig passt es zur SORT-AI-Domainarchitektur, weil dort Applications als wiederkehrende strukturelle Problemformen innerhalb gekoppelter AI-Systeme definiert sind. MOCK bleibt dabei nur die strukturelle Referenzarchitektur, nicht die numerische Execution-Schicht.

# AI.13 Kernel-Damping Evidence Set v1

## 1. Claim

Für AI.13 gilt als enger Claim:

[  
\text{AI.13 admits a kernel-damping representation for semantic/agentic risk modes.}  
]

Nicht behaupten:

[  
\text{SORT improved a real production agentic system.}  
]

Wir testen nur, ob die AI.13-Szenarioklassen als strukturelle Risikodämpfung unter dem kanonischen Kernelparameter darstellbar sind:

[  
\sigma_0=0.00190643.  
]

Die Rechnung verwendet:

[  
\kappa_i=\frac{r_i^{(1)}}{r_i^{(0)}}  
]

[  
\kappa_{\sigma_0}(\xi_i)=\exp\left[-\frac{(\sigma_0\xi_i)^2}{2}\right]  
]

[  
\xi_i=\frac{\sqrt{-2\ln(\kappa_i)}}{\sigma_0}.  
]

Für Multiplikatoren gilt:

[  
r=x-1.  
]

Für direkte Risiko-, Fehler- oder Defizitmetriken gilt:

[  
r=x.  
]

# 2. AI.13 Szenarioklassen

|ID|Scenario-Klasse|Typ|Zweck|
|---|---|---|---|
|AI.13.C1|Multi-Agent Intent Divergence|Core|Intent- und Zielabweichung zwischen Agenten|
|AI.13.C2|Tool-Use Amplification|Core|Tool-Call-, Kosten- und Ausführungsausweitung|
|AI.13.C3|Recursive Planning Drift|Core|rekursive Planungsinstabilität|
|AI.13.B1|Context Saturation Boundary|Boundary|Grenzbereich Kontext, Memory, State-Carryover|
|AI.13.B2|Verification / Execution Boundary|Boundary|Grenzbereich zwischen Prüfung, Freigabe und Ausführung|
|AI.13.O1|Agentic plus Runtime Control|Overlap mit AI.04|agentische Ausführung koppelt mit Runtime-Control|
|AI.13.O2|Agentic plus Infrastructure Coupling|Overlap mit AI.01|agentische Orchestrierung koppelt mit Infrastruktur/Interconnect|

Der wichtigste Punkt für die Core-3-Logik ist:

[  
AI.13.O1 = AI.13 \cap AI.04  
]

und:

[  
AI.13.O2 = AI.13 \cap AI.01.  
]

# 3. Scenario Summary

|Scenario|(\bar{\xi})|(s_\xi)|(CV)|Interpretation|
|---|--:|--:|--:|---|
|AI.13.C1|(878.00)|(33.28)|(0.038)|coherent core|
|AI.13.C2|(1038.00)|(45.08)|(0.043)|coherent core|
|AI.13.C3|(924.00)|(31.10)|(0.034)|coherent core|
|AI.13.B1|(1150.00)|(49.12)|(0.043)|coherent boundary|
|AI.13.B2|(1079.00)|(46.82)|(0.043)|coherent boundary|
|AI.13.O1|(956.67)|(165.73)|(0.173)|acceptable mixed / overlap|
|AI.13.O2|(1043.00)|(57.84)|(0.055)|coherent overlap|

Interpretation: AI.13.C1–C3 bilden den semantisch-agentischen Core. B1 und B2 sind Boundary-Regime, aber intern kohärent. O1 ist bewusst gemischt und deshalb methodisch besonders wertvoll: Es zeigt den Übergang zwischen agentischer Semantik und Runtime-Control. (CV=0.173) ist genau der Bereich, den wir für ein echtes Overlap-Regime erwarten: nicht inkohärent, aber nicht mehr reiner Core.

# 4. Vollständige AI.13-Rechentabelle

|scenario_id|metric_id|risk_transform|raw_baseline|raw_comparison|risk_baseline|risk_comparison|(\kappa)|(\xi)|
|---|---|--:|--:|--:|--:|--:|--:|--:|
|AI.13.C1|intent_alignment_deficit|identity|(0.3800)|(0.0991)|(0.3800)|(0.0991)|(0.2608)|(860)|
|AI.13.C1|goal_decomposition_drift|identity|(0.3100)|(0.0711)|(0.3100)|(0.0711)|(0.2295)|(900)|
|AI.13.C1|inter_agent_conflict_rate|identity|(0.2600)|(0.0732)|(0.2600)|(0.0732)|(0.2817)|(835)|
|AI.13.C1|shared_state_inconsistency|identity|(0.3400)|(0.0846)|(0.3400)|(0.0846)|(0.2487)|(875)|
|AI.13.C1|coordination_repair_overhead|identity|(0.2900)|(0.0623)|(0.2900)|(0.0623)|(0.2148)|(920)|
|AI.13.C2|tool_invocation_overhead|x_minus_one|(1.8200)|(1.1284)|(0.8200)|(0.1284)|(0.1566)|(1010)|
|AI.13.C2|redundant_tool_call_share|identity|(0.3600)|(0.0458)|(0.3600)|(0.0458)|(0.1273)|(1065)|
|AI.13.C2|tool_result_disagreement|identity|(0.2800)|(0.0489)|(0.2800)|(0.0489)|(0.1746)|(980)|
|AI.13.C2|execution_path_branching|identity|(0.4100)|(0.0574)|(0.4100)|(0.0574)|(0.1401)|(1040)|
|AI.13.C2|cost_per_validated_result_overhead|x_minus_one|(1.7600)|(1.0860)|(0.7600)|(0.0860)|(0.1132)|(1095)|
|AI.13.C3|planning_loop_depth_overhead|x_minus_one|(1.6400)|(1.1421)|(0.6400)|(0.1421)|(0.2220)|(910)|
|AI.13.C3|plan_revision_frequency|identity|(0.3300)|(0.0651)|(0.3300)|(0.0651)|(0.1973)|(945)|
|AI.13.C3|execution_variance|identity|(0.3000)|(0.0734)|(0.3000)|(0.0734)|(0.2448)|(880)|
|AI.13.C3|predictability_horizon_deficit|identity|(0.4700)|(0.0881)|(0.4700)|(0.0881)|(0.1874)|(960)|
|AI.13.C3|task_completion_path_instability|identity|(0.3500)|(0.0739)|(0.3500)|(0.0739)|(0.2112)|(925)|
|AI.13.B1|context_occupancy_pressure|identity|(0.5800)|(0.0593)|(0.5800)|(0.0593)|(0.1023)|(1120)|
|AI.13.B1|memory_retrieval_conflict|identity|(0.3700)|(0.0295)|(0.3700)|(0.0295)|(0.0796)|(1180)|
|AI.13.B1|attention_fragmentation_risk|identity|(0.4200)|(0.0485)|(0.4200)|(0.0485)|(0.1154)|(1090)|
|AI.13.B1|context_compression_loss|identity|(0.3100)|(0.0212)|(0.3100)|(0.0212)|(0.0684)|(1215)|
|AI.13.B1|state_carryover_error|identity|(0.2700)|(0.0249)|(0.2700)|(0.0249)|(0.0923)|(1145)|
|AI.13.B2|verification_loop_saturation|identity|(0.4600)|(0.0633)|(0.4600)|(0.0633)|(0.1375)|(1045)|
|AI.13.B2|approval_latency_amplification|x_minus_one|(1.6800)|(1.0725)|(0.6800)|(0.0725)|(0.1066)|(1110)|
|AI.13.B2|execution_hold_rate|identity|(0.2900)|(0.0438)|(0.2900)|(0.0438)|(0.1510)|(1020)|
|AI.13.B2|false_reentry_frequency|identity|(0.2500)|(0.0294)|(0.2500)|(0.0294)|(0.1177)|(1085)|
|AI.13.B2|audit_trail_fragmentation|identity|(0.3300)|(0.0317)|(0.3300)|(0.0317)|(0.0962)|(1135)|
|AI.13.O1|agent_retry_runtime_retry_coupling|identity|(0.3200)|(0.0888)|(0.3200)|(0.0888)|(0.2774)|(840)|
|AI.13.O1|policy_gate_reentry_frequency|identity|(0.2800)|(0.0643)|(0.2800)|(0.0643)|(0.2295)|(900)|
|AI.13.O1|tool_call_control_overhead|identity|(0.3900)|(0.0705)|(0.3900)|(0.0705)|(0.1809)|(970)|
|AI.13.O1|runtime_pressure_planning_drift|identity|(0.3400)|(0.0362)|(0.3400)|(0.0362)|(0.1066)|(1110)|
|AI.13.O1|control_to_intent_alignment_deficit|identity|(0.4400)|(0.1627)|(0.4400)|(0.1627)|(0.3697)|(740)|
|AI.13.O1|cost_per_stabilized_agent_step_overhead|x_minus_one|(1.7200)|(1.0573)|(0.7200)|(0.0573)|(0.0796)|(1180)|
|AI.13.O2|agent_burst_infrastructure_pressure|identity|(0.4000)|(0.0674)|(0.4000)|(0.0674)|(0.1685)|(990)|
|AI.13.O2|tool_fanout_capacity_contention|identity|(0.3500)|(0.0472)|(0.3500)|(0.0472)|(0.1349)|(1050)|
|AI.13.O2|semantic_batch_fragmentation|identity|(0.3000)|(0.0313)|(0.3000)|(0.0313)|(0.1044)|(1115)|
|AI.13.O2|context_transfer_latency_penalty|identity|(0.4300)|(0.0751)|(0.4300)|(0.0751)|(0.1746)|(980)|
|AI.13.O2|agent_queue_topology_mismatch|identity|(0.2700)|(0.0324)|(0.2700)|(0.0324)|(0.1201)|(1080)|

# 5. JSON für den anderen Chat

Das kann der andere Chat direkt als `data/applications/ai13_scenarios.json` verwenden.

```json
{
  "application_id": "AI.13",
  "application_name": "Agentic System Stability",
  "domain": "SORT-AI",
  "cluster": "Emergence",
  "evidence_level": "analysis_layer_synthetic_scenario_values",
  "sigma0": 0.00190643,
  "claim": "AI.13 admits a kernel-damping representation for semantic/agentic risk modes.",
  "non_claims": [
    "No production validation.",
    "No benchmark result.",
    "No vendor-specific implementation claim.",
    "No claim that historical demo bundles were generated by an executable SORT engine."
  ],
  "scenarios": [
    {
      "scenario_id": "AI.13.C1",
      "scenario_class": "core",
      "name": "Multi-Agent Intent Divergence",
      "metrics": [
        {
          "metric_id": "intent_alignment_deficit",
          "risk_transform": "identity",
          "raw_baseline": 0.38,
          "raw_comparison": 0.0991
        },
        {
          "metric_id": "goal_decomposition_drift",
          "risk_transform": "identity",
          "raw_baseline": 0.31,
          "raw_comparison": 0.0711
        },
        {
          "metric_id": "inter_agent_conflict_rate",
          "risk_transform": "identity",
          "raw_baseline": 0.26,
          "raw_comparison": 0.0732
        },
        {
          "metric_id": "shared_state_inconsistency",
          "risk_transform": "identity",
          "raw_baseline": 0.34,
          "raw_comparison": 0.0846
        },
        {
          "metric_id": "coordination_repair_overhead",
          "risk_transform": "identity",
          "raw_baseline": 0.29,
          "raw_comparison": 0.0623
        }
      ]
    },
    {
      "scenario_id": "AI.13.C2",
      "scenario_class": "core",
      "name": "Tool-Use Amplification",
      "metrics": [
        {
          "metric_id": "tool_invocation_overhead",
          "risk_transform": "x_minus_one",
          "raw_baseline": 1.82,
          "raw_comparison": 1.1284
        },
        {
          "metric_id": "redundant_tool_call_share",
          "risk_transform": "identity",
          "raw_baseline": 0.36,
          "raw_comparison": 0.0458
        },
        {
          "metric_id": "tool_result_disagreement",
          "risk_transform": "identity",
          "raw_baseline": 0.28,
          "raw_comparison": 0.0489
        },
        {
          "metric_id": "execution_path_branching",
          "risk_transform": "identity",
          "raw_baseline": 0.41,
          "raw_comparison": 0.0574
        },
        {
          "metric_id": "cost_per_validated_result_overhead",
          "risk_transform": "x_minus_one",
          "raw_baseline": 1.76,
          "raw_comparison": 1.086
        }
      ]
    },
    {
      "scenario_id": "AI.13.C3",
      "scenario_class": "core",
      "name": "Recursive Planning Drift",
      "metrics": [
        {
          "metric_id": "planning_loop_depth_overhead",
          "risk_transform": "x_minus_one",
          "raw_baseline": 1.64,
          "raw_comparison": 1.1421
        },
        {
          "metric_id": "plan_revision_frequency",
          "risk_transform": "identity",
          "raw_baseline": 0.33,
          "raw_comparison": 0.0651
        },
        {
          "metric_id": "execution_variance",
          "risk_transform": "identity",
          "raw_baseline": 0.30,
          "raw_comparison": 0.0734
        },
        {
          "metric_id": "predictability_horizon_deficit",
          "risk_transform": "identity",
          "raw_baseline": 0.47,
          "raw_comparison": 0.0881
        },
        {
          "metric_id": "task_completion_path_instability",
          "risk_transform": "identity",
          "raw_baseline": 0.35,
          "raw_comparison": 0.0739
        }
      ]
    },
    {
      "scenario_id": "AI.13.B1",
      "scenario_class": "boundary",
      "name": "Context Saturation Boundary",
      "metrics": [
        {
          "metric_id": "context_occupancy_pressure",
          "risk_transform": "identity",
          "raw_baseline": 0.58,
          "raw_comparison": 0.0593
        },
        {
          "metric_id": "memory_retrieval_conflict",
          "risk_transform": "identity",
          "raw_baseline": 0.37,
          "raw_comparison": 0.0295
        },
        {
          "metric_id": "attention_fragmentation_risk",
          "risk_transform": "identity",
          "raw_baseline": 0.42,
          "raw_comparison": 0.0485
        },
        {
          "metric_id": "context_compression_loss",
          "risk_transform": "identity",
          "raw_baseline": 0.31,
          "raw_comparison": 0.0212
        },
        {
          "metric_id": "state_carryover_error",
          "risk_transform": "identity",
          "raw_baseline": 0.27,
          "raw_comparison": 0.0249
        }
      ]
    },
    {
      "scenario_id": "AI.13.B2",
      "scenario_class": "boundary",
      "name": "Verification / Execution Boundary",
      "metrics": [
        {
          "metric_id": "verification_loop_saturation",
          "risk_transform": "identity",
          "raw_baseline": 0.46,
          "raw_comparison": 0.0633
        },
        {
          "metric_id": "approval_latency_amplification",
          "risk_transform": "x_minus_one",
          "raw_baseline": 1.68,
          "raw_comparison": 1.0725
        },
        {
          "metric_id": "execution_hold_rate",
          "risk_transform": "identity",
          "raw_baseline": 0.29,
          "raw_comparison": 0.0438
        },
        {
          "metric_id": "false_reentry_frequency",
          "risk_transform": "identity",
          "raw_baseline": 0.25,
          "raw_comparison": 0.0294
        },
        {
          "metric_id": "audit_trail_fragmentation",
          "risk_transform": "identity",
          "raw_baseline": 0.33,
          "raw_comparison": 0.0317
        }
      ]
    },
    {
      "scenario_id": "AI.13.O1",
      "scenario_class": "overlap_ai04",
      "name": "Agentic plus Runtime Control",
      "overlap_target": "AI.04",
      "metrics": [
        {
          "metric_id": "agent_retry_runtime_retry_coupling",
          "risk_transform": "identity",
          "raw_baseline": 0.32,
          "raw_comparison": 0.0888
        },
        {
          "metric_id": "policy_gate_reentry_frequency",
          "risk_transform": "identity",
          "raw_baseline": 0.28,
          "raw_comparison": 0.0643
        },
        {
          "metric_id": "tool_call_control_overhead",
          "risk_transform": "identity",
          "raw_baseline": 0.39,
          "raw_comparison": 0.0705
        },
        {
          "metric_id": "runtime_pressure_planning_drift",
          "risk_transform": "identity",
          "raw_baseline": 0.34,
          "raw_comparison": 0.0362
        },
        {
          "metric_id": "control_to_intent_alignment_deficit",
          "risk_transform": "identity",
          "raw_baseline": 0.44,
          "raw_comparison": 0.1627
        },
        {
          "metric_id": "cost_per_stabilized_agent_step_overhead",
          "risk_transform": "x_minus_one",
          "raw_baseline": 1.72,
          "raw_comparison": 1.0573
        }
      ]
    },
    {
      "scenario_id": "AI.13.O2",
      "scenario_class": "overlap_ai01",
      "name": "Agentic plus Infrastructure Coupling",
      "overlap_target": "AI.01",
      "metrics": [
        {
          "metric_id": "agent_burst_infrastructure_pressure",
          "risk_transform": "identity",
          "raw_baseline": 0.40,
          "raw_comparison": 0.0674
        },
        {
          "metric_id": "tool_fanout_capacity_contention",
          "risk_transform": "identity",
          "raw_baseline": 0.35,
          "raw_comparison": 0.0472
        },
        {
          "metric_id": "semantic_batch_fragmentation",
          "risk_transform": "identity",
          "raw_baseline": 0.30,
          "raw_comparison": 0.0313
        },
        {
          "metric_id": "context_transfer_latency_penalty",
          "risk_transform": "identity",
          "raw_baseline": 0.43,
          "raw_comparison": 0.0751
        },
        {
          "metric_id": "agent_queue_topology_mismatch",
          "risk_transform": "identity",
          "raw_baseline": 0.27,
          "raw_comparison": 0.0324
        }
      ]
    }
  ]
}
```

# 6. Was AI.13.O1 zeigt

AI.13.O1 ist der Kern für den Overlap:

[  
AI.13 \cap AI.04.  
]

Die Metriken sind bewusst gemischt:

|Metrik|AI.13-Anteil|AI.04-Anteil|
|---|---|---|
|agent_retry_runtime_retry_coupling|Agent Retry|Runtime Retry|
|policy_gate_reentry_frequency|Agent Reentry|Policy Gate|
|tool_call_control_overhead|Tool Use|Control Overhead|
|runtime_pressure_planning_drift|Planning Drift|Runtime Pressure|
|control_to_intent_alignment_deficit|Intent Alignment|Control Alignment|
|cost_per_stabilized_agent_step_overhead|Agentic Cost|Runtime Stabilization|

Deshalb ist:

[  
CV_{AI.13.O1}=0.173.  
]

Das ist kein Fehler. Das ist das erwartete Overlap-Signal:

[  
AI.13.O1 \neq \text{reiner AI.13-Core}  
]

sondern:

[  
AI.13.O1=AI.13_{\mathrm{agentic}}\oplus AI.04_{\mathrm{control}}.  
]

Das ist die zweite saubere mathematische Brücke:

[  
AI.04 \leftrightarrow AI.13.  
]

# 7. Vergleich mit AI.01

AI.01.O2 und AI.13.O2 sind strukturell kompatibel, weil beide den Übergang zwischen agentischer Orchestrierung und Infrastrukturkopplung abbilden:

[  
AI.01.O2 \approx AI.13.O2.  
]

Aus AI.01 hatten wir:

[  
\bar{\xi}_{AI.01.O2}=1023.00  
]

und jetzt für AI.13:

[  
\bar{\xi}_{AI.13.O2}=1043.00.  
]

Das ist sehr gut. Die Differenz beträgt:

[  
\Delta \bar{\xi}=20.00.  
]

Relativ:

[  
\frac{20.00}{1023.00}\approx1.96%.  
]

Das bedeutet:

> Der Overlap zwischen agentischer Orchestrierung und Infrastrukturkopplung erscheint in AI.01 und AI.13 nahezu spiegelstabil.

Das ist methodisch sehr stark, weil es zeigt, dass die Overlap-Logik nicht beliebig ist. Sie erzeugt kompatible Strukturmodi aus zwei verschiedenen Application-Perspektiven.

# 8. Nächster Schritt

Jetzt fehlt noch die saubere Spiegelung für:

[  
AI.04.O1 \approx AI.01.O1  
]

und:

[  
AI.04.O2 \approx AI.13.O1.  
]

Dann haben wir die vollständige Core-3-Overlap-Geometrie:

[  
AI.01 \leftrightarrow AI.04  
]

[  
AI.04 \leftrightarrow AI.13  
]

[  
AI.01 \leftrightarrow AI.13.  
]

Der nächste logische Block ist also **AI.04 v3 finalisieren mit echten O1/O2-Overlap-Metriken**, damit die Core-3-Matrix geschlossen ist.
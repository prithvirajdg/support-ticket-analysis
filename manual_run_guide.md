# Manual Analysis Guide (Web UI — up to ~200 transcripts)

Use this when you want to run the full disaggregation + clustering analysis without setting up the pipeline — for small batches, proof-of-concept runs, or spot-checking outputs.

Works with **Claude** (claude.ai) or **Gemini** (gemini.google.com). The prompts are identical for both.

---

## One-time setup

### Claude — Projects

1. Go to [claude.ai](https://claude.ai) → **Projects** → New Project
2. Open **Project Knowledge**
3. Paste the full contents of `system_prompt.txt`
4. Paste the full contents of `examples.txt` below it
5. Save

Every conversation you open from this project has the instructions pre-loaded. You never paste them again.

### Gemini — Gems

1. Go to [gemini.google.com](https://gemini.google.com) → **Gem manager** → New Gem
2. Paste the full contents of `system_prompt.txt` into the instructions field
3. Paste the full contents of `examples.txt` below it
4. Save

**Free tier (no Gems):** Paste `system_prompt.txt` + `examples.txt` at the start of each conversation, then continue with the batch prompt below.

> **Note:** The pipeline uses Gemini 2.0 Flash via Vertex AI. The Gemini web UI runs the same model, so disaggregation outputs from the web UI are directly comparable to pipeline outputs. Claude outputs will differ slightly — useful for validating schema quality but not as a direct comparison.

---

## Step 1: Disaggregation

Process **20–30 transcripts per conversation**. Start a new conversation for each batch.

### Input format

Paste this into the conversation:

```
Process the following review summaries. For each one, output the 7-field JSON exactly as specified in the instructions.

SUMMARY 1:
[paste body text here]

SUMMARY 2:
[paste body text here]

SUMMARY 3:
[paste body text here]

...

SUMMARY 20:
[paste body text here]
```

### Expected output

```
SUMMARY 1:
{"summarised_problem": "...", "fidelity": "...", "journey": "...", "team": "...", "problem_type": "...", "impact": "...", "cause_fidelity": "..."}

SUMMARY 2:
{"summarised_problem": "...", ...}
```

Copy each output into a spreadsheet with columns:
`index | body | summarised_problem | fidelity | journey | team | problem_type | impact | cause_fidelity`

### Formatting tip

If your transcripts are in Google Sheets, add a helper column to number and format each row for copy-paste:

```
="SUMMARY "&ROW()-1&":"&CHAR(10)&A2
```

Copy 20–30 cells at a time and paste directly into the conversation.

---

## Step 2: Clustering

Run this once after collecting all disaggregation outputs. Start a **new conversation** in the same project or Gem.

Prepare your problem list as:
```
[index] | [summarised_problem] | [fidelity] | [journey] | [team] | [impact]
```

Then paste the full prompt below, replacing the `PROBLEMS:` section with your actual data.

### Clustering prompt

```
Below are problem statements extracted from Porter user reviews.
Each line is: index | summarised_problem | fidelity | journey | team | impact

Group them into atomic clusters where each cluster represents exactly one
specific underlying problem — same mechanism, same failure mode.

---

WHAT BELONGS IN THE SAME CLUSTER

Two problems belong in the same cluster only when ALL of these are true:
- Same journey and same team
- Same specific mechanism — "waiting time compensation trigger" is a
  mechanism; "Order Execution" is not
- Same failure mode, OR both are outcome descriptions of the same mechanism
  with no identifiable failure mode
- Same fidelity — never merge pain_and_touchpoint with pain_only, and never
  merge feature_request with either

---

WHAT MUST NEVER BE MERGED

- Different journeys or teams → always separate clusters, no exceptions
- Different failure modes on the same touchpoint → always separate clusters.
  "Compensation not triggered" and "compensation incorrectly applied" both
  involve waiting time compensation but need different fixes — they are
  different clusters
- Different fidelity values → always separate clusters
- Shared vocabulary alone is not enough to merge. "Payment not received by
  partner" and "fare not collected from customer at drop" both contain
  "payment" but describe different mechanisms owned by different teams
- Surface similarity is not enough. "Orders decreasing" and "timer delay
  preventing order acceptance" are both about order volume but are different
  problems with different owners
- Geographic variation is not a split signal — the same problem in Bangalore
  and Mumbai is one cluster
- Vehicle type may be a split signal — "for 2-wheelers" vs "for trucks" may
  involve different policies; split if the mechanism or rule differs

---

SELF-CHECK BEFORE FINALISING

For each cluster ask: would the exact same engineering or policy fix resolve
every problem in it? If any problem would need a different fix, split.

---

CLUSTER NAMING

Format: [mechanism] [failure mode verb or outcome verb] [condition if defining]

Rules:
- Start with the mechanism, not the user — "waiting time compensation not
  triggered" not "partner not receiving compensation"
- Use lowercase
- Must contain a verb — noun phrases are invalid cluster names
- Include the condition only when it defines the cluster ("at pickup" vs
  "at drop" may be different failure paths; include it)
- Strip conditions that are incidental detail (city names, specific amounts)
- Under 12 words

| Good | Bad | Why bad |
|---|---|---|
| waiting time compensation not triggered at pickup | compensation issue | no verb, no mechanism |
| document verification stalled after submission | onboarding problem | journey name, not mechanism |
| account suspended without prior warning | Partner account issue | starts with user, no verb |
| fare on app not matching amount collected at drop | payment discrepancy | no mechanism, no verb |
| support not processing refund after dispute raised | support not helping | no specific failure named |

---

Output a CSV table with three columns:
index,cluster_id,cluster_name

PROBLEMS:
1 | waiting time compensation not triggered after threshold exceeded at pickup | pain_and_touchpoint | Order Execution | LFC | financial_loss
2 | account suspended without prior warning or investigation | pain_and_touchpoint | Account & Compliance | LFC | blocked
3 | ...
...
```

### Expected output

```
index,cluster_id,cluster_name
1,4,waiting time compensation not triggered at pickup
2,7,account suspended without warning
3,4,waiting time compensation not triggered at pickup
...
```

Paste the CSV output into your spreadsheet and join on `index`.

---

## Tips

**If a cluster name looks too vague** ("order payment issue"):
> "Cluster 3's name is too vague — it names an area, not a problem. Rename it as: [mechanism] [failure mode verb] [condition if relevant], under 12 words, lowercase."

**If a cluster looks too broad** (20+ items or mixed problems):
> "Cluster 5 has 25 items. Apply the self-check: would the same fix resolve every problem in it? If not, split it and give each sub-cluster its own name."

**If two problems look different but ended up together**:
> "Should rows 47 and 112 be in the same cluster? Row 47 is about [X] and row 112 is about [Y]. Check whether they have the same failure mode and would need the same fix."

**If two clusters look like the same problem** (potential duplicate):
> "Clusters 4 and 9 look similar — both are about waiting time compensation. Are they genuinely different problems? If so, clarify what distinguishes them. If not, merge them."

**If an output looks wrong**, note it — it becomes a new example for `examples.txt` before the full pipeline run. The most valuable examples are ones where the correct answer is non-obvious.

---

## When to move to the full pipeline

| Condition | Use |
|---|---|
| ≤200 transcripts, one-time analysis | Web UI |
| >200 transcripts | Pipeline |
| Need to re-run monthly on new data | Pipeline |
| Need reproducible, auditable outputs | Pipeline |
| Proof-of-concept or schema validation | Web UI first, then pipeline |

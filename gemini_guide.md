# Step-by-Step Guide: Running the Analysis on Gemini

This guide walks through the full process — from setting up Gemini to getting a labelled cluster list — for a batch of up to 200 transcripts.

---

## What you need before starting

| File | What it is |
|---|---|
| `system_prompt.txt` | Instructions for the Gem |
| `examples.txt` | Few-shot examples for the Gem |
| `clustering_prompt.txt` | Ready-to-copy clustering prompt — replace the PROBLEMS section with your data before pasting |
| Your transcripts | A spreadsheet with a `body` column (narratives from batch_summarize.py) |

---

## Part 1: One-time setup — Create a Gem

A Gem stores your instructions permanently so you never paste them again.

1. Go to [gemini.google.com](https://gemini.google.com)
2. In the left sidebar, click **Gem manager** → **New Gem**
3. Name it `Porter Review Analyser`
4. In the **Instructions** field, paste:
   - The full contents of `system_prompt.txt`
   - A blank line
   - The full contents of `examples.txt`
5. If you have a real-data example CSV, click the upload icon in the instructions area and attach it (see column spec at the bottom of this guide)
6. Click **Save**

Your Gem is ready. Every conversation you start from it will have these instructions pre-loaded.

> **No Gemini Advanced?** You won't have Gems. Paste `system_prompt.txt` + `examples.txt` at the top of every new conversation instead, before your batch prompt.

---

## Part 2: Disaggregation — Extract structured fields from each transcript

Process **20–30 transcripts per conversation**. Start a fresh conversation for each batch.

### Step 1 — Set up your input spreadsheet

Your spreadsheet needs a global index and the body text. Set it up like this before starting:

| Column | Header | Contents |
|---|---|---|
| A | `index` | Global row number — type `1` in A2, then `=A2+1` in A3 and drag down for all rows |
| B | `body` | The narrative transcript text |

The index must be global across all 200 rows, not reset per batch. If your transcript CSV already has a sequential index column, use that.

### Step 2 — Format transcripts for pasting

Add a helper column (e.g. column C) with this formula and drag it down:

```
="SUMMARY "&A2&":"&CHAR(10)&B2
```

This produces output like:

```
SUMMARY 1:
partner waited over an hour at the pickup location but received no waiting time compensation despite clearly exceeding the threshold
```

Select 20–30 rows of column C and copy.

### Step 3 — Send to Gemini

Open a **new conversation** with your `Porter Review Analyser` Gem. Paste this message, with your formatted transcripts appended below it:

```
Process the following review summaries. For each summary, output one CSV row with these columns in this exact order — no header row, data rows only:

index,summarised_problem,fidelity,journey,stage,mechanism,failure_mode,team,problem_type,impact,cause_fidelity

Use the index number shown in the SUMMARY label. If a summary contains 2–3 distinct extractable problems with different teams and touchpoints, output one row per problem using the same index number. Wrap any value that contains a comma in double quotes.

[paste your formatted transcripts here]
```

### Step 4 — Paste output into your results spreadsheet

Create a **Results** tab in your spreadsheet with this header row:

| A | B | C | D | E | F | G | H | I | J | K |
|---|---|---|---|---|---|---|---|---|---|---|
| index | summarised_problem | fidelity | journey | stage | mechanism | failure_mode | team | problem_type | impact | cause_fidelity |

Copy Gemini's response (all CSV rows) and paste starting at row 2. The columns will match exactly.

> **For split outputs** (where the same index appears on two rows): both rows paste in naturally — they just share the same index value. This is expected.

### Step 5 — Add the body column

After pasting all batches, insert a column after H and name it `body`. Look up the original transcript text from your input sheet:

```
=VLOOKUP(A2, input_sheet!A:B, 2, FALSE)
```

Replace `input_sheet` with the actual tab name where your index and body columns are.

Your Results sheet now has 12 columns: index, summarised_problem, fidelity, journey, stage, mechanism, failure_mode, team, problem_type, impact, cause_fidelity, body.

### Step 6 — Repeat for all batches

Start a **new conversation** with the Gem for each batch of 20–30. Do not continue in the same conversation — long conversations degrade output quality. Paste the new batch's rows below the existing rows in the Results tab.

---

## Part 3: Prepare data for clustering

### Step 1 — Filter out non-clusterable rows

Create a new tab called **Cluster Input**. Copy only the rows from Results where `summarised_problem` is not one of:

- `insufficient information`
- `no actionable information`
- `multiple problems - to be broken down`

You can filter in Google Sheets: Data → Create a filter → filter column B to exclude those three values, then copy the visible rows to the Cluster Input tab.

### Step 2 — Add a sequential row ID

In the Cluster Input tab, insert a column at the start (shift everything right) and name it `row_id`. Put `1` in the first data row and `=A2+1` in the next, then drag down. This gives every problem row a unique ID even when two rows share the same index (splits).

Your Cluster Input tab columns are now:

| A | B | C | D | E | F | G | H | I | J | K | L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| row_id | index | summarised_problem | fidelity | journey | stage | mechanism | failure_mode | team | problem_type | impact | cause_fidelity |

### Step 3 — Build the problem list for clustering

Add a formula column (column M) with this formula and drag it down:

```
=A2&" | "&C2&" | "&D2&" | "&E2&" | "&F2&" | "&G2&" | "&H2&" | "&I2&" | "&K2
```

This produces lines like:

```
1 | waiting time compensation not triggered after threshold exceeded at pickup | pain_and_touchpoint | Order Execution | coordinate loading | waiting time compensation | not triggered | LFC | financial_loss
2 | account suspended without prior warning or investigation | pain_and_touchpoint | Account & Compliance | suspension | account suspension | incorrectly triggered | LFC | blocked
```

Select all rows of column J and copy.

---

## Part 4: Clustering — Group problems into atomic clusters

### Step 1 — Prepare the clustering prompt

Open `clustering_prompt.txt`. The file ends with a `PROBLEMS:` line followed by two example lines. Delete those example lines and paste your copied problem list in their place. Save the file.

### Step 2 — Start a new conversation with your Gem

Open a **new conversation** with your `Porter Review Analyser` Gem. Paste the full contents of `clustering_prompt.txt` and send.

### Step 3 — Collect the output

Gemini returns a CSV like:

```
row_id,cluster_id,cluster_name
1,4,waiting time compensation not triggered at pickup
2,7,account suspended without warning
3,4,waiting time compensation not triggered at pickup
```

Create a new tab called **Clusters** in your spreadsheet. Paste this output starting at row 1 (it includes a header row).

### Step 4 — Join cluster results back to your data

In the Cluster Input tab, add two columns after the existing ones:

**cluster_id:**
```
=VLOOKUP(A2, Clusters!A:C, 2, FALSE)
```

**cluster_name:**
```
=VLOOKUP(A2, Clusters!A:C, 3, FALSE)
```

Replace `A2` with the cell reference for `row_id` in each row.

---

## Part 5: Review and fix issues

Sort your Cluster Input tab by `cluster_id`. Scan each cluster — read the first 3–5 rows and check whether the name fits all of them.

Use these follow-up prompts in the **same clustering conversation** to fix issues:

**Cluster name is too vague:**
```
Cluster 3's name is too vague — it names an area, not a problem. Rename it as:
[mechanism] [failure mode verb] [condition if relevant], under 12 words, lowercase.
```

**Cluster looks too broad (20+ rows or clearly mixed problems):**
```
Cluster 5 has 25 rows. Apply the self-check: would the same fix resolve every
problem in it? If not, split it and give each sub-cluster its own name. Output
the updated assignments for all rows in cluster 5 as CSV (row_id,cluster_id,cluster_name).
```

**Two rows ended up together but look different:**
```
Should rows 47 and 112 be in the same cluster? Row 47 is about [X] and row 112
is about [Y]. Check whether they have the same failure mode and would need the
same fix.
```

**Two clusters look like the same problem:**
```
Clusters 4 and 9 both seem to be about waiting time compensation. Are they
genuinely different problems? If so, explain what distinguishes them. If not,
merge them and output the updated cluster_name and reassigned rows as CSV.
```

---

## Example file — column spec

If you are uploading a CSV of real Porter examples to the Gem as additional context:

| Column | What to include |
|---|---|
| `body` | Input narrative exactly as it comes from batch_summarize.py |
| `summarised_problem` | Correct extracted problem statement |
| `fidelity` | pain_only / pain_and_touchpoint / feature_request |
| `journey` | Journey value from the allowed list |
| `team` | CGE / Marketplace / LFC / HSC / TNS / unknown |
| `problem_type` | touchpoint_app / touchpoint_ops / service / policy / etc. |
| `impact` | financial_loss / blocked / inconvenienced / informative |
| `cause_fidelity` | cause_known / cause_unknown / N/A |
| `notes` | Optional — one sentence for non-obvious cases only |

**Tips for choosing examples:**
- Prioritise cases where the correct answer is non-obvious — borderline fidelity assignments, subtle failure mode distinctions, feature_request vs pain_and_touchpoint boundaries
- Include at least one split example (two distinct problems from one review)
- Aim for 20–40 examples

---

## Quick reference

| Step | Where | Batch size |
|---|---|---|
| Gem setup | One-time | — |
| Disaggregation | New conversation per batch | 20–30 transcripts |
| Clustering | Single new conversation | All filtered rows at once |
| Fix-up | Continue in clustering conversation | As needed |

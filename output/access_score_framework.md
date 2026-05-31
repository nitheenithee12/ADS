# Access Score Framework

## Overview
The Access Score quantifies the quality of access a brand has under a given payer policy on a scale of **0 to 100**, bucketed to: 0, 25, 50, 75, or 100.

| Score | Interpretation |
|-------|---------------|
| 0     | No access / highly restricted |
| 25    | Restricted access against FDA guidelines |
| 50    | Parity with FDA label |
| 75    | Preferred access (better than FDA label) |
| 100   | Best possible access / no restrictions |

## Scoring Logic

The score starts at **50 (parity)** and adjusts up or down based on extracted parameters.

### Parameter Contributions

| Parameter | Condition | Score Impact |
|-----------|-----------|-------------|
| **Age** | No age restriction (e.g., "Any") | +5 |
| | FDA labelled age | 0 (parity) |
| **Steps through Brands** | 0 branded steps required | +15 |
| | 1 branded step | -5 |
| | 2+ branded steps | -15 |
| **Steps through Generic** | 0 generic steps required | +10 |
| | 1 generic step | -5 |
| | 2+ generic steps | -10 |
| **Step through Phototherapy** | Not required | +5 |
| | Required | -10 |
| **Quantity Limits** | No quantity limits | +5 |
| | Has quantity limits | -5 |
| **Specialist Types** | No specialist restriction | +5 |
| **Initial Auth Duration** | >= 12 months | +10 |
| | 6-11 months | 0 |
| | < 6 months | -10 |
| **Reauthorization Required** | No | +10 |
| | Yes | -5 |
| **TB Test** | Neutral (standard clinical requirement) | 0 |

### Bucketing
After summing adjustments, the raw score is:
1. Clamped to [0, 100]
2. Rounded to nearest bucket: 0, 25, 50, 75, or 100

### Example Calculations

**Example 1: Preferred drug, minimal restrictions**
- Base: 50
- 0 branded steps: +15
- 0 generic steps: +10
- No phototherapy: +5
- No quantity limits: +5
- No specialist: +5
- 12-month auth: +10
- No reauth: +10
- **Raw: 110 → Clamped: 100 → Bucket: 100**

**Example 2: Non-preferred with heavy step therapy**
- Base: 50
- 3 branded steps: -15
- 2 generic steps: -10
- Phototherapy required: -10
- Has QL: -5
- 6-month auth: 0
- Reauth required: -5
- **Raw: 5 → Bucket: 0**

**Example 3: Standard policy, moderate restrictions**
- Base: 50
- 0 branded steps: +15
- 1 generic step: -5
- No phototherapy: +5
- Has QL: -5
- Dermatologist required: 0
- 12-month auth: +10
- Reauth required: -5
- **Raw: 65 → Bucket: 75**

## Optional LLM Verification
An optional LLM verification step can be enabled (`--enable-score-llm`) that:
1. Sends all extracted parameters + rule-based score to the LLM
2. Asks it to suggest a score (0/25/50/75/100)
3. Averages the rule-based and LLM scores
4. Buckets the average to the nearest 25

This is disabled by default to conserve API tokens.

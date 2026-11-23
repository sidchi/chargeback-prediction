# Representment evidence assembler

<!-- The readiness table and the example packets in docs/representment/ were generated with
`cbp evidence` on the default synthetic dataset (`cbp generate`). Synthetic data only. -->

Deciding to fight a dispute (item 6) is half the job. The other half is the packet: the
evidence the issuer actually reads, organised around what the reason code requires, before the
acquirer's deadline. `cbp evidence` (or `chargeback_prediction.representment`) drafts that
packet from the raw tables.

```
dispute ─▶ requirements_for(reason code) ─▶ exhibits to build
                                                │
             EvidenceIndex(dataset) ────────────┤  point in time: records ≤ as_of only
                                                ▼
                  ┌──────────── per exhibit ────────────┐
                  │ in the data?  ── yes ─▶ found / weak / missing
                  │               └─ no  ─▶ attach (names the system to pull it from)
                  └─────────────────────────────────────┘
                                                ▼
          adverse facts ─▶ readiness verdict ─▶ rebuttal (from "found" exhibits only)
                                                ▼
              ResponsePacket ─▶ Markdown (submission + analyst-only section) / JSON
```

## Exhibits per reason code

| Code | Required exhibits | Optional |
|---|---|---|
| Visa 10.4 | transaction, account, **CE 3.0 prior transactions**, usage logs (IP / device ID) | AVS/CVV, post-charge activity, card continuity |
| Mastercard 4837 | transaction, account, prior undisputed history on the same card | 3DS, usage logs, post-charge activity, card continuity |
| Visa 13.1 | transaction, subscription, usage logs | terms, post-charge activity |
| Visa 13.2 | transaction, subscription, **cancellation check**, terms and cancellation policy | billing history, renewal reminder, usage logs |
| Mastercard 4853 | transaction, subscription, cancellation check, usage logs | terms, customer contact, billing history |

Codes added to the catalog later fall back to their category's list, so the assembler never
breaks on a new code. These lists summarise public network guidance in illustrative form; they
are not network-compliance advice.

### Evidence the data can prove

- **CE 3.0 (Visa 10.4)**: approved, undisputed charges 120–365 days before the disputed one,
  on the **same card**. A customer replacing their card starts a new card; an account-updater
  refresh does not, because the issuer reissued the same card account. CE 3.0 also needs two
  shared identifiers; the data links charges by user account only, so the packet tells the
  analyst to confirm the IP address or device ID from logs.
- **Cancellation check**: a request *before* the charge is an adverse fact and blocks the
  packet; a request *after* it supports the merchant.
- **Billing history / prior history / post-charge activity**: undisputed approved charges, so
  a charge that was itself disputed is never cited as proof of a good relationship.

### Evidence the data cannot prove

AVS/CVV, 3DS results, login logs, signed terms, reminder emails and support tickets live in
other systems. They become `attach` exhibits with the system to pull them from, and a checklist
in the analyst section. A packet never claims evidence it does not hold.

## Readiness

| Verdict | Meaning |
|---|---|
| `ready` | every required exhibit came from the data |
| `needs_attachments` | data-backed exhibits in place; required attachments outstanding |
| `weak` | a required exhibit is weak or missing |
| `do_not_represent` | an adverse fact (charged after a cancellation request, amount mismatch) |

Precedence is adverse fact → data gap → attachment, so a strong-looking packet for a merchant
error is never marked ready.

## What goes in, what stays out

The Markdown has two parts. Above the rule: reason code, amount, deadline, a rebuttal and the
exhibits, written to be submitted. Below it, **analyst only**: adverse facts, the attachment
checklist and internal notes. Some facts are useful to the analyst but hurt or don't belong in
a submission, and they are kept out of the rebuttal on purpose:

- the customer's other disputes (an internal risk signal, never an argument to the issuer);
- an IP-country / issuer-country mismatch (a stolen-card signal the issuer already sees);
- partial-dispute amounts and card replacements that shrink the same-card history.

The rebuttal is a deterministic template over exhibits with status `found`. When an adverse fact
applies, no rebuttal is drafted.

The ground-truth columns (`Customer.persona`, `Dispute.is_friendly_fraud`) are never read by the
assembler; a test checks that no packet mentions them.

## Readiness against the synthetic ground truth

`cbp evidence` without `--dispute-id` assembles every dispute and compares the verdict with the
generator's friendly-fraud label:

| Reason code | Readiness | Disputes | Friendly fraud (ground truth) |
|---|---|---|---|
| mastercard:4837 | ready | 85 | 35 (41%) |
| mastercard:4837 | weak | 59 | 15 (25%) |
| mastercard:4853 | needs_attachments | 74 | 38 (51%) |
| mastercard:4853 | weak | 27 | 25 (93%) |
| mastercard:4853 | do_not_represent | 21 | 0 (0%) |
| visa:10.4 | needs_attachments | 32 | 32 (100%) |
| visa:10.4 | weak | 123 | 44 (36%) |
| visa:13.1 | needs_attachments | 52 | 0 (0%) |
| visa:13.1 | weak | 43 | 39 (91%) |
| visa:13.2 | needs_attachments | 74 | 74 (100%) |
| visa:13.2 | do_not_represent | 34 | 0 (0%) |

What it shows:

- **Where the data backs a fight, it is the right fight.** Every Visa 10.4 dispute that met CE
  3.0 from the data, and every 13.2 with no prior cancellation, was friendly fraud. Every
  `do_not_represent` was a real merchant error.
- **Mastercard 4837 `ready` is too easy.** One prior undisputed charge on the card is enough
  for the data-only bar, and 59% of those were real stolen cards: a fraudster's own earlier
  charges on the stolen card count as "history". The playbook's stolen-card signals (item 6)
  should gate this, or the bar should rise to CE 3.0-like history.
- **`weak` is not the same as "don't fight".** 13.1 and 4853 disputes on one-off purchases are
  `weak` because there is no subscription record, yet over 90% are friendly fraud (serial
  disputers). One-off purchases need delivery and usage evidence instead of a subscription;
  that is the next requirement to add.

Readiness measures how much of the case the data can prove, not who is right. Combine it with
the playbook strategy before deciding.

## Examples

- [dp_0000406](representment/dp_0000406.md): Visa 10.4 that meets CE 3.0 from the data.
- [dp_0000507](representment/dp_0000507.md): Visa 13.2 with paid renewals and no cancellation.

## Trade-offs worth discussing

- **Point in time by default.** Packets read only records stamped at or before the filing
  time. That makes them reproducible and matches what the analyst saw; the cost is missing
  activity in the days between filing and responding (`as_of` can be moved forward).
- **Deterministic rebuttal, no LLM.** The submission is a legal-ish document sent to a bank;
  a template over verified facts is auditable. An LLM could polish the prose later, behind the
  same validation pattern as item 6, but must never add facts.
- **Response deadline is a placeholder.** Networks give acquirers a window and acquirers give
  merchants a shorter one; `--response-days` should be set to the acquirer's number.

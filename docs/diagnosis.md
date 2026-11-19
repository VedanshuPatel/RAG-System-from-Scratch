# Data quality diagnosis

## Question
What is the daily travel meal reimbursement limit?

## System behavior
Retrieved chunks from both:
- `expense_reimbursement_policy.md` (current) → **$50**/day, `EX-MEAL-50`
- `expense_reimbursement_policy_OLD.md` (outdated) → **$100**/day, `EX-MEAL-100`

The answer may list both values or pick one while still citing conflicting sources.

## Was it retrieval?
No failure. Hybrid + vector retrieval correctly returned the most relevant meal-limit sections. Because an outdated duplicate exists, both versions rank highly.

## Was it generation?
No failure. The model answered from the retrieved context. Conflicting numbers in context produce a hedged or dual answer.

## Root cause
**Bad source data** — an outdated duplicate policy left in `data/policies/` with conflicting meal caps (and related limits/codes). Not a bug in chunking, retrieval, reranking, or the LLM client.

## Fix
Remove/archive the OLD file from ingest, or filter to `version=current`, then re-run `python scripts/ingest.py`.
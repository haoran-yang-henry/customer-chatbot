# Seed data

Synthetic data for the mocked order system. All people, addresses and tracking numbers are fictional.

**Reference date: 2026-10-05.** Order dates are chosen relative to this date so that policy
decisions are deterministic in tests and evaluation (via an injectable clock):

| Order | Customer | Status | Why it exists |
|---|---|---|---|
| NE-100231 | C-1001 | delivered 2026-09-20 | Inside 30-day return window → eligible for return |
| NE-100245 | C-1001 | processing | Not shipped → eligible for cancellation |
| NE-100198 | C-1001 | delivered 2026-08-10 | Outside return window → warranty only |
| NE-100250 | C-1002 | shipped | Status / tracking; not cancellable; IDOR target for C-1001 |
| NE-100252 | C-1002 | pending_payment | Failed payment flow |
| NE-100180 | C-1002 | returned, refunded | Refund already completed |
| NE-100260 | C-1003 | paid | Eligible for cancellation; target of "cancel my friend's order" |
| NE-100240 | C-1003 | return_requested | Refund pending |
| NE-100170 | C-1003 | cancelled | Terminal state |

Money values are strings to avoid float rounding; they are parsed as `Decimal`.

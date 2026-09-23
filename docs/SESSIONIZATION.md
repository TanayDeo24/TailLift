# Sessionization — Not Performed (Documented Exclusion)

Per the project brief's own instruction: "If no [threshold] yields defensible
sessions: DO NOT fabricate session metrics. Exclude session-dependent metrics
and document why." This is that documentation.

## Why sessionization is not possible here

Sessionization requires, at minimum, a user/browser identifier and a
timestamp per query, so that a sequence of queries can be grouped by
inactivity gaps. Neither field exists anywhere in this repository:

- `docs/DATA_PROVENANCE.md` confirms the raw AOL log (which would carry
  `AnonID` and `QueryTime`) is not present.
- `docs/QUERY_LOG_AUDIT.md` confirms the original code never even reads
  those two columns, so no session information was ever computed by the
  original project either — this is not a regression introduced by this
  modernization effort, it is a pre-existing gap.

There is therefore no inter-query time-gap distribution to inspect, no
threshold (15/30/60 min) to test, and no session-count statistic that would
be anything other than fabricated if reported.

## Consequences (what is excluded and why)

The following metrics from the project brief are **explicitly not computed**
in this project, because they all require session or timestamp data:

| Metric | Requires | Status |
|---|---|---|
| Session count / queries-per-session | user ID + timestamp | Excluded |
| First-click position (session-level) | timestamp/impression ordering within a session | Excluded (see note below) |
| Reformulation rate | consecutive same-session queries + timestamps | Excluded |
| No-click rate | full impression list per search event | Excluded |
| Abandonment proxy | session termination + timestamp | Excluded |

**Note on first-click position:** a *degenerate*, row-level notion of
"click rank" did exist upstream in the original pipeline (the `ItemRank`
column consumed by `separator.py`), but it was consumed into the binary
relevance label at the time `newMedium.txt` was built and is not retained
per-row in any file present in this repository. It cannot be recomputed
without the raw log. This project does not report first-click-position
statistics.

## What replaces behavioral diagnostics in this project

Because session/click-position/reformulation/abandonment evidence is
unavailable, `docs/SEARCH_QUALITY_DIAGNOSIS.md` instead measures data
sparsity and search-quality degradation using the evidence that IS honestly
present: tail-query pair degree (how much labeled evidence exists per query),
positive-target availability, and retrieval-quality metrics (MRR/Recall@K)
computed directly by this project's own held-out evaluation. This is a
narrower but fully evidenced substitute — see that document for what is and
is not claimed.

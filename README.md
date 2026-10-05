# SnapTriage, GDG KAU AI Engineering Workshop

Screenshot support-ticket triage that calls an LLM only when it has to.

## Run it
1. Copy `.env.example` to `.env` and paste your OpenRouter key.
2. `python snaptriage.py` opens the visualiser at http://localhost:8000
   (or `python snaptriage.py samples/01_session_expired.png` for terminal output)

Python 3.9+, no pip installs needed.

## The pipeline
| Stage | Model | Notes |
|---|---|---|
| 1 Read | cheap Gemini Flash-Lite as OCR | stronger Gemini Flash only if OCR finds < 25 characters |
| 2 Triage | Jev call 1 | category, team, severity; ticket routed at team confidence >= 0.6 |
| 3 Pick a fix | Jev call 2 | chooses one known fix in that category, or "none" |
| 4 Answer | template, or Claude Opus | template if fix confidence >= 0.85, else the top-tier LLM drafts a reply |

Thresholds and model preferences are at the top of `snaptriage.py`. Model ids are checked
against the live OpenRouter catalog at startup, so they keep working when new versions ship.
Approved answers are saved to `known_issues.json` (created on first run).

## Demo script (about 10 minutes)
1. Run samples 01 to 05: each is answered by a template with 0 LLM calls.
2. Run 06 (certificate error): no fix matches, so Claude Opus writes the answer.
   Click "Approve into library", then run 06 again: now it is a template, with no LLM call.
3. Run 07 (blank page): OCR finds almost no text, so the vision fallback kicks in.
4. Keep "Also run the baseline" on and compare cost and time per ticket in the top bar.
5. Change FIX_THRESHOLD live (e.g. 0.95) and watch more tickets go to the LLM.

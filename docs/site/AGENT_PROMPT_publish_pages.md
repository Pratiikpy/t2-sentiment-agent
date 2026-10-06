You are working on my Windows machine. Goal: put the t2-sentiment-agent proof deck and whitepaper
live on my Vercel site https://t2-sentiment-agent-run2.vercel.app at /deck.html and /whitepaper.html,
without disturbing the paper-trading agent that is running right now.

CONTEXT (read before doing anything)
- Project: t2-sentiment-agent (https://github.com/Pratiikpy/t2-sentiment-agent), Bitget hackathon
  Track 2. Run 2 is LIVE: `t2sa go-live` (or scripts\watchdog.ps1) is trading on Bitget Demo, and
  scripts\publish_site.ps1 -Project t2-sentiment-agent-run2 deploys the folder public\ to Vercel
  once an hour (about :10 past the hour), only when a newer record has been exported.
- The run-2 checkout is a separate folder from run 1 (often a git worktree of `run2-prep`). Find it:
  it is the folder whose var\ holds the run-2 paper ledger and whose public\ contains index.html,
  ledger.jsonl and genesis.json with hash starting fb66b6bf. Confirm with:
    Get-Content public\genesis.json | Select-String fb66b6bf
  If you find more than one candidate, STOP and ask me which one.

HARD RULES
1. Do NOT run `git pull`, `git checkout`, `git merge` or change any code in the run-2 folder. Run 2's
   pre-registration pins the code commit; changing it can make a restart refuse to start.
2. Do NOT stop, restart or kill the agent, the watchdog or the publisher.
3. Do NOT edit or delete anything already in public\. Only ADD the two files below.
4. Never print or open .secrets\ or any *.env file.

STEPS
1. cd into the run-2 folder (see CONTEXT). Show me the path and the genesis check output.
2. Download the two ready-built pages into public\:
     iwr https://raw.githubusercontent.com/Pratiikpy/t2-sentiment-agent/master/docs/site/deck.html -OutFile public\deck.html
     iwr https://raw.githubusercontent.com/Pratiikpy/t2-sentiment-agent/master/docs/site/whitepaper.html -OutFile public\whitepaper.html
3. Check them: both files exist, deck.html is about 2 MB and contains "t2-sentiment-agent proof deck",
   whitepaper.html is about 45 KB and contains "t2-sentiment-agent: whitepaper". Neither may contain
   a path like C:\Users\ (the publisher refuses to deploy a folder with a personal path in it):
     Select-String -Path public\deck.html,public\whitepaper.html -Pattern 'C:\\Users' -SimpleMatch
   must print nothing.
4. Wait for the next hourly publish. Find the publisher's log (in the window running
   publish_site.ps1, or under var\logs\) and wait for the next line that says `published`. If the
   line says `not redeployed` or `not published`, tell me the exact line and stop.
5. Verify live (both must return 200):
     (iwr https://t2-sentiment-agent-run2.vercel.app/deck.html -UseBasicParsing).StatusCode
     (iwr https://t2-sentiment-agent-run2.vercel.app/whitepaper.html -UseBasicParsing).StatusCode
   Open both in the browser and confirm the deck shows 16 slides and the whitepaper shows
   "1. Abstract".
6. Report: the folder used, the publish log line, and the two status codes.

IF THE PUBLISHER IS NOT RUNNING
Tell me, and do not start it yourself. I will decide whether to start
`powershell -ExecutionPolicy Bypass -File scripts\publish_site.ps1 -Project t2-sentiment-agent-run2`.

AFTER 2026-10-07 00:00 UTC ONLY (when the scoring window has closed), and only when I say so:
  git pull origin master
  t2sa export --mode paper
  python scripts\recompute.py public
  t2sa verify --mode paper
The export now writes deck.html and whitepaper.html itself, and the record's top navigation links
to both. Then let the publisher deploy, and re-check the two URLs.

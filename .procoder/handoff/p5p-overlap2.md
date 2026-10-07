# Handoff: p5p-overlap2 (decision "5p Task 4" B) — done, awaiting the user's decision

Branch `p5p-overlap2` (p6b-stack merged in at 3f834236): a9f7779a channel after the serial wait; 4a03d787 failed
launch completes under its iteration (fixed the `overlap_failed_steps_fail_their_requests_only` hang); 36edc3bc
the rule "the turn after a request ended completes the iteration in flight serially" + test
`overlap_admits_resubmissions_in_the_next_plan` (red without it). Gate ok. A/B (perf log 5p "S-3: overlap
completes serially the turn after a request ended"): TTFT p50 1.00x on Llama c16 and OLMoE c16, tok/s +1.2 % /
+3.6 %, golden PASS everywhere.

Next: the user decides "5p overlap: serial turn after a finish — new S-3 rule" (A flip default on / B keep off /
C reject). On A: change the `execution.overlap_scheduling` default (config + contract + docs), gate, merge.

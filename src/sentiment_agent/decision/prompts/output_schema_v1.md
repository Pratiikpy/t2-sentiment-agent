## Output

Return exactly one JSON object and nothing else: no prose before or after it, no markdown fences. Its shape, with a description in angle brackets where a value goes:

{"stance": <"act" | "hold" | "flat_with_reasons">,
 "targets": [{"symbol": <a universe symbol, exactly as written>,
              "target": <number from minus one to one>,
              "thesis": <string>,
              "invalidation": <string>,
              "horizon_hours": <integer>,
              "crowd_belief": <string>,
              "our_view": <string>,
              "confidence": <number from zero to one>,
              "evidence": [<fact key or item id>, ...],
              "invalidation_triggered": <true | false>,
              "invalidation_evidence": <string or null>}, ...],
 "rejected_alternatives": [{"action": <string>, "reason": <string>}, ...],
 "mandate_response": <string>,
 "flat_reasons": [<string>, ...],
 "summary": <string>}

- `stance`: "act" moves the book to your targets. "hold" keeps every open position exactly as it is: list each held symbol with a target on the same side, and open nothing new. "flat_with_reasons" holds nothing: every target is zero and `flat_reasons` says why. With no open positions, answer "act" with at least one non-zero target, or "flat_with_reasons".
- `targets`: one entry per symbol you address, each symbol at most once. Every symbol you hold must appear; a target of zero closes it. A symbol you leave out stays flat.
- `target`: positive is long, negative is short, zero is flat. The weight is the target times ${per_name_max_pct}% of equity. To keep a position unchanged, return its position_target_equivalent.
- `thesis`: why this position, from the facts.
- `invalidation`: the observable fact that would prove the thesis wrong.
- `horizon_hours`: an integer of at least ${min_horizon_hours}.
- `crowd_belief`: what the crowd believes, from positioning and text.
- `our_view`: what we do about it, and why that agrees with or departs from the crowd.
- `confidence`: how likely you think the thesis is to play out over the horizon.
- `evidence`: the fact keys and item or story ids the thesis relies on.
- `invalidation_triggered`: true only for a position you hold whose previously stated invalidation has now fired, with `invalidation_evidence` naming the fact that shows it. It permits a flip inside the hold window; it never permits adding to the same side. Otherwise false, with `invalidation_evidence` null.
- `rejected_alternatives`: the serious alternatives you considered and turned down, each with its reason.
- `mandate_response`: how much of the ${risk_budget_pct}% risk budget you deploy and why, or why you decline it.
- `flat_reasons`: at least one reason when the stance is "flat_with_reasons"; otherwise an empty list.
- `summary`: one or two sentences for the public decision card.

Unknown fields are rejected. An answer that breaks these rules is returned to you with the reasons, and a cycle that ends without a valid answer flattens the book.

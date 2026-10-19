# Limitations

- **Toy demo apps.** `demo_rag` and `demo_agent` are deterministic and rule-based. They show how the harness behaves;
  they say nothing about real LLM systems.
- **Small, author-written test sets.** 13 and 8 cases, written by me. With n around 10 the bootstrap intervals are
  wide (see the README comparison). A drop of 0.18 cannot be told from noise. Use 100+ cases before trusting a
  threshold near the baseline.
- **Judge scorers are unvalidated.** The faithfulness and relevance scorers have only been tested with scripted
  judges. Agreement with human labels has not been measured yet (see `docs/RELIABILITY.md`). Do not gate a release
  on them until it has.
- **Judges favour their own style and can be fooled.** The judge prompts say to treat the judged text as data, but
  that is an instruction, not a guarantee.
- **Token totals only, no price table.** Judge and target token counts are reported; converting them to money is
  left to the user because prices change.
- **Refusal detection is pattern-based.** The safety scorer checks for common refusal phrases. A refusal worded
  differently is scored as a failure; a non-answer containing "cannot" passes. Add `forbidden_strings` for
  stricter checks.
- **Tool trace is names only.** Arguments are carried in the output but not compared.
- **Single process, no concurrency.** Cases run one after another.
- **No human-review workflow** beyond the label export used for the reliability study.

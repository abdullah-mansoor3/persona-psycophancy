# Member 1 AI-Agent Context — Data

Read `shared_agent_context.md`.

Own only `src/data_pipeline/**`, `tests/data_pipeline/**`, audit notebook, and agreed data docs. Do not edit prediction or agent internals.

First inspect official Twin-2K-500 configs, actual fields/types/IDs, question text/options, wave/item mapping, variants, missingness, and participant identifiers. Record facts before coding.

Export the files in `context/06_shared_contracts.md`. Keep target and Wave 4 answers out of persona profiles. Use participant-disjoint splits. Exclude focal participant from peer candidates. Match exact item and condition. Preserve source IDs and original labels. Add leakage, split, and mapping tests. Do not invent normalization semantics.

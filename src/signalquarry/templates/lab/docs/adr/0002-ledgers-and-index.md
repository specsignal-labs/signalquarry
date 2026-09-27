# 0002. Per-family ledgers chained into a project index

Status: accepted

Decision: `[evidence] layout = "per_family"`. Each family's appends are mirrored in
`evidence/project_index.jsonl`; the DSR denominator counts every family. A family log that
disagrees with the index fails `sqy evidence verify`.

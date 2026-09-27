# 0001. One repository, isolated families

Status: accepted

Decision: all families share one repository, lock file and trial count, but each lives in
`families/<family>/` with its own package, ledgers and publication policy; commits touch
one family; families never import each other. Consequence: a family can be extracted
with `git filter-repo` and still verifies on its own.

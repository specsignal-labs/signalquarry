# 0003. Paper runners on a dedicated host

Status: accepted

Decision: a small dedicated VM; one unix user, one paper account and one set of keys per
deployment; decide in an isolated process; deploys only outside market hours ±30 minutes
and never while configured markers of other systems exist. Consequence: a failure in one
deployment cannot reach another system's accounts or services.

# Security policy

Report vulnerabilities privately through GitHub's **Report a vulnerability**
button (private vulnerability reporting) on this repository. Please do not open
public issues for security problems.

In scope: anything that could leak credentials, submit orders a human did not
arm, reach a non-paper broker endpoint, bypass the trial ledger, holdout seal or
arm token, or execute code during `sqy check` beyond the strategy under test.

You will get an acknowledgement within 7 days. Fixes are released as patch
versions and credited unless you prefer otherwise. Only the latest minor
version receives security fixes during 0.x.

SignalQuarry sends no telemetry. It contacts only the Alpaca endpoints you
configure, with your own keys.

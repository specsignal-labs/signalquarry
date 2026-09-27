# SPDX-License-Identifier: Apache-2.0
"""Paper-trading kernel: lease, journal, arm token, reconcile-before-decide, brokers.

No live-trading path exists. Every broker declares ``paper_only = True`` and the
Alpaca adapter hard-codes the paper origin.
"""

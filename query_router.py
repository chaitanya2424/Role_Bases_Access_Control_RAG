"""
query_router.py
===============
Classifies the user's query into a structured intent and routes it
to the appropriate data sources — before RBAC checks and retrieval.

Two-stage process:
  1. Rule-based keyword classification (fast, zero-latency)
  2. Falls back to "general" if no specific intent matched

Intent taxonomy (must match RBAC policy intents):
  hr_policy       → leave, conduct, performance, benefits
  leave_info      → leave, vacation, holiday, PTO
  salary_info     → salary, pay, compensation, CTC, payroll
  payroll_data    → payroll, deductions, net pay, HRA, PF
  department_report → reports, metrics, KPIs, team performance
  security_report → security, incidents, vulnerabilities, breaches
  audit_data      → audit, logs, access logs, system events
  financial_data  → finance, budget, revenue, P&L
  remote_work     → remote, WFH, hybrid, work from home
  performance_info → appraisal, rating, increment, review
  conduct_info    → code of conduct, behaviour, ethics, bribery
  general         → anything else
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RouteResult:
    """
    Result of query routing.

    intent         : Classified intent string (maps to RBAC intent keys)
    target_sources : Ordered list of source keys to search first
    confidence     : Router confidence 0.0–1.0
    matched_keywords : Keywords that triggered this routing decision
    query_type     : "specific" or "general"
    """
    intent:           str
    target_sources:   List[str]
    confidence:       float
    matched_keywords: List[str] = field(default_factory=list)
    query_type:       str = "general"
    explanation:      str = ""


# ════════════════════════════════════════════════════════════════════════════
# INTENT → SOURCE MAPPING
# ════════════════════════════════════════════════════════════════════════════

# Maps each intent to the data sources it should search (priority order)
INTENT_SOURCE_MAP: Dict[str, List[str]] = {
    "hr_policy":         ["leave_policy", "code_of_conduct", "performance_review_policy",
                          "remote_work_policy", "data_privacy_policy"],
    "leave_info":        ["leave_policy"],
    "salary_info":       ["payroll", "employees"],
    "payroll_data":      ["payroll"],
    "department_report": ["employees", "q1_security_report"],
    "security_report":   ["q1_security_report", "information_security_policy"],
    "audit_data":        ["audit_logs"],
    "financial_data":    ["payroll", "employees"],
    "remote_work":       ["remote_work_policy"],
    "performance_info":  ["performance_review_policy"],
    "conduct_info":      ["code_of_conduct", "data_privacy_policy"],
    "compliance_info":   ["data_privacy_policy", "information_security_policy"],
    "employee_info":     ["employees"],
    "general":           ["leave_policy", "code_of_conduct", "performance_review_policy",
                          "remote_work_policy", "data_privacy_policy",
                          "information_security_policy", "employees"],
}


# ════════════════════════════════════════════════════════════════════════════
# KEYWORD RULES
# Each rule: (intent, [keyword_patterns], confidence)
# Patterns are regex-like but simple — case-insensitive substring match.
# Rules are evaluated in order; first match wins (most specific first).
# ════════════════════════════════════════════════════════════════════════════

ROUTING_RULES: List[Tuple[str, List[str], float]] = [
    # ── High-confidence specific intents ──────────────────────────────────────
    ("salary_info",    ["salary", "salaries", "pay slip", "payslip", "ctc",
                        "compensation", "earning", "take home", "wage",
                        "show.*salary", "salary information"],                  0.95),

    ("payroll_data",   ["payroll", "net pay", "gross pay", "deduction", "pf ",
                        "provident fund", "tds ", "hra ", "tax deducted"],      0.95),

    ("leave_info",     ["leave", "vacation", "holiday", "pto", "annual leave",
                        "sick leave", "maternity", "paternity", "bereavement",
                        "time off", "days off"],                                0.90),

    ("remote_work",    ["remote", "wfh", "work from home", "hybrid", "home office",
                        "work remotely", "remote work policy"],                 0.90),

    ("performance_info", ["appraisal", "performance review", "performance rating",
                          "increment", "annual review", "pip ", "performance plan",
                          "rating scale", "self.?assessment"],                  0.90),

    ("conduct_info",   ["code of conduct", "misconduct", "ethics", "bribery",
                        "anti.?corruption", "whistleblower", "discipline",
                        "harassment", "conflict of interest"],                  0.88),

    ("audit_data",     ["audit log", "access log", "system log", "login log",
                        "user activity", "event log", "system event"],          0.90),

    ("security_report",["security incident", "security report", "breach",
                        "vulnerability", "phishing", "malware", "cyber",
                        "incident report", "security metrics"],                 0.90),

    ("financial_data", ["revenue", "profit", "budget", "p&l", "balance sheet",
                        "financial report", "quarterly earnings"],              0.88),

    ("department_report", ["department report", "team report", "team metrics",
                           "department metrics", "team performance"],           0.85),

    ("compliance_info",["gdpr", "pdpa", "data privacy", "data protection",
                        "compliance policy", "regulatory", "data breach",
                        "data retention"],                                      0.88),

    ("employee_info",  ["employee list", "headcount", "org chart", "staff",
                        "employee database", "employee record"],                0.82),

    # ── Broad HR policy catch-all ──────────────────────────────────────────────
    ("hr_policy",      ["policy", "policies", "hr policy", "company policy",
                        "employee handbook", "work policy", "benefit"],         0.75),
]


# ════════════════════════════════════════════════════════════════════════════
# QUERY ROUTER
# ════════════════════════════════════════════════════════════════════════════

class QueryRouter:
    """
    Classifies a natural-language query into an intent and maps it
    to the appropriate data sources for retrieval.
    """

    def __init__(self):
        print("[✓] QueryRouter initialised with rule-based classifier.")

    def route(self, query: str) -> RouteResult:
        """
        Main routing entry point.

        1. Normalises the query
        2. Applies keyword rules (most specific first)
        3. Falls back to "general" if nothing matches

        Returns a RouteResult with intent, sources, and confidence.
        """
        normalised = query.lower().strip()
        normalised = re.sub(r"\s+", " ", normalised)

        # Evaluate each rule
        for intent, keywords, confidence in ROUTING_RULES:
            matched = self._match_keywords(normalised, keywords)
            if matched:
                sources = INTENT_SOURCE_MAP.get(intent, INTENT_SOURCE_MAP["general"])
                return RouteResult(
                    intent=intent,
                    target_sources=sources,
                    confidence=confidence,
                    matched_keywords=matched,
                    query_type="specific",
                    explanation=f"Matched keywords: {', '.join(matched)} → intent: {intent}",
                )

        # No match → general
        return RouteResult(
            intent="general",
            target_sources=INTENT_SOURCE_MAP["general"],
            confidence=0.50,
            matched_keywords=[],
            query_type="general",
            explanation="No specific intent matched. Routing to general knowledge base.",
        )

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _match_keywords(self, text: str, patterns: List[str]) -> List[str]:
        """Returns list of matched keyword patterns for the given text."""
        matched = []
        for pattern in patterns:
            # Support simple regex (e.g. "show.*salary")
            try:
                if re.search(pattern, text):
                    matched.append(pattern)
            except re.error:
                if pattern in text:
                    matched.append(pattern)
        return matched

    def explain(self, query: str) -> str:
        """Returns a human-readable explanation of routing decision."""
        result = self.route(query)
        lines = [
            f"**Query:** {query}",
            f"**Intent:** `{result.intent}`",
            f"**Confidence:** {result.confidence:.0%}",
            f"**Target Sources:** {', '.join(result.target_sources)}",
            f"**Matched Keywords:** {', '.join(result.matched_keywords) or 'none'}",
            f"**Explanation:** {result.explanation}",
        ]
        return "\n".join(lines)

    def get_all_intents(self) -> List[str]:
        """Returns all known intent strings."""
        return list(INTENT_SOURCE_MAP.keys())


# ════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    router = QueryRouter()

    queries = [
        "What is the leave policy?",
        "Show salary information for all employees",
        "Can I work from home on Mondays?",
        "What happened in the Q1 security incident?",
        "How is the performance review conducted?",
        "What is the code of conduct regarding gifts?",
        "Show me the audit logs for March",
        "What is the GDPR data retention policy?",
        "Tell me about the company",
    ]

    print("\n" + "="*70)
    for q in queries:
        result = router.route(q)
        print(f"Q: {q}")
        print(f"   → intent={result.intent:20s} confidence={result.confidence:.0%} "
              f"sources={result.target_sources[:2]}")
    print("="*70)

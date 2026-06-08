"""
rbac.py
=======
Role-Based Access Control (RBAC) enforcement layer.

Responsibilities:
  1. Load user → role mappings from disk
  2. Load source → permission policies from disk
  3. Validate whether a (user, intent, source) triple is permitted
  4. Return structured AccessDecision objects with audit trail
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class UserContext:
    """Represents the authenticated user making a request."""
    username:    str
    role:        str               # "employee" | "manager" | "admin"
    employee_id: str = ""
    department:  str = ""


@dataclass
class AccessDecision:
    """
    Result of an RBAC check.

    allowed          : True if access is granted
    reason           : Human-readable explanation
    denied_sources   : List of sources that were filtered out
    allowed_sources  : List of sources the user may access (["*"] for admin)
    audit_entry      : Dict for audit logging
    """
    allowed:         bool
    reason:          str
    denied_sources:  List[str] = field(default_factory=list)
    allowed_sources: List[str] = field(default_factory=list)
    audit_entry:     Dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# RBAC MANAGER
# ════════════════════════════════════════════════════════════════════════════

class RBACManager:
    """
    Loads and enforces access control policies.

    Policy files:
      - data/metadata/user_roles.json       : username → role/dept
      - data/policies/access_control_policies.json : role → allowed/denied sources & intents
    """

    def __init__(
        self,
        user_roles_path: str  = "data/metadata/user_roles.json",
        policy_path:     str  = "data/policies/access_control_policies.json",
    ):
        self.user_roles: Dict[str, Any] = {}
        self.policies:   Dict[str, Any] = {}
        self._load_policies(user_roles_path, policy_path)

    # ── LOADING ───────────────────────────────────────────────────────────────

    def _load_policies(self, user_roles_path: str, policy_path: str) -> None:
        """Loads both policy files from disk."""
        # User-role mappings
        if os.path.exists(user_roles_path):
            with open(user_roles_path) as f:
                data = json.load(f)
                self.user_roles = data.get("users", {})
            print(f"[✓] RBAC: Loaded {len(self.user_roles)} user-role mappings.")
        else:
            print(f"[!] RBAC: User roles file not found at {user_roles_path}")

        # Access control policies
        if os.path.exists(policy_path):
            with open(policy_path) as f:
                data = json.load(f)
                self.policies = data.get("roles", {})
            print(f"[✓] RBAC: Loaded policies for roles: {list(self.policies.keys())}")
        else:
            print(f"[!] RBAC: Policy file not found at {policy_path}")

    # ── USER RESOLUTION ───────────────────────────────────────────────────────

    def get_user_context(self, username: str) -> Optional[UserContext]:
        """
        Resolves a username to a UserContext object.
        Returns None if the user is not found in the role mapping.
        """
        info = self.user_roles.get(username)
        if not info:
            return None
        return UserContext(
            username=username,
            role=info.get("role", "employee"),
            employee_id=info.get("employee_id", ""),
            department=info.get("department", ""),
        )

    def get_role_info(self, role: str) -> Dict[str, Any]:
        """Returns the full policy dict for a role."""
        return self.policies.get(role, {})

    # ── ALLOWED SOURCES ───────────────────────────────────────────────────────

    def get_allowed_sources(self, role: str) -> List[str]:
        """
        Returns the list of data sources a role may access.
        Admin → ["*"] meaning unrestricted.
        """
        role_policy = self.policies.get(role, {})
        allowed = role_policy.get("allowed_sources", [])
        return allowed  # ["*"] for admin

    def get_denied_sources(self, role: str) -> List[str]:
        """Returns sources explicitly denied for a role."""
        role_policy = self.policies.get(role, {})
        return role_policy.get("denied_sources", [])

    # ── INTENT VALIDATION ─────────────────────────────────────────────────────

    def is_intent_allowed(self, role: str, intent: str) -> bool:
        """
        Checks if the query intent is permitted for the given role.
        """
        role_policy = self.policies.get(role, {})
        allowed_intents = role_policy.get("allowed_query_intents", [])
        denied_intents  = role_policy.get("denied_query_intents", [])

        # Admin wildcard
        if "*" in allowed_intents:
            return True

        # Explicit deny overrides allow
        if intent in denied_intents:
            return False

        return intent in allowed_intents or "general" in allowed_intents

    # ── MAIN ACCESS DECISION ──────────────────────────────────────────────────

    def check_access(
        self,
        user_context: UserContext,
        intent: str,
        requested_sources: Optional[List[str]] = None,
    ) -> AccessDecision:
        """
        Core RBAC decision point.

        Parameters
        ----------
        user_context       : The authenticated user
        intent             : Classified query intent (e.g. "salary_info")
        requested_sources  : (optional) specific sources the query targets

        Returns
        -------
        AccessDecision with allowed=True/False and full audit info.
        """
        role = user_context.role

        # ── 1. Check intent permission ────────────────────────────────────────
        if not self.is_intent_allowed(role, intent):
            decision = AccessDecision(
                allowed=False,
                reason=(
                    f"🔒 **Access Denied** — Your role (`{role}`) does not have permission "
                    f"to access `{intent}` information.\n\n"
                    f"If you believe this is an error, please contact your HR Business Partner "
                    f"or System Administrator."
                ),
                denied_sources=self.get_denied_sources(role),
                allowed_sources=self.get_allowed_sources(role),
            )
            decision.audit_entry = self._build_audit_entry(
                user_context, intent, "DENIED", "Intent not permitted for role"
            )
            return decision

        # ── 2. Resolve allowed sources for this role ───────────────────────────
        allowed_sources = self.get_allowed_sources(role)  # ["*"] for admin
        denied_sources  = self.get_denied_sources(role)

        # ── 3. Check specific source access (if caller specified sources) ──────
        if requested_sources and "*" not in allowed_sources:
            blocked = [s for s in requested_sources if s in denied_sources
                       or s not in allowed_sources]
            if blocked:
                decision = AccessDecision(
                    allowed=False,
                    reason=(
                        f"🔒 **Access Denied** — You do not have permission to access "
                        f"the following data source(s): `{', '.join(blocked)}`.\n\n"
                        f"Your role (`{role}`) is restricted from this information."
                    ),
                    denied_sources=blocked,
                    allowed_sources=allowed_sources,
                )
                decision.audit_entry = self._build_audit_entry(
                    user_context, intent, "DENIED",
                    f"Blocked sources: {blocked}"
                )
                return decision

        # ── 4. Access granted ─────────────────────────────────────────────────
        decision = AccessDecision(
            allowed=True,
            reason=f"Access granted for role `{role}` with intent `{intent}`.",
            denied_sources=denied_sources,
            allowed_sources=allowed_sources,
        )
        decision.audit_entry = self._build_audit_entry(
            user_context, intent, "GRANTED", "All checks passed"
        )
        return decision

    # ── DOCUMENT FILTER ───────────────────────────────────────────────────────

    def filter_documents(
        self,
        documents: list,
        user_context: UserContext,
    ) -> list:
        """
        Post-retrieval filter: removes any retrieved documents the user
        shouldn't see, even if they slipped through the vector search filter.

        This is a defence-in-depth measure.
        """
        allowed = self.get_allowed_sources(user_context.role)
        denied  = self.get_denied_sources(user_context.role)

        if "*" in allowed:
            return documents  # admin sees all

        filtered = []
        for doc in documents:
            source = doc.metadata.get("source", "")
            if source in denied:
                continue   # silently drop denied docs
            if source in allowed:
                filtered.append(doc)

        return filtered

    # ── AUDIT ─────────────────────────────────────────────────────────────────

    def _build_audit_entry(
        self,
        user: UserContext,
        intent: str,
        decision: str,
        detail: str,
    ) -> Dict[str, Any]:
        """Builds a structured audit log entry."""
        from datetime import datetime
        return {
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "username":    user.username,
            "employee_id": user.employee_id,
            "role":        user.role,
            "intent":      intent,
            "decision":    decision,
            "detail":      detail,
        }


# ════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rbac = RBACManager()

    # Test employee trying to access salary
    user = rbac.get_user_context("employee_demo")
    print(f"\nUser: {user}")

    decision = rbac.check_access(user, "salary_info")
    print(f"Salary query → allowed={decision.allowed}")
    print(f"  Reason: {decision.reason}")

    decision = rbac.check_access(user, "hr_policy")
    print(f"\nLeave policy query → allowed={decision.allowed}")
    print(f"  Reason: {decision.reason}")

    # Test admin
    admin = rbac.get_user_context("admin_demo")
    decision = rbac.check_access(admin, "salary_info")
    print(f"\nAdmin salary query → allowed={decision.allowed}")

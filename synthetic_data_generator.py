"""
synthetic_data_generator.py
============================
Generates synthetic sample data needed by the RAG system:
- HR Policy documents (.txt simulating PDFs)
- Employee & Payroll CSV databases
- Compliance documents
- Security reports
- JSON audit logs
- User-role mappings
- Access control policies
"""

import os
import json
import csv
import random
from datetime import datetime, timedelta

# -- Output directories ------------------------------------------------------
DOCS_DIR    = "data/documents"
CSV_DIR     = "data/csv"
JSON_DIR    = "data/json"
META_DIR    = "data/metadata"
POLICY_DIR  = "data/policies"

for d in [DOCS_DIR, CSV_DIR, JSON_DIR, META_DIR, POLICY_DIR]:
    os.makedirs(d, exist_ok=True)


# 
# 1. HR POLICY DOCUMENTS
# 

def generate_hr_policies():
    """Creates realistic HR policy text files."""

    # Leave Policy
    with open(f"{DOCS_DIR}/leave_policy.txt", "w") as f:
        f.write("""DemoCo — LEAVE POLICY
Document ID: HR-POL-001 | Version: 3.2 | Effective: January 1, 2024
Classification: General — All Employees

1. ANNUAL LEAVE
   Employees are entitled to 21 working days of paid annual leave per calendar year.
   Leave accrues at 1.75 days per month from the date of joining.
   Unused leave up to 10 days may be carried forward to the next calendar year.
   Carry-forward leave must be consumed by March 31 of the following year.

2. SICK LEAVE
   All employees are entitled to 12 days of paid sick leave per year.
   Sick leave does not carry forward to the next year.
   Medical certificate required for sick leave exceeding 3 consecutive days.
   Sick leave cannot be encashed.

3. MATERNITY / PATERNITY LEAVE
   Maternity Leave: 26 weeks of paid leave for the first two children.
   Paternity Leave: 10 working days of paid leave within 6 months of childbirth.
   Adoption Leave: 12 weeks for primary caregiver; 5 days for secondary caregiver.

4. BEREAVEMENT LEAVE
   Immediate family (spouse, child, parent): 5 working days paid.
   Extended family (sibling, grandparent, in-law): 3 working days paid.

5. LEAVE APPLICATION PROCESS
   Annual leave must be applied minimum 7 days in advance via the HR portal.
   Emergency leave requests can be submitted same-day with manager approval.
   Leave approval is at the discretion of the reporting manager.
   HR Business Partner must be notified for leave exceeding 15 consecutive days.

6. LEAVE ENCASHMENT
   Up to 30 days of accumulated leave can be encashed at the time of resignation.
   Leave encashment is calculated on the basis of last drawn basic salary.
   Encashment is subject to applicable income tax deductions.

7. CONTACT
   For leave-related queries: hr@democo.com | Ext: 1001
""")

    # Code of Conduct
    with open(f"{DOCS_DIR}/code_of_conduct.txt", "w") as f:
        f.write("""DemoCo — CODE OF CONDUCT
Document ID: HR-POL-002 | Version: 2.1 | Effective: January 1, 2024
Classification: General — All Employees

1. PROFESSIONAL BEHAVIOR
   All employees are expected to maintain the highest standards of professional conduct.
   Respect for colleagues, clients, and vendors is mandatory at all times.
   Discrimination, harassment, or bullying of any form will result in immediate disciplinary action.

2. CONFLICT OF INTEREST
   Employees must disclose any personal or financial interests that may conflict with company interests.
   Engaging in competing business activities without written approval is strictly prohibited.
   Gifts exceeding ₹5,000 in value from vendors or clients must be disclosed to the Ethics Committee.

3. CONFIDENTIALITY
   All business information, strategies, client data, and intellectual property are confidential.
   Employees must sign the Non-Disclosure Agreement (NDA) upon joining.
   Sharing confidential information on social media or with unauthorized parties is a terminable offense.

4. USE OF COMPANY RESOURCES
   Company devices, internet, and tools must be used primarily for business purposes.
   Reasonable personal use is permitted, but accessing inappropriate content is prohibited.
   Company assets must be returned in good condition upon separation.

5. ANTI-BRIBERY & CORRUPTION
   Offering or accepting bribes in any form is strictly prohibited and may result in legal action.
   Facilitation payments are not permitted under any circumstances.
   Suspected bribery must be reported to the Compliance Team immediately.

6. DISCIPLINARY PROCESS
   Violations of the Code of Conduct will be investigated by HR.
   Penalties range from verbal warning to termination, depending on severity.
   Employees have the right to representation during disciplinary proceedings.

7. WHISTLEBLOWER PROTECTION
   Employees reporting violations in good faith are protected from retaliation.
   Anonymous reporting available at: ethics@democo.com
""")

    # Performance Review Policy
    with open(f"{DOCS_DIR}/performance_review_policy.txt", "w") as f:
        f.write("""DemoCo — PERFORMANCE REVIEW POLICY
Document ID: HR-POL-003 | Version: 1.8 | Effective: January 1, 2024
Classification: General — All Employees

1. REVIEW CYCLE
   Annual Performance Reviews are conducted in December for the January-December cycle.
   Mid-year check-ins are mandatory and occur in June.
   New joinees are reviewed at 3-month and 6-month probation milestones.

2. RATING SCALE
   5 — Exceptional: Consistently exceeds all performance expectations.
   4 — Exceeds Expectations: Regularly surpasses key performance indicators.
   3 — Meets Expectations: Reliably meets all assigned targets and responsibilities.
   2 — Needs Improvement: Partially meets expectations; improvement plan required.
   1 — Unsatisfactory: Fails to meet core responsibilities; may lead to PIP or separation.

3. PERFORMANCE IMPROVEMENT PLAN (PIP)
   Employees rated 2 or below for two consecutive cycles are placed on a PIP.
   PIP duration: 60-90 days with fortnightly HR review meetings.
   Successful completion of PIP leads to reassessment; failure may lead to separation.

4. INCREMENTS & BONUSES
   Annual increment eligibility begins after 6 months of service.
   Increment percentages are linked to performance ratings (see HR-FIN-007).
   Performance bonuses are disbursed in January for the previous year.

5. SELF-ASSESSMENT
   Employees must complete self-assessment forms by November 30 each year.
   Self-assessment is a critical input to the review discussion; omission may delay review.

6. APPEALS
   Employees may appeal their performance rating within 15 days of receiving it.
   Appeals are reviewed by the Skip-Level Manager and HR Business Partner.

7. CONTACT
   Performance review queries: performance@democo.com
""")

    # Remote Work Policy
    with open(f"{DOCS_DIR}/remote_work_policy.txt", "w") as f:
        f.write("""DemoCo — REMOTE WORK POLICY
Document ID: HR-POL-004 | Version: 1.3 | Effective: March 1, 2024
Classification: General — All Employees

1. ELIGIBILITY
   Employees who have completed 6 months of service are eligible for hybrid/remote work.
   Remote work is subject to role suitability as determined by the reporting manager and HR.
   Employees on PIP are not eligible for remote work arrangements.

2. HYBRID WORK MODEL
   Default hybrid schedule: 3 days office, 2 days remote per week.
   Mandatory office days: Tuesday and Thursday (team collaboration days).
   Exceptions require written approval from the Department Head.

3. HOME OFFICE REQUIREMENTS
   Employees must have a dedicated, secure, and distraction-free workspace.
   Stable internet connection with minimum 20 Mbps speed is mandatory.
   Use of VPN is compulsory when accessing company systems remotely.
   Employees must be available during core hours: 10:00 AM - 5:00 PM (IST).

4. EQUIPMENT & EXPENSES
   The company provides a laptop and necessary peripherals for remote work.
   Internet allowance of ₹1,000/month is reimbursable with manager approval.
   Personal devices must not be used to store confidential company data.

5. SECURITY OBLIGATIONS
   VPN must be active at all times while accessing internal systems.
   Screen lock must be enabled and auto-activates after 5 minutes of inactivity.
   Confidential discussions must not be conducted in public spaces.

6. MONITORING & COMPLIANCE
   Employees are expected to log hours via the HR portal on remote days.
   Managers may request office presence on remote days with 24-hour notice for business needs.

7. CONTACT
   Remote work requests: hybrid@democo.com
""")

    print("[✔] HR Policy documents generated.")


# 
# 2. EMPLOYEE & PAYROLL CSV DATABASES
# 

DEPARTMENTS = ["Engineering", "Finance", "HR", "Sales", "Marketing", "Legal", "Operations"]
DESIGNATIONS = {
    "Engineering": ["Software Engineer", "Senior Engineer", "Tech Lead", "Engineering Manager"],
    "Finance":     ["Analyst", "Senior Analyst", "Finance Manager", "CFO"],
    "HR":          ["HR Executive", "HR Manager", "HR Business Partner", "CHRO"],
    "Sales":       ["Sales Executive", "Account Manager", "Sales Manager", "VP Sales"],
    "Marketing":   ["Marketing Executive", "Content Strategist", "Marketing Manager", "CMO"],
    "Legal":       ["Legal Executive", "Senior Counsel", "Legal Manager", "CLO"],
    "Operations":  ["Operations Executive", "Process Lead", "Operations Manager", "COO"],
}

FIRST_NAMES = ["Arjun","Priya","Rahul","Sneha","Vikram","Ananya","Rohan","Nisha","Amit","Kavya",
               "Sanjay","Deepa","Arun","Meera","Kiran","Pooja","Suresh","Lakshmi","Nikhil","Divya"]
LAST_NAMES  = ["Sharma","Patel","Reddy","Nair","Iyer","Gupta","Mehta","Singh","Kumar","Rao"]

def generate_employee_database():
    """Creates employee master data CSV."""
    employees = []
    emp_id = 1001
    join_date = datetime(2019, 1, 1)

    for _ in range(30):
        dept = random.choice(DEPARTMENTS)
        designation = random.choice(DESIGNATIONS[dept])
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        jd = join_date + timedelta(days=random.randint(0, 1800))
        employees.append({
            "employee_id":   f"EMP{emp_id}",
            "name":          name,
            "email":         f"{name.lower().replace(' ', '.')}@democo.com",
            "department":    dept,
            "designation":   designation,
            "location":      random.choice(["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Pune"]),
            "join_date":     jd.strftime("%Y-%m-%d"),
            "manager_id":    f"EMP{random.randint(1001, 1005)}",
            "status":        "Active",
        })
        emp_id += 1

    path = f"{CSV_DIR}/employees.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=employees[0].keys())
        writer.writeheader()
        writer.writerows(employees)

    print(f"[✔] Employee database -> {path} ({len(employees)} records)")
    return employees


def generate_payroll_database(employees):
    """Creates confidential payroll data CSV (Admin-only access)."""
    salary_map = {
        "Software Engineer": 850000, "Senior Engineer": 1400000, "Tech Lead": 1800000,
        "Engineering Manager": 2400000, "Analyst": 700000, "Senior Analyst": 1100000,
        "Finance Manager": 1900000, "CFO": 4500000, "HR Executive": 650000,
        "HR Manager": 1600000, "HR Business Partner": 2000000, "CHRO": 4000000,
        "Sales Executive": 600000, "Account Manager": 1200000, "Sales Manager": 2100000,
        "VP Sales": 3800000, "Marketing Executive": 700000, "Content Strategist": 900000,
        "Marketing Manager": 1700000, "CMO": 4200000, "Legal Executive": 800000,
        "Senior Counsel": 1600000, "Legal Manager": 2200000, "CLO": 4100000,
        "Operations Executive": 650000, "Process Lead": 1100000, "Operations Manager": 1900000,
        "COO": 4300000,
    }

    payroll = []
    for emp in employees:
        base = salary_map.get(emp["designation"], 900000)
        base += random.randint(-50000, 100000)
        hra      = round(base * 0.40)
        convey   = 19200
        medical  = 15000
        gross    = base + hra + convey + medical
        pf       = round(base * 0.12)
        tds      = round(gross * 0.10)
        net      = gross - pf - tds

        payroll.append({
            "employee_id":    emp["employee_id"],
            "name":           emp["name"],
            "department":     emp["department"],
            "designation":    emp["designation"],
            "basic_salary":   base,
            "hra":            hra,
            "conveyance":     convey,
            "medical":        medical,
            "gross_salary":   gross,
            "pf_deduction":   pf,
            "tds_deduction":  tds,
            "net_salary":     net,
            "bank_account":   f"HDFC{random.randint(10000000000, 99999999999)}",
            "pan_number":     f"ABCDE{random.randint(1000, 9999)}F",
        })

    path = f"{CSV_DIR}/payroll.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=payroll[0].keys())
        writer.writeheader()
        writer.writerows(payroll)

    print(f"[✔] Payroll database -> {path} ({len(payroll)} records)")


# 
# 3. COMPLIANCE DOCUMENTS
# 

def generate_compliance_docs():
    with open(f"{DOCS_DIR}/data_privacy_policy.txt", "w") as f:
        f.write("""DemoCo — DATA PRIVACY & GDPR COMPLIANCE POLICY
Document ID: COMP-001 | Version: 2.0 | Effective: May 25, 2023
Classification: Internal — All Employees

1. SCOPE & PURPOSE
   This policy governs the collection, processing, storage, and sharing of personal data
   across all DemoCo operations in compliance with GDPR, PDPA, and IT Act 2000.

2. DATA CLASSIFICATION
   Tier 1 (Public): Marketing materials, press releases, published reports.
   Tier 2 (Internal): Policies, procedures, internal communications.
   Tier 3 (Confidential): Employee data, financial data, client contracts.
   Tier 4 (Restricted): Payroll, medical records, source code, trade secrets.

3. DATA COLLECTION PRINCIPLES
   Data must be collected for specified, explicit, and legitimate purposes only.
   Consent must be obtained before collecting personal data from customers or prospects.
   Minimum necessary data must be collected (data minimization principle).

4. DATA RETENTION
   Customer data: 7 years post-contract termination.
   Employee data: 5 years post-separation.
   Financial records: 10 years (statutory requirement).
   Logs and audit trails: 2 years.

5. DATA SUBJECT RIGHTS
   Right to Access: Individuals may request a copy of their personal data within 30 days.
   Right to Rectification: Incorrect data must be corrected within 15 days of request.
   Right to Erasure: Data deletion requests processed within 30 days (subject to legal holds).
   Right to Portability: Data provided in machine-readable format on request.

6. DATA BREACH RESPONSE
   Breaches must be reported to the DPO within 24 hours of discovery.
   Regulatory authorities notified within 72 hours if breach affects personal data.
   Affected individuals notified within 7 days if breach poses high risk.

7. PENALTIES
   Non-compliance may result in fines up to €20 million or 4% of global annual turnover.
   Individual employees may face disciplinary action up to and including termination.

8. CONTACT
   Data Protection Officer: dpo@democo.com | Ext: 2020
""")

    with open(f"{DOCS_DIR}/information_security_policy.txt", "w") as f:
        f.write("""DemoCo — INFORMATION SECURITY POLICY
Document ID: COMP-002 | Version: 3.5 | Effective: January 1, 2024
Classification: Internal — All Employees

1. PURPOSE
   To protect the confidentiality, integrity, and availability of DemoCo's information assets.

2. ACCESS CONTROL
   Access to systems follows the principle of least privilege.
   Multi-Factor Authentication (MFA) is mandatory for all internal systems.
   Privileged Access Management (PAM) required for admin-level access.
   Password must be minimum 12 characters with complexity requirements.
   Passwords must be rotated every 90 days.

3. NETWORK SECURITY
   All remote access must use company-approved VPN.
   Unencrypted WiFi networks must not be used for business communications.
   Network segmentation enforced between production, development, and corporate zones.
   Firewall rules reviewed and updated quarterly.

4. ENDPOINT SECURITY
   All endpoints must have company-approved antivirus/EDR software installed.
   Full-disk encryption (BitLocker/FileVault) is mandatory on all company devices.
   USB storage devices are blocked by default; exceptions require CISO approval.
   Patch management: Critical patches applied within 24 hours; others within 7 days.

5. INCIDENT RESPONSE
   Security incidents must be reported to security@democo.com immediately.
   Incident response plan activated within 1 hour of confirmed security incident.
   Post-incident review mandatory within 5 business days.

6. THIRD-PARTY RISK
   All vendors handling company data must complete security assessment.
   Data Processing Agreements (DPAs) required with all data processors.
   Annual security review of critical vendors.

7. TRAINING & AWARENESS
   Mandatory security awareness training: annually for all employees.
   Phishing simulation exercises conducted quarterly.
   New joiners must complete security induction within first 30 days.

8. CONTACT
   CISO: ciso@democo.com | Security Hotline: 1800-SEC-DEMO
""")

    print("[✔] Compliance documents generated.")


# 
# 4. SECURITY REPORTS
# 

def generate_security_reports():
    with open(f"{DOCS_DIR}/q1_security_report.txt", "w") as f:
        f.write("""DemoCo — Q1 2024 SECURITY INCIDENT REPORT
Document ID: SEC-RPT-2024-Q1 | Classification: CONFIDENTIAL — Admin/CISO Only
Prepared by: Security Operations Center | Date: April 5, 2024

EXECUTIVE SUMMARY
Q1 2024 saw 3 confirmed security incidents, 12 near-misses, and 0 data breaches.
Overall security posture improved by 18% compared to Q4 2023.

INCIDENT SUMMARY TABLE
+------------------+------------+----------+----------+---------------------+
| Incident ID      | Date       | Severity | Type     | Status              |
+------------------+------------+----------+----------+---------------------+
| INC-2024-0001    | 2024-01-12 | HIGH     | Phishing | Resolved            |
| INC-2024-0002    | 2024-02-03 | MEDIUM   | Malware  | Resolved            |
| INC-2024-0003    | 2024-03-21 | LOW      | Policy   | Under Review        |
+------------------+------------+----------+----------+---------------------+

INCIDENT DETAILS

INC-2024-0001: Targeted Phishing Attack (HIGH)
- 45 employees received spear-phishing emails impersonating the CFO.
- 3 employees clicked the malicious link; credentials compromised for 1 user.
- Immediate password reset and MFA enforcement triggered.
- No data exfiltration detected.
- Remediation: Phishing awareness session conducted for all employees.

INC-2024-0002: Malware Detection on Endpoint (MEDIUM)
- Trojan detected on a developer's workstation in Engineering department.
- Isolated within 20 minutes; forensic analysis confirmed no lateral movement.
- Source: Unauthorized software download from personal USB.
- Remediation: USB block policy reinforced; device reimaged.

INC-2024-0003: Policy Violation — Unauthorized Cloud Upload (LOW)
- Employee uploaded client contracts to personal Google Drive.
- DLP tool triggered alert; investigation ongoing.
- Employee placed on compliance training program.

METRICS
- Mean Time to Detect (MTTD): 4.2 hours
- Mean Time to Respond (MTTR): 8.7 hours
- Phishing simulation click rate: 6.8% (down from 11.2% in Q4 2023)
- Patch compliance rate: 96.4%
- MFA adoption rate: 99.1%

RECOMMENDATIONS
1. Implement AI-powered email filtering for enhanced phishing detection.
2. Enforce DLP policies on all cloud storage integrations.
3. Conduct red team exercise in Q2 2024.

Report prepared by: Security Operations Center
Approved by: CISO, Amit Kumar | Date: April 8, 2024
""")

    print("[✔] Security reports generated.")


# 
# 5. JSON AUDIT LOGS
# 

def generate_audit_logs():
    """Creates system audit logs as JSON."""
    log_events = []
    actions = ["LOGIN", "LOGOUT", "FILE_ACCESS", "DATA_EXPORT", "CONFIG_CHANGE",
               "USER_CREATED", "PASSWORD_RESET", "PERMISSION_CHANGE", "REPORT_VIEW"]
    users = [f"EMP{i}" for i in range(1001, 1010)]
    resources = ["HR_PORTAL", "PAYROLL_SYSTEM", "CRM", "CODE_REPO",
                 "FINANCE_DB", "COMPLIANCE_DOCS", "AUDIT_LOGS"]

    base_time = datetime(2024, 3, 1, 8, 0, 0)
    for i in range(100):
        ts = base_time + timedelta(minutes=random.randint(0, 60*24*30))
        action = random.choice(actions)
        status = "SUCCESS" if random.random() > 0.15 else "FAILED"
        log_events.append({
            "log_id":     f"LOG-{2024001 + i}",
            "timestamp":  ts.isoformat(),
            "user_id":    random.choice(users),
            "action":     action,
            "resource":   random.choice(resources),
            "ip_address": f"10.{random.randint(0,5)}.{random.randint(0,255)}.{random.randint(1,254)}",
            "status":     status,
            "details":    f"{action} on {random.choice(resources)} — {status}",
            "session_id": f"SESS-{random.randint(100000, 999999)}",
        })

    path = f"{JSON_DIR}/audit_logs.json"
    with open(path, "w") as f:
        json.dump(log_events, f, indent=2)
    print(f"[✔] Audit logs -> {path} ({len(log_events)} events)")


# 
# 6. USER-ROLE MAPPINGS
# 

def generate_user_role_mappings():
    """Creates user -> role mapping JSON."""
    mappings = {
        "users": {
            # Employees
            "alice_johnson":  {"role": "employee", "employee_id": "EMP1001", "department": "Engineering"},
            "bob_smith":      {"role": "employee", "employee_id": "EMP1002", "department": "Sales"},
            "carol_white":    {"role": "employee", "employee_id": "EMP1003", "department": "Marketing"},
            # Managers
            "dave_manager":   {"role": "manager",  "employee_id": "EMP1004", "department": "Engineering"},
            "eve_manager":    {"role": "manager",  "employee_id": "EMP1005", "department": "HR"},
            "frank_mgr":      {"role": "manager",  "employee_id": "EMP1006", "department": "Finance"},
            # Admins
            "grace_admin":    {"role": "admin",    "employee_id": "EMP1007", "department": "IT"},
            "henry_admin":    {"role": "admin",    "employee_id": "EMP1008", "department": "HR"},
            # Demo users matching the UI
            "employee_demo":  {"role": "employee", "employee_id": "EMP2001", "department": "Engineering"},
            "manager_demo":   {"role": "manager",  "employee_id": "EMP2002", "department": "Engineering"},
            "admin_demo":     {"role": "admin",    "employee_id": "EMP2003", "department": "IT"},
        }
    }

    path = f"{META_DIR}/user_roles.json"
    with open(path, "w") as f:
        json.dump(mappings, f, indent=2)
    print(f"[✔] User-role mappings -> {path}")


# 
# 7. ACCESS CONTROL POLICIES
# 

def generate_access_policies():
    """Creates the master RBAC policy file."""
    policies = {
        "version": "1.0",
        "description": "DemoCo — Role-Based Access Control Policies",
        "roles": {
            "employee": {
                "description": "Regular employee — limited read access",
                "allowed_sources": [
                    "leave_policy",
                    "code_of_conduct",
                    "performance_review_policy",
                    "remote_work_policy",
                    "data_privacy_policy",
                    "information_security_policy",
                    "employees",        # own record only
                ],
                "denied_sources": [
                    "payroll",
                    "q1_security_report",
                    "audit_logs",
                ],
                "allowed_query_intents": [
                    "hr_policy",
                    "leave_info",
                    "conduct_info",
                    "remote_work",
                    "performance_info",
                    "general",
                ],
                "denied_query_intents": [
                    "salary_info",
                    "payroll_data",
                    "security_report",
                    "audit_data",
                    "financial_data",
                ],
            },
            "manager": {
                "description": "Manager — extended read access including department reports",
                "allowed_sources": [
                    "leave_policy",
                    "code_of_conduct",
                    "performance_review_policy",
                    "remote_work_policy",
                    "data_privacy_policy",
                    "information_security_policy",
                    "employees",
                    "q1_security_report",
                    "audit_logs",
                ],
                "denied_sources": [
                    "payroll",
                ],
                "allowed_query_intents": [
                    "hr_policy",
                    "leave_info",
                    "conduct_info",
                    "remote_work",
                    "performance_info",
                    "department_report",
                    "security_report",
                    "audit_data",
                    "general",
                ],
                "denied_query_intents": [
                    "salary_info",
                    "payroll_data",
                    "financial_data",
                ],
            },
            "admin": {
                "description": "System administrator — full access",
                "allowed_sources": ["*"],
                "denied_sources": [],
                "allowed_query_intents": ["*"],
                "denied_query_intents": [],
            },
        },
        "data_source_classification": {
            "leave_policy":                {"sensitivity": "low",      "owner": "HR"},
            "code_of_conduct":             {"sensitivity": "low",      "owner": "HR"},
            "performance_review_policy":   {"sensitivity": "low",      "owner": "HR"},
            "remote_work_policy":          {"sensitivity": "low",      "owner": "HR"},
            "data_privacy_policy":         {"sensitivity": "medium",   "owner": "Legal"},
            "information_security_policy": {"sensitivity": "medium",   "owner": "IT"},
            "employees":                   {"sensitivity": "medium",   "owner": "HR"},
            "payroll":                     {"sensitivity": "high",     "owner": "Finance"},
            "q1_security_report":          {"sensitivity": "high",     "owner": "Security"},
            "audit_logs":                  {"sensitivity": "high",     "owner": "IT"},
        },
    }

    path = f"{POLICY_DIR}/access_control_policies.json"
    with open(path, "w") as f:
        json.dump(policies, f, indent=2)
    print(f"[✔] Access control policies -> {path}")


# 
# 8. DOCUMENT METADATA
# 

def generate_metadata():
    """Creates a metadata catalog for all documents."""
    catalog = [
        {"doc_id": "HR-POL-001", "source": "leave_policy",               "type": "policy",   "owner": "HR",       "sensitivity": "low",    "path": "data/documents/leave_policy.txt"},
        {"doc_id": "HR-POL-002", "source": "code_of_conduct",             "type": "policy",   "owner": "HR",       "sensitivity": "low",    "path": "data/documents/code_of_conduct.txt"},
        {"doc_id": "HR-POL-003", "source": "performance_review_policy",   "type": "policy",   "owner": "HR",       "sensitivity": "low",    "path": "data/documents/performance_review_policy.txt"},
        {"doc_id": "HR-POL-004", "source": "remote_work_policy",          "type": "policy",   "owner": "HR",       "sensitivity": "low",    "path": "data/documents/remote_work_policy.txt"},
        {"doc_id": "COMP-001",   "source": "data_privacy_policy",         "type": "compliance","owner": "Legal",   "sensitivity": "medium", "path": "data/documents/data_privacy_policy.txt"},
        {"doc_id": "COMP-002",   "source": "information_security_policy", "type": "compliance","owner": "IT",      "sensitivity": "medium", "path": "data/documents/information_security_policy.txt"},
        {"doc_id": "SEC-RPT-Q1", "source": "q1_security_report",          "type": "report",   "owner": "Security", "sensitivity": "high",   "path": "data/documents/q1_security_report.txt"},
        {"doc_id": "DB-EMP",     "source": "employees",                   "type": "database", "owner": "HR",       "sensitivity": "medium", "path": "data/csv/employees.csv"},
        {"doc_id": "DB-PAY",     "source": "payroll",                     "type": "database", "owner": "Finance",  "sensitivity": "high",   "path": "data/csv/payroll.csv"},
        {"doc_id": "LOG-AUDIT",  "source": "audit_logs",                  "type": "log",      "owner": "IT",       "sensitivity": "high",   "path": "data/json/audit_logs.json"},
    ]

    path = f"{META_DIR}/document_catalog.json"
    with open(path, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"[✔] Document catalog -> {path}")


# 
# MAIN RUNNER
# 

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DemoCo — Synthetic Demo Data Generator")
    print("="*60 + "\n")

    generate_hr_policies()
    employees = generate_employee_database()
    generate_payroll_database(employees)
    generate_compliance_docs()
    generate_security_reports()
    generate_audit_logs()
    generate_user_role_mappings()
    generate_access_policies()
    generate_metadata()

    print("\n" + "="*60)
    print("  ✔ All synthetic data generated successfully!")
    print("="*60 + "\n")






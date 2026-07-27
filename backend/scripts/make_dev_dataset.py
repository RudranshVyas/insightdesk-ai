"""Generate a SYNTHETIC development dataset.

This exists so the pipeline can be exercised end to end before the real Kaggle
CSV is dropped into data/raw/. It is not a substitute for real data and every
artifact built from it inherits the synthetic-data limitations recorded in
diagnostics.

Output is written to data/raw/ and is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

AREAS = {
    "Payments": [
        ("Payment failed but amount deducted", "The payment for my {plan} plan failed but {amount} was still deducted from my card.",
         "Confirmed a duplicate authorization hold. Voided the hold and advised the customer to allow five business days for the bank to release the funds."),
        ("Charged twice for the same invoice", "We were billed twice for invoice {inv}. Only one charge should have applied.",
         "Located two identical charges caused by a client-side retry. Refunded the duplicate and enabled idempotency keys on the account."),
        ("Refund not received", "A refund was promised {days} days ago and nothing has reached our account yet.",
         "Refund had been issued to an expired card and bounced to the merchant balance. Re-issued to the current payment method."),
        ("Tax not applied to invoice", "Our invoices show no VAT even though we supplied VAT number {inv}.",
         "Tax profile was missing the registration number. Added it and re-issued the affected invoices with tax applied."),
    ],
    "Auth": [
        ("OTP not received on login", "I never receive the OTP code when signing in. Tried {days} times today.",
         "Carrier filtering blocked the SMS sender ID. Moved the user to the email OTP channel and raised a routing ticket with the SMS vendor."),
        ("SSO login redirect loop", "Signing in with SSO bounces between the app and our IdP without ever completing.",
         "SAML assertion was missing the NameID attribute after the IdP upgrade. Corrected the attribute mapping and verified login."),
        ("Session expires too quickly", "Users are signed out after about {days} minutes of activity.",
         "Session cookie lifetime had been overridden by a stale workspace security policy. Restored the default twelve hour lifetime."),
        ("Password reset email never arrives", "Requesting a password reset produces no email, checked spam already.",
         "Reset mails were being rejected by the customer's mail gateway. Allow-listed the sending domain and confirmed delivery."),
    ],
    "Reporting": [
        ("Dashboard export returns HTTP 500", "Exporting the monthly report returns a 500 error, trace id {inv}.",
         "Export job exceeded the worker memory limit for large accounts. Increased the worker tier and paginated the export."),
        ("Metrics lag behind real time", "The analytics dashboard is about {days} hours behind live data.",
         "Ingestion backlog caused by a stuck consumer group. Restarted the consumers and backfilled the delayed window."),
        ("Charts render blank", "All dashboard charts are blank in {platform} but fine elsewhere.",
         "A cached bundle referenced a removed chart module. Cleared the CDN cache and asked the customer to hard refresh."),
        ("Scheduled report never sends", "Our weekly scheduled report has not been emailed for {days} weeks.",
         "Schedule owner had been deactivated, which silently paused the job. Reassigned ownership and resumed the schedule."),
    ],
    "Integrations": [
        ("Webhook deliveries stopped", "Our webhook endpoint stopped receiving events {days} days ago with no config change.",
         "Endpoint was auto-disabled after repeated non-2xx responses. Re-enabled it and replayed the missed events."),
        ("API returns 429 constantly", "We hit rate limits after only a handful of requests to the {plan} API.",
         "Client was retrying without backoff, multiplying request volume. Implemented exponential backoff and raised the burst limit."),
        ("Slack notifications duplicated", "Every alert posts {days} times into our Slack channel.",
         "Multiple Slack integrations were installed against the same channel. Removed the redundant installs."),
        ("Connector disconnects nightly", "Our {plan} connector drops every night and must be re-authorised.",
         "Refresh token rotation was failing behind the customer's proxy. Allow-listed the token endpoint and the connection stayed up."),
    ],
    "Onboarding": [
        ("Cannot find team invite option", "I want to add teammates but cannot locate the invite screen on the {plan} plan.",
         "Plan seat cap had been reached, which hides the invite control. Explained the cap and shared upgrade options."),
        ("CSV import fails with column mismatch", "Our CSV import fails with a column mismatch error, the file has {days} columns.",
         "Header row contained a trailing empty column from an Excel export. Re-saving as UTF-8 CSV resolved the import."),
        ("Sandbox environment access", "How do we request a sandbox environment before going live on {plan}?",
         "Provisioned a sandbox workspace and sent the credential request process to the customer."),
        ("Data migration timeline question", "How long does migrating {days} thousand records usually take?",
         "Shared the migration runbook and scheduled a guided import window with the onboarding engineer."),
    ],
    "Mobile": [
        ("App crashes on launch", "The app crashes immediately on {platform} after the latest update.",
         "Crash traced to a null workspace preference written by an older build. Shipped a patch release and cleared the bad preference."),
        ("Push notifications not delivered", "No push notifications arrive on {platform} devices for {days} days.",
         "Push certificate had expired. Renewed the certificate and confirmed delivery on test devices."),
        ("Offline mode loses edits", "Edits made offline on {platform} disappear once we reconnect.",
         "Sync conflict resolution discarded local edits with older timestamps. Corrected the merge rule and restored the affected records."),
        ("Biometric login unavailable", "Face unlock is missing on {platform} even though it is enabled in settings.",
         "Feature flag had not been rolled out to the customer's region. Enabled the flag for the account."),
    ],
}

ISSUE_TYPE = {
    "Payments": "Billing inquiry",
    "Auth": "Technical issue",
    "Reporting": "Technical issue",
    "Integrations": "Technical issue",
    "Onboarding": "Product inquiry",
    "Mobile": "Technical issue",
}

CHANNELS = ["Email", "Chat", "Phone", "Social media"]
PLATFORMS = ["Windows", "macOS", "iOS", "Android", "Web"]
PLANS = ["Starter", "Business", "Enterprise"]
REGIONS = ["EMEA", "AMER", "APAC"]
SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]

# Optional clauses, sampled independently, so two tickets from the same template
# rarely share wording. Without this the corpus collapses to ~24 distinct texts
# and the Phase 2 clustering gate correctly refuses to run.
IMPACT = [
    "This is blocking our month end close.",
    "Roughly forty people are affected.",
    "Our whole support team is stuck on this.",
    "It only affects one workspace so far.",
    "Two customers have already complained to us about it.",
    "We have had to fall back to a manual process meanwhile.",
    "This started right after our last release.",
    "Nothing else in the product seems affected.",
]
ENVIRONMENT = [
    "We are on the {plan} plan in {region}.",
    "Reproduced on {platform} and on a colleague's machine.",
    "Happens in both our staging and production workspaces.",
    "Only one of our three environments shows the problem.",
    "We use SSO through an external identity provider.",
    "Our workspace was migrated from a legacy account last quarter.",
]
ATTEMPTED = [
    "We already tried clearing the cache and signing out.",
    "I have restarted the app several times with no change.",
    "A colleague tried from a different network with the same result.",
    "We rolled back our own recent config change and it persists.",
    "Following the help centre article did not resolve it.",
    "We waited a full day in case it was transient.",
]
CLOSERS = [
    "Please advise on next steps.",
    "Any pointers would be appreciated.",
    "Let us know what diagnostics you need.",
    "Happy to jump on a call if that is faster.",
    "We would like an update today if possible.",
]

FIRST = ["Alice", "Brian", "Carla", "Dan", "Elena", "Farid", "Grace", "Hugo", "Ines",
         "Jonas", "Kira", "Luis", "Maya", "Nils", "Omar", "Priya", "Rosa", "Sven",
         "Tara", "Umar", "Vera", "Will", "Yasmin", "Zach"]
LAST = ["Moreno", "Ochoa", "Devi", "Whitfield", "Fischer", "Haddad", "Lin", "Reyes",
        "Kovac", "Bauer", "Nakamura", "Ferreira", "Oduya", "Andersen", "Saleh",
        "Raman", "Iglesias", "Larsson", "Nguyen", "Farouk", "Popov", "Turner"]


def escalation_probability(area: str, priority: str, channel: str, segment: str) -> float:
    """A deliberate, documented signal so the Phase 8 ladder has something real
    to find. All inputs are known at ticket creation."""
    p = 0.06
    p += {"Payments": 0.10, "Auth": 0.08, "Integrations": 0.07,
          "Reporting": 0.05, "Mobile": 0.03, "Onboarding": 0.00}[area]
    p += {"Critical": 0.22, "High": 0.12, "Medium": 0.03, "Low": 0.0}[priority]
    p += {"Phone": 0.05, "Social media": 0.04, "Email": 0.01, "Chat": 0.0}[channel]
    p += {"Enterprise": 0.06, "Mid-Market": 0.02, "SMB": 0.0}[segment]
    return min(p, 0.85)


def generate(n: int, seed: int, out: Path) -> None:
    rng = random.Random(seed)
    start = datetime(2023, 1, 1, 8, 0, 0)
    rows = []

    for i in range(n):
        area = rng.choice(list(AREAS))
        subject, body, resolution = rng.choice(AREAS[area])
        plan = rng.choice(PLANS)
        platform = rng.choice(PLATFORMS)
        segment = rng.choice(SEGMENTS)
        channel = rng.choice(CHANNELS)
        priority = rng.choices(PRIORITIES, weights=[3, 4, 3, 1])[0]

        desc = body.format(
            plan=plan,
            platform=platform,
            amount=f"${rng.randint(9, 900)}.{rng.randint(10, 99)}",
            inv=f"{rng.randint(10000, 99999)}",
            days=rng.randint(2, 9),
        )
        extras = []
        if rng.random() < 0.75:
            extras.append(rng.choice(IMPACT))
        if rng.random() < 0.70:
            extras.append(rng.choice(ENVIRONMENT).format(plan=plan, platform=platform,
                                                         region=rng.choice(REGIONS)))
        if rng.random() < 0.65:
            extras.append(rng.choice(ATTEMPTED))
        if rng.random() < 0.55:
            extras.append(rng.choice(CLOSERS))
        rng.shuffle(extras)
        if extras:
            desc = desc + " " + " ".join(extras)

        # ~8% of tickets carry a contact detail, so redaction has real work to do.
        if rng.random() < 0.08:
            first, last = rng.choice(FIRST), rng.choice(LAST)
            desc += f" Reach me at {first.lower()}.{last.lower()}@example.com."

        created = start + timedelta(minutes=rng.randint(0, 525_600))
        p_esc = escalation_probability(area, priority, channel, segment)
        escalated = rng.random() < p_esc

        open_ticket = rng.random() < 0.12
        if open_ticket:
            status = rng.choice(["Open", "Pending Customer Response"])
            first_resp = created + timedelta(hours=rng.uniform(0.5, 20)) if rng.random() < 0.6 else None
            resolved_at = None
            res_notes = ""
            csat = 0
        else:
            status = "Closed"
            first_resp = created + timedelta(hours=rng.uniform(0.2, 24))
            base = rng.gammavariate(2.0, 9.0) + (28 if escalated else 0)
            resolved_at = first_resp + timedelta(hours=base)
            res_notes = resolution
            # Occasional boilerplate, as real exports contain.
            if rng.random() < 0.05:
                res_notes = rng.choice(["N/A", "resolved", ""])
            csat = 0 if rng.random() < 0.3 else rng.choices([1, 2, 3, 4, 5], weights=[1, 1, 2, 4, 4])[0]

        rows.append(
            {
                "Ticket ID": f"T{i + 1:06d}",
                "Customer Name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                "Customer Email": f"user{i}@example.com",
                "Customer Age": rng.randint(20, 65),
                "Customer Gender": rng.choice(["Male", "Female", "Other"]),
                "Customer Segment": segment,
                "Product Purchased": area,
                "Date of Purchase": (created - timedelta(days=rng.randint(30, 900))).date(),
                "Ticket Type": ISSUE_TYPE[area],
                "Ticket Subject": subject,
                "Ticket Description": desc,
                "Ticket Status": status,
                "Resolution": res_notes,
                "Ticket Priority": priority,
                "Ticket Channel": channel,
                "Platform": platform,
                "Region": rng.choice(REGIONS),
                "SLA Plan": plan,
                "Ticket Created At": created.strftime("%Y-%m-%d %H:%M:%S"),
                "First Response Time": first_resp.strftime("%Y-%m-%d %H:%M:%S") if first_resp else "",
                "Time to Resolution": resolved_at.strftime("%Y-%m-%d %H:%M:%S") if resolved_at else "",
                "Customer Satisfaction Rating": csat,
                "Escalated": "Yes" if escalated else "No",
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} SYNTHETIC rows to {out}")
    print("This is development data. Replace it with the real dataset before")
    print("reporting any number anywhere.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate a synthetic dev dataset.")
    p.add_argument("--rows", type=int, default=6000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", type=Path, default=Path("data/raw/dev_synthetic_tickets.csv"))
    args = p.parse_args(argv)
    generate(args.rows, args.seed, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

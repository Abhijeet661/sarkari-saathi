"""Rule-based eligibility check.

This part deliberately does NOT use the LLM. Age and category maths has one
right answer, and a model that is "usually right" about someone's age limit
is not good enough. The LLM only does the reading (extractor.py); the
deciding happens here, in code you can test.

Qualification matching is the fuzzy part - notifications write the same
degree ten different ways. When we can't tell, we say "check_manually"
instead of guessing.
"""

import re
from datetime import date

from .schemas import NotificationDetails, PostEligibility, UserProfile

# canonical level -> patterns that mean it
QUALIFICATION_PATTERNS = {
    "phd": [r"\bph\.?\s?d\b", r"\bdoctorate\b"],
    "postgrad": [r"\bm\.?\s?tech\b", r"\bm\.?\s?e\.?\b", r"\bm\.?\s?sc\b", r"\bmca\b", r"\bmba\b",
                 r"\bpost\s?graduat", r"\bmaster"],
    "engineering": [r"\bb\.?\s?tech\b", r"\bb\.?\s?e\.?\b", r"\bbachelor of (engineering|technology)\b"],
    "graduate": [r"\bb\.?\s?sc\b", r"\bb\.?\s?com\b", r"\bb\.?\s?a\b", r"\bbca\b", r"\bgraduat",
                 r"\bbachelor'?s? degree\b", r"\bdegree\b"],
    "diploma": [r"\bdiploma\b"],
    "12th": [r"\b12th\b", r"\bhsc\b", r"\bintermediate\b", r"\b10\s?\+\s?2\b", r"\bhigher secondary\b"],
    "10th": [r"\b10th\b", r"\bssc\b", r"\bmatriculat", r"\bhigh school\b"],
}

# a higher level also satisfies a lower one (B.Tech holder has passed 12th)
LEVEL_ORDER = ["10th", "12th", "diploma", "graduate", "engineering", "postgrad", "phd"]

SUBJECT_KEYWORDS = {
    "computer": [r"computer", r"\bcse\b", r"\bit\b", r"information technology", r"\bmca\b", r"\bbca\b"],
    "electronics": [r"electronics", r"\bece\b", r"\bentc\b", r"telecommunication"],
    "electrical": [r"electrical", r"\beee\b"],
    "mechanical": [r"mechanical"],
    "civil": [r"\bcivil\b"],
    "agriculture": [r"agricultur"],
}


def _levels(text: str) -> set[str]:
    t = text.lower()
    return {lvl for lvl, pats in QUALIFICATION_PATTERNS.items() if any(re.search(p, t) for p in pats)}


def _subjects(text: str) -> set[str]:
    t = text.lower()
    return {s for s, pats in SUBJECT_KEYWORDS.items() if any(re.search(p, t) for p in pats)}


def age_on(dob: date, on: date) -> int:
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


def relaxation_for(profile: UserProfile, relaxation: dict[str, int]) -> int:
    rel = {k.upper(): v for k, v in relaxation.items()}
    years = rel.get(profile.category, 0)
    if profile.pwbd:
        years += rel.get("PWBD", rel.get("PWD", 0))
    return years


def check_qualification(required: str, user_quals: list[str]) -> tuple[str, str]:
    need_levels = _levels(required)
    have_levels = set().union(*[_levels(q) for q in user_quals]) if user_quals else set()
    if not need_levels:
        return "check_manually", f"Couldn't read the required qualification: '{required}'"

    need_rank = min(LEVEL_ORDER.index(l) for l in need_levels)
    have_rank = max((LEVEL_ORDER.index(l) for l in have_levels), default=-1)
    if have_rank < need_rank:
        return "not_eligible", f"Needs {LEVEL_ORDER[need_rank]} level; your highest is " + (
            LEVEL_ORDER[have_rank] if have_rank >= 0 else "unknown")

    need_subjects = _subjects(required)
    if need_subjects:
        have_subjects = set().union(*[_subjects(q) for q in user_quals])
        if not need_subjects & have_subjects:
            return "not_eligible", f"Subject must be one of: {', '.join(sorted(need_subjects))}"
        return "eligible", "Qualification level and subject match"

    return "eligible", "Qualification level matches"


def check_eligibility(details: NotificationDetails, profile: UserProfile, today: date | None = None) -> list[PostEligibility]:
    cutoff = details.age_cutoff_date or details.application_end or today or date.today()
    age = age_on(profile.date_of_birth, cutoff)
    extra = relaxation_for(profile, details.age_relaxation)
    results = []

    for post in details.posts:
        reasons, statuses = [], []

        if post.age_max is not None:
            limit = post.age_max + extra
            if age > limit:
                statuses.append("not_eligible")
                reasons.append(f"Age {age} on {cutoff} is above the limit of {limit} (incl. {extra} yrs relaxation)")
            else:
                reasons.append(f"Age {age} on {cutoff} is within the limit of {limit}")
        else:
            statuses.append("check_manually")
            reasons.append("Upper age limit not found in the notification")

        if post.age_min is not None and age < post.age_min:
            statuses.append("not_eligible")
            reasons.append(f"Minimum age is {post.age_min}")

        q_status, q_reason = check_qualification(post.qualification, profile.qualifications)
        statuses.append(q_status)
        reasons.append(q_reason)

        if "not_eligible" in statuses:
            status = "not_eligible"
        elif "check_manually" in statuses:
            status = "check_manually"
        else:
            status = "eligible"
        results.append(PostEligibility(post=post.name, status=status, reasons=reasons))
    return results

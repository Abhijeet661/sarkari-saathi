import json
from datetime import date

import pytest

from saathi.eligibility import age_on, check_eligibility, check_qualification
from saathi.extractor import ExtractionError, extract_details
from saathi.llm import FakeLLM
from saathi.schemas import NotificationDetails, UserProfile

GOOD_JSON = json.dumps({
    "organization": "National Institute of Rural Technology (Sample)",
    "advertisement_no": "NIRT/REC/07/2026",
    "posts": [
        {"name": "Technical Assistant (Computer)", "vacancies": 12, "qualification": "B.E./B.Tech in Computer Science or IT", "age_min": 18, "age_max": 30, "pay_level": "Level 6"},
        {"name": "Junior Engineer (Electrical)", "vacancies": 5, "qualification": "Diploma in Electrical Engineering", "age_min": 18, "age_max": 30, "pay_level": "Level 6"},
        {"name": "Data Entry Operator", "vacancies": 8, "qualification": "12th pass with typing speed 35 wpm", "age_min": 18, "age_max": 27, "pay_level": "Level 4"},
    ],
    "age_cutoff_date": "2026-12-31",
    "age_relaxation": {"OBC": 3, "SC": 5, "ST": 5, "PwBD": 10},
    "application_start": "2026-12-01",
    "application_end": "2026-12-31",
    "fee_general": 500,
    "fee_reserved": 0,
    "selection_process": ["Computer Based Test", "Skill Test"],
    "apply_url": "https://nirt-sample.example.in/recruitment",
})


@pytest.fixture
def details():
    return NotificationDetails.model_validate_json(GOOD_JSON)


def test_extraction_first_try(sample_pages):
    result = extract_details(sample_pages, FakeLLM([GOOD_JSON]))
    assert result.advertisement_no == "NIRT/REC/07/2026"
    assert len(result.posts) == 3


def test_extraction_self_corrects(sample_pages):
    broken = GOOD_JSON.replace('"vacancies": 12', '"vacancies": "12 (UR-5)"')
    llm = FakeLLM([broken, "```json\n" + GOOD_JSON + "\n```"])
    result = extract_details(sample_pages, llm)
    assert result.posts[0].vacancies == 12
    assert "invalid" in llm.prompts[1]


def test_extraction_gives_up(sample_pages):
    with pytest.raises(ExtractionError):
        extract_details(sample_pages, FakeLLM(["not json"] * 3), max_retries=2)


def test_age_on_birthday_edge():
    assert age_on(date(2003, 6, 26), date(2026, 6, 25)) == 22
    assert age_on(date(2003, 6, 26), date(2026, 6, 26)) == 23


@pytest.mark.parametrize("required, quals, expected", [
    ("B.E./B.Tech in Computer Science or IT", ["B.Tech Computer Science"], "eligible"),
    ("B.E./B.Tech in Computer Science or IT", ["B.Tech Mechanical"], "not_eligible"),
    ("Diploma in Electrical Engineering", ["12th"], "not_eligible"),
    ("12th pass with typing speed 35 wpm", ["B.Sc Physics"], "eligible"),
    ("Experience in handling lab equipment", ["B.Tech CSE"], "check_manually"),
])
def test_qualification(required, quals, expected):
    assert check_qualification(required, quals)[0] == expected


def test_eligibility_general_candidate(details):
    profile = UserProfile(date_of_birth=date(2003, 6, 26), category="GEN", qualifications=["B.Tech Computer Science", "12th"])
    results = {r.post: r.status for r in check_eligibility(details, profile)}
    assert results["Technical Assistant (Computer)"] == "eligible"
    assert results["Data Entry Operator"] == "eligible"
    # B.Tech CSE does not satisfy "Diploma in Electrical": subject mismatch
    assert results["Junior Engineer (Electrical)"] == "not_eligible"


def test_relaxation_makes_obc_eligible(details):
    # 32 on cutoff: over 30 for GEN, within 33 for OBC
    gen = UserProfile(date_of_birth=date(1994, 6, 1), category="GEN", qualifications=["B.Tech IT"])
    obc = gen.model_copy(update={"category": "OBC"})
    ta = "Technical Assistant (Computer)"
    assert {r.post: r.status for r in check_eligibility(details, gen)}[ta] == "not_eligible"
    assert {r.post: r.status for r in check_eligibility(details, obc)}[ta] == "eligible"

"""Builds data/sample/sample_notification.pdf - a FICTIONAL notification.

Used by the tests and the eval set so they always run on the same file.
Real notifications go in data/notifications/ (not committed, see README).
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

OUT = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_notification.pdf"

PAGES = [
    [
        ("h", "National Institute of Rural Technology (Sample)"),
        ("p", "SAMPLE DOCUMENT FOR TESTING - THIS IS NOT A REAL NOTIFICATION"),
        ("p", "Advertisement No. NIRT/REC/07/2026"),
        ("p", "Recruitment of Technical Assistant, Junior Engineer and Data Entry Operator on direct recruitment basis."),
        ("p", "Online applications are invited from eligible Indian citizens. Candidates must apply only through the "
              "official website https://nirt-sample.example.in/recruitment."),
        ("h2", "Important Dates"),
        ("p", "Opening date of online application: 01 December 2026."),
        ("p", "Last date of online application: 31 December 2026 (till 11:59 PM)."),
        ("p", "Tentative date of Computer Based Test (CBT): February 2027."),
    ],
    [
        ("h2", "1. Details of Posts and Vacancies"),
        ("t", [["Post", "Vacancies", "Pay Level", "Essential Qualification"],
               ["Technical Assistant (Computer)", "12", "Level 6", "B.E./B.Tech in Computer Science or IT"],
               ["Junior Engineer (Electrical)", "5", "Level 6", "Diploma in Electrical Engineering"],
               ["Data Entry Operator", "8", "Level 4", "12th pass with typing speed 35 wpm"]]),
        ("p", "Category-wise break-up of the 12 Technical Assistant posts: UR-5, EWS-1, OBC-3, SC-2, ST-1."),
        ("p", "The number of vacancies is provisional and may change as per the requirement of the Institute."),
    ],
    [
        ("h2", "2. Age Limit"),
        ("p", "Age will be counted as on 31 December 2026, the last date of online application."),
        ("p", "Technical Assistant (Computer): 18 to 30 years. Junior Engineer (Electrical): 18 to 30 years. "
              "Data Entry Operator: 18 to 27 years."),
        ("h2", "3. Age Relaxation"),
        ("p", "Upper age limit is relaxable by 3 years for OBC (Non-Creamy Layer) candidates and by 5 years for "
              "SC/ST candidates. Persons with Benchmark Disabilities (PwBD) get an additional 10 years. "
              "No relaxation is available for UR and EWS candidates."),
    ],
    [
        ("h2", "4. Application Fee"),
        ("p", "Application fee for UR, EWS and OBC candidates is Rs. 500 (non-refundable)."),
        ("p", "SC, ST, PwBD and Women candidates are exempted from payment of fee."),
        ("p", "Fee must be paid online through debit card, credit card, net banking or UPI."),
        ("h2", "5. How to Apply"),
        ("p", "Register on the website, fill the form, upload a recent photograph and signature, pay the fee and "
              "keep a printout of the submitted form. Applications sent by post will not be accepted."),
    ],
    [
        ("h2", "6. Selection Process"),
        ("p", "Selection will be through a Computer Based Test of 120 marks followed by a Skill Test. "
              "The Skill Test is qualifying in nature. Final merit is based on CBT marks only."),
        ("p", "For Data Entry Operator, the Skill Test is a typing test on computer."),
        ("h2", "7. General Instructions"),
        ("p", "Candidates must have the essential qualification on or before the last date of application. "
              "Any canvassing will lead to disqualification. The decision of the Institute is final."),
    ],
]


def build(out: Path = OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for i, page in enumerate(PAGES):
        for kind, content in page:
            if kind == "t":
                story.append(Table(content))
            else:
                style = {"h": "Title", "h2": "Heading2", "p": "BodyText"}[kind]
                story.append(Paragraph(content, styles[style]))
            story.append(Spacer(1, 6))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(out), pagesize=A4, title="Sample notification").build(story)
    return out


if __name__ == "__main__":
    print("wrote", build())

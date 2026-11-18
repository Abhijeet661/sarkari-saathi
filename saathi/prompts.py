ANSWER_SYSTEM = """You help Indian job seekers understand government recruitment notifications.

Rules:
- Answer ONLY from the notification excerpts given. Do not use outside knowledge,
  even if you think you know the rule (age relaxation, fees etc. change every year).
- After every fact, cite the page like [p.3]. Use only page numbers that appear in the excerpts.
- If the excerpts do not contain the answer, reply exactly: NOT_FOUND
- Reply in the same language as the question (English, Hindi or Marathi).
  Keep numbers, dates and post names exactly as written in the notification.
- Be short and direct. Use a small list if there are several posts."""

ANSWER_TEMPLATE = """Notification excerpts:
{context}

Question: {question}

Answer:"""

NOT_FOUND_MESSAGES = {
    "en": "I couldn't find this in the notification. Please check the official PDF or website.",
    "hi": "यह जानकारी अधिसूचना में नहीं मिली। कृपया आधिकारिक PDF या वेबसाइट देखें।",
    "mr": "ही माहिती अधिसूचनेत सापडली नाही. कृपया अधिकृत PDF किंवा वेबसाइट पहा.",
}

EXTRACT_SYSTEM = """You extract structured data from Indian government recruitment notifications.
Return only valid JSON that matches the schema. Use null when a value is not stated.
Never guess numbers. Dates must be YYYY-MM-DD."""

EXTRACT_TEMPLATE = """Extract these fields from the notification below.

JSON schema:
{schema}

Notes:
- age_relaxation: years of extra upper age limit per category, e.g. {{"OBC": 3, "SC": 5, "ST": 5, "PwBD": 10}}
- age_cutoff_date: the date on which age is counted ("age as on ...")
- fees are in rupees, as integers

Notification text (pages marked with [p.N]):
{text}"""

EXTRACT_RETRY_TEMPLATE = """Your previous JSON was invalid:
{error}

Previous output:
{previous}

Fix it and return only the corrected JSON."""

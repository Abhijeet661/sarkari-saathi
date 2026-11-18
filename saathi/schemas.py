from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Post(BaseModel):
    name: str
    vacancies: int | None = None
    qualification: str = Field(description="Essential qualification as written in the notification")
    age_min: int | None = None
    age_max: int | None = None
    pay_level: str | None = None


class NotificationDetails(BaseModel):
    organization: str
    advertisement_no: str | None = None
    posts: list[Post]
    age_cutoff_date: date | None = None
    age_relaxation: dict[str, int] = Field(default_factory=dict)
    application_start: date | None = None
    application_end: date | None = None
    fee_general: int | None = None
    fee_reserved: int | None = Field(default=None, description="Fee for SC/ST/PwBD/Women if different")
    selection_process: list[str] = Field(default_factory=list)
    apply_url: str | None = None


Category = Literal["GEN", "EWS", "OBC", "SC", "ST"]


class UserProfile(BaseModel):
    date_of_birth: date
    category: Category = "GEN"
    pwbd: bool = False
    qualifications: list[str] = Field(description='e.g. ["B.Tech Computer Science", "12th"]')


class PostEligibility(BaseModel):
    post: str
    status: Literal["eligible", "not_eligible", "check_manually"]
    reasons: list[str]

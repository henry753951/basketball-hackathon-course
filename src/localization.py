"""Select learner-facing text for the active course notebook."""

import os


def localize(chinese: str, english: str, japanese: str) -> str:
    language = os.environ.get("BASKETBALL_COURSE_LANGUAGE", "zh")
    return {"en": english, "ja": japanese}.get(language, chinese)

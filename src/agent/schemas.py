"""Subject-type schema registry.

Each schema must have top-level `title` and `description` fields
and avoid deep nesting (LLM extraction degrades with nested objects).
"""

EXECUTIVE_SCHEMA = {
    "type": "object",
    "title": "Executive",
    "description": "Professional background for a business executive, founder, or operator",
    "required": ["role", "current_company", "prior_companies", "years_experience"],
    "properties": {
        "role": {
            "type": "string",
            "description": "Current role/title (e.g. 'Founder & CEO', 'CTO')",
        },
        "current_company": {
            "type": "string",
            "description": "Single most prominent current employer or company the person leads",
        },
        "current_companies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All companies the person currently leads or actively runs (populate when running multiple simultaneously)",
        },
        "prior_companies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Companies the person has FULLY DEPARTED — exclude any org they currently lead or own",
        },
        "years_experience": {
            "type": "number",
            "description": "Years of full-time paid work experience (exclude internships and academic years)",
        },
    },
}

POLITICIAN_SCHEMA = {
    "type": "object",
    "title": "Politician",
    "description": "Profile for an elected official, political leader, or government minister",
    "required": ["office", "party", "country", "political_history"],
    "properties": {
        "office": {
            "type": "string",
            "description": "Current office or title (e.g. 'Prime Minister', 'Member of European Parliament')",
        },
        "party": {
            "type": "string",
            "description": "Current political party or movement",
        },
        "country": {
            "type": "string",
            "description": "Country or jurisdiction the person represents or governs",
        },
        "constituency": {
            "type": "string",
            "description": "Electoral district, region, or constituency represented (if applicable)",
        },
        "political_history": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Previous offices held, parties, or significant political roles in chronological order",
        },
        "notable_positions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key policy positions, stances, or political platforms",
        },
    },
}

ENTERTAINER_SCHEMA = {
    "type": "object",
    "title": "Entertainer",
    "description": "Profile for a musician, actor, comedian, band, or other entertainer",
    "required": ["medium", "notable_works", "active_since"],
    "properties": {
        "medium": {
            "type": "string",
            "description": "Primary creative medium (e.g. 'actor', 'rock band', 'stand-up comedian', 'rapper')",
        },
        "active_since": {
            "type": "string",
            "description": "Year or period when professional career began",
        },
        "label_or_studio": {
            "type": "string",
            "description": "Current record label, film studio, or production company (if applicable)",
        },
        "notable_works": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Most significant albums, films, TV shows, specials, or other creative works",
        },
        "awards": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Major awards or nominations (e.g. 'Grammy Award for Best Album 2023')",
        },
        "genre": {
            "type": "string",
            "description": "Primary genre or style (e.g. 'indie rock', 'romantic comedy', 'hip-hop')",
        },
    },
}

ATHLETE_SCHEMA = {
    "type": "object",
    "title": "Athlete",
    "description": "Profile for a professional or elite amateur athlete",
    "required": ["sport", "current_team", "career_highlights"],
    "properties": {
        "sport": {
            "type": "string",
            "description": "Primary sport (e.g. 'basketball', 'tennis', 'Formula 1')",
        },
        "current_team": {
            "type": "string",
            "description": "Current team, club, or 'Free agent' / 'Retired'",
        },
        "position": {
            "type": "string",
            "description": "Playing position or event specialty (if applicable)",
        },
        "nationality": {
            "type": "string",
            "description": "Country the athlete represents",
        },
        "active_since": {
            "type": "string",
            "description": "Year professional career began",
        },
        "career_highlights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Major titles, records, championships, or career milestones",
        },
        "prior_teams": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Previous clubs or teams in chronological order",
        },
    },
}

ACADEMIC_SCHEMA = {
    "type": "object",
    "title": "Academic",
    "description": "Profile for a researcher, professor, or scientist",
    "required": ["institution", "research_areas", "notable_publications"],
    "properties": {
        "institution": {
            "type": "string",
            "description": "Current university, research institute, or lab",
        },
        "department": {
            "type": "string",
            "description": "Department or faculty (e.g. 'Department of Computer Science')",
        },
        "title": {
            "type": "string",
            "description": "Academic title (e.g. 'Professor', 'Associate Professor', 'Research Scientist')",
        },
        "research_areas": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Primary fields of research or academic focus",
        },
        "notable_publications": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Most cited or significant papers, books, or research contributions",
        },
        "prior_institutions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Previous universities or research institutions",
        },
    },
}

JOURNALIST_SCHEMA = {
    "type": "object",
    "title": "Journalist",
    "description": "Profile for a journalist, reporter, or media personality",
    "required": ["outlet", "beat", "notable_stories"],
    "properties": {
        "outlet": {
            "type": "string",
            "description": "Current employer or publication (e.g. 'The New York Times', 'CNN')",
        },
        "beat": {
            "type": "string",
            "description": "Primary coverage area (e.g. 'technology', 'politics', 'sports')",
        },
        "role": {
            "type": "string",
            "description": "Job title (e.g. 'Senior Reporter', 'Anchor', 'Investigative Journalist')",
        },
        "notable_stories": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Significant investigations, scoops, or notable pieces of work",
        },
        "awards": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Journalism awards or recognition (e.g. 'Pulitzer Prize 2022')",
        },
        "prior_outlets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Previous employers or publications",
        },
    },
}

SCHEMAS: dict[str, dict] = {
    "executive": EXECUTIVE_SCHEMA,
    "politician": POLITICIAN_SCHEMA,
    "entertainer": ENTERTAINER_SCHEMA,
    "athlete": ATHLETE_SCHEMA,
    "academic": ACADEMIC_SCHEMA,
    "journalist": JOURNALIST_SCHEMA,
}

SUBJECT_TYPES = list(SCHEMAS.keys())

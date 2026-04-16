from dataclasses import dataclass, field
from typing import Any, Optional, Annotated
import operator

from pydantic import BaseModel


DEFAULT_EXTRACTION_SCHEMA = {
    "type": "object",
    "required": [
        "years_experience",
        "current_company",
        "role",
        "prior_companies",
    ],
    "properties": {
        "role": {"type": "string", "description": "Current role of the person."},
        "years_experience": {
            "type": "number",
            "description": "How many years of full time work experience (excluding internships) does this person have.",
        },
        "current_company": {
            "type": "string",
            "description": "The name of the current company the person works at.",
        },
        "prior_companies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of previous companies where the person has worked",
        },
    },
    "description": "Person information",
    "title": "Person",
}


class Person(BaseModel):
    """A class representing a person to research."""

    name: Optional[str] = None
    """The name of the person."""
    company: Optional[str] = None
    """The current company of the person."""
    linkedin: Optional[str] = None
    """The Linkedin URL of the person."""
    email: Optional[str] = None
    """The email of the person."""
    role: Optional[str] = None
    """The current title of the person."""


@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""

    person: Person
    "Person to research."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: Optional[dict[str, Any]] = field(default=None)
    "Any notes from the user to start the research process."


@dataclass(kw_only=True)
class OverallState:
    """Input state defines the interface between the graph and the user (external API)."""

    person: Person
    "Person to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: str = field(default=None)
    "Any notes from the user to start the research process."

    subject_type: str = field(default="executive")
    "Detected subject type used to select the extraction schema (executive/politician/entertainer/athlete/academic/journalist)"

    abort_reason: Optional[str] = field(default=None)
    "Set by classify_subject when the subject is fictional, ambiguous beyond use, or otherwise unresearchable"

    search_queries: list[str] = field(default=None)
    "List of generated search queries to find relevant information"

    # Add default values for required fields
    completed_notes: Annotated[list, operator.add] = field(default_factory=list)
    "Notes from completed research related to the schema"

    info: dict[str, Any] = field(default=None)
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """

    verification: dict[str, Any] = field(default=None)
    "Fact-check results for each extracted field"

    is_satisfactory: bool = field(default=None)
    "True if all required fields are well populated, False otherwise"

    reflection_steps_taken: int = field(default=0)
    "Number of times the reflection node has been executed"

    bio: Optional[str] = field(default=None)
    "Narrative biography synthesized from research notes and web sources."

    bio_sources: list[str] = field(default_factory=list)
    "URLs cited in the bio."


@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """

    verification: Optional[dict[str, Any]] = None
    "Fact-check results for each extracted field"

    subject_type: Optional[str] = None
    "Detected subject type (executive/politician/entertainer/athlete/academic/journalist)"

    bio: Optional[str] = None
    "Narrative biography synthesized from research notes and web sources."

    bio_sources: list[str] = field(default_factory=list)
    "URLs cited in the bio."

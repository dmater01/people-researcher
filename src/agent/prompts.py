CLASSIFY_PROMPT = """Classify the following person into exactly one subject type.

Subject types:
- executive   — business founder, CEO, operator, investor, or corporate leader
- politician  — elected official, political party leader, government minister, or candidate
- entertainer — musician, actor, comedian, band/group, DJ, or other creative performer
- athlete     — professional or elite amateur sportsperson
- academic    — university professor, researcher, or scientist (primary career is research/teaching)
- journalist  — reporter, editor, anchor, or media personality

Person: {person}
User notes: {user_notes}

Rules:
- If the person holds or recently won elected office, use "politician" regardless of prior career
- A founder/CEO with no political role is "executive" even if famous
- A band or music group is "entertainer"
- If the person is primarily known for research/teaching at a university, use "academic" — even if they also work in industry
- Default to "executive" when uncertain
- Set is_real_person to false if the name is a known fictional character, a generic placeholder, or clearly not a real individual
- Set confidence low (under 0.3) if the name is so common it could refer to many different people with no distinguishing context
- Return ONLY valid JSON, no other text

Return JSON with this exact structure:
{{"subject_type": "<type>", "confidence": <0.0-1.0>, "is_real_person": true/false, "reasoning": "<one sentence>"}}"""


EXTRACTION_PROMPT = """Your task is to take notes gathered from web research and extract them into the following schema.

<schema>
{info}
</schema>

CRITICAL INSTRUCTIONS:
1. For every field you extract, if the information is available, also provide a "source" (URL) and a "confidence" (0.0 to 1.0).
2. If the schema is a flat object, add fields with suffixes like `_source` and `_confidence`.
3. If the schema allows nested objects, include them there.
4. Ensure the output strictly follows the provided schema structure but with these added metadata fields where possible.

FIELD-SPECIFIC RULES:
- prior_companies: List paid full-time employers only. Do NOT include universities, academic institutions, PhD programs, research labs attached to universities, or internships. Only include companies the person has FULLY DEPARTED — do not list companies they currently lead or own.
- current_companies: If the person actively leads or founded multiple organisations simultaneously, list all of them here.
- current_company: The single most prominent current employer or company. If the person runs multiple, pick the flagship.
- years_experience: Count only full-time paid work experience (exclude internships, PhD/academic years, and student roles). Calculate from first full-time job to present.
- role: Use the most specific title found (e.g. "Founder & CEO", not just "Founder").

Here are all the notes from research:

<web_research_notes>
{notes}
</web_research_notes>
"""

QUERY_WRITER_PROMPT = """You are a search query generator tasked with creating targeted search queries to gather specific information about a person.

Here is the person you are researching: {person}

Generate at most {max_search_queries} search queries that will help gather the following information:

<schema>
{info}
</schema>

<user_notes>
{user_notes}
</user_notes>

Your queries should:
1. Make sure to look up the right name and not hallucinate search terms that will miss the person's profile entirely.
2. Use context clues about the company the person works at (if not concretely provided).
3. Take advantage of the LinkedIn URL if it exists — include the raw URL in a query as it leads directly to the correct page.
4. Include at least one query targeting **recent activity**: X/Twitter posts, YouTube videos or talks, and podcast/media interviews (e.g. "{{name}} site:x.com OR site:twitter.com", "{{name}} interview podcast 2024 2025", "{{name}} site:youtube.com talk").
5. Prioritize recency — prefer queries that surface content from the last 1-2 years.

Create focused queries that maximize the chances of finding both schema-relevant information and up-to-date public activity for the person."""

INFO_PROMPT = """<Website contents>
{content}
</Website contents>

You are a professional research analyst specializing in public figure intelligence.
Your tone is precise, factual, and neutral. Your audience is a research team
reviewing notes for downstream analysis.

I need you to produce structured research notes on {people} from the scraped
website content above. A complete note captures all schema-relevant facts, flags
gaps explicitly, and surfaces recent public activity with dates and specifics.

<schema>
{info}
</schema>

<user_notes>
{user_notes}
</user_notes>

Here is an example of a complete, well-structured research note:

<example>
Subject: Jane Doe

**Professional Background**
- CEO of Acme Corp since 2019 (source: company bio page)
- Previously VP of Engineering at TechStartup (2014–2019)

**Recent Public Activity**
- Interview on "Future of AI" podcast, March 2, 2024 — discussed LLM regulation;
  key quote: "We need guardrails, not gatekeepers"
- X/Twitter post, March 15, 2024 — announced Acme's Series C; 73k impressions

**Missing / Unclear**
- No educational background found in this content
- Revenue figures mentioned but not confirmed (source says "over $50M" without citation)
</example>

Rules:
- Only include information explicitly present in the website content — never infer or invent
- If a schema field has no matching content, write: [NOT FOUND IN SOURCE]
- If information is ambiguous or unverifiable, flag it inline: [UNVERIFIED]
- If the subject ({people}) does not appear in the content, say so immediately and stop
- Do not reformat output to match the schema structure — write flowing notes
- Always include a "Missing / Unclear" section, even if brief

Return your notes using these sections (add or remove as content warrants):
- Professional Background
- Key Affiliations / Organizations
- Recent Public Activity (posts, interviews, videos — include dates)
- Notable Statements or Positions
- Missing / Unclear"""


VERIFICATION_PROMPT = """You are a fact-checker tasked with verifying whether extracted information is actually supported by the research notes.

For each field in the extracted info, find the specific evidence in the notes that supports or contradicts it.

<extracted_info>
{info}
</extracted_info>

<research_notes>
{notes}
</research_notes>

For every field, assign one of three verdicts:
- VERIFIED   — a specific quote or passage in the notes directly supports this value
- UNVERIFIED — the value may be plausible but no passage in the notes clearly confirms it
- CONTRADICTED — the notes contain information that conflicts with this value

Special rules for multi-company executives:
- If `current_companies` is present and lists multiple organisations, treat `current_company` as the intentional flagship pick — do NOT mark it CONTRADICTED simply because other companies exist. Mark it VERIFIED if the flagship appears in the notes and UNVERIFIED only if the flagship itself is not mentioned.
- `prior_companies` should only be CONTRADICTED if the notes explicitly state the person currently leads a company listed there.

Return a JSON object with this structure:
{{
  "fields": {{
    "<field_name>": {{
      "verdict": "VERIFIED" | "UNVERIFIED" | "CONTRADICTED",
      "evidence": "<exact quote or passage from the notes, or 'No supporting evidence found'>",
      "note": "<brief explanation if UNVERIFIED or CONTRADICTED>"
    }}
  }},
  "overall_confidence": "HIGH" | "MEDIUM" | "LOW",
  "flags": ["<any serious concerns about accuracy>"]
}}

Be strict: only mark VERIFIED if you can quote the evidence directly from the notes."""

BIO_PROMPT = """You are a professional biographer. Write a concise, factual biography of {person} using only the sources below.

Rules:
- Write in third person, past and present tense as appropriate
- Lead with who they are and what they are best known for
- Cover career arc, notable achievements, and current role
- Mention recent public activity (talks, interviews, posts) if present and noteworthy — include dates
- Do NOT invent or infer facts absent from the sources
- If a fact is flagged [UNVERIFIED] in the notes, omit it or add "(unconfirmed)"
- Length: 2–4 paragraphs; be concise and journalist-quality

<wikipedia_summary>
{wiki_summary}
</wikipedia_summary>

<research_notes>
{notes}
</research_notes>

<extracted_facts>
{info}
</extracted_facts>

Write the biography now. Plain prose only — no headings, no bullet points."""


REFLECTION_PROMPT = """You are a research analyst tasked with reviewing the quality and completeness of extracted person information.

The subject being researched: {person}

Compare the extracted information with the required schema:

<Schema>
{schema}
</Schema>

Here is the extracted information:
<extracted_info>
{info}
</extracted_info>

Analyze if all required fields are present and sufficiently populated. Consider:
1. Are any required fields missing?
2. Are any fields incomplete or containing uncertain information?
3. Are there fields with placeholder values or "unknown" markers?

If generating follow-up search queries, always use the subject's actual name ({person}) — never use placeholders like "[Person Name]".
"""

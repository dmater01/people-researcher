import asyncio
from typing import cast, Any, Literal
import json
import os
import re
import httpx

from tavily import AsyncTavilyClient
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_community.utilities import BraveSearchWrapper
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

from agent.configuration import Configuration
from agent.state import InputState, OutputState, OverallState
from agent.utils import deduplicate_and_format_sources, format_all_notes
from agent.prompts import (
    BIO_PROMPT,
    CLASSIFY_PROMPT,
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    VERIFICATION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
)
from agent.schemas import SCHEMAS

# LLMs

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)

def get_model(config: Configuration):
    if config.llm_provider == "ollama":
        return ChatOllama(model=config.ollama_model, temperature=0)
    else:
        return ChatAnthropic(
            model="claude-sonnet-4-5", temperature=0, rate_limiter=rate_limiter
        )

# Search Clients
tavily_async_client = AsyncTavilyClient()

async def brave_search(query: str, max_results: int = 3) -> list[dict]:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_SEARCH_API_KEY not set")
    wrapper = BraveSearchWrapper(api_key=api_key)
    results = await asyncio.to_thread(wrapper.results, query, count=max_results)
    return [{"url": r["link"], "content": r["snippet"], "title": r["title"]} for r in results]

async def serper_search(query: str, max_results: int = 3) -> list[dict]:
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY not set")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results}
        )
        data = response.json()
        results = data.get("organic", [])
        return [{"url": r["link"], "content": r.get("snippet", ""), "title": r.get("title", "")} for r in results]

async def jina_scrape(url: str) -> str:
    """Deep scrape a URL using Jina Reader."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://r.jina.ai/{url}", timeout=10)
            return response.text
    except Exception as e:
        return f"Failed to scrape {url}: {str(e)}"

class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


def classify_subject(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Classify the subject into a type and select the appropriate extraction schema."""
    configurable = Configuration.from_runnable_config(config)

    # If schema was pinned via CLI, skip LLM classification entirely
    if state.subject_type == "custom":
        # Custom JSON schema provided by user — use as-is
        print(f"[classify_subject] Using custom schema: {state.extraction_schema.get('title', '(untitled)')}")
        return {"subject_type": "custom", "extraction_schema": state.extraction_schema}
    if state.subject_type and state.subject_type != "executive":
        schema = SCHEMAS.get(state.subject_type, SCHEMAS["executive"])
        print(f"[classify_subject] Using pinned schema: {state.subject_type}")
        return {"subject_type": state.subject_type, "extraction_schema": schema}

    model = get_model(configurable)

    person_str = state.person.name or state.person.email or "Unknown"
    if state.person.role:
        person_str += f", {state.person.role}"
    if state.person.company:
        person_str += f" at {state.person.company}"

    prompt = CLASSIFY_PROMPT.format(
        person=person_str,
        user_notes=state.user_notes or "",
    )

    result = model.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Classify this person and return JSON."},
    ])

    subject_type = "executive"
    confidence = 0.0
    is_real_person = True
    reasoning = ""
    try:
        raw = str(result.content).strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)
        subject_type = parsed.get("subject_type", "executive")
        confidence = float(parsed.get("confidence", 0.0))
        is_real_person = bool(parsed.get("is_real_person", True))
        reasoning = parsed.get("reasoning", "")
        if subject_type not in SCHEMAS:
            subject_type = "executive"
    except Exception:
        pass

    print(f"[classify_subject] type={subject_type}  confidence={confidence:.2f}  real={is_real_person}  reason={reasoning}")

    # Determine if we should abort before spending API budget on research
    abort_reason = None
    if not is_real_person:
        abort_reason = f"'{person_str}' appears to be fictional or not a real person ({reasoning})"
    elif confidence < configurable.min_classify_confidence:
        abort_reason = (
            f"Name '{person_str}' is too ambiguous to research without more context "
            f"(confidence={confidence:.2f}). Try adding --company, --role, or --notes."
        )

    if abort_reason:
        print(f"[classify_subject] ABORT: {abort_reason}")

    schema = SCHEMAS.get(subject_type, SCHEMAS["executive"])
    return {
        "subject_type": subject_type,
        "extraction_schema": schema,
        "abort_reason": abort_reason,
    }


def route_from_classify(
    state: OverallState,
) -> Literal["generate_queries", END]:  # type: ignore
    """Route to research if subject is valid, or END immediately if fictional/too ambiguous."""
    if state.abort_reason:
        return END
    return "generate_queries"


def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries
    model = get_model(configurable)

    # Generate search queries
    structured_llm = model.with_structured_output(Queries)

    # Format system instructions
    person_str = f"Email: {state.person.email}" if state.person.email else ""
    if state.person.name:
        person_str += f" Name: {state.person.name}"
    if state.person.linkedin:
        person_str += f" LinkedIn URL: {state.person.linkedin}"
    if state.person.role:
        person_str += f" Role: {state.person.role}"
    if state.person.company:
        person_str += f" Company: {state.person.company}"

    query_instructions = QUERY_WRITER_PROMPT.format(
        person=person_str,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )

    # Generate queries
    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        ),
    )

    # Queries
    query_list = [query for query in results.queries]
    return {"search_queries": query_list}


async def research_person(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process."""

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results
    provider = configurable.search_provider
    model = get_model(configurable)

    person_name = state.person.name or (state.person.email.split("@")[0] if state.person.email else "unknown")

    # LinkedIn lookup via Tavily (cached index — Jina is blocked by LinkedIn)
    # If no LinkedIn URL was provided, auto-discover it first via name search.
    linkedin_url: str | None = state.person.linkedin
    linkedin_task = None

    if provider == "tavily" and not linkedin_url:
        try:
            li_discovery = await tavily_async_client.search(
                f'"{person_name}" site:linkedin.com/in',
                max_results=5,
                include_raw_content=False,
                topic="general",
            )
            for r in (li_discovery.get("results", []) if isinstance(li_discovery, dict) else []):
                url = r.get("url", "")
                # Accept only /in/ profile URLs — reject /company/, /posts/, /pub/, /school/
                if (
                    "linkedin.com/in/" in url
                    and not any(p in url for p in ["/posts/", "/activity", "/feed/", "/pulse/", "/detail/"])
                ):
                    linkedin_url = url.split("?")[0].rstrip("/")  # strip tracking params
                    print(f"LinkedIn discovered: {linkedin_url}")
                    break
            if not linkedin_url:
                print("LinkedIn: profile URL not found via search")
        except Exception as e:
            print(f"LinkedIn discovery error: {e}")

    if linkedin_url and provider == "tavily":
        linkedin_task = asyncio.create_task(
            tavily_async_client.search(
                linkedin_url,
                max_results=3,
                include_raw_content=True,
                topic="general",
                include_domains=["linkedin.com"],
            )
        )
        if state.person.linkedin:
            print(f"Fetching LinkedIn: {linkedin_url}")  # user-supplied, already printed above if discovered

    # Web search
    search_tasks = []
    for query in state.search_queries:
        if provider == "tavily":
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    days=configurable.search_days,
                    max_results=max_search_results,
                    include_raw_content=True,
                    topic="general",
                )
            )
        elif provider == "brave":
            search_tasks.append(brave_search(query, max_results=max_search_results))
        elif provider == "serper":
            search_tasks.append(serper_search(query, max_results=max_search_results))

    # Additional targeted searches on social/interview domains (Tavily only)
    if provider == "tavily":
        twitter_domains = ["x.com", "twitter.com"]
        interview_domains = ["podcasts.apple.com", "open.spotify.com", "lexfridman.com",
                             "hubermanlab.com", "tim.blog", "ycombinator.com", "techcrunch.com",
                             "theverge.com", "wired.com", "bloomberg.com", "ft.com",
                             "wsj.com", "forbes.com", "businessinsider.com"]
        search_tasks.append(
            tavily_async_client.search(
                f"{person_name}",
                days=configurable.search_days,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
                include_domains=twitter_domains,
            )
        )
        search_tasks.append(
            tavily_async_client.search(
                f"{person_name} interview podcast talk",
                days=configurable.search_days,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
                include_domains=interview_domains,
            )
        )

    # Execute all searches concurrently (LinkedIn task already running in background)
    search_results = await asyncio.gather(*search_tasks)

    # Standardize results for Tavily vs others
    all_docs = []
    if provider == "tavily":
        for res in search_results:
            if isinstance(res, dict) and "results" in res:
                all_docs.extend(res["results"])
            else:
                all_docs.extend(res)
    else:
        for res in search_results:
            all_docs.extend(res)

    # Collect LinkedIn result — keep only the actual profile page, skip posts/activity
    linkedin_url_confirmed: str | None = None
    if linkedin_task:
        linkedin_res = await linkedin_task
        linkedin_docs = linkedin_res.get("results", []) if isinstance(linkedin_res, dict) else []
        profile_added = False
        for doc in linkedin_docs:
            url = doc.get("url", "")
            # Skip posts, activity, and comment pages — only use profile URLs
            if any(p in url for p in ["/posts/", "/activity", "/feed/", "/pulse/"]):
                continue
            if "linkedin.com/in/" not in url:
                continue
            content = (doc.get("raw_content") or doc.get("content") or "").strip()
            if content:
                all_docs.append({
                    "url": url,
                    "content": content[:4000],
                    "title": f"LinkedIn profile — {doc.get('title', state.person.name or state.person.email)}",
                })
                linkedin_url_confirmed = url.split("?")[0].rstrip("/")
                profile_added = True
                # Print first substantive line
                for line in content.splitlines():
                    line = line.strip()
                    if len(line) > 20 and not line.startswith("http") and not line.startswith("["):
                        print(f"LinkedIn profile: {line[:120]}")
                        break
                break
        if not profile_added:
            print("LinkedIn: no profile content retrieved")

    # Company website scrape — search for the person's bio on their current company site
    if state.person.company and provider == "tavily":
        company_res = await tavily_async_client.search(
            f"{state.person.name or person_name} {state.person.company} founder CEO role",
            max_results=3,
            include_raw_content=True,
            topic="general",
        )
        for doc in (company_res.get("results", []) if isinstance(company_res, dict) else []):
            content = (doc.get("raw_content") or doc.get("content") or "").strip()
            if content:
                all_docs.append({
                    "url": doc["url"],
                    "content": content[:3000],
                    "title": f"Company bio — {doc.get('title', state.person.company)}",
                })
                print(f"Company source: {doc['url']}")
                break

    # --- YouTube channel discovery (first pass only) ---
    # Find the person's YouTube channel URL, then Jina-scrape it for recent videos.
    # Skip on reflection loops — channel won't have changed and re-discovery adds noise.
    yt_channel_out: str | None = None
    yt_videos_out: list[dict] = []
    if provider == "tavily" and state.reflection_steps_taken == 0:
        yt_discovery = await tavily_async_client.search(
            f"{person_name} youtube channel",
            max_results=3,
            include_raw_content=False,
            topic="general",
            include_domains=["youtube.com"],
        )
        yt_results = yt_discovery.get("results", []) if isinstance(yt_discovery, dict) else []
        # Normalise a raw YouTube URL to the channel root
        _YT_TAB_SUFFIXES = ("/videos", "/playlists", "/about", "/community",
                            "/shorts", "/streams", "/featured")

        def _yt_channel_root(url: str) -> str | None:
            """Return the desktop channel root URL, or None if it's a video/watch page."""
            if "/watch?" in url or "/shorts/" in url or "/live/" in url:
                return None
            # Normalize mobile and music URLs to desktop
            url = url.replace("m.youtube.com", "www.youtube.com")
            url = url.replace("music.youtube.com", "www.youtube.com")
            for suffix in _YT_TAB_SUFFIXES:
                if url.rstrip("/").endswith(suffix):
                    url = url.rstrip("/")[: -len(suffix)]
            return url.rstrip("/")

        # Prefer @handle URLs, then /channel/, then /user/
        channel_url = None
        for pattern in ["/@", "/channel/", "/user/"]:
            for r in yt_results:
                root = _yt_channel_root(r.get("url", ""))
                if root and pattern in root:
                    channel_url = root
                    break
            if channel_url:
                break
        if not channel_url and yt_results:
            channel_url = _yt_channel_root(yt_results[0].get("url", ""))

        if channel_url:
            handle = next(
                (seg for seg in channel_url.split("/") if seg.startswith("@")),
                None,
            )
            channel_label = handle or channel_url
            print(f"YouTube channel found: {channel_label} ({channel_url})")

            # Scrape About and Videos tab concurrently
            about_content, videos_content = await asyncio.gather(
                jina_scrape(f"{channel_url}/about"),
                jina_scrape(f"{channel_url}/videos"),
            )

            # Parse video titles + URLs — try strict heading pattern first, fall back to any link
            _DURATION_RE = re.compile(r"^\d+:\d+")
            video_entries = re.findall(
                r"###\s+\[([^\]]+)\]\((https://www\.youtube\.com/watch[^\s\)\"]+)",
                videos_content,
            )
            if not video_entries:
                # Fallback: grab all youtube watch links with their link text
                raw = re.findall(
                    r"\[([^\]]{8,})\]\((https://www\.youtube\.com/watch\?v=[^)\s\"]+)",
                    videos_content,
                )
                # Deduplicate by URL, skip duration-only labels ("18:59 18:59 Now playing")
                seen_urls: set[str] = set()
                for title, url in raw:
                    if url in seen_urls or _DURATION_RE.match(title.strip()):
                        continue
                    seen_urls.add(url)
                    video_entries.append((title.strip(), url))

            print(f"\n[YouTube Videos — {channel_label}] ({len(video_entries)} found)")
            for title, url in video_entries:
                print(f"  • {title}  →  {url}")

            # Build a clean plain-text video list for the LLM
            video_list_text = "\n".join(
                f"- {title} ({url})" for title, url in video_entries
            ) or videos_content[:4000]

            all_docs.append({
                "url": f"{channel_url}/about",
                "content": about_content[:3000],
                "title": f"YouTube channel About — {channel_label}",
            })
            all_docs.append({
                "url": f"{channel_url}/videos",
                "content": video_list_text,
                "title": f"YouTube video list — {channel_label} ({len(video_entries)} videos)",
            })

            # Store for output file
            yt_channel_out = channel_url
            yt_videos_out = [{"title": t, "url": u} for t, u in video_entries]

        else:
            print(f"No personal YouTube channel found for {person_name} — searching for featured videos...")
            # Fall back: find YouTube videos the person appears in (interviews, talks, etc.)
            yt_featured = await tavily_async_client.search(
                f"{person_name} interview talk lecture",
                days=configurable.search_days,
                max_results=5,
                include_raw_content=False,
                topic="general",
                include_domains=["youtube.com"],
            )
            featured_results = yt_featured.get("results", []) if isinstance(yt_featured, dict) else []
            # Only keep video watch pages
            video_results = [r for r in featured_results if "/watch?" in r.get("url", "")]
            if video_results:
                print(f"\n[YouTube Featured Videos — {person_name}] ({len(video_results)} found)")
                video_list_lines = []
                for r in video_results:
                    title = r.get("title", "Untitled")
                    url = r.get("url", "")
                    snippet = r.get("content", "")[:120].replace("\n", " ")
                    print(f"  • {title}")
                    print(f"    {url}")
                    video_list_lines.append(f"- {title} ({url}): {snippet}")
                all_docs.append({
                    "url": "https://youtube.com/search",
                    "content": "\n".join(video_list_lines),
                    "title": f"YouTube videos featuring {person_name} ({len(video_results)} results)",
                })
                # Store for output file
                yt_videos_out = [{"title": r.get("title", "Untitled"), "url": r.get("url", "")} for r in video_results]
            else:
                print(f"No YouTube videos found for {person_name}")

    # Optional Deep Scrape with Jina (top non-YouTube/social URLs)
    if configurable.enable_deep_scrape:
        skip_domains = {"x.com", "twitter.com", "youtube.com",
                        "open.spotify.com", "podcasts.apple.com"}
        scrapeable = [
            d["url"] for d in all_docs
            if not any(s in d.get("url", "") for s in skip_domains)
        ]
        unique_urls = list(dict.fromkeys(scrapeable))[:2]
        scraped_contents = await asyncio.gather(*[jina_scrape(u) for u in unique_urls])
        for url, content in zip(unique_urls, scraped_contents):
            all_docs.append({
                "url": url,
                "content": content[:5000],
                "title": f"Deep Scrape of {url}",
            })

    # Deduplicate and format sources
    # Note: deduplicate_and_format_sources expects a specific format if it's from Tavily
    # We'll just pass all_docs. If they have 'raw_content', it uses it.
    source_str = deduplicate_and_format_sources(
        all_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        people=state.person.model_dump(),
        user_notes=state.user_notes,
    )
    result = await model.ainvoke(p)
    out: dict[str, Any] = {"completed_notes": [str(result.content)]}
    if yt_videos_out:
        out["youtube_videos"] = yt_videos_out
    if yt_channel_out:
        out["youtube_channel"] = yt_channel_out
    # Prefer confirmed profile URL; fall back to discovered URL if content wasn't retrieved
    final_linkedin = linkedin_url_confirmed or linkedin_url
    if final_linkedin:
        out["linkedin_url"] = final_linkedin
    return out


def gather_notes_extract_schema(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""
    configurable = Configuration.from_runnable_config(config)
    model = get_model(configurable)

    # Format all notes
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    # Sanitize the title for Anthropic tool-name requirements: ^[a-zA-Z0-9_-]{1,128}$
    safe_schema = dict(state.extraction_schema)
    if "title" in safe_schema:
        safe_schema["title"] = re.sub(r"[^a-zA-Z0-9_-]", "_", safe_schema["title"])[:128]

    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )
    structured_llm = model.with_structured_output(safe_schema)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Produce a structured output from these notes. For each field, if possible, include a 'source' and 'confidence' (0-1).",
            },
        ]
    )
    return {"info": result}


def verify_extraction(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Cross-check every extracted field against the raw research notes."""
    configurable = Configuration.from_runnable_config(config)
    model = get_model(configurable)

    notes = format_all_notes(state.completed_notes)
    prompt = VERIFICATION_PROMPT.format(
        info=json.dumps(state.info, indent=2),
        notes=notes,
    )

    result = model.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Verify each extracted field and return the JSON verdict."},
    ])

    try:
        # Strip markdown fences if present
        raw = result.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        verification = json.loads(raw)
    except Exception:
        verification = {"error": "Could not parse verification output", "raw": result.content}

    # Print a compact truth table to the terminal
    print("\n[Verification]")
    fields = verification.get("fields", {})
    for field, detail in fields.items():
        verdict = detail.get("verdict", "?")
        icon = {"VERIFIED": "✓", "UNVERIFIED": "?", "CONTRADICTED": "✗"}.get(verdict, "?")
        print(f"  {icon} {field}: {verdict}")
        if verdict != "VERIFIED":
            print(f"    → {detail.get('note') or detail.get('evidence', '')}")
    confidence = verification.get("overall_confidence", "?")
    flags = verification.get("flags", [])
    print(f"  Overall confidence: {confidence}")
    for flag in flags:
        print(f"  ⚠ {flag}")

    return {"verification": verification}


def reflection(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    configurable = Configuration.from_runnable_config(config)
    model = get_model(configurable)
    structured_llm = model.with_structured_output(ReflectionOutput)

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        person=state.person.name or state.person.email or "Unknown",
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info,
    )

    # Invoke
    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."},
            ]
        ),
    )

    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }


async def generate_bio(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Fetch a Wikipedia summary and synthesize a narrative biography."""
    configurable = Configuration.from_runnable_config(config)
    model = get_model(configurable)

    person_name = state.person.name or state.person.email or "Unknown"
    wiki_summary = ""
    bio_sources: list[str] = []

    # 1. Try Wikipedia REST API first
    encoded = person_name.replace(" ", "_")
    wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(wiki_url, follow_redirects=True)
            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract", "")
                page_type = data.get("type", "")
                # Reject disambiguation pages and stub extracts
                if page_type != "disambiguation" and len(extract) > 150:
                    wiki_summary = extract
                    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    if page_url:
                        bio_sources.append(page_url)
                    print(f"Wikipedia: found article for '{person_name}' ({len(extract)} chars)")
                else:
                    print(f"Wikipedia: disambiguation or stub for '{person_name}' — skipping")
    except Exception as e:
        print(f"Wikipedia API error: {e}")

    # 2. Fallback: Tavily search scoped to Wikipedia
    if not wiki_summary:
        try:
            wiki_search = await tavily_async_client.search(
                f"{person_name} biography",
                max_results=3,
                include_raw_content=False,
                topic="general",
                include_domains=["en.wikipedia.org", "britannica.com"],
            )
            wiki_results = wiki_search.get("results", []) if isinstance(wiki_search, dict) else []
            # Pick English Wikipedia first, then any Wikipedia, then Britannica
            for r in wiki_results:
                if "en.wikipedia.org/wiki/" in r.get("url", ""):
                    wiki_summary = r.get("content", "")[:2000]
                    bio_sources.append(r["url"])
                    print(f"Wikipedia fallback: {r['url']}")
                    break
            if not wiki_summary:
                for r in wiki_results:
                    if "wikipedia.org/wiki/" in r.get("url", ""):
                        wiki_summary = r.get("content", "")[:2000]
                        bio_sources.append(r["url"])
                        print(f"Wikipedia fallback (non-EN): {r['url']}")
                        break
            # If still nothing, use Britannica snippet
            if not wiki_summary and wiki_results:
                wiki_summary = wiki_results[0].get("content", "")[:2000]
                bio_sources.append(wiki_results[0].get("url", ""))
                print(f"Bio fallback source: {wiki_results[0].get('url', '')}")
        except Exception as e:
            print(f"Bio search fallback error: {e}")

    if not wiki_summary:
        wiki_summary = "[No Wikipedia or encyclopedic summary found]"

    # 3. Synthesize bio with LLM
    notes = format_all_notes(state.completed_notes)
    prompt = BIO_PROMPT.format(
        person=person_name,
        wiki_summary=wiki_summary,
        notes=notes,
        info=json.dumps(state.info, indent=2) if state.info else "{}",
    )

    result = await model.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Write the biography of {person_name}."},
    ])

    bio_text = str(result.content).strip()
    print(f"\n--- Biography ({len(bio_text)} chars) ---\n{bio_text}\n")
    return {"bio": bio_text, "bio_sources": bio_sources}


def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_person", "generate_bio"]:  # type: ignore
    """Route the graph based on the reflection output."""
    configurable = Configuration.from_runnable_config(config)
    terminal = "generate_bio" if configurable.generate_bio else END

    if state.is_satisfactory:
        return terminal

    # Abort early if extraction is mostly unknown — more loops won't help
    if state.info:
        unknown_count = sum(
            1 for v in state.info.values()
            if v in (None, "", 0, [], "<UNKNOWN>", "UNKNOWN")
        )
        total = len(state.info)
        if total > 0 and unknown_count / total >= 0.5:
            print(f"Stopping: {unknown_count}/{total} fields unresolvable — more context needed.")
            return terminal

    # Abort early if verification shows persistent contradictions on core fields
    if state.verification:
        fields = state.verification.get("fields", {})
        contradicted = [f for f, d in fields.items() if d.get("verdict") == "CONTRADICTED"]
        if contradicted and state.reflection_steps_taken >= configurable.max_reflection_steps:
            print(f"Stopping: fields still contradicted after max reflections: {contradicted}")
            return terminal

    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_person"

    return terminal


# Add nodes and edges
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
builder.add_node("classify_subject", classify_subject)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("generate_queries", generate_queries)
builder.add_node("research_person", research_person)
builder.add_node("verify_extraction", verify_extraction)
builder.add_node("reflection", reflection)
builder.add_node("generate_bio", generate_bio)

builder.add_edge(START, "classify_subject")
builder.add_conditional_edges("classify_subject", route_from_classify)
builder.add_edge("generate_queries", "research_person")
builder.add_edge("research_person", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "verify_extraction")
builder.add_edge("verify_extraction", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)
builder.add_edge("generate_bio", END)

# Compile
graph = builder.compile()

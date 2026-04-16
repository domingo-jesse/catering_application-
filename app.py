"""
CaterCraft AI Streamlit App

How to set your API key:
1) Streamlit secrets (recommended): create `.streamlit/secrets.toml` with:
   OPENAI_API_KEY = "your_api_key_here"
2) Or export an environment variable before running:
   export OPENAI_API_KEY="your_api_key_here"
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List
from urllib.parse import quote_plus

import streamlit as st


def load_api_key() -> str | None:
    """Read API key from Streamlit secrets first, then environment variable."""
    return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def build_prompt(
    event_type: str,
    guests: int,
    budget: float,
    cuisine: str,
    dietary: str,
    vibe: str,
) -> str:
    budget_per_person = budget / guests if guests else 0
    return f"""
You are a senior catering strategist and chef.
Generate exactly 3 distinct catering menu options for the event below.

EVENT DETAILS
- Event type: {event_type}
- Number of guests: {guests}
- Total budget (USD): {budget:.2f}
- Budget per person target (USD): {budget_per_person:.2f}
- Cuisine preferences: {cuisine}
- Dietary restrictions: {dietary}
- Event vibe/style: {vibe}

CRITICAL RULES
1) Return VALID JSON only. No markdown. No code fences. No commentary.
2) Return exactly this top-level object shape:
{{
  "menus": [
    {{
      "title": "...",
      "theme": "...",
      "appetizer": "...",
      "main": "...",
      "side": "...",
      "dessert": "...",
      "drink": "...",
      "cost_per_person": 0,
      "total_estimated_cost": 0,
      "notes": "...",
      "recipes": {{
        "appetizer": "step-by-step recipe",
        "main": "step-by-step recipe",
        "side": "step-by-step recipe",
        "dessert": "step-by-step recipe",
        "drink": "step-by-step recipe"
      }}
    }}
  ],
  "design_explanation": "How this menu set was designed, including pricing logic and constraints handling"
}}
3) Menus must be realistic, premium, and clearly distinct.
4) Respect dietary restrictions exactly.
5) Ground pricing in the provided budget and guest count.
6) cost_per_person and total_estimated_cost must be numbers, not strings.
7) total_estimated_cost should approximately equal cost_per_person * guests.
8) Keep notes practical and concise (prep/service considerations).
""".strip()


def extract_json(text: str) -> str:
    """Best-effort JSON extraction for malformed responses."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    raise ValueError("No JSON object found in model response")


def parse_menu_json(raw_text: str) -> Dict[str, Any]:
    payload = json.loads(extract_json(raw_text))

    if not isinstance(payload, dict) or "menus" not in payload:
        raise ValueError("Response JSON missing top-level 'menus' field")

    menus = payload.get("menus")
    if not isinstance(menus, list) or len(menus) != 3:
        raise ValueError("Expected exactly 3 menu options in 'menus'")

    required_fields = {
        "title",
        "theme",
        "appetizer",
        "main",
        "side",
        "dessert",
        "drink",
        "cost_per_person",
        "total_estimated_cost",
        "notes",
        "recipes",
    }

    for idx, menu in enumerate(menus, start=1):
        missing = required_fields - set(menu.keys())
        if missing:
            raise ValueError(f"Menu {idx} missing fields: {', '.join(sorted(missing))}")
        if not isinstance(menu["recipes"], dict):
            raise ValueError(f"Menu {idx} 'recipes' must be an object")

    return payload


def generate_recipe_search_link(dish_name: str) -> str:
    query = quote_plus(f"{dish_name} recipe")
    return f"https://www.google.com/search?q={query}"


def get_openai_client(api_key: str):
    """Create an OpenAI client while handling missing SDK dependency."""
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `openai` package is not installed. Add `openai` to your dependencies."
        ) from exc

    return OpenAI(api_key=api_key)


def call_openai_for_menus(client: Any, prompt: str) -> Dict[str, Any]:
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": "You produce reliable JSON for catering menu planning.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    if not response.output_text:
        raise ValueError("OpenAI returned an empty response.")

    return parse_menu_json(response.output_text)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main {background: linear-gradient(180deg, #fffdf8 0%, #fff9f1 100%);}
            .hero {
                padding: 1.5rem 1.8rem;
                border-radius: 20px;
                background: linear-gradient(120deg, #fdf4ff 0%, #eef7ff 100%);
                color: #1f2a44;
                box-shadow: 0 8px 24px rgba(31, 42, 68, 0.12);
                margin-bottom: 1rem;
            }
            .hero h1 {
                margin: 0;
                font-size: 2rem;
                letter-spacing: 0.4px;
                display: inline-block;
                padding: 0.2rem 0.7rem;
                border-radius: 12px;
                background: linear-gradient(90deg, #ffb1df 0%, #ff8bd0 55%, #ff70c4 100%);
                color: #3d2331;
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.3);
            }
            .hero p {margin: 0.45rem 0 0; opacity: 0.95;}
            .menu-card {
                background: #ffffff;
                border-radius: 18px;
                padding: 1.25rem;
                border: 1px solid #f1e2d2;
                box-shadow: 0 8px 20px rgba(117, 79, 46, 0.10);
                margin: 1rem 0;
            }
            .menu-title {font-size: 1.35rem; font-weight: 700; color: #4b2e2e; margin-bottom: 0.25rem;}
            .menu-theme {font-size: 0.95rem; color: #8a5a35; margin-bottom: 0.85rem;}
            .chip {
                display: inline-block;
                padding: 0.3rem 0.55rem;
                border-radius: 999px;
                margin: 0.2rem 0.35rem 0.2rem 0;
                background: #fdf0e3;
                border: 1px solid #f1d2b5;
                color: #6d3f19;
                font-size: 0.85rem;
            }
            .divider {height: 1px; background: #f0dfcd; margin: 0.95rem 0 0.85rem;}
            .section-heading {font-weight: 650; color: #5a3518; margin: 0.3rem 0 0.55rem;}
            .footer-note {
                margin-top: 1.4rem;
                padding: 0.9rem 1rem;
                background: #fff5e9;
                border-radius: 12px;
                border: 1px solid #f1dcc4;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_menu_card(menu: Dict[str, Any], idx: int) -> None:
    st.markdown(f"<div class='menu-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='menu-title'>Menu {idx}: {menu['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='menu-theme'>{menu['theme']}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-heading'>Course Lineup</div>", unsafe_allow_html=True)
    st.markdown(
        "".join(
            [
                f"<span class='chip'>Appetizer: {menu['appetizer']}</span>",
                f"<span class='chip'>Main: {menu['main']}</span>",
                f"<span class='chip'>Side: {menu['side']}</span>",
                f"<span class='chip'>Dessert: {menu['dessert']}</span>",
                f"<span class='chip'>Drink: {menu['drink']}</span>",
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated Cost / Person", f"${menu['cost_per_person']:.2f}")
    with col2:
        st.metric("Estimated Total Event Cost", f"${menu['total_estimated_cost']:.2f}")

    st.markdown("<div class='section-heading'>Preparation Notes</div>", unsafe_allow_html=True)
    st.write(menu["notes"])

    st.markdown("<div class='section-heading'>Recipe search links</div>", unsafe_allow_html=True)
    for course in ["appetizer", "main", "side", "dessert", "drink"]:
        dish = menu[course]
        st.markdown(f"- **{dish}** — [Recipe search link]({generate_recipe_search_link(dish)})")

    st.markdown("<div class='section-heading'>AI-generated recipe instructions</div>", unsafe_allow_html=True)
    recipes = menu.get("recipes", {})
    for course in ["appetizer", "main", "side", "dessert", "drink"]:
        st.markdown(f"**{course.capitalize()} ({menu[course]}):**")
        st.write(recipes.get(course, "No recipe instructions provided."))

    st.markdown("</div>", unsafe_allow_html=True)


def app() -> None:
    st.set_page_config(page_title="Catering with Olive", page_icon="🍽️", layout="wide")
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>🍽️ Catering with Olive</h1>
            <p>Premium catering menu concepts tailored to your event goals, budget, and guest profile.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_key = load_api_key()
    if not api_key:
        st.error(
            "Missing OpenAI API key. Add OPENAI_API_KEY to Streamlit secrets or environment variables and restart the app."
        )
        st.stop()

    with st.container(border=True):
        st.subheader("Event Brief")
        sample_text = (
            "Ex: Elegant rooftop engagement party for 80 guests, $6,000 budget, "
            "Mediterranean preference, pescatarian-friendly, warm modern vibe."
        )
        summary = st.text_area(
            "Describe your event",
            placeholder=sample_text,
            height=110,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            event_type = st.text_input("Event type", value="Corporate networking dinner")
            guests = st.number_input("Number of guests", min_value=5, max_value=2000, value=120, step=5)
        with c2:
            budget = st.number_input("Total budget (USD)", min_value=200.0, value=8500.0, step=100.0)
            cuisine = st.text_input("Cuisine preferences", value="Californian, Mediterranean")
        with c3:
            dietary = st.text_input("Dietary restrictions", value="Nut-free, vegetarian options")
            vibe = st.text_input("Event vibe/style", value="Modern upscale with relaxed energy")

        generate_col, regen_col = st.columns([1, 1])
        with generate_col:
            generate_clicked = st.button("Generate Menus", use_container_width=True, type="primary")
        with regen_col:
            regenerate_clicked = st.button("Regenerate Menus", use_container_width=True)

    should_run = generate_clicked or regenerate_clicked

    if should_run:
        if not event_type.strip() or guests <= 0 or budget <= 0:
            st.error("Please provide valid event type, guest count, and budget.")
            st.stop()

        prompt = build_prompt(
            event_type=event_type,
            guests=int(guests),
            budget=float(budget),
            cuisine=(cuisine + (f" | Extra brief: {summary}" if summary else "")).strip(),
            dietary=dietary,
            vibe=vibe,
        )

        try:
            client = get_openai_client(api_key=api_key)
        except ModuleNotFoundError as exc:
            st.error(str(exc))
            st.info("Install with: `pip install openai` and restart the app.")
            st.stop()

        with st.spinner("Designing premium menu concepts..."):
            try:
                result = call_openai_for_menus(client, prompt)
                st.session_state["generated_result"] = result
            except json.JSONDecodeError:
                st.error("We received an invalid JSON response. Please regenerate menus.")
            except Exception as exc:
                st.error(f"Could not generate menus right now: {exc}")

    result = st.session_state.get("generated_result")
    if result:
        st.subheader("Your Catering Concepts")
        for i, menu in enumerate(result.get("menus", []), start=1):
            try:
                render_menu_card(menu, i)
            except Exception as exc:
                st.warning(f"Menu {i} could not be fully rendered: {exc}")

        st.markdown("### How this menu was designed")
        st.markdown(
            f"<div class='footer-note'>{result.get('design_explanation', 'Built from your constraints with balanced pricing and operational realism.')}</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    app()

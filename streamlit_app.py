import os
import json
import time
import smtplib
import ssl
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import io
import pandas as pd
from serpapi import Client
from PyPDF2 import PdfReader
import streamlit as st
import openai
from openai import OpenAI          # ← add
from xlsxwriter import Workbook

# ==============================================================================
# --- ✅ 1. User Configuration ---
# ==============================================================================


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")

if not OPENAI_API_KEY or not SERPAPI_KEY:
    st.error(
        "Missing OPENAI_API_KEY or SERPAPI_KEY.\n"
        "Add them in Streamlit Cloud (Settings → Secrets) OR create `.streamlit/secrets.toml` locally "
        "OR export them as environment variables before running."
    )
    st.stop()

oa_client = OpenAI(api_key=OPENAI_API_KEY)   # ← good
  


# ==============================================================================
# --- ⚙️ 2. Core Application Logic ---
# ==============================================================================

def read_pdf_text(file_like_object):
    """Extracts text content from a PDF file-like object."""
    text = ""
    try:
        reader = PdfReader(file_like_object)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# CHANGE 2: Replaced the entire 'call_gemini_with_backoff' function
# with a new, universal function for calling the OpenAI API.
def call_openai_with_backoff(system_prompt, user_prompt, is_json=False, max_retries=5):
    retries = 0
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    while retries < max_retries:
        try:
            kwargs = {"model": "gpt-5-nano", "messages": messages}
            if is_json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = oa_client.chat.completions.create(**kwargs)
            time.sleep(0.5)
            return resp.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait = (2 ** retries) + 1
            st.toast(f"Rate limited; retrying in {wait}s…")
            time.sleep(wait); retries += 1
        except Exception as e:
            st.warning(f"OpenAI error: {e}")
            return None
    st.error("Max retries reached."); return None


# CHANGE 3: The 'generate_cover_letter_openai' function is now a simple wrapper
# around the new centralized 'call_openai_with_backoff' function.
def generate_cover_letter(company_name, fit_analysis):
    """Uses OpenAI GPT-5 Nano to generate a cover letter."""
    st.write(f"    - ✍️ Writing cover letter for {company_name}...")
    
    system_prompt = "You are an expert career coach who writes compelling, concise, and professional cover letters under 250 words."
    user_prompt = f"""
    Based on the following analysis of my skills and the job requirements, write a professional cover letter for a role at {company_name}.

    The letter must be written from the perspective of the job applicant. It must highlight why I am a strong fit by directly referencing the key points from the 'Fit Analysis' provided below. Keep the tone enthusiastic and professional.

    **Fit Analysis:**
    {fit_analysis}
    """
    
    cover_letter = call_openai_with_backoff(system_prompt, user_prompt)
    
    if not cover_letter:
        st.warning(f"Could not generate cover letter for {company_name}.")
        return "Error: Could not generate cover letter."
    return cover_letter

def search_and_extract_jobs(query, serpapi_key, num_results=1):
    st.write(f"    - 🔍 Searching Google Jobs for: '{query}'")
    extracted_jobs = []
    try:
        params = {"q": query, "engine": "google_jobs", "num": str(num_results)}
        serp = Client(api_key=serpapi_key)     # ← renamed
        result = serp.search(params)
        jobs_results = result.get("jobs_results", [])

        if "error" in result:
            st.error(f"    - ❌ SerpApi search error: {result.get('error')}")
            return []
        if not jobs_results:
            st.warning(f"    - ⚠️ No jobs found on Google for query: '{query}'.")
            return []

        for job in jobs_results:
            application_url = (job.get("apply_options", [{}])[0].get("link", ""))
            extracted_jobs.append({
                "job_id": job.get("job_id"), "title": job.get("title"),
                "company": job.get("company_name"), "description": job.get("description"),
                "application_url": application_url
            })
        return extracted_jobs
    except Exception as e:
        st.error(f"    - ❌ An unexpected error occurred during SerpApi search: {e}")
        return []

# CHANGE 4: This function now calls the new OpenAI function instead of Gemini.
def generate_initial_search_queries(preferences, resume_text):
    """Uses OpenAI to generate initial search queries."""
    system_prompt = """
    You are an expert career search strategist. Your task is to generate 2 diverse and effective Google search queries.
    
    **VERY IMPORTANT INSTRUCTIONS:**
    1. The queries MUST be simple strings. For example: "Executive Communication Director Houston TX".
    2. **DO NOT USE BOOLEAN OPERATORS.** Do not use "OR", "AND", or parentheses.
    3. Return a valid JSON object with a single key "queries", which is an array of these simple query strings.
    """
    user_prompt = f"""
    Based on the following job preferences and resume summary, generate the queries.
    
    **Preferences:** {json.dumps(preferences, indent=2)}
    **Resume Summary:** {resume_text[:1000]}
    """
    response_text = call_openai_with_backoff(system_prompt, user_prompt, is_json=True)
    if not response_text:
        return []
    try:
        return json.loads(response_text).get("queries", [])
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON for initial queries. Error: {e}")
        return []

# CHANGE 5: This function now also calls the new OpenAI function instead of Gemini.
def analyze_job_posting(job_text, resume_text, preferences, url):
    """Uses OpenAI to analyze a job posting against a resume."""
    st.write(f"    - 🤔 Analyzing job posting from: {url or 'Source not found'}")
    
    system_prompt = """
    You are a meticulous hiring manager. Your task is to analyze the provided job description and determine if the candidate's resume is a strong match based on their stated preferences.
    Your response MUST be a single, valid JSON object without any markdown formatting.

    **Response Format:**
    - If it is a good match, the JSON must contain:
      `"is_match": true`,
      `"company": "Extracted Company Name"`,
      `"role": "Extracted Role Title"`,
      `"salary": "Extracted Salary or 'Not specified'"`,
      `"reason_for_fit":` an array of objects, each with "job_requirement", "resume_evidence", and "explanation".
    - If it is NOT a match, the JSON must contain: `"is_match": false`, `"reason_for_fit": []`.

    **CRITICAL INSTRUCTIONS:**
    1. Base your analysis *strictly* on the provided Resume, Preferences, and Job Description.
    2. Extract the company name and role title directly from the Job Description text.
    """
    user_prompt = f"""
    ---
    **Job Preferences:** {json.dumps(preferences, indent=2)}
    ---
    **Candidate Resume:** {resume_text}
    ---
    **Job Description:** {job_text}
    ---
    """
    
    response_text = call_openai_with_backoff(system_prompt, user_prompt, is_json=True)
    if not response_text:
        return {"is_match": False, "reason_for_fit": []}
    try:
        analysis_result = json.loads(response_text)
        analysis_result['application_url'] = url
        return analysis_result
    except json.JSONDecodeError as e:
        st.warning(f"Failed to decode JSON from job analysis: {e}. Raw text was: {response_text}")
        return {"is_match": False, "reason_for_fit": []}

# ==============================================================================
# --- 🖥️ 3. Streamlit UI Application ---
# ==============================================================================

st.set_page_config(page_title="MatchmAIker: AI Matching Resumes to Job Descriptions", layout="wide")
st.title("🚀 MatchmAIker: AI Matching Resumes to Job Descriptions")
st.markdown("This app now uses **OpenAI GPT-5 Nano** to find and analyze job postings based on your resume and preferences.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Note: we are using the free tier of AI so there will be daily limits.")
    EXCEL_FILENAME = "job_search_results.xlsx"

with st.form(key="job_search_form"):
    st.header("📝 Your Job Preferences")
    col1, col2 = st.columns(2)
    with col1:
        job_titles = st.text_input("Desired Job Titles (comma-separated)", "Executive Communication Director, Internal Communications Director")
        location = st.text_input("Location(s) or 'remote'", "Houston Texas or Remote")
    with col2:
        employment_type = st.text_input("Employment Type", "Full Time")
        salary = st.text_input("Desired Salary", "$150,000")
    uploaded_resume = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
    submit_button = st.form_submit_button(label="✨ Start AI Job Search")

if submit_button:
    # CHANGE 6: Updated the API key check to only look for OpenAI and SerpApi keys.
    if "REDACTED" in SERPAPI_KEY or "REDACTED" in OPENAI_API_KEY:
        st.error("Please hardcode your SerpApi and OpenAI API keys at the top of the script.")
    elif not uploaded_resume:
        st.error("Please upload your resume.")
    else:
        # CHANGE 7: Removed the 'genai.configure' block as it's no longer needed.
        preferences = {
            "job_titles": job_titles, "location": location,
            "employment_type": employment_type, "salary": salary
        }
        resume_text = read_pdf_text(uploaded_resume)
        if not resume_text:
            st.error("Could not read text from the uploaded PDF. Please try another file.")
            st.stop()

        with st.status("🤖 AI agent at work...", expanded=True) as status:
            st.write("[Phase 1] Reading resume and generating initial search queries...")
            initial_queries = generate_initial_search_queries(preferences, resume_text)

            if not initial_queries:
                status.update(label="Error!", state="error", expanded=False)
                st.error("Could not generate initial search queries. Check OpenAI API key and try different preferences. Exiting.")
                st.stop()
            st.write(f"✅ AI generated queries: {initial_queries}")

            st.write("\n--- [Phase 2] Searching for jobs and extracting details ---")
            unique_jobs = {}
            for query in initial_queries:
                job_posts = search_and_extract_jobs(query, SERPAPI_KEY)
                if job_posts:
                    st.write(f"    - ✅ Extracted {len(job_posts)} job details for query '{query}'.")
                    for post in job_posts:
                        if post.get("job_id"):
                            unique_jobs[post["job_id"]] = post

            st.write(f"\n--- [Phase 3] Analyzing {len(unique_jobs)} unique job postings ---")
            if not unique_jobs:
                st.warning("Initial search did not return any jobs to analyze. Try broader search terms.")
                status.update(label="Search complete!", state="complete", expanded=False)
                st.stop()

            all_found_jobs = []
            job_progress_bar = st.progress(0, text=f"Analyzing 0 / {len(unique_jobs)} jobs...")
            for i, (job_id, details) in enumerate(unique_jobs.items()):
                job_progress_bar.progress((i + 1) / len(unique_jobs), text=f"Analyzing {i+1} / {len(unique_jobs)} jobs...")
                if details and details.get("description") and len(details.get("description")) > 100:
                    # CHANGE 9: Pass the OpenAI API key to the analysis function.
                    analysis = analyze_job_posting(
                        job_text=details["description"], resume_text=resume_text,
                        preferences=preferences, url=details["application_url"]
                    )

                    if analysis.get("is_match"):
                        all_found_jobs.append(analysis)
                        st.success(f"    - ✅ MATCH FOUND: {analysis.get('role')} at {analysis.get('company')}")
                else:
                    st.warning(f"    - ⚠️ Skipping job ID {job_id[:20]}... due to insufficient description.")
            
            st.write("\n--- [Phase 4] Generating cover letters for matched jobs ---")
            if all_found_jobs:
                jobs_list_for_df = []
                cover_letter_progress = st.progress(0, text=f"Writing 0 / {len(all_found_jobs)} cover letters...")
                for i, job in enumerate(all_found_jobs):
                    cover_letter_progress.progress((i + 1) / len(all_found_jobs), text=f"Writing {i+1} / {len(all_found_jobs)} cover letters...")
                    fit_summary = ""
                    for reason in job.get("reason_for_fit", []):
                        fit_summary += (
                            f"• Requirement: {reason.get('job_requirement', 'N/A')}\n"
                            f"  - My Experience: {reason.get('resume_evidence', 'N/A')}\n\n"
                        )
                    fit_summary = fit_summary.strip()
                    
                    # CHANGE 10: Call the renamed 'generate_cover_letter' function and pass the key.
                    generated_cover_letter = generate_cover_letter(
                        company_name=job.get('company'),
                        fit_analysis=fit_summary
                    )

                    jobs_list_for_df.append({
                        'Company': job.get('company'), 'Role': job.get('role'),
                        'Salary': job.get('salary'), 'Fit Analysis': fit_summary,
                        'URL': job.get('application_url'),
                        'Cover Letter': generated_cover_letter
                    })
                
                results_df = pd.DataFrame(jobs_list_for_df)
            
            status.update(label="Search complete!", state="complete", expanded=False)

    st.header("🏆 Job Search Results")
    if 'results_df' in locals() and not results_df.empty:
        st.success(f"Found and processed {len(results_df)} matching jobs!")
        
        for index, row in results_df.iterrows():
            with st.expander(f"**{row['Role']}** at **{row['Company']}**"):
                st.markdown(f"**Salary:** {row['Salary']}")
                st.markdown(f"**Application Link:** [Apply Here]({row['URL']})")
                st.markdown("**AI Fit Analysis:**")
                st.text(row['Fit Analysis'])
                
                st.markdown("**AI Generated Cover Letter:**")
                st.text_area("Cover Letter", value=row['Cover Letter'], height=300, key=f"cover_letter_{index}")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False, sheet_name='AI Job Search Results')
        st.download_button(
            label="📥 Download Results as Excel", data=output.getvalue(),
            file_name=EXCEL_FILENAME,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No matching jobs were found in this run. Check the debug messages in the status box above for clues. You may need to broaden your search preferences.")
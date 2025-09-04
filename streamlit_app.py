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
# CHANGE 1: The correct class to import is 'Client'.
from serpapi import Client
from PyPDF2 import PdfReader
import streamlit as st
import google.generativeai as genai

# ==============================================================================
# --- ✅ 1. User Configuration ---
# ==============================================================================
# --- HARDCODE YOUR CREDENTIALS HERE ---
# No need to enter them in the app anymore.

GEMINI_API_KEY = "AIzaSyB1OQZIC_NzDLtrCM1jnxaV7F9HVrNTnoI"
SERPAPI_KEY = "89071354e319d81a379c83ff8952e6c0b4d91b0927f8c78b0c8f708e16ee5ad9"

# --- Optional Email Settings ---
GMAIL_SENDER = "dg182364@gmail.com"
GMAIL_APP_PASSWORD = "cvgm oalu tpew qtun" # Use a Google App Password
GMAIL_RECIPIENT = "mdgraham2288@gmail.com"


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

def call_gemini_with_backoff(prompt, is_json=False, max_retries=5):
    """Calls the Gemini API with exponential backoff for rate limiting."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    generation_config = genai.GenerationConfig(response_mime_type="application/json") if is_json else None
    retries = 0
    while retries < max_retries:
        try:
            response = model.generate_content(prompt, generation_config=generation_config)
            time.sleep(1) # Be respectful of API rate limits
            clean_text = response.text.strip().replace('```json', '').replace('```', '')
            return clean_text
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait_time = (2 ** retries) + 1
                st.toast(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                st.warning(f"Unexpected Gemini API error: {e}")
                return None
    st.error("Max retries reached. Failed to get Gemini response.")
    return None

def search_and_extract_jobs(query, serpapi_key, num_results=10):
    """
    Searches Google Jobs and extracts full job details from the initial search.
    Returns a list of dictionaries, where each dict is a job posting.
    """
    st.write(f"  - 🔍 Searching Google Jobs for: '{query}'")
    extracted_jobs = []
    try:
        # NOTE: The api_key is now passed to the Client, not the params dict.
        params = {
            "q": query,
            "engine": "google_jobs",
            "num": str(num_results) # Using string for parameter is safer
        }
        
        # CHANGE 2: Instantiate the 'Client' with your API key.
        client = Client(api_key=serpapi_key)
        
        # CHANGE 3: Use the client.search(params) method to get results dictionary.
        result = client.search(params)

        jobs_results = result.get("jobs_results", [])

        if "error" in result:
            st.error(f"  - ❌ SerpApi search error: {result.get('error')}")
            return []

        if not jobs_results:
            st.warning(f"  - ⚠️ No jobs found on Google for query: '{query}'.")
            return []

        for job in jobs_results:
            application_url = ""
            if apply_options := job.get("apply_options", []):
                application_url = apply_options[0].get("link")

            extracted_jobs.append({
                "job_id": job.get("job_id"),
                "title": job.get("title"),
                "company": job.get("company_name"),
                "description": job.get("description"),
                "application_url": application_url
            })
        return extracted_jobs
    except Exception as e:
        st.error(f"  - ❌ An unexpected error occurred during SerpApi search: {e}")
        return []


def generate_initial_search_queries(preferences, resume_text):
    """Uses Gemini to generate initial search queries."""
    prompt = f"""
    You are an expert career search strategist. Based on the following job preferences and resume text,
    generate 4 diverse and effective Google search queries.

    **VERY IMPORTANT INSTRUCTIONS:**
    1.  The queries MUST be simple strings. For example: "Executive Communication Director Houston TX".
    2.  **DO NOT USE BOOLEAN OPERATORS.** Do not use "OR", "AND", or parentheses.
    3.  Return a valid JSON object with a single key "queries", which is an array of these simple query strings.

    **Preferences:** {json.dumps(preferences, indent=2)}
    **Resume Summary:** {resume_text[:1000]}
    """
    response_text = call_gemini_with_backoff(prompt, is_json=True)
    if not response_text:
        return []
    try:
        return json.loads(response_text).get("queries", [])
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON for initial queries. Error: {e}")
        return []

def analyze_job_posting(job_text, resume_text, preferences, url):
    """Uses Gemini to analyze a job posting against a resume."""
    st.write(f"  - 🤔 Analyzing job posting from: {url or 'Source not found'}")
    prompt = f"""
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
    1.  Base your analysis *strictly* on the provided Resume, Preferences, and Job Description.
    2.  Extract the company name and role title directly from the Job Description text.

    ---
    **Job Preferences:** {json.dumps(preferences, indent=2)}
    ---
    **Candidate Resume:** {resume_text}
    ---
    **Job Description:** {job_text}
    ---
    """
    response_text = call_gemini_with_backoff(prompt, is_json=True)
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

st.set_page_config(page_title="Resume MatchmAIker", layout="wide")
st.title("🚀 Resume MatchmAIker")
st.markdown("This app uses AI to find and analyze job postings based on your resume and preferences.")

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
    if "PASTE_YOUR" in GEMINI_API_KEY or "PASTE_YOUR" in SERPAPI_KEY:
        st.error("Please hardcode your Gemini and SerpApi keys at the top of the script.")
    elif not uploaded_resume:
        st.error("Please upload your resume.")
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            st.success("Google Gemini API configured successfully.")
        except Exception as e:
            st.error(f"Error configuring Google Gemini API: {e}")
            st.stop()

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
                st.error("Could not generate initial search queries. Check Gemini API key and try different preferences. Exiting.")
                st.stop()

            st.write(f"✅ AI generated queries: {initial_queries}")

            st.write("\n--- [Phase 2] Searching for jobs and extracting details ---")
            unique_jobs = {} # Use a dict to store unique jobs by job_id
            for query in initial_queries:
                job_posts = search_and_extract_jobs(query, SERPAPI_KEY)
                if job_posts:
                    st.write(f"  - ✅ Extracted {len(job_posts)} job details for query '{query}'.")
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
                    analysis = analyze_job_posting(
                        job_text=details["description"],
                        resume_text=resume_text,
                        preferences=preferences,
                        url=details["application_url"]
                    )
                    if analysis.get("is_match"):
                        all_found_jobs.append(analysis)
                        st.success(f"  - ✅ MATCH FOUND: {analysis.get('role')} at {analysis.get('company')}")
                else:
                    st.warning(f"  - ⚠️ Skipping job ID {job_id[:20]}... due to insufficient description.")
            status.update(label="Search complete!", state="complete", expanded=False)

        st.header("🏆 Job Search Results")
        if all_found_jobs:
            st.success(f"Found {len(all_found_jobs)} potentially matching jobs!")
            jobs_list_for_df = []
            for job in all_found_jobs:
                fit_summary = ""
                for reason in job.get("reason_for_fit", []):
                    fit_summary += (
                        f"• Requirement: {reason.get('job_requirement', 'N/A')}\n"
                        f"  - My Experience: {reason.get('resume_evidence', 'N/A')}\n\n"
                    )
                jobs_list_for_df.append({
                    'Company': job.get('company'), 'Role': job.get('role'),
                    'Salary': job.get('salary'), 'Fit Analysis': fit_summary.strip(),
                    'URL': job.get('application_url')
                })
            results_df = pd.DataFrame(jobs_list_for_df)
            for index, row in results_df.iterrows():
                with st.expander(f"**{row['Role']}** at **{row['Company']}**"):
                    st.markdown(f"**Salary:** {row['Salary']}")
                    st.markdown(f"**Application Link:** [Apply Here]({row['URL']})")
                    st.markdown("**AI Fit Analysis:**")
                    st.text(row['Fit Analysis'])

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
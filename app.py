import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
import docx
import requests
import re

def setup_gemini_api(api_key):
    """Initialize the Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

# ---------- Job Description Input ----------
def get_job_description():
    st.subheader("Job Description Input")
    method = st.radio("Select input method:", ["Paste text", "Upload file"], horizontal=True)

    if method == "Paste text":
        job_description = st.text_area("Paste the job description here:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload job description (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
        job_description = ""
        if uploaded_file:
            with open(f"temp_job.{uploaded_file.name.split('.')[-1]}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            job_description = extract_text_from_file(f"temp_job.{uploaded_file.name.split('.')[-1]}")

    return job_description.strip()


def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def extract_text_from_docx(docx_path):
    """Extract text content from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text

def extract_text_from_file(file_path):
    """Extract text from PDF, DOCX, or TXT file."""
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        raise ValueError("Unsupported file format. Please provide PDF, DOCX, or TXT file.")

def analyze_resume_job_match(model, resume_text, job_description):
    """Use Gemini 2.0 Flash to analyze how well the resume matches the job description."""

    with open("resume_review_rules.txt", "r") as f:
      rules_text = f.read()

    prompt = f"""
    You are an expert in resume analysis and career coaching.
    Use the following resume review rules to guide your feedback:
    {rules_text}

    Please analyze the resume against the job description provided and give detailed feedback on:

    1. Match Score (0-100%): How well the candidate's qualifications match the job requirements
    2. Strengths: Key strengths and qualifications that align well with the job
    3. Gaps: Skills, experiences, or qualifications mentioned in the job description that are missing or not clearly demonstrated in the resume
    4. Improvement Suggestions: Specific recommendations for improving the resume to better match this job description
    5. Keywords: Important keywords from the job description that should be emphasized in the resume

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}

    Provide your analysis in a structured format with clear headings and actionable feedback.
    """

    response = model.generate_content(prompt)
    return response.text

def analyze_skill_gaps_with_resources(model, resume_text, job_description):
    prompt = f"""
    Analyze the skills mentioned in the job description that are missing from the resume.
    For each missing skill:
    1. Identify the skill gap
    2. Explain its importance for the role
    3. Suggest specific online courses, certifications, or resources to develop this skill
    4. Estimate the time investment needed to acquire basic proficiency

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}
    """

    response = model.generate_content(prompt)
    return response.text

def generate_cover_letter(model, resume_text, job_description):
    prompt = f"""
    Create a professional cover letter based on the candidate's resume and the job description.
    The cover letter should:
    1. Have a professional greeting and introduction
    2. Highlight the most relevant experiences and skills from the resume that match the job
    3. Address any potential concerns or gaps identified in the resume analysis
    4. Include a compelling closing paragraph
    5. Maintain a professional but personalized tone

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}
    """

    response = model.generate_content(prompt)
    return response.text

def generate_interview_prep(model, resume_text, job_description):
    prompt = f"""
    Based on this resume and job description, create an interview preparation guide with:
    1. 10 likely technical questions specific to this role and the candidate's background
    2. 5 behavioral questions that might probe potential gaps in experience
    3. Suggested answer frameworks for each question, incorporating the candidate's specific experiences
    4. 3 questions the candidate should ask the interviewer

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}
    """

    response = model.generate_content(prompt)
    return response.text

def generate_resume_versions(model, resume_text, job_description):
    prompt = f"""
    Create 3 different versions of bullet points for the candidate's most recent roles, each emphasizing different aspects:
    1. Version focusing on technical skills and achievements
    2. Version emphasizing leadership and collaboration
    3. Version highlighting business impact and results

    For each version, rewrite the experience section to best position the candidate for this specific job.

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}
    """

    response = model.generate_content(prompt)
    return response.text

def grammar_check_resume(resume_text):
    """Check the resume text for grammar issues using LanguageTool API."""
    api_url = "https://api.languagetool.org/v2/check"
    data = {
        'text': resume_text,
        'language': 'en-US',
    }
    try:
        response = requests.post(api_url, data=data)
        result = response.json()

        grammar_issues = []
        for match in result.get('matches', []):
            message = match.get('message', '')
            rule_id = match.get('rule', {}).get('id', '')
            replacements = match.get('replacements', [])
            context = match.get('context', {})
            sentence = context.get('text', '')
            offset = context.get('offset', 0)
            length = context.get('length', 0)
            error_word = sentence[offset:offset+length]

            # Skip whitespace issues
            if "whitespace" in message.lower():
                continue

            # Skip proper nouns flagged by spell checker
            if rule_id == "MORFOLOGIK_RULE_EN_US" and error_word.istitle() and offset > 0:
                continue

            # Format replacement suggestions
            if replacements:
                suggestions = ', '.join(rep.get('value', '') for rep in replacements if rep.get('value', '').strip())
            else:
                suggestions = match.get('shortMessage', '').strip()

            if not suggestions:
                suggestions = "(No clear suggestion provided)"

            # Highlight error word in sentence
            highlighted = sentence[:offset] + "**" + sentence[offset:offset+length] + "**" + sentence[offset+length:]

            issue = f"""🔹 **Issue:** {message}
🔸 **Line:** {highlighted}
💡 **Suggestion:** {suggestions}
"""
            grammar_issues.append(issue)

        if not grammar_issues:
            return "✅ No major grammar issues found!"
        else:
            return "### Grammar Issues Found:\n\n" + "\n".join(grammar_issues)

    except Exception as e:
        return f"❌ Error checking grammar: {e}"


def get_industry_specific_feedback(model, resume_text, job_description):
    # First determine the industry
    industry_prompt = f"""
    Based on this job description, identify the specific industry and role category (e.g., 'Tech - Software Engineering',
    'Finance - Investment Banking', 'Healthcare - Nursing'). Return only the category name.

    JOB DESCRIPTION:
    {job_description}
    """

    industry_response = model.generate_content(industry_prompt)
    industry = industry_response.text.strip()

    # Then get industry-specific feedback
    feedback_prompt = f"""
    Provide industry-specific resume feedback for a {industry} position.
    Include:
    1. Industry-specific conventions and expectations for resumes in this field
    2. Key certifications or credentials that are valued but missing
    3. Industry jargon or technical terms that should be included
    4. Format and presentation norms for this specific industry

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}
    """

    feedback_response = model.generate_content(feedback_prompt)
    return feedback_response.text

# ---------- Utility Functions for Metrics ----------
def resume_length_score(resume_text):
    word_count = len(resume_text.split())
    if word_count < 250:
        return "Too short"
    elif word_count > 900:
        return "Too long"
    return "Appropriate"


def tone_score(resume_text, model):
    prompt = f"""
    Evaluate the professionalism and tone of the following resume text. Give a score out of 100,
    considering clarity, formality, confidence, and conciseness. Only return a number like:
    Tone Score: 85%

    RESUME:
    {resume_text}
    """
    try:
        response = model.generate_content(prompt)
        match = re.search(r"(\d{1,3})\s*%", response.text)
        if match:
            return int(match.group(1))
    except:
        return None
    return None

def compute_keyword_overlap_score(resume_text, job_description,model):
    prompt = f"""
    Compare the following resume and job description and estimate a keyword relevance score (0 to 100).
    Focus on whether the resume captures the key responsibilities, terminologies, and themes mentioned in the job post,
    even if exact keywords are not repeated.

    Provide the result as: Keyword Match Score: XX%

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        match = re.search(r"(\d{1,3})\s*%", text)
        if match:
            score = int(match.group(1))
            return score
    except Exception as e:
        st.warning(f"⚠️ Keyword match scoring failed: {e}")
    return 0

def skill_coverage_score(resume_text, job_description,model):
    prompt = f"""
    Compare the following resume and job description and estimate a skill match score (0 to 100).
    Focus on whether the candidate's resume demonstrates proficiency in the skills required by the job,
    even if exact keywords are not matched.

    Provide the result as: Skill Match Score: XX%

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
     
        match = re.search(r"(\d{1,3})\s*%", text)
        if match:
            score = int(match.group(1))
            return score
    except Exception as e:
        st.warning(f"Skill match scoring failed: {e}")
    return 0


def formatting_consistency_score(resume_text):
    lines = resume_text.split('\n')
    indent_counts = [len(line) - len(line.lstrip(' ')) for line in lines if line.strip()]
    most_common_indent = Counter(indent_counts).most_common(1)[0][0] if indent_counts else 0
    consistent_lines = sum(1 for count in indent_counts if count == most_common_indent)
    consistency = consistent_lines / len(indent_counts) if indent_counts else 1
    return round(consistency * 100, 2)

# ---------- Streamlit Metrics Dashboard ----------
def display_metrics(resume_text, job_description,model):

    keyword_score = compute_keyword_overlap_score(resume_text, job_description,model)
    skill_score = skill_coverage_score(resume_text, job_description,model)
    formatting = formatting_consistency_score(resume_text)
    length_status = resume_length_score(resume_text)
    tone = tone_score(resume_text, model)

    st.markdown("## 📊 Resume Metrics Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Keyword Match %", f"{keyword_score}%")
        st.metric("Skill Match %", f"{skill_score}%")
        st.metric("Resume Length", length_status)

    with col2:
      st.metric("Formatting Consistency %", f"{formatting}%")
      if tone is not None:
          st.metric("Professional Tone Score", f"{tone}%")
      else:
          st.caption("Professional Tone Score: Could not be determined")
          
def create_streamlit_app():
    st.title("AI Career Companion")

    api_key = st.secrets["gemini"]["api_key"]

    uploaded_resume = st.file_uploader("Upload your resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    job_description=get_job_description()

    if api_key and uploaded_resume and job_description:
        # Save uploaded files with extensions
        with open(f"temp_resume.{uploaded_resume.name.split('.')[-1]}", "wb") as f:
            f.write(uploaded_resume.getbuffer())

        # Update the file paths
        resume_file = f"temp_resume.{uploaded_resume.name.split('.')[-1]}"
        # Extract text
        resume_text = extract_text_from_file(resume_file)

        # Initialize model
        model = setup_gemini_api(api_key)

        # Store previous selection to detect change
        if "prev_analysis_type" not in st.session_state:
            st.session_state.prev_analysis_type = None

        analysis_type = st.selectbox(
            "Select analysis type:",
            ["Basic Resume Analysis", "Skill Gap Analysis", "Cover Letter Generation",
            "Interview Preparation", "Resume Versions", "Industry-Specific Feedback", "Grammar Check on Resume"]
        )

        # Clear previous results and chat if analysis type has changed
        if st.session_state.prev_analysis_type != analysis_type:
            st.session_state.prev_analysis_type = analysis_type
            st.session_state.pop("result", None)
            st.session_state.pop("messages", None)

        if st.button("Generate Analysis"):
            with st.spinner("Analyzing..."):
                if analysis_type == "Basic Resume Analysis":
                    result = analyze_resume_job_match(model, resume_text, job_description)
                    # Show metrics dashboard
                    display_metrics(resume_text, job_description,model)
                elif analysis_type == "Skill Gap Analysis":
                    result = analyze_skill_gaps_with_resources(model, resume_text, job_description)
                elif analysis_type == "Cover Letter Generation":
                    result = generate_cover_letter(model, resume_text, job_description)
                elif analysis_type == "Interview Preparation":
                    result = generate_interview_prep(model, resume_text, job_description)
                elif analysis_type == "Resume Versions":
                    result = generate_resume_versions(model, resume_text, job_description)
                elif analysis_type == "Industry-Specific Feedback":
                    result = get_industry_specific_feedback(model, resume_text, job_description)
                elif analysis_type == "Grammar Check on Resume":
                    result = grammar_check_resume(resume_text)


                # Store result in session state
                st.session_state.result = result

                # Only show it in the results section
                st.markdown("## Analysis Results")
                st.markdown(result)

                # Don't add it to the chat history yet
                st.session_state.messages = []


        # Add chat interface after displaying analysis results
        if "result" in st.session_state:
            st.markdown("## Ask Follow-up Questions")
            st.markdown("You can ask specific questions about the analysis to get more detailed information.")

            # Display previous messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Get user input
            user_question = st.chat_input("Ask a follow-up question about the analysis...")

            if user_question:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_question})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_question)

                # Generate response
                prompt = f"""
                Based on the previous resume analysis and the user's question: "{user_question}"

                RESUME:
                {resume_text}

                JOB DESCRIPTION:
                {job_description}

                Provide a helpful, specific response to their question.
                """

                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        response = model.generate_content(prompt)
                        feedback = response.text
                        st.markdown(feedback)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": feedback})
    
    else:
        st.warning("Please upload both the resume and job description to proceed.")
        st.stop()


if __name__ == "__main__":
    create_streamlit_app()

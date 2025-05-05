import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
import docx
import requests


def setup_gemini_api(api_key):
    """Initialize the Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

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
    prompt = f"""
    You are an expert in resume analysis and career coaching.

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
            replacements = match.get('replacements', [])
            context = match.get('context', {})
            sentence = context.get('text', '')
            offset = context.get('offset', 0)
            length = context.get('length', 0)

            # Ignore suggestions with no replacements and unhelpful messages
            if not replacements or "whitespace" in message.lower():
                continue

            suggestions = ', '.join(rep['value'] for rep in replacements)
            highlighted = sentence[:offset] + "**" + sentence[offset:offset+length] + "**" + sentence[offset+length:]

            issue = f"""🔹 **Issue:** {message}
🔸 **Line:** {highlighted}
💡 **Suggestion:** {suggestions}
"""
            grammar_issues.append(issue)

        if not grammar_issues:
            return "No major grammar issues found!"
        else:
            return "### Grammar Issues Found:\n\n" + "\n".join(grammar_issues)

    except Exception as e:
        return f"Error checking grammar: {e}"

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

def create_streamlit_app():
    st.title("AI Career Companion")

    api_key = st.text_input("Enter your Gemini API key:", type="password")

    uploaded_resume = st.file_uploader("Upload your resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    uploaded_job = st.file_uploader("Upload job description (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

    if api_key and uploaded_resume and uploaded_job:
        # Save uploaded files with extensions
        with open(f"temp_resume.{uploaded_resume.name.split('.')[-1]}", "wb") as f:
            f.write(uploaded_resume.getbuffer())
        with open(f"temp_job.{uploaded_job.name.split('.')[-1]}", "wb") as f:
            f.write(uploaded_job.getbuffer())

        # Update the file paths
        resume_file = f"temp_resume.{uploaded_resume.name.split('.')[-1]}"
        job_file = f"temp_job.{uploaded_job.name.split('.')[-1]}"

        # Extract text
        resume_text = extract_text_from_file(resume_file)
        job_description = extract_text_from_file(job_file)

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
                # elif analysis_type == "Grammar Check on Resume":
                #     result = grammar_check_resume(resume_text)


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



if __name__ == "__main__":
    create_streamlit_app()

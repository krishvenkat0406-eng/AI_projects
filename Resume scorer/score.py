import os
import json
from pathlib import Path
from pypdf import PdfReader
from google import genai
from google.genai import types
from IPython.display import HTML, display

# Step 1: Initialize Gemini Client
# Reads GEMINI_API_KEY directly from environment variable / GitHub Codespaces secret
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing or empty.")

client = genai.Client(api_key=api_key)

# Step 2: Function to extract text from all PDFs in the repository
def extract_repo_pdfs(directory="."):
    extracted_data = {}
    pdf_files = list(Path(directory).rglob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF file(s) in repository.")
    
    for pdf_path in pdf_files:
        # Skip hidden directories like .git
        if any(part.startswith('.') for part in pdf_path.parts):
            continue
            
        try:
            reader = PdfReader(pdf_path)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            extracted_data[pdf_path.name] = text
        except Exception as e:
            print(f"Failed to read {pdf_path.name}: {e}")
            
    return extracted_data

# Step 3: Function to evaluate a resume using Gemini API
def evaluate_resume(filename, resume_text):
    prompt = f"""
    Evaluate the following candidate resume for a Data & BI Engineer role.
    Provide a JSON response with the following keys strictly:
    - "candidate_name": Name extracted from resume (or filename if not found)
    - "score": Integer rating from 0 to 100 based on qualification match
    - "strengths": Array of 3 key strengths
    - "weaknesses": Array of 3 key gaps or areas for improvement

    Resume text:
    {resume_text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )
    
    return json.loads(response.text)

# Step 4: Generate HTML Report
def generate_html_report(evaluations):
    cards_html = ""
    for eval_item in evaluations:
        score_color = "#16a34a" if eval_item['score'] >= 80 else ("#ca8a04" if eval_item['score'] >= 60 else "#dc2626")
        
        strengths_list = "".join([f"<li>{s}</li>" for s in eval_item['strengths']])
        weaknesses_list = "".join([f"<li>{w}</li>" for w in eval_item['weaknesses']])
        
        cards_html += f"""
        <div style="background: white; border-radius: 8px; border: 1px solid #e2e8f0; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; margin-bottom: 16px;">
                <h3 style="margin: 0; color: #0f172a; font-family: sans-serif;">{eval_item['candidate_name']} <span style="font-size: 12px; color: #64748b; font-weight: normal;">({eval_item['filename']})</span></h3>
                <span style="background-color: {score_color}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-family: sans-serif;">
                    Score: {eval_item['score']}/100
                </span>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-family: sans-serif; font-size: 14px;">
                <div style="background: #f0fdf4; border-left: 4px solid #16a34a; padding: 12px; border-radius: 4px;">
                    <strong style="color: #15803d;">Key Strengths</strong>
                    <ul style="margin: 8px 0 0 0; padding-left: 20px; color: #166534;">{strengths_list}</ul>
                </div>
                <div style="background: #fef2f2; border-left: 4px solid #dc2626; padding: 12px; border-radius: 4px;">
                    <strong style="color: #b91c1c;">Gaps / Weaknesses</strong>
                    <ul style="margin: 8px 0 0 0; padding-left: 20px; color: #991b1b;">{weaknesses_list}</ul>
                </div>
            </div>
        </div>
        """
        
    full_html = f"""
    <div style="font-family: sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f8fafc;">
        <h2 style="color: #0f172a; border-bottom: 2px solid #0284c7; padding-bottom: 8px;">Resume Evaluation Dashboard</h2>
        {cards_html}
    </div>
    """
    return full_html

# --- Pipeline Execution ---
if __name__ == "__main__":
    # 1. Read files
    resumes = extract_repo_pdfs(".")

    if not resumes:
        print("No PDF files found to evaluate.")
    else:
        # 2. Evaluate with Gemini
        evaluations = []
        for filename, text in resumes.items():
            print(f"Evaluating {filename} with Gemini...")
            result = evaluate_resume(filename, text)
            result['filename'] = filename
            evaluations.append(result)

        # 3. Generate and output HTML
        html_output = generate_html_report(evaluations)

        # Save to disk
        with open("resume_evaluation_report.html", "w", encoding="utf-8") as f:
            f.write(html_output)

        print("Successfully generated resume_evaluation_report.html!")

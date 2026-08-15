import os
import sys
from pathlib import Path
from google import genai

# 1. Initialize Gemini Client
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY environment variable not found.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# 2. Find PDF resumes in the directory
pdf_files = [p for p in Path(".").rglob("*.pdf") if not any(part.startswith('.') for part in p.parts)]

if not pdf_files:
    print("⚠️ No PDF files found in directory.")
    sys.exit(0)

# 3. Let Gemini Agent read the PDF and generate HTML directly
for pdf_path in pdf_files:
    print(f"🤖 Uploading and sending {pdf_path.name} to Gemini Agent...")
    
    # Upload PDF directly to Gemini Files API
    uploaded_file = client.files.upload(file=str(pdf_path))
    
    prompt = """
    You are an AI Talent Evaluator. Analyze the attached resume for a Data & BI Engineer role.
    
    Generate a complete, production-ready, styled HTML document (single file with inline CSS) containing:
    1. Candidate name and resume file context.
    2. Qualification Match Score (0 to 100) displayed prominently with color coding (Green for >=80, Yellow for >=60, Red for <60).
    3. A 2-column card layout displaying 3 Key Strengths and 3 Gaps/Weaknesses.
    
    IMPORTANT: Output ONLY valid raw HTML code. Do NOT wrap it in markdown backticks (e.g. do not use ```html).
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[uploaded_file, prompt]
    )
    
    # Clean output string just in case
    html_output = response.text.replace("```html", "").replace("```", "").strip()
    
    # Save Agent's HTML directly
    output_filename = "resume_evaluation_report.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_output)
        
    print(f"✅ Gemini Agent generated direct HTML report: {output_filename}")

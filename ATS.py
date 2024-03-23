
import streamlit as st
from fpdf import FPDF
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import re
import string
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import pdfplumber
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfminer.high_level
import pytesseract
from PIL import Image
from weasyprint import HTML
import os
import json
import re
import pdfkit
import subprocess
import threading
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import matplotlib.pyplot as plt 


def txt_to_pdf(txt_file, pdf_file):
    # Open the TXT file and read its content
    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
        text_content = f.readlines()

    # Create a PDF canvas
    c = canvas.Canvas(pdf_file, pagesize=letter)

    # Set font properties and line height
    c.setFont("Helvetica", 12)  # Adjust font size and type as needed
    line_height = 14  # Adjust line spacing as needed

    # Write the text content line by line with a small gap
    y_pos = 750  # Start position for the text (adjust as needed)
    for line in text_content:
        # Replace non-encodable characters with whitespace
        cleaned_line = ''.join(c if ord(c) < 128 else ' ' for c in line)
        c.drawString(30, y_pos, cleaned_line.strip())  # Set horizontal offset and remove newline characters
        y_pos -= line_height

    # Save the PDF document
    c.save()



class ResumeScoreCalculator:
    def __init__(self, resume_data):
        self.resume_data = resume_data
        self.default_skills_file = "skills.csv"
        
        self.sw = stopwords.words('english')
        self.stemmer = SnowballStemmer('english')
        
    def _extract_skill_keywords(self, skills_file):
        self.skills = pd.read_csv(skills_file)
        if 'Keywords' in self.skills.columns:
            self.skills['Clean Keywords'] = self.skills['Keywords'].astype(str).apply(lambda x: x.lower())
            self.skills['Clean Keywords'] = self.skills['Clean Keywords'].apply(lambda x: ' '.join([word for word in x.split() if word not in self.sw]))
            self.skills['Clean Keywords'] = self.skills['Clean Keywords'].apply(lambda x: ' '.join([self.stemmer.stem(word) for word in x.split()]))
        else:
            raise ValueError("Column 'Keywords' not found in skills file.")
        
    def _extract_resume_data(self):
        # Extract resume data from UploadedFile object
        resume_bytes = self.resume_data.getvalue()
        
        # Save resume data to a temporary file
        with open("temp_resume.pdf", "wb") as temp_file:
            temp_file.write(resume_bytes)
        
        # Open the temporary file for processing
        with pdfplumber.open("temp_resume.pdf") as pdf:
            self.resume_text = []
            for page in pdf.pages:
                text = page.extract_text()
                self.resume_text.append(text)
        
        # Remove the temporary file
        os.remove("temp_resume.pdf")
                
    def _text_cleaning(self):
        text = ','.join(self.resume_text)
        text = self.resume_text[0].replace("\n", " ")
        text = text.replace("|", " ")
        text = text.replace(":", " ")
        text = text.replace(";", " ")
        text = text.replace("-", " ")
        text = text.replace("/", " ")
        text = text.replace("•", " ")
        text = str.lower(text)
        
        self.cleaned_text = text
        
    def _remove_stopwords(self):
        li = []
        source = ''
        for word in self.cleaned_text.split():
            if word not in self.sw and word not in string.punctuation:
                if not word.isdigit():
                    li.append(word)
                    source = ' '.join(li)
                    
        self.stopwords_removed_text = source
        
    def _tokenize(self):
        vectorizer = TfidfVectorizer()
        combine = []
        combine.append(self.stopwords_removed_text)
        
        for i in self.skills['Clean Keywords']:
            combine.append(i)
            
        self.embeddings = vectorizer.fit_transform(combine)
        
    def _calculate_scores(self):
        cosine_similarities = cosine_similarity(self.embeddings, self.embeddings)*100
        
        for i in range(1, len(cosine_similarities[0][1:]) + 1):
            com = cosine_similarities[0][i]
            if com > float(100):
                cosine_similarities[0][i] = float(100)
        
        per = (cosine_similarities[0][1:]).tolist()
        skill = self.skills['Skills'].tolist()
        
        new_df = pd.DataFrame({'Skills': skill, 'Percentage': per})
        return new_df
    
    def calculate_score(self, skills_file=None):
        if skills_file:
            self._extract_skill_keywords(skills_file)
        else:
            self._extract_skill_keywords(self.default_skills_file)
        
        self._extract_resume_data()
        self._text_cleaning()
        self._remove_stopwords()
        self._tokenize()
        return (self._calculate_scores()['Percentage'].mean(), self._calculate_scores())
def calculate_score(resume_paths, skills_file=None):
    scores = {}
    for path in resume_paths:
        resume_score_calculator = ResumeScoreCalculator(resume_data=path)
        score, _ = resume_score_calculator.calculate_score(skills_file)
        scores[path] = score
    return scores




def extract_text_from_pdf(pdf_file_path, output_file):
    try:
        with open(output_file, "w", encoding='utf-8') as out_file:
            text = pdfminer.high_level.extract_text(pdf_file_path)
            out_file.write(text)
        print(f"Text extracted successfully from PDF and saved to {output_file}.txt")
        return output_file
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_file_path}: {e}")
        return None

def extract_text_from_image(image_file_path, output_file):
    try:
        extracted_text = pytesseract.image_to_string(Image.open(image_file_path))
        with open(output_file, "w", encoding='utf-8') as out_file:
            out_file.write(extracted_text)
        print(f"Text extracted successfully from image and saved to {output_file}.txt")
        return output_file
    except Exception as e:
        print(f"Error extracting text from image {image_file_path}: {e}")
        return None

def edit_text_in_editor(file_path):
    editor_command = f" notepad {file_path}"  # Use nano editor, you can change this to your preferred text editor
    subprocess.run(editor_command, shell=True)

def extract_information_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            unstructured_text = file.read()
            structured_data = {
                "personal_info": [],
                "soft_skills": [],
                "technical_skills": [],
                "hard_skills": [],
                "education": [],
                "gender": [],
                "projects": []
            }
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            mobile_pattern = r'\b\d{10}\b'
            github_pattern = r'github\.com\/[A-Za-z0-9_-]+'
            soft_skills_keywords = ["communication", "leadership", "teamwork", "problem-solving", "time management"]
            technical_skills_keywords = ["languages", "frameworks", "tools", "technologies"]
            specific_skills_keywords = ["designer", "editor"]
            education_keywords = ["school", "college", "pcu", "institution"]
            projects_keywords = ["project", "system"]

            for line in unstructured_text.split('\n'):
                emails = re.findall(email_pattern, line)
                mobiles = re.findall(mobile_pattern, line)
                githubs = re.findall(github_pattern, line)

                if emails:
                    structured_data["personal_info"].extend(emails)
                if mobiles:
                    structured_data["personal_info"].extend(mobiles)
                if githubs:
                    structured_data["personal_info"].extend(githubs)

                if re.match(r'\b[a-zA-Z]+\.[a-zA-Z]+\b', line):
                    structured_data["personal_info"].append(line)               
                if re.match(r'\b[a-zA-Z]+\.[a-zA-Z]+\b', line):
                    structured_data["personal_info"].append(line)

                line_lower = line.lower()
                if any(keyword in line_lower for keyword in soft_skills_keywords):
                    structured_data["soft_skills"].append(line)
                if any(keyword in line_lower for keyword in technical_skills_keywords):
                    structured_data["technical_skills"].append(line)
                if any(keyword in line_lower for keyword in specific_skills_keywords):
                    structured_data["hard_skills"].append(line)
                if any(keyword in line_lower for keyword in education_keywords):
                    structured_data["education"].append(line)
                if any(keyword in line_lower for keyword in projects_keywords):
                    structured_data["projects"].append(line)

                if "male" in line_lower or "female" in line_lower:
                    structured_data["gender"].append(line)

            return structured_data
    except FileNotFoundError:
        print("File not found.")
        return None
def save_to_txt(structured_data, output_file):
    try:
        with open(output_file, "w", encoding='utf-8') as txt_file:
            for section, content in structured_data.items():
                if content:
                    txt_file.write(f"{section.replace('_', ' ').title()}:\n")
                    for item in content:
                        txt_file.write(f"{item}\n")
                    txt_file.write("\n")
        print(f"Structured data saved to {output_file} in TXT format.")
    except Exception as e:
        print(f"Error saving TXT file: {e}")

def save_to_json(structured_data, output_file):
    with open(output_file, "w", encoding='utf-8') as json_file:
        json.dump(structured_data, json_file, indent=2, ensure_ascii=False)
    print(f"Structured data saved to {output_file} in JSON format.")
file_lock = threading.Lock()
   

def save_to_pdf(structured_data, output_file):
    try:
        # Convert structured data to PDF format
        html_content = "<html><head><title>Structured Resume</title></head><body>"
        for section, content in structured_data.items():
            if content:
                html_content += f"<h2>{section.replace('_', ' ').title()}</h2>"
                html_content += "<ul>"
                for item in content:
                    html_content += f"<li>{item}</li>"
                html_content += "</ul>"
        html_content += "</body></html>"
        with open("temp.html", "w") as temp_file:
            temp_file.write(html_content)
        
        HTML(string=html_content).write_pdf(output_file)  # Convert HTML to PDF using WeasyPrint
        os.remove("temp.html")  # Remove temporary HTML file
        print(f"Structured data saved to {output_file} in PDF format.")
        return True
    except Exception as e:
        print(f"Error saving PDF: {e}")
        return False




def main():

    st.title("Resume Processing and Scoring")
    st.sidebar.title("Menu")
    options = ["Extract Structured Data", "Calculate Resume Scores", "Sorting the resumes based on the score" ,"Exit"]
    choice = st.sidebar.selectbox("Select an Option", options)
    if choice == "Extract Structured Data":
        st.subheader("Extract Structured Data")
        uploaded_file = st.file_uploader("Upload a PDF or Image file", type=["pdf", "png", "jpg"])

        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
            st.write(file_details)

            if uploaded_file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(uploaded_file.name, "extracted_text.txt")
            else:
                extracted_text = extract_text_from_image(uploaded_file.name, "extracted_text.txt")

            if extracted_text:
                st.write("Extracted Text:")
                st.write(extracted_text)

                st.subheader("Edit Extracted Text")
                edit_button = st.button("Edit Text in Editor")
                if edit_button:
                    edit_text_in_editor("extracted_text.txt")

                continue_button = st.button("Continue")
                if continue_button:
                    # Convert the saved extracted code to structured data
                    structured_data = extract_information_from_file("extracted_text.txt")
                    if structured_data:
                        # Save structured data to TXT file
                        edit_text_in_editor("structured_data.txt")
                        save_to_txt(structured_data, "structured_data.txt")
                        st.write("Structured Data Saved to TXT")
                        


                        # Download the TXT file
                        if os.path.isfile("structured_data.txt"):
                            with open("structured_data.txt", "rb") as txt_file:
                                txt_bytes = txt_file.read()

                            st.download_button(label="Download TXT", data=txt_bytes, file_name="structured_data.txt", mime="text/plain")
                        else:
                            st.warning("TXT file not generated or file path invalid.")
        # this is for pdf
        st.title("Convert TXT to PDF")
        uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])

        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
            st.write(file_details)

            if uploaded_file.type == "text/plain":
                file_path = f"uploaded_file.txt"
                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getvalue())

                st.write("TXT file uploaded successfully.")

                convert_button = st.button("Convert to PDF")
                if convert_button:
                # Convert TXT to PDF
                    txt_to_pdf(file_path, "converted_file__.pdf" )

                # Download the PDF file
                    if os.path.isfile("converted_file__.pdf"):
                        with open("converted_file__.pdf", "rb") as pdf_file:
                         pdf_bytes = pdf_file.read()

                        st.download_button(label="Download PDF", data=pdf_bytes, file_name="converted_file__.pdf", mime="application/pdf")
                    else:
                        st.warning("PDF file not generated or file path invalid.")
                else:
                    st.warning("Please upload a TXT file.")
                                           
    elif choice == "Calculate Resume Scores":
        st.subheader("Calculate Resume Scores")
        
        upload_csv = st.radio("Do you want to upload a CSV file?", ("Yes", "No"))
        if upload_csv == "Yes":
            skills_file = st.file_uploader("Upload Skills CSV File", type=["csv"])
        else:
            skills_file = None
        
        resume_data_path = st.file_uploader("Upload the resume data file (PDF)", type=["pdf"])
        if resume_data_path:
            resume_score_calculator = ResumeScoreCalculator(resume_data=resume_data_path)
            score, score_table = resume_score_calculator.calculate_score(skills_file)
            st.write("Average Resume Score:", score)
            
            



    elif choice == "Sort Resumes":
        st.title("Resume Sorting App")

    # Upload multiple resumes
        uploaded_files = st.file_uploader("Upload resumes (PDF)", accept_multiple_files=True, type="pdf")
        if uploaded_files:
            resume_paths = [file for file in uploaded_files]

        # Calculate scores and sort resumes
            if st.button("Sort Resumes"):
                st.write("Calculating scores and sorting resumes...")

            # Create a list to store scores
                scores = []

            # Iterate over each uploaded resume
                for resume_file in resume_paths:
                # Create a ResumeScoreCalculator object for each resume
                    resume_score_calculator = ResumeScoreCalculator(resume_data=resume_file)
                    score, _ = resume_score_calculator.calculate_score()
                    scores.append((resume_file.name, score))  # Store the filename and score

            # Sort resumes based on scores
            sorted_resumes = sorted(scores, key=lambda x: x[1], reverse=True)

            # Display sorted resumes with scores
            st.subheader("Sorted Resumes")
            for resume, score in sorted_resumes:
                st.write(f"Resume: {resume}, Score: {score}")


        
    elif choice == "Exit":
        st.write("Thank you for using the app!")

if __name__ == "__main__":
    main()



import streamlit as st
import pickle
import re
import nltk
import os
import fitz  # PDF parsing

#  NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

# Load models safely
if os.path.exists("clf.pkl") and os.path.exists("tfidf.pkl"):
    with open("clf.pkl", "rb") as f:
        clf = pickle.load(f)

    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
else:
    st.error("Model files not found! Make sure `clf.pkl` and `tfidf.pkl` exist.")
    st.stop()

# Mapping for category predictions
category_mapping = {
    15: 'Java Developer', 23: 'Testing', 8: 'DevOps Engineer', 20: 'Python Developer',
    24: 'Web Designing', 12: 'HR', 13: 'Hadoop', 3: 'Blockchain', 10: 'ETL Developer',
    18: 'Operations Manager', 6: 'Data Science', 22: 'Sales', 16: 'Mechanical Engineer',
    1: 'Arts', 7: 'Database', 11: 'Electrical Engineering', 14: 'Health and Fitness',
    19: 'PMO', 4: 'Business Analyst', 9: 'Dotnet Developer', 2: 'Automation Testing',
    17: 'Network Engineer', 21: 'SAP Developer', 5: 'Civil Engineer', 0: 'Advocate'
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Streamlit Web App
def main():
    st.title("📄 Resume Screening App")
    st.write("Upload a **TXT** or **PDF** file, and we will predict the job category.")

    upload_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])

    if upload_file is not None:
        with st.spinner("Processing..."):
            # Read file content
            if upload_file.type == "text/plain":
                resume_bytes = upload_file.read()
                try:
                    resume_text = resume_bytes.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    resume_text = resume_bytes.decode('latin-1', errors='ignore')
            elif upload_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(upload_file)
            else:
                st.error("Unsupported file format. Please upload a .txt or .pdf file.")
                return

            # Ensure some text is extracted
            if not resume_text:
                st.warning("No readable text found in the file. Please check your document.")
                return

            # Clean and transform text
            cleaned_resume = cleanResume(resume_text)
            cleaned_resume = tfidf.transform([cleaned_resume])

            # Predict category
            prediction_id = clf.predict(cleaned_resume)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")

            # Display result
            st.success(f"✅ **Predicted Job Category: {category_name}**")

if __name__ == "__main__":
    main()

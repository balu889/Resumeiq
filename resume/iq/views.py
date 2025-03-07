from django.shortcuts import render
import re
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout, login, authenticate
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
import io
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader
import docx
import spacy
import torch
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from django.urls import reverse
from django.contrib.sessions.models import Session
from plotly.offline import plot
import plotly.graph_objects as go

@login_required(login_url='login')
def home(request):
    return render(request, 'index.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            # messages.success(request, 'You are now logged in.')
            # Replace 'home' with the name of your homepage view
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')

    return render(request, 'login.html')


def signup_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']

        if len(username) < 6:
            messages.error(
                request, "Username must be at least 6 characters long")
            return redirect('signup')

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            messages.error(request, "Invalid email address")
            return redirect('signup')

        password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
        if not re.match(password_pattern, password1):
            messages.error(
                request, "Password must be at least 8 characters long, include numbers, special characters, one uppercase letter, and one lowercase letter.")
            return redirect('signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken")
            return redirect('signup')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already taken")
            return redirect('signup')

        # if password1 != password2:
        #     messages.error(request, "Passwords do not match")
        #     return redirect('signup')
        user = User.objects.create_user(
            username=username, email=email, password=password1)
        user.save()
        messages.success(request, "User added Successfully")
        return redirect('login')

    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('home')


# Load NLP models
nlp = spacy.load('en_core_web_sm')
transformer_nlp = pipeline(
    'feature-extraction', model='distilbert-base-uncased', framework='pt')

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

# Helper function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# Function to extract a specific section from resume text
def extract_section(text, section_titles):
    """Extracts the section text based on multiple possible section titles."""
    section = []
    in_section = False
    for line in text.split("\n"):
        if any(title.lower() in line.lower() for title in section_titles):
            in_section = True
        elif in_section and line.strip() == "":
            break
        elif in_section:
            section.append(line)
    return "\n".join(section)

# Function to extract keywords from text using SpaCy
def extract_keywords(text):
    doc = nlp(text)
    return set(token.lemma_.lower() for token in doc if token.pos_ in {'NOUN', 'PROPN', 'ADJ'} and not token.is_stop)

# Function to parse the resume and extract sections with flexible titles
def parse_resume(resume_text):
    # Expanding possible titles for each section to handle naming variations
    skills_titles = ['skills', 'competency', 'proficiency']
    experience_titles = ['experience',
                         'work history', 'professional experience']
    education_titles = ['education', 'academic background', 'qualifications']

    skills = extract_section(resume_text, skills_titles)
    experience = extract_section(resume_text, experience_titles)
    education = extract_section(resume_text, education_titles)

    return {
        'skills': skills.lower(),
        'experience': experience.lower(),
        'education': education.lower(),
        'entities': [ent.text for ent in nlp(resume_text).ents],
        'full_text': resume_text
    }

# Function to calculate similarity between two text sections
def calculate_similarity(text1, text2):
    vec1 = transformer_nlp(text1)
    vec2 = transformer_nlp(text2)
    vec1_mean = torch.mean(torch.tensor(vec1), dim=1)
    vec2_mean = torch.mean(torch.tensor(vec2), dim=1)
    return cosine_similarity(vec1_mean, vec2_mean)[0][0]

# Function to score the resume based on job description keywords and sections
def score_resume(resume_data, job_data):
    # Extract and normalize entities from resume and job description
    resume_entities = set(ent.lower() for ent in resume_data['entities'])
    job_keywords = set(keyword.lower() for keyword in job_data['keywords'])

    # Debugging: Print extracted entities for both resume and job description
    print("Debug: Resume Entities:", resume_entities)
    print("Debug: Job Description Entities:", job_keywords)

    # Calculate Entity Score based on the intersection of entities
    matched_entities = resume_entities & job_keywords
    entity_score = len(matched_entities) / len(job_keywords) * 100 if job_keywords else 0

    # Debugging: Print matched entities and entity score
    print("Debug: Matched Entities:", matched_entities)
    print("Debug: Entity Score:", entity_score)

    # Calculate other scores for skills, experience, and education
    skills_score = calculate_similarity(resume_data['skills'], job_data['skills']) * 100
    experience_score = calculate_similarity(resume_data['experience'], job_data['experience']) * 100
    education_score = calculate_similarity(resume_data['education'], job_data['education']) * 100

    # Combine all scores to calculate the total score
    total_score = 0.3 * entity_score + 0.3 * skills_score + 0.2 * experience_score + 0.2 * education_score
    return {
        'entity_score': round(entity_score, 2),
        'skills_score': round(skills_score, 2),
        'experience_score': round(experience_score, 2),
        'education_score': round(education_score, 2),
        'total_score': round(total_score, 2)
    }


# Function to generate feedback based on missing keywords and section scores
def generate_feedback(resume_data, job_data):
    feedback = []
    missing_skills = job_data['keywords'] - set(resume_data['skills'].split())
    if missing_skills:
        feedback.append(
            f"Consider including skills like: {', '.join(list(missing_skills)[:10])}.")
    return ' '.join(feedback)

# Main view for resume analysis
@csrf_exempt
def analyze_resume(request):
    # Clear only the analyze_resume-related session data
    if request.method == 'GET':
        keys_to_clear = ['resume_data', 'analysis_results', 'feedback']
        for key in keys_to_clear:
            if key in request.session:
                del request.session[key]

    if request.method == 'POST':
        resume_file = request.FILES['resume']
        job_title = request.POST['job_title']
        job_description = request.POST['job_description']
        job_responsibilities = request.POST['job_responsibilities']
        job_experience = request.POST['job_experience']
        job_skills = request.POST['job_skills']
        job_education = request.POST['job_education']

        if resume_file.name.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.name.endswith('.docx'):
            resume_text = extract_text_from_docx(resume_file)
        else:
            return render(request, 'analyze_resume.html', {'error': 'Unsupported file format'})

        # Process and store the results in the session
        resume_data = parse_resume(resume_text)
        job_data = {
            'description': job_description,
            'responsibilities': job_responsibilities,
            'experience': job_experience,
            'skills': job_skills,
            'education': job_education,
            'keywords': extract_keywords(job_description + job_responsibilities + job_experience + job_skills + job_education)
        }
        analysis_results = score_resume(resume_data, job_data)
        feedback = generate_feedback(resume_data, job_data)

        # Store data in session
        request.session['resume_data'] = resume_data
        request.session['analysis_results'] = analysis_results
        request.session['feedback'] = feedback
        return redirect(reverse('results'))

    return render(request, 'analyze_resume.html')



def results(request):
    resume_data = request.session.get('resume_data')
    analysis_results = request.session.get('analysis_results')
    feedback = request.session.get('feedback')

    # Create Plotly bar chart
    scores = go.Bar(
        x=['Entity Score', 'Skills Score', 'Experience Score', 'Education Score'],
        y=[
            analysis_results['entity_score'],
            analysis_results['skills_score'],
            analysis_results['experience_score'],
            analysis_results['education_score']
        ],
        marker_color='rgb(255,99,132)'
    )

    layout = go.Layout(
        title='Resume Analysis Scores',
        xaxis=dict(title='Categories'),
        yaxis=dict(title='Score (%)', range=[0, 100])
    )

    fig = go.Figure(data=[scores], layout=layout)
    plot_div = plot(fig, output_type='div')

    return render(request, 'results.html', {
        'resume_data': resume_data,
        'analysis_results': analysis_results,
        'feedback': feedback,
        'plot_div': plot_div
    })

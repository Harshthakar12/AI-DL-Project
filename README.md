Smart ATS Resume Analyzer

An AI-powered Resume Analyzer built using Python and Streamlit that compares multiple resumes with a Job Description (JD) and ranks candidates based on ATS score, keyword matching, TF-IDF similarity, and Deep Learning semantic similarity.

Features
Upload multiple resumes at once
ATS Score Calculation
Resume Ranking System
Resume vs Resume Comparison
Matching Skills Detection
Missing Skills Detection
Keyword Density Analysis
Deep Learning Semantic Matching
TF-IDF Similarity Matching
Resume Score Breakdown Graph
Download Top Ranked Resume
Section Analysis
Weak Word Detection
Impact Analysis
Bullet Feedback Analysis
Interactive Dashboard
Modern UI using Streamlit
Technologies Used
Frontend
Streamlit
Backend
Python
AI / Machine Learning
TF-IDF Vectorization
Cosine Similarity
Sentence Transformers
Semantic Similarity
Libraries Used
streamlit
pandas
re
sentence-transformers
scikit-learn
PyPDF2
Deep Learning Concept Used

This project uses Sentence Transformers for semantic similarity matching.

Previously, the system only checked:

Exact keyword matching
TF-IDF similarity

After adding Deep Learning:

The system now understands contextual meaning
Detects similar skills even if exact words are different
Gives smarter ATS ranking
Improves candidate matching accuracy

Example:

Resume contains: "Network Protection"
JD contains: "Cyber Security"

Traditional systems may fail.

Deep Learning can understand both are related concepts.

ATS Score Formula
Final ATS Score =
20% Skill Match +
20% TF-IDF Similarity +
60% AI Semantic Similarity
Functionalities Explained
1. Resume Ranking

Ranks resumes from highest ATS score to lowest.

2. Resume Comparison

Compare two resumes side-by-side.

3. Skill Matching

Shows:

Matching skills
Missing skills
4. Keyword Analysis

Counts how many times important keywords appear.

5. Deep Learning Matching

Uses semantic understanding instead of only exact keywords.

Future Enhancements
Resume Improvement Suggestions
AI Chatbot for Resume Review
Email Notification System
Resume PDF Report Download
Real-Time Job API Integration
Dark / Light Theme Toggle
Grammar Checking
Resume Template Scoring
LinkedIn Profile Analyzer
Sample Job Roles Supported
Software Developer
Data Scientist
Machine Learning Engineer
Cyber Security Analyst
Cloud Engineer
Web Developer
Backend Developer
Developed By

Harsh Thakar
Smart ATS Resume Analyzer Project

"""
Synthetic CV-Job Dataset Generator
Generates realistic CV and Job profiles for the bipartite graph project.
"""
import numpy as np
import random

random.seed(42)
np.random.seed(42)

SKILLS_POOL = [
    "Python", "Java", "C++", "JavaScript", "SQL", "R", "Scala", "Go",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Statistics",
    "Data Analysis", "Data Engineering", "Cloud (AWS)", "Cloud (GCP)", "Cloud (Azure)",
    "Docker", "Kubernetes", "Spark", "Hadoop", "TensorFlow", "PyTorch", "Scikit-learn",
    "React", "Node.js", "Django", "FastAPI", "Spring Boot",
    "Project Management", "Agile/Scrum", "Communication", "Leadership", "Problem Solving",
    "DevOps", "CI/CD", "Git", "Linux", "MongoDB", "PostgreSQL", "Elasticsearch"
]

DOMAINS = ["Data Science", "Software Engineering", "DevOps", "Backend", "Frontend", "ML Engineering", "Data Engineering"]
LEVELS = ["junior", "intermediate", "senior"]
DIPLOMAS = ["Bac+3", "Master", "PhD", "Ingénieur", "MBA"]
UNIVERSITIES = ["Université Paris-Saclay", "ENS", "Polytechnique", "INSA Lyon", "Télécom Paris",
                "Sorbonne", "Université de Bordeaux", "ENSEEIHT", "CentraleSupélec", "EPFL"]

def generate_cv_text(skills, level, domain, diploma, years_exp, university):
    exp_desc = {
        "junior": f"Diplômé récent avec {years_exp} an(s) d'expérience.",
        "intermediate": f"Professionnel avec {years_exp} ans d'expérience solide.",
        "senior": f"Expert expérimenté avec {years_exp} ans dans le domaine."
    }
    skills_str = ", ".join(skills)
    return (f"{exp_desc[level]} Domaine principal: {domain}. "
            f"Formation: {diploma} obtenu à {university}. "
            f"Compétences clés: {skills_str}. "
            f"Orienté résultats, capacité d'adaptation et travail en équipe.")

def generate_job_text(skills_required, level, domain, company_size):
    skills_str = ", ".join(skills_required)
    return (f"Nous recherchons un profil {level} en {domain} pour rejoindre notre équipe ({company_size}). "
            f"Compétences requises: {skills_str}. "
            f"Environnement dynamique, projets innovants, culture de collaboration.")

def generate_dataset(n_cv=40, n_jobs=25):
    cvs = []
    for i in range(n_cv):
        level = random.choices(LEVELS, weights=[0.35, 0.40, 0.25])[0]
        domain = random.choice(DOMAINS)
        years_exp = {"junior": random.randint(0, 2), "intermediate": random.randint(3, 7), "senior": random.randint(8, 20)}[level]
        diploma = random.choice(DIPLOMAS)
        university = random.choice(UNIVERSITIES)
        n_skills = random.randint(4, 10)
        skills = random.sample(SKILLS_POOL, n_skills)
        # Bias skills towards domain
        domain_skills = {
            "Data Science": ["Python", "Machine Learning", "Statistics", "Scikit-learn", "R"],
            "Software Engineering": ["Java", "C++", "Git", "Docker", "Spring Boot"],
            "DevOps": ["Docker", "Kubernetes", "CI/CD", "Linux", "Cloud (AWS)"],
            "Backend": ["Python", "Django", "FastAPI", "PostgreSQL", "Docker"],
            "Frontend": ["JavaScript", "React", "Node.js", "Git"],
            "ML Engineering": ["Python", "TensorFlow", "PyTorch", "Docker", "Spark"],
            "Data Engineering": ["Python", "Spark", "Hadoop", "SQL", "Cloud (GCP)"]
        }
        core = domain_skills.get(domain, [])
        skills = list(set(skills + random.sample(core, min(2, len(core)))))[:10]
        
        cv = {
            "id": f"CV_{i:03d}",
            "level": level,
            "domain": domain,
            "years_exp": years_exp,
            "diploma": diploma,
            "university": university,
            "skills": skills,
            "polyvalent": len(set([s for s in skills if s in SKILLS_POOL[:15]])) > 4,
            "text": generate_cv_text(skills, level, domain, diploma, years_exp, university)
        }
        cvs.append(cv)
    
    jobs = []
    for j in range(n_jobs):
        level = random.choices(LEVELS, weights=[0.3, 0.45, 0.25])[0]
        domain = random.choice(DOMAINS)
        n_skills = random.randint(3, 8)
        skills_required = random.sample(SKILLS_POOL, n_skills)
        company_size = random.choice(["startup", "PME", "grande entreprise", "multinationale"])
        
        job = {
            "id": f"JOB_{j:03d}",
            "level": level,
            "domain": domain,
            "skills_required": skills_required,
            "company_size": company_size,
            "text": generate_job_text(skills_required, level, domain, company_size)
        }
        jobs.append(job)
    
    return cvs, jobs

def generate_edges(cvs, jobs, edge_prob=0.15):
    """Generate edges based on skill overlap and level compatibility"""
    edges = []
    for cv in cvs:
        for job in jobs:
            cv_skills = set(cv["skills"])
            job_skills = set(job["skills_required"])
            overlap = len(cv_skills & job_skills) / max(len(job_skills), 1)
            level_match = cv["level"] == job["level"]
            domain_match = cv["domain"] == job["domain"]
            
            score = overlap * 0.5 + level_match * 0.25 + domain_match * 0.25
            if random.random() < score * edge_prob * 4:
                edges.append((cv["id"], job["id"], {"weight": round(score, 3)}))
    return edges
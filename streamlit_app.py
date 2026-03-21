"""
Agentic Resume Chatbot with MCP-like Tools
Features: Tool Use, Multi-step Reasoning, ReAct Pattern
"""

import streamlit as st
import requests
import numpy as np
import json
import re
from typing import List, Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Vijai Venkatesan - Agentic Resume Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# GROQ MODELS
# =====================================================
GROQ_MODELS = {
    "Llama 3.1 8B (Fast)": {
        "id": "llama-3.1-8b-instant",
        "description": "Fast responses, good for quick queries",
        "icon": "⚡"
    },
    "Llama 3.3 70B (Powerful)": {
        "id": "llama-3.3-70b-versatile",
        "description": "Most powerful, best for complex reasoning",
        "icon": "🚀"
    },
    "Llama 4 Scout 17B": {
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "description": "Latest Llama 4, balanced",
        "icon": "🦙"
    },
    "Qwen 3 32B": {
        "id": "qwen/qwen3-32b",
        "description": "Great for reasoning tasks",
        "icon": "🌟"
    }
}

# =====================================================
# RESUME DATA (Structured for Tools)
# =====================================================
RESUME_DATA = {
    "personal": {
        "name": "Vijai Venkatesan",
        "title": "Associate Consultant - AI/ML",
        "company": "Datamatics (TruAI Division)",
        "location": "Pondicherry, India",
        "email": "vijaibt1@gmail.com",
        "phone": "+91 8825947952",
        "linkedin": "linkedin.com/in/vijai-v-2b89841a3"
    },
    "summary": "Results-driven AI/ML Engineer with nearly 7+ years of experience in designing and deploying scalable AI solutions, including Generative AI and Large Language Models. Expertise in Python, machine learning, natural language processing, and intelligent document processing.",
    "experience": [
        {
            "title": "Associate Consultant - AI/ML",
            "company": "Datamatics (TruAI Division)",
            "location": "Pondicherry",
            "start_date": "April 2022",
            "end_date": "Present",
            "duration_years": 3.11,
            "projects": [
                "Named Entity Recognition for ADB",
                "Image Classification for UHG and Star Health",
                "Receipt Extraction from Images",
                "Photo Matching",
                "Resume Data Extraction",
                "Trepp Field Extraction from URLs",
                "Web Scraper for Fiercepharma",
                "TruAI GPT R&D",
                "Trepp Newsfeed",
                "Resume AI",
                "Azure OpenAI GPT-4 Integration for XPO Project"
            ],
            "key_projects": [
                {"name": "Ingram Micro Invoice Automation", "accuracy": "90%"},
                {"name": "BelleTire TruAI Automation", "accuracy": "93.40%"}
            ],
            "achievements": [
                "Achieved 90% extraction accuracy for Ingram Micro",
                "Achieved 93.40% accuracy for BelleTire",
                "Optimized processing to 10-11 seconds per page",
                "Leading end-to-end production ownership",
                "Managing multiple live production pipelines"
            ],
            "tech_stack": ["Python", "Django REST APIs", "Gemini 2.5 Flash", "Azure", "GCP"]
        },
        {
            "title": "Data Science Intern",
            "company": "Innodatatics",
            "location": "Hyderabad",
            "start_date": "October 2021",
            "end_date": "April 2022",
            "duration_years": 0.5,
            "projects": [
                {"name": "Recommendation Engine for Career Transition", "accuracy": "85%"},
                {"name": "Named Entity Recognition on Medical Journals"}
            ],
            "tech_stack": ["Python", "Scikit-Learn", "TensorFlow", "Streamlit", "NLP"]
        },
        {
            "title": "Associate (Medical Summarizer)",
            "company": "Aosta Software Technologies",
            "location": "Chennai",
            "start_date": "June 2021",
            "end_date": "August 2021",
            "duration_years": 0.25,
            "responsibilities": ["Medical record summarization", "AI/ML and NLP techniques"]
        },
        {
            "title": "Medical Record Analyst",
            "company": "Rapid Care Transcription",
            "location": "Pondicherry",
            "start_date": "April 2018",
            "end_date": "May 2021",
            "duration_years": 3.1,
            "responsibilities": ["Healthcare data analysis", "SQL and visualization"]
        },
        {
            "title": "Project Intern",
            "company": "CSIR-Central Leather Research Institute",
            "location": "Chennai",
            "start_date": "December 2016",
            "end_date": "June 2017",
            "duration_years": 0.5,
            "responsibilities": ["Research on antimicrobial activity"]
        }
    ],
    "skills": {
        "programming": ["Python", "R"],
        "frameworks": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "Keras", "Matplotlib", "Seaborn"],
        "ai_ml": ["Generative AI", "LLM", "Transformer Models", "BERT", "NLP", "NER", "Elasticsearch"],
        "cloud": ["GCP", "AWS", "Microsoft Azure"],
        "databases": ["MySQL"],
        "tools": ["Django REST API", "Postman", "Power BI", "Tableau", "GitHub"],
        "ides": ["VS Code", "PyCharm", "Jupyter Notebook", "Google Colab"]
    },
    "education": [
        {
            "degree": "B.Tech in Biotechnology",
            "institution": "Shri Andal Alagar College of Engineering, Anna University",
            "gpa": "72.9%",
            "year": "2013-2017"
        }
    ],
    "certifications": [
        "AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents (Udemy)",
        "AI Engineer Agentic Track: Complete Agent & MCP Course (Udemy)",
        "MCP Masterclass: Complete Guide to MCP in Python (Udemy)",
        "Data Science Certification (Panasonic CareerEx & 360DigiTMG)",
        "Machine Learning with Python (IBM)",
        "Deep Learning Fundamentals (IBM)",
        "Google Analytics for Beginners",
        "Cloud Learning: AWS DevOps, Azure DevOps, Docker"
    ],
    "awards": [
        {"name": "L&D Trainer Felicitation", "date": "May 2025", "org": "Datamatics"},
        {"name": "Spot Individual Award Winner", "date": "July 2024", "org": "Datamatics"},
        {"name": "L&D Trainer Felicitation", "date": "May 2024", "org": "Datamatics"},
        {"name": "Spot Individual Award Winner", "date": "June 2023", "org": "Datamatics"}
    ]
}

# Calculate total experience
def calculate_total_experience() -> float:
    return sum(exp.get("duration_years", 0) for exp in RESUME_DATA["experience"])

RESUME_DATA["total_experience_years"] = calculate_total_experience()


# =====================================================
# MCP-LIKE TOOL DEFINITIONS
# =====================================================

@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None


class Tool:
    """Base class for MCP-like tools"""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError


class SearchResumeTool(Tool):
    name = "search_resume"
    description = "Search the resume for specific information using semantic search"
    parameters = {
        "query": {"type": "string", "description": "Search query"}
    }
    
    def __init__(self, rag):
        self.rag = rag
    
    def execute(self, query: str) -> ToolResult:
        try:
            results = self.rag.search(query, top_k=3)
            data = [{"text": chunk.text, "section": chunk.section, "score": score} 
                    for chunk, score in results]
            return ToolResult(success=True, data=data)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetSkillsTool(Tool):
    name = "get_skills"
    description = "Get all skills categorized by type (programming, frameworks, AI/ML, cloud, etc.)"
    parameters = {
        "category": {"type": "string", "description": "Optional: specific category (programming, frameworks, ai_ml, cloud, databases, tools)", "required": False}
    }
    
    def execute(self, category: str = None) -> ToolResult:
        try:
            skills = RESUME_DATA["skills"]
            if category and category in skills:
                return ToolResult(success=True, data={category: skills[category]})
            return ToolResult(success=True, data=skills)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetExperienceTool(Tool):
    name = "get_experience"
    description = "Get work experience details. Can filter by company or get all."
    parameters = {
        "company": {"type": "string", "description": "Optional: specific company name", "required": False}
    }
    
    def execute(self, company: str = None) -> ToolResult:
        try:
            experience = RESUME_DATA["experience"]
            if company:
                filtered = [exp for exp in experience if company.lower() in exp["company"].lower()]
                return ToolResult(success=True, data=filtered)
            return ToolResult(success=True, data=experience)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetProjectsTool(Tool):
    name = "get_projects"
    description = "Get all projects worked on with details"
    parameters = {}
    
    def execute(self) -> ToolResult:
        try:
            all_projects = []
            for exp in RESUME_DATA["experience"]:
                company = exp["company"]
                if "projects" in exp:
                    for proj in exp["projects"]:
                        if isinstance(proj, dict):
                            all_projects.append({**proj, "company": company})
                        else:
                            all_projects.append({"name": proj, "company": company})
                if "key_projects" in exp:
                    for proj in exp["key_projects"]:
                        all_projects.append({**proj, "company": company, "is_key": True})
            return ToolResult(success=True, data=all_projects)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetAchievementsTool(Tool):
    name = "get_achievements"
    description = "Get key achievements and metrics"
    parameters = {}
    
    def execute(self) -> ToolResult:
        try:
            achievements = []
            for exp in RESUME_DATA["experience"]:
                if "achievements" in exp:
                    for ach in exp["achievements"]:
                        achievements.append({"achievement": ach, "company": exp["company"]})
                if "key_projects" in exp:
                    for proj in exp["key_projects"]:
                        if "accuracy" in proj:
                            achievements.append({
                                "achievement": f"{proj['name']}: {proj['accuracy']} accuracy",
                                "company": exp["company"]
                            })
            return ToolResult(success=True, data=achievements)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetContactTool(Tool):
    name = "get_contact"
    description = "Get contact information"
    parameters = {}
    
    def execute(self) -> ToolResult:
        try:
            return ToolResult(success=True, data=RESUME_DATA["personal"])
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetCertificationsTool(Tool):
    name = "get_certifications"
    description = "Get all certifications"
    parameters = {}
    
    def execute(self) -> ToolResult:
        try:
            return ToolResult(success=True, data=RESUME_DATA["certifications"])
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetAwardsTool(Tool):
    name = "get_awards"
    description = "Get all awards and recognition"
    parameters = {}
    
    def execute(self) -> ToolResult:
        try:
            return ToolResult(success=True, data=RESUME_DATA["awards"])
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class CalculateExperienceTool(Tool):
    name = "calculate_experience"
    description = "Calculate total years of experience or experience in specific area"
    parameters = {
        "area": {"type": "string", "description": "Optional: specific area like 'AI/ML', 'Data Science'", "required": False}
    }
    
    def execute(self, area: str = None) -> ToolResult:
        try:
            total = RESUME_DATA["total_experience_years"]
            if area:
                # Calculate area-specific experience
                area_lower = area.lower()
                area_exp = 0
                for exp in RESUME_DATA["experience"]:
                    title = exp["title"].lower()
                    if any(term in title for term in ["ai", "ml", "data science", "machine learning"]):
                        area_exp += exp.get("duration_years", 0)
                return ToolResult(success=True, data={
                    "total_experience": f"{total:.1f} years",
                    f"{area}_experience": f"{area_exp:.1f} years"
                })
            return ToolResult(success=True, data={"total_experience": f"{total:.1f} years"})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetEducationTool(Tool):
    name = "get_education"
    description = "Get educational background"
    parameters = {}
    
    def execute(self) -> ToolResult:
        try:
            return ToolResult(success=True, data=RESUME_DATA["education"])
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class CompareTool(Tool):
    name = "compare_roles"
    description = "Compare different roles/positions held"
    parameters = {}
    
    def execute(self) -> ToolResult:
        try:
            comparison = []
            for exp in RESUME_DATA["experience"]:
                comparison.append({
                    "title": exp["title"],
                    "company": exp["company"],
                    "duration": f"{exp.get('duration_years', 0):.1f} years",
                    "period": f"{exp['start_date']} - {exp['end_date']}"
                })
            return ToolResult(success=True, data=comparison)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


# =====================================================
# AGENT CLASS
# =====================================================

class ResumeAgent:
    """Agentic AI for Resume Q&A with Tool Use"""
    
    def __init__(self, rag, model_id: str):
        self.rag = rag
        self.model_id = model_id
        self.tools = self._register_tools()
        self.conversation_history = []
    
    def _register_tools(self) -> Dict[str, Tool]:
        """Register all available tools"""
        return {
            "search_resume": SearchResumeTool(self.rag),
            "get_skills": GetSkillsTool(),
            "get_experience": GetExperienceTool(),
            "get_projects": GetProjectsTool(),
            "get_achievements": GetAchievementsTool(),
            "get_contact": GetContactTool(),
            "get_certifications": GetCertificationsTool(),
            "get_awards": GetAwardsTool(),
            "calculate_experience": CalculateExperienceTool(),
            "get_education": GetEducationTool(),
            "compare_roles": CompareTool()
        }
    
    def _get_tools_description(self) -> str:
        """Generate tools description for the LLM"""
        tools_desc = []
        for name, tool in self.tools.items():
            params = getattr(tool, 'parameters', {})
            params_str = json.dumps(params) if params else "none"
            tools_desc.append(f"- {name}: {tool.description}\n  Parameters: {params_str}")
        return "\n".join(tools_desc)
    
    def _parse_tool_calls(self, response: str) -> List[Dict]:
        """Parse tool calls from LLM response"""
        tool_calls = []
        
        # Pattern: [TOOL: tool_name(param1="value1", param2="value2")]
        pattern = r'\[TOOL:\s*(\w+)\((.*?)\)\]'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for tool_name, params_str in matches:
            params = {}
            if params_str.strip():
                # Parse parameters
                param_pattern = r'(\w+)\s*=\s*["\']?([^"\']+)["\']?'
                param_matches = re.findall(param_pattern, params_str)
                for key, value in param_matches:
                    params[key] = value.strip()
            
            tool_calls.append({"tool": tool_name, "params": params})
        
        return tool_calls
    
    def _execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute parsed tool calls"""
        results = []
        for call in tool_calls:
            tool_name = call["tool"]
            params = call["params"]
            
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                result = tool.execute(**params)
                results.append({
                    "tool": tool_name,
                    "params": params,
                    "result": result.data if result.success else f"Error: {result.error}",
                    "success": result.success
                })
            else:
                results.append({
                    "tool": tool_name,
                    "params": params,
                    "result": f"Unknown tool: {tool_name}",
                    "success": False
                })
        
        return results
    
    def _call_llm(self, messages: List[Dict]) -> str:
        """Call Groq API"""
        api_key = st.secrets.get("GROQ_API_KEY", "")
        
        if not api_key:
            return "Error: API key not configured"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def run(self, user_query: str) -> Dict:
        """Run the agent with ReAct pattern"""
        
        # Step 1: Planning - Determine which tools to use
        system_prompt = f"""You are an AI assistant for Vijai Venkatesan's resume. You have access to tools to retrieve accurate information.

AVAILABLE TOOLS:
{self._get_tools_description()}

INSTRUCTIONS:
1. Analyze the user's question
2. Decide which tools to use to gather information
3. Call tools using this format: [TOOL: tool_name(param="value")]
4. You can call multiple tools if needed
5. After gathering information, provide a clear answer

IMPORTANT: Always use tools to get accurate data. Don't make up information.

Example:
User: "What are Vijai's skills?"
Response: Let me check the skills information.
[TOOL: get_skills()]

User: "How many years of experience does he have?"
Response: Let me calculate the total experience.
[TOOL: calculate_experience()]
"""
        
        # Initial tool selection
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        planning_response = self._call_llm(messages)
        
        # Parse and execute tools
        tool_calls = self._parse_tool_calls(planning_response)
        tool_results = []
        
        if tool_calls:
            tool_results = self._execute_tools(tool_calls)
        else:
            # If no tools called, do a semantic search as fallback
            search_result = self.tools["search_resume"].execute(query=user_query)
            tool_results = [{
                "tool": "search_resume",
                "params": {"query": user_query},
                "result": search_result.data,
                "success": search_result.success
            }]
        
        # Step 2: Generate final response with tool results
        tool_results_str = json.dumps(tool_results, indent=2, default=str)
        
        final_prompt = f"""Based on the tool results below, provide a helpful and accurate answer to the user's question.

USER QUESTION: {user_query}

TOOL RESULTS:
{tool_results_str}

INSTRUCTIONS:
- Use the tool results to provide accurate information
- Be professional and friendly
- Include specific numbers and details when available
- If the information isn't in the results, say so politely
- Format the response clearly

ANSWER:"""
        
        messages = [
            {"role": "system", "content": "You are a professional AI assistant representing Vijai Venkatesan. Provide accurate, helpful responses based on the tool results."},
            {"role": "user", "content": final_prompt}
        ]
        
        final_response = self._call_llm(messages)
        
        return {
            "answer": final_response,
            "tools_used": [t["tool"] for t in tool_results],
            "tool_results": tool_results,
            "reasoning": planning_response
        }


# =====================================================
# RAG (for semantic search tool)
# =====================================================

RESUME_TEXT = """
VIJAI VENKATESAN
Contact: vijaibt1@gmail.com | +91 8825947952 | linkedin.com/in/vijai-v-2b89841a3

PROFESSIONAL SUMMARY
Results-driven AI/ML Engineer with nearly 7+ years of experience in designing and deploying scalable AI solutions, including Generative AI and Large Language Models. Expertise in Python, machine learning, natural language processing, and intelligent document processing.

WORK EXPERIENCE

Associate Consultant - AI/ML at Datamatics (TruAI Division) | April 2022 - Present
- Leading end-to-end production ownership of Ingram Micro and BelleTire TruAI automation systems
- Achieved 90% extraction accuracy for Ingram Micro invoice processing
- Achieved 93.40% accuracy for BelleTire invoice processing
- Optimized processing to 10-11 seconds per page
- Projects: Named Entity Recognition, Image Classification, Receipt Extraction, Resume AI, Azure OpenAI GPT-4 Integration

Data Science Intern at Innodatatics | October 2021 - April 2022
- Recommendation Engine for Career Transition (85% accuracy)
- Named Entity Recognition on Medical Journals

SKILLS
Programming: Python, R
Frameworks: Pandas, NumPy, Scikit-learn, TensorFlow, Keras
AI/ML: Generative AI, LLM, BERT, NLP, NER
Cloud: GCP, AWS, Azure
Tools: Django REST API, Power BI, Tableau

CERTIFICATIONS
- AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents
- AI Engineer Agentic Track: Complete Agent & MCP Course
- MCP Masterclass: Complete Guide to MCP in Python
- Data Science Certification (IBM, 360DigiTMG)

AWARDS
- L&D Trainer Felicitation (2025, 2024)
- Spot Individual Award Winner (2024, 2023)
"""


@dataclass
class Chunk:
    text: str
    section: str
    index: int


class LightweightRAG:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectorizer: TfidfVectorizer = None
        self.tfidf_matrix = None
    
    def split_and_index(self, text: str):
        sections = text.split('\n\n')
        self.chunks = []
        
        for i, section in enumerate(sections):
            if section.strip():
                self.chunks.append(Chunk(
                    text=section.strip(),
                    section=self._identify_section(section),
                    index=i
                ))
        
        chunk_texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
    
    def _identify_section(self, text: str) -> str:
        text_upper = text.upper()
        if "SUMMARY" in text_upper:
            return "Summary"
        elif any(kw in text_upper for kw in ["EXPERIENCE", "POSITION", "COMPANY"]):
            return "Experience"
        elif "SKILL" in text_upper:
            return "Skills"
        elif "CERTIFICATION" in text_upper:
            return "Certifications"
        elif "AWARD" in text_upper:
            return "Awards"
        elif "EDUCATION" in text_upper:
            return "Education"
        elif "CONTACT" in text_upper:
            return "Contact"
        return "General"
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_idx = np.argsort(similarities)[::-1][:top_k]
        return [(self.chunks[i], similarities[i]) for i in top_idx]


@st.cache_resource
def initialize_rag() -> LightweightRAG:
    rag = LightweightRAG()
    rag.split_and_index(RESUME_TEXT)
    return rag


# =====================================================
# CSS STYLING
# =====================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .agent-badge {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    .profile-card h3 { margin: 0 0 0.5rem 0; font-size: 1.2rem; }
    .profile-card p { margin: 0.25rem 0; font-size: 0.85rem; opacity: 0.95; }
    .profile-card a { color: white !important; text-decoration: underline; }
    
    .tool-card {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 0.5rem;
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0;
        font-size: 0.8rem;
    }
    
    .tool-name {
        color: #059669;
        font-weight: 600;
    }
    
    .reasoning-card {
        background: #fefce8;
        border: 1px solid #fde047;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    
    .model-info {
        background: #e3f2fd;
        border-radius: 0.5rem;
        padding: 0.5rem 0.75rem;
        font-size: 0.8rem;
        color: #1565c0;
        margin-top: 0.5rem;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf0 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    .powered-by {
        background: linear-gradient(90deg, #f97316 0%, #ea580c 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 2rem;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Llama 3.1 8B (Fast)"

if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = False


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div class="profile-card">
        <h3>👤 Vijai Venkatesan</h3>
        <p><strong>Associate Consultant - AI/ML</strong></p>
        <p>🏢 Datamatics (TruAI Division)</p>
        <p>📍 Pondicherry, India</p>
        <p>📧 vijaibt1@gmail.com</p>
        <p>🔗 <a href="https://linkedin.com/in/vijai-v-2b89841a3" target="_blank">LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Quick Stats")
    c1, c2 = st.columns(2)
    c1.metric("Experience", "~7+ Years")
    c2.metric("Projects", "12+")
    c3, c4 = st.columns(2)
    c3.metric("Certs", "9+")
    c4.metric("Awards", "4")
    
    st.divider()
    
    st.markdown("### 🤖 Agent Configuration")
    
    selected_model = st.selectbox(
        "AI Model",
        options=list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.selected_model),
        format_func=lambda x: f"{GROQ_MODELS[x]['icon']} {x}",
        label_visibility="collapsed"
    )
    st.session_state.selected_model = selected_model
    
    model_info = GROQ_MODELS[selected_model]
    st.markdown(f'<div class="model-info">ℹ️ {model_info["description"]}</div>', unsafe_allow_html=True)
    
    st.session_state.show_reasoning = st.checkbox("🔍 Show Agent Reasoning", value=st.session_state.show_reasoning)
    
    st.markdown('<div class="powered-by">🤖 Agentic AI + Tools</div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🛠️ Available Tools")
    tools_list = [
        "🔍 search_resume",
        "💻 get_skills",
        "💼 get_experience",
        "📁 get_projects",
        "🏆 get_achievements",
        "📧 get_contact",
        "📜 get_certifications",
        "🥇 get_awards",
        "📊 calculate_experience",
        "🎓 get_education"
    ]
    for tool in tools_list:
        st.markdown(f"<small>{tool}</small>", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 💡 Try These")
    questions = [
        "What skills does Vijai have?",
        "Calculate total experience",
        "What projects has he done?",
        "What are his key achievements?",
        "Compare his different roles",
        "What certifications does he have?"
    ]
    
    for i, q in enumerate(questions):
        if st.button(f"📌 {q}", key=f"q_{i}", use_container_width=True):
            st.session_state.pending_question = q
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =====================================================
# MAIN CONTENT
# =====================================================

st.markdown('<h1 class="main-header">🤖 Agentic Resume Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by AI Agents with Tool Use & Multi-step Reasoning</p>', unsafe_allow_html=True)

# Initialize
rag = initialize_rag()
model_id = GROQ_MODELS[st.session_state.selected_model]["id"]
agent = ResumeAgent(rag, model_id)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and msg.get("tools_used"):
            st.markdown(f'<span class="agent-badge">🛠️ Tools: {", ".join(msg["tools_used"])}</span>', unsafe_allow_html=True)
            
            if st.session_state.show_reasoning and msg.get("reasoning"):
                with st.expander("🧠 Agent Reasoning"):
                    st.markdown(f'<div class="reasoning-card">{msg["reasoning"]}</div>', unsafe_allow_html=True)
            
            if msg.get("tool_results"):
                with st.expander("📊 Tool Results"):
                    for result in msg["tool_results"]:
                        st.markdown(f'<div class="tool-card"><span class="tool-name">{result["tool"]}</span></div>', unsafe_allow_html=True)
                        st.json(result["result"])

# Handle pending question
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question
    
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.spinner(f"🤖 Agent thinking with {st.session_state.selected_model}..."):
        result = agent.run(question)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tools_used": result["tools_used"],
        "tool_results": result["tool_results"],
        "reasoning": result["reasoning"]
    })
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about experience, skills, projects... (Agent will use tools!)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(f"🤖 Agent reasoning with {st.session_state.selected_model}..."):
            result = agent.run(prompt)
        
        st.markdown(result["answer"])
        st.markdown(f'<span class="agent-badge">🛠️ Tools: {", ".join(result["tools_used"])}</span>', unsafe_allow_html=True)
        
        if st.session_state.show_reasoning:
            with st.expander("🧠 Agent Reasoning"):
                st.markdown(f'<div class="reasoning-card">{result["reasoning"]}</div>', unsafe_allow_html=True)
        
        with st.expander("📊 Tool Results"):
            for tr in result["tool_results"]:
                st.markdown(f'<div class="tool-card"><span class="tool-name">{tr["tool"]}</span></div>', unsafe_allow_html=True)
                st.json(tr["result"])
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tools_used": result["tools_used"],
        "tool_results": result["tool_results"],
        "reasoning": result["reasoning"]
    })

st.markdown("""
<style>
.welcome-card {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 100%) !important;
    border-radius: 15px !important;
    padding: 25px !important;
    margin: 20px 0 !important;
    border-left: 5px solid #4CAF50 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
}

.welcome-card h3 {
    color: #ffffff !important;
    font-size: 1.4rem !important;
    margin-bottom: 15px !important;
    font-weight: 700 !important;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5) !important;
}

.welcome-card p {
    color: #ffffff !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    margin: 8px 0 !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.4) !important;
}

.welcome-card strong {
    color: #5dde6e !important;
    font-weight: 600 !important;
}

.welcome-card em {
    color: #ffe44d !important;
    font-style: normal !important;
}
</style>
""", unsafe_allow_html=True)

# Welcome Card
if not st.session_state.messages:
    st.markdown(f"""
    <div class="welcome-card">
        <h3>👋 Welcome to the Agentic Resume Assistant!</h3>
        <p>This AI uses <strong>Agentic Architecture</strong> with <strong>MCP-like Tools</strong></p>
        <br>
        <p><strong>🛠️ The agent can:</strong></p>
        <p>• Use tools to retrieve accurate information</p>
        <p>• Reason step-by-step (ReAct pattern)</p>
        <p>• Combine multiple data sources</p>
        <p>• Calculate and compare information</p>
        <br>
        <p>Toggle <strong>"Show Agent Reasoning"</strong> to see how the AI thinks!</p>
        <br>
        <p><em>👈 Try a quick question or type your own!</em></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
    🤖 Agentic AI with Tool Use | 
    💼 <a href="https://linkedin.com/in/vijai-v-2b89841a3" target="_blank">LinkedIn</a> |
    📧 <a href="mailto:vijaibt1@gmail.com">Email</a>
</div>
""", unsafe_allow_html=True)

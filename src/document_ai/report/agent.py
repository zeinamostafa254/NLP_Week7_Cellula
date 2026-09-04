from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from document_ai.llm.model import get_llm
from document_ai.schemas.answer import FinalAnswer
from document_ai.report.pdf_tools import generate_styled_pdf

import logging
logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = """
You are the Report Design Agent.
Your job is to take a final answered text and convert it into a beautiful, styled PDF report.

You have access to design tools, specifically `generate_styled_pdf`.
You must choose an appropriate aesthetic (fonts, title color, text color, background color) 
based on the tone or topic of the question.

Colors must be valid Hex codes (e.g. #2C3E50, #EAEAEA).
Fonts must be one of: Helvetica, Times, Courier.

When a user provides a Final Answer, call the tool to generate the PDF, and then return a 
friendly message including the file path to the generated PDF.
"""

class ReportAgent:
    """
    An Agent that designs and generates PDFs from FinalAnswers.
    """
    def __init__(self):
        self.agent = create_react_agent(
            model=get_llm(),
            tools=[generate_styled_pdf],
            prompt=REPORT_SYSTEM_PROMPT
        )

    def generate_report(self, final_answer: FinalAnswer, custom_instructions: str = "") -> str:
        logger.info("    [Report] Generating styled PDF report...")
        
        # Sanitize text for standard PDF fonts (Latin-1)
        safe_answer = final_answer.answer.encode('latin-1', 'replace').decode('latin-1')
        safe_sources = final_answer.sources.encode('latin-1', 'replace').decode('latin-1')
        
        content = f"""
Please generate a PDF for the following response.

Title / Question:
{final_answer.question}

Content:
{safe_answer}

Sources:
{safe_sources}

User Design Requests:
{custom_instructions if custom_instructions else "None. Please decide the best aesthetic yourself."}
"""
        
        response = self.agent.invoke(
            {"messages": [("user", content)]}
        )
        
        return response["messages"][-1].content

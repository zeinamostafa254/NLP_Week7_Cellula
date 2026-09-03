import uuid
from pathlib import Path
from fpdf import FPDF
from langchain_core.tools import tool

# Create a reports directory
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

class ReportPDF(FPDF):
    def __init__(self, title_color=(0, 0, 0), bg_color=(255, 255, 255), **kwargs):
        super().__init__(**kwargs)
        self.title_color = title_color
        self.bg_color = bg_color

    def header(self):
        # We can add a simple header
        self.set_fill_color(*self.bg_color)
        self.rect(0, 0, 210, 297, "F") # Fill background

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def hex_to_rgb(hex_code: str) -> tuple:
    hex_code = hex_code.lstrip('#')
    if len(hex_code) != 6:
        return (0, 0, 0)
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

@tool
def generate_styled_pdf(
    title: str, 
    content: str, 
    sources: str, 
    font_family: str = "Helvetica", 
    title_color_hex: str = "#2C3E50", 
    text_color_hex: str = "#333333",
    bg_color_hex: str = "#FFFFFF"
) -> str:
    """
    Generates a beautifully styled PDF report.
    Args:
        title: The main title of the report.
        content: The main body text of the report (use simple text/paragraphs, avoid complex markdown).
        sources: A string containing the list of sources/bibliography.
        font_family: The font to use (e.g., 'Helvetica', 'Times', 'Courier').
        title_color_hex: Hex color for the title (e.g., '#FF5733').
        text_color_hex: Hex color for the main text.
        bg_color_hex: Hex color for the page background.
    Returns:
        The file path of the generated PDF.
    """
    try:
        t_color = hex_to_rgb(title_color_hex)
        txt_color = hex_to_rgb(text_color_hex)
        b_color = hex_to_rgb(bg_color_hex)
        
        pdf = ReportPDF(title_color=t_color, bg_color=b_color)
        pdf.add_page()
        
        # Safe font fallback
        valid_fonts = ["Helvetica", "Times", "Courier"]
        if font_family not in valid_fonts:
            font_family = "Helvetica"

        # Title
        pdf.set_font(font_family, "B", 24)
        pdf.set_text_color(*t_color)
        pdf.multi_cell(0, 12, txt=title, align="C")
        pdf.ln(10)
        
        # Content
        pdf.set_font(font_family, "", 12)
        pdf.set_text_color(*txt_color)
        
        # FPDF handles multi_cell well for paragraphs. Replace some markdown chars if present.
        clean_content = content.replace('**', '').replace('__', '')
        for paragraph in clean_content.split('\n'):
            if paragraph.strip():
                pdf.multi_cell(0, 8, txt=paragraph.strip())
                pdf.ln(2)
        
        # Sources
        if sources.strip():
            pdf.ln(10)
            pdf.set_font(font_family, "B", 16)
            pdf.set_text_color(*t_color)
            pdf.cell(0, 10, txt="Sources & References", ln=True)
            pdf.set_font(font_family, "I", 10)
            pdf.set_text_color(*txt_color)
            for src_line in sources.split('\n'):
                if src_line.strip():
                    pdf.multi_cell(0, 6, txt=src_line.strip())
                    
        file_id = str(uuid.uuid4())[:8]
        file_path = REPORTS_DIR / f"report_{file_id}.pdf"
        
        # encode output safely (latin-1 by default for standard fonts in FPDF)
        pdf.output(str(file_path))
        return f"Successfully generated PDF report at: {file_path}"
    
    except Exception as e:
        return f"Error generating PDF: {str(e)}"

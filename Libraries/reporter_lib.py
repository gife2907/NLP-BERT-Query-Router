# reporter_lib.py

import sys
import os
import datetime
import io
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
import matplotlib.pyplot as plt

# Additional imports for SVG conversion
try:
    from cairosvg import svg2png
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False
    
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

class OutputLogger:
    """
    A class to redirect stdout to both terminal and a text file.
    """
    def __init__(self, log_directory="Results", filename="stdout.txt"):
        # Create the directory if it doesn't exist
        os.makedirs(log_directory, exist_ok=True)
        
        # Set up the file path
        self.log_file_path = os.path.join(log_directory, filename)
        
        # Store the original stdout
        self.terminal = sys.stdout
        
        # Open the log file
        self.log = open(self.log_file_path, "w", encoding="utf-8")
        
        # Report data storage
        self.report_data = []
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        if self.log:
            self.log.close()
            
    def __del__(self):
        self.close()

    def add_to_report(self, content_type, content, **kwargs):
        """
        Add content to the report data structure
        
        Parameters:
        - content_type: "heading1", "heading2", "text", "image", "table", etc.
        - content: The actual content (text, image path, etc.)
        - kwargs: Additional arguments specific to the content type
        """
        self.report_data.append({
            "type": content_type,
            "content": content,
            "options": kwargs
        })
        
        # If it's regular text, also print to stdout and log
        if content_type == "text":
            self.write(content + "\n")


class ReportGenerator:
    """
    Class to generate reports from the logged data and additional content.
    """
    def __init__(self, logger, report_directory="Results"):
        self.logger = logger
        self.report_directory = report_directory
        os.makedirs(report_directory, exist_ok=True)
        
    def _create_front_page(self, doc, title, subtitle=None, author=None, organization=None):
        """Create a professional front page for the report"""
        section = doc.sections[0]
        
        # Add title
        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title_para.add_run(title)
        title_run.bold = True
        title_run.font.size = Pt(24)
        
        # Add subtitle if provided
        if subtitle:
            subtitle_para = doc.add_paragraph()
            subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            subtitle_run = subtitle_para.add_run(subtitle)
            subtitle_run.italic = True
            subtitle_run.font.size = Pt(16)
        
        # Add date
        date_para = doc.add_paragraph()
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_run = date_para.add_run(datetime.datetime.now().strftime("%B %d, %Y"))
        date_run.font.size = Pt(12)
        
        # Add author if provided
        if author:
            author_para = doc.add_paragraph()
            author_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            author_para.add_run(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPrepared by: {author}")
        
        # Add organization if provided
        if organization:
            org_para = doc.add_paragraph()
            org_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            org_para.add_run(organization)
            
        # Add page break after cover page
        doc.add_page_break()
    
    def _insert_toc(self, doc):
        """Insert table of contents"""
        doc.add_paragraph("Table of Contents").style = 'Heading 1'
        
        # Add TOC field
        doc.add_paragraph().add_run("TOC WILL BE GENERATED AUTOMATICALLY").bold = True
        
        # In a real Word document, you would use this field code:
        # {TOC \o "1-2" \h \z \u}
        # But we use a placeholder as python-docx doesn't directly support field codes
        
        doc.add_page_break()
    
    def _insert_external_content(self, doc, file_path):
        """Insert content from an external document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                doc.add_paragraph(content)
        except Exception as e:
            doc.add_paragraph(f"Error including external content: {str(e)}")
            
    def _convert_svg_to_png(self, svg_path):
        """Convert SVG to PNG for Word compatibility"""
        if not os.path.exists(svg_path):
            print(f"SVG file not found: {svg_path}")
            return None
        
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(svg_path))[0]
        png_path = os.path.join(self.report_directory, f"{base_name}.png")
        
        try:
            if HAS_CAIROSVG:
                # Use cairosvg to convert SVG to PNG
                with open(svg_path, 'rb') as svg_file:
                    svg_content = svg_file.read()
                    svg2png(bytestring=svg_content, write_to=png_path, output_width=800)
                return png_path
            elif HAS_PIL:
                # Fallback warning
                print("Warning: CairoSVG not found. SVG conversion may be suboptimal.")
                # Try using other methods (this is a placeholder - real implementation would need a different approach)
                # This won't actually work but shows where to integrate alternate conversion
                return None
            else:
                print("Warning: SVG conversion libraries not found. Skipping SVG image.")
                return None
        except Exception as e:
            print(f"Error converting SVG to PNG: {str(e)}")
            return None
    
    def generate_report(self, output_filename="report.docx", title="ML/NLP Research Report", 
                        subtitle="Results and Analysis", author=None, organization=None, intro_file=None):
        """
        Generate a professional report based on collected data
        """
        # Create document
        doc = Document()
        
        # Front page
        self._create_front_page(doc, title, subtitle, author, organization)
        
        # Table of contents
        self._insert_toc(doc)
        
        # Introduction section
        doc.add_heading("Introduction", level=1)
        if intro_file and os.path.exists(intro_file):
            self._insert_external_content(doc, intro_file)
        else:
            doc.add_paragraph("Introduction content not found or not specified.")
        
        # Process all the content collected in the logger
        for item in self.logger.report_data:
            content_type = item["type"]
            content = item["content"]
            options = item["options"]
            
            if content_type == "heading1":
                doc.add_heading(content, level=1)
            
            elif content_type == "heading2":
                doc.add_heading(content, level=2)
            
            elif content_type == "text":
                paragraph = doc.add_paragraph(content)
                
                # Apply formatting options if provided
                if options.get("bold", False):
                    for run in paragraph.runs:
                        run.bold = True
                if options.get("italic", False):
                    for run in paragraph.runs:
                        run.italic = True
            
            elif content_type == "image":
                # Check if the file exists
                if os.path.exists(content):
                    paragraph = doc.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    width = options.get("width", 6)  # Default width in inches
                    
                    # Check if it's an SVG file
                    if content.lower().endswith('.svg'):
                        # Try to convert SVG to PNG
                        png_path = self._convert_svg_to_png(content)
                        if png_path and os.path.exists(png_path):
                            doc.add_picture(png_path, width=Inches(width))
                        else:
                            # If conversion fails, add a placeholder text
                            doc.add_paragraph(f"[SVG image: {os.path.basename(content)}]")
                            print(f"Warning: Could not convert SVG to PNG: {content}")
                    else:
                        # For non-SVG images
                        try:
                            doc.add_picture(content, width=Inches(width))
                        except Exception as e:
                            doc.add_paragraph(f"[Image could not be added: {os.path.basename(content)}]")
                            print(f"Error adding image: {str(e)}")
                else:
                    doc.add_paragraph(f"[Image not found: {content}]")
            
            elif content_type == "table":
                # Example for adding tables, can be expanded as needed
                table_data = content
                if table_data and isinstance(table_data, list) and len(table_data) > 0:
                    # Create table with appropriate dimensions
                    num_rows = len(table_data)
                    num_cols = len(table_data[0]) if num_rows > 0 else 0
                    
                    table = doc.add_table(rows=num_rows, cols=num_cols)
                    table.style = 'Table Grid'
                    
                    # Fill the table with data
                    for i, row_data in enumerate(table_data):
                        for j, cell_data in enumerate(row_data):
                            table.cell(i, j).text = str(cell_data)
        
        # Save the document
        output_path = os.path.join(self.report_directory, output_filename)
        doc.save(output_path)
        print(f"Report saved to {output_path}")
        return output_path

# Function to set up the logger and reporter
def setup_logging_and_reporting(log_dir="Results", log_filename="stdout.txt"):
    """
    Set up the logging and reporting system
    
    Returns a tuple containing:
    - logger: The OutputLogger instance
    - reporter: The ReportGenerator instance
    """
    logger = OutputLogger(log_directory=log_dir, filename=log_filename)
    # Redirect stdout to our logger
    sys.stdout = logger
    
    # Create the reporter
    reporter = ReportGenerator(logger, report_directory=log_dir)
    
    return logger, reporter
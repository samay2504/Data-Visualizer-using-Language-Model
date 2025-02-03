import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from io import BytesIO
import re
import base64
import svgwrite
from pdfminer.high_level import extract_text
from jinja2 import Template
from groq import Groq

# Initialize Groq client
client = Groq(api_key="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")

def extract_text_from_pdf(pdf_file_path):
    """Extracts text from PDF using pdfminer."""
    return extract_text(pdf_file_path)


def chunk_text(text, max_length=1000):
    """Chunks text into manageable parts."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def generate_summary_and_extract_tables(text):
    """Generates concise summaries and extracts tables using LLM."""
    chunks = chunk_text(text)
    summaries = []
    extracted_tables = []

    for chunk in chunks:
        response = client.chat.completions.create(
            messages=[{"role": "user",
                       "content": f"Please summarize the following text by breaking it down by headings (e.g., Introduction, Methods, Results, Conclusion)."
                                  f"Provide concise key points (2-3 max per section) and suggest how to layout the information visually."
                                  f"Extract any tables with numerical data, including their context and title, while omitting non-numerical information."
                                  f"Focus on clarity and visual appeal:\n\n{chunk}"}],
            model="llama3-8b-8192",
        )
        content = response.choices[0].message.content
        summaries.append(content)

        # Extract tables using regex for numerical data
        tables = re.findall(r'(?:Table|Data|Results|Statistics).*?:\s*(.*?)(?:\n\s*\n|$)', content, re.DOTALL)
        extracted_tables.extend(
            [t for t in tables if not re.match(r'(None|No tables|no tables|No tabular)', t, re.IGNORECASE)])

    # Split summary into bullet points
    summary_points = re.split(r'(?<=\.) ', " ".join(summaries))

    return summary_points, extracted_tables

def convert_csv_to_dataframe(csv_tables):
    """Converts extracted CSV tables to DataFrames."""
    dataframes = []
    for csv_table in csv_tables:
        try:
            if csv_table.strip():
                cleaned_csv = re.sub(r'[ \t]+', ',', csv_table.strip())
                data = pd.read_csv(StringIO(cleaned_csv), sep=',', on_bad_lines='skip', engine='python')
                if not data.empty:
                    dataframes.append(data)
        except Exception as e:
            print(f"Error processing table: {e}")
    return dataframes


def sanitize_filename(filename):
    """Sanitize the file name to remove invalid characters."""
    return re.sub(r'[\\/*?:"<>|,]', "_", filename)

def create_plots(dataframes):
    """Generate a variety of plots based on the data in each DataFrame."""
    plots = []
    for i, df in enumerate(dataframes):
        if df.empty or df.select_dtypes(include='number').empty:
            continue

        numeric_cols = df.select_dtypes(include='number').columns
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()

        plt.figure(figsize=(8, 6))

        try:
            if len(numeric_cols) == 1:
                sns.histplot(df[numeric_cols[0]], kde=True)
            elif len(numeric_cols) > 1 and len(df) <= 1000:
                sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(categorical_cols) > 0:
                sns.countplot(x=categorical_cols[0], data=df)
            else:
                sns.lineplot(data=df)
        except Exception as e:
            print(f"Error creating plot for DataFrame {i}: {e}")
            continue

        plt.tight_layout()
        img_data = BytesIO()  # Use BytesIO for binary data
        plt.savefig(img_data, format='png')
        plt.close()
        img_data.seek(0)  # Rewind the buffer to the beginning
        plots.append(base64.b64encode(img_data.getvalue()).decode('utf-8'))

    return plots

def table_to_svg(df):
    """Converts a DataFrame to an SVG image."""
    dwg = svgwrite.Drawing(size=(600, 400))
    x, y = 10, 20

    for j, col in enumerate(df.columns):
        dwg.add(dwg.text(col, insert=(x + j * 100, y), font_size='14px', font_weight='bold'))

    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, value in enumerate(row):
            dwg.add(dwg.text(str(value), insert=(x + j * 100, y + i * 20), font_size='12px'))

    svg_code = dwg.tostring()
    return base64.b64encode(svg_code.encode('utf-8')).decode('utf-8')


def generate_poster(template_file, output_file, title, summary_text, plots, tables):
    """Renders and saves the poster as an HTML file."""
    with open(template_file, 'r', encoding='utf-8') as file:
        template = Template(file.read())

    # Render the poster content
    poster_content = template.render(
        title=title,
        summary_text=summary_text,
        plots=plots,
        tables=tables
    )

    # Save the poster content with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(poster_content)

    print(f"Poster created successfully as {output_file}.")

if __name__ == '__main__':
    pdf_path = "D:/Projects/Meta/23.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)

    summary_text, csv_tables = generate_summary_and_extract_tables(extracted_text)
    dataframes = convert_csv_to_dataframe(csv_tables)

    title = "Generated Research Poster"
    plots = create_plots(dataframes)
    tables = [table_to_svg(df) for df in dataframes]

    generate_poster("template0.html", "poster.html", title, summary_text, plots, tables)

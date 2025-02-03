import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re
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
                                  f" Provide concise key points (2-3 max per section) and suggest how to layout the information visually."
                                  f" Extract any tables with numerical data, including their context and title, while omitting non-numerical information."
                                  f" Focus on clarity and visual appeal:\n\n{chunk}"}],
            model="llama3-8b-8192",
        )
        content = response.choices[0].message.content
        summaries.append(content)

        # Extract tables using regex for numerical data
        tables = re.findall(r'(?:Table|Data|Results|Statistics).*?:\s*(.*?)(?:\n\s*\n|$)', content, re.DOTALL)
        extracted_tables.extend(
            [t for t in tables if not re.match(r'(None|No tables|no tables|No tabular)', t, re.IGNORECASE)])

    return " ".join(summaries), extracted_tables


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

        # Determine the appropriate plot type
        numeric_cols = df.select_dtypes(include='number').columns
        categorical_cols = df.select_dtypes(include='category').columns.tolist() + df.select_dtypes(
            include='object').columns.tolist()

        plt.figure(figsize=(8, 6))

        try:
            # Choose plot type based on data context
            if len(numeric_cols) == 1:
                sns.histplot(df[numeric_cols[0]], kde=True)  # Histogram for single numeric columns
            elif len(numeric_cols) > 1 and len(df) <= 1000:  # Limit pairplot to smaller datasets
                sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1])  # Scatterplot for numeric columns
            elif len(categorical_cols) > 0:
                sns.countplot(x=categorical_cols[0], data=df)  # Count plot for categorical columns
            else:
                sns.lineplot(data=df)  # Default to line plot if no clear choice
        except Exception as e:
            print(f"Error creating plot for DataFrame {i}: {e}")
            continue

        plot_filename = f"plot_{i}.png"
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to free memory
        plots.append(plot_filename)

    return plots


def table_to_svg(df):
    """Converts a DataFrame to an SVG image."""
    dwg = svgwrite.Drawing(size=(600, 400))
    x, y = 10, 20

    # Draw table header
    for j, col in enumerate(df.columns):
        dwg.add(dwg.text(col, insert=(x + j * 100, y), font_size='14px', font_weight='bold'))
    y += 20  # Move y position down for the next row

    # Draw table rows
    for i, row in df.iterrows():
        for j, cell in enumerate(row):
            dwg.add(dwg.text(str(cell), insert=(x + j * 100, y)))
        y += 20  # Move to next row

    sanitized_filename = sanitize_filename(f"table_{df.index[0]}.svg")
    dwg.saveas(sanitized_filename)
    return sanitized_filename


def create_poster_html(title, summary_text, plots, tables):
    """Renders HTML content for the poster using Jinja2."""
    with open('template2.html') as f:
        template = Template(f.read())
    return template.render(title=title, summary_text=summary_text, plots=plots, tables=tables)


def main():
    pdf_file_path = "D:/Projects/Meta/31.pdf"  # Adjust path as necessary
    text = extract_text_from_pdf(pdf_file_path)

    # Extract summary and tables
    summary, csv_tables = generate_summary_and_extract_tables(text)
    dataframes = convert_csv_to_dataframe(csv_tables)

    # Ensure there are DataFrames available
    if not dataframes:
        print("No valid dataframes found. Exiting.")
        return

    # Generate plots
    plots = create_plots(dataframes)

    # Generate tables as SVG
    tables = [table_to_svg(df) for df in dataframes if not df.empty]

    # Ensure that we have valid plots and tables generated
    if not plots and not tables:
        print("No plots or tables generated. Exiting.")
        return

    # Generate the poster HTML
    html_content = create_poster_html("Research Poster Example", summary, plots, tables)

    with open('poster2.html', 'w') as f:
        f.write(html_content)
    print("Poster created successfully as poster2.html.")


if __name__ == "__main__":
    main()

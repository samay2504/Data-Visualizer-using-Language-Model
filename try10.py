import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import re
import base64
import svgwrite
from pdfminer.high_level import extract_text
from jinja2 import Template
from groq import Groq
import spacy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import BayesianGaussianMixture

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load NLP model for context-aware extraction (SpaCy)
nlp = spacy.load("en_core_web_sm")

# Initialize Groq client
client = Groq(api_key="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")


def extract_text_from_pdf(pdf_file_path):
    """Extracts text from PDF using pdfminer."""
    return extract_text(pdf_file_path)


def chunk_text(text, max_length=1000):
    """Chunks text into manageable parts."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def extract_numerical_data_with_context(text):
    """
    Extracts numerical data along with their context (headings, units, etc.) using NLP.
    """
    doc = nlp(text)
    numerical_data = []

    for token in doc:
        if token.like_num:  # Identifies numerical values
            context = token.head.text  # Get the most relevant context word (usually the head of the numerical phrase)
            numerical_data.append({"value": token.text, "context": context})

    return numerical_data


def filter_meaningless_contexts(contexts):
    """Filters out meaningless or empty contexts."""
    meaningful_contexts = [context for context in contexts if len(context.split()) > 1]
    return meaningful_contexts


def cluster_data(numerical_data):
    """
    Clusters the numerical data using Bayesian Gaussian Mixture Models.
    """
    contexts = [data['context'] for data in numerical_data]

    # Remove duplicate or meaningless contexts
    contexts = list(set(filter_meaningless_contexts(contexts)))

    # Convert contexts into numerical vectors using TF-IDF
    if len(contexts) < 2:
        print("Not enough unique contexts for clustering.")
        return {}

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(contexts).toarray()

    # Use Bayesian Gaussian Mixture Model (BGMM) for clustering
    bgmm = BayesianGaussianMixture(n_components=5, random_state=42)
    bgmm.fit(X)
    labels = bgmm.predict(X)

    # Group numerical data by cluster and create tables
    tables = defaultdict(list)
    for i, data in enumerate(numerical_data):
        if i < len(labels):  # Ensure valid indexing
            cluster = labels[i]
            tables[cluster].append(data['value'])

    return tables


def create_table_from_clusters(tables):
    """
    Converts clustered numerical data into structured tables using Pandas.
    """
    table_dict = {}
    for cluster, values in tables.items():
        table_dict[f"Cluster {cluster}"] = pd.DataFrame(values, columns=["Numerical Values"])

    return table_dict


def generate_summary_and_extract_tables(text):
    """
    Generates concise summaries and extracts tables using LLM and contextual intelligence.
    """
    chunks = chunk_text(text)
    summaries = []
    extracted_tables = []

    for chunk in chunks:
        # LLM-based summary
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": f"Summarize the introduction of this input in not more than 2000 words:\n\n{chunk}"}],
            model="llama3-8b-8192"
        )

        content = response.choices[0].message.content
        summaries.append(content)

        # Apply contextual extraction to the current chunk
        numerical_data = extract_numerical_data_with_context(chunk)

        # Cluster the numerical data by context
        tables = cluster_data(numerical_data)

        # Create table from clustered numerical data
        table_data = create_table_from_clusters(tables)
        extracted_tables.append(table_data)

    # Split summary into bullet points
    summary_points = re.split(r'(?<=\.) ', " ".join(summaries))

    return summary_points, extracted_tables


def convert_csv_to_dataframe(extracted_tables):
    """Converts extracted numerical data into DataFrames."""
    dataframes = []
    for table_cluster in extracted_tables:
        for table in table_cluster.values():
            if not table.empty:
                dataframes.append(table)
    return dataframes


def create_plots(dataframes):
    """Generates a variety of plots based on the data in each DataFrame."""
    plots = []
    for i, df in enumerate(dataframes):
        if df.empty or df.select_dtypes(include='number').empty:
            continue

        numeric_cols = df.select_dtypes(include='number').columns

        plt.figure(figsize=(8, 6))

        try:
            if len(numeric_cols) == 1:
                sns.histplot(df[numeric_cols[0]], kde=True)
            elif len(numeric_cols) > 1 and len(df) <= 1000:
                sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
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

    summary_text, extracted_tables = generate_summary_and_extract_tables(extracted_text)
    dataframes = convert_csv_to_dataframe(extracted_tables)

    title = "Generated Research Poster"
    plots = create_plots(dataframes)
    tables = [table_to_svg(df) for df in dataframes]

    generate_poster("template1.html", "poster_23.html", title, summary_text, plots, tables)

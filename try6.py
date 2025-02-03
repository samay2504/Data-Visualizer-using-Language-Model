import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import BytesIO, StringIO
import pandas as pd

from groq import Groq
from pdfminer.high_level import extract_text

# Initialize Groq client
client = Groq(api_key="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")

# Function to extract text from a PDF using pdfminer
def extract_text_from_pdf(pdf_file_path):
    return extract_text(pdf_file_path)

# Function to chunk text
def chunk_text(text, max_length=1000):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# Generate summary and extract tables using Groq's LLaMA3-8B model
def generate_summary_and_extract_tables(text):
    chunks = chunk_text(text)
    summaries = []
    extracted_tables = []

    for chunk in chunks:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Please summarize the following text and identify any tables. "
                               f"For tables, provide them in CSV format:\n\n{chunk}",
                }
            ],
            model="llama3-8b-8192",
        )
        content = response.choices[0].message.content
        summaries.append(content)

        # Enhanced Table Extraction using regex
        tables = re.findall(r'(?:Table|Data).*?:\s*(.*?)(?:\n\s*\n|$)', content, re.DOTALL)
        # Filter out non-table data
        filtered_tables = [t for t in tables if
                           not re.match(r'(None|No tables|no tables|No tabular)', t, re.IGNORECASE)]
        extracted_tables.extend(filtered_tables)

    return " ".join(summaries), extracted_tables

# Convert extracted CSV tables to DataFrames
def convert_csv_to_dataframe(csv_tables):
    dataframes = []
    for csv_table in csv_tables:
        try:
            if csv_table.strip():
                # Replace multiple spaces or irregular separators with a comma
                cleaned_csv = re.sub(r'\s+', ',', csv_table.strip())
                data = pd.read_csv(StringIO(cleaned_csv), sep=',', on_bad_lines='skip')
                if not data.empty:
                    dataframes.append(data)
        except Exception as e:
            print(f"Error processing table: {e}")
    return dataframes

# Plotting functions
def create_bar_chart(df, output_path):
    plt.figure(figsize=(10, 7))
    sns.barplot(x=df.columns[0], y=df.columns[1], data=df, palette='viridis')
    plt.savefig(output_path)
    plt.close()

def create_pie_chart(df, output_path):
    plt.figure(figsize=(10, 7))
    plt.pie(df[df.columns[1]], labels=df[df.columns[0]], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.savefig(output_path)
    plt.close()

def create_line_chart(df, output_path):
    plt.figure(figsize=(10, 7))
    plt.plot(df[df.columns[0]], df[df.columns[1]], marker='o', linestyle='-', color='b')
    plt.savefig(output_path)
    plt.close()

def create_scatter_plot(df, output_path):
    plt.figure(figsize=(10, 7))
    plt.scatter(df[df.columns[0]], df[df.columns[1]], c='g', alpha=0.5)
    plt.savefig(output_path)
    plt.close()

def create_histogram(df, output_path):
    plt.figure(figsize=(10, 7))
    plt.hist(df[df.columns[1]], bins=10, color='purple', alpha=0.7)
    plt.savefig(output_path)
    plt.close()

# Layout Manager class to manage layout
class LayoutManager:
    def __init__(self, poster_width, margin=50):
        self.poster_width = poster_width
        self.margin = margin
        self.current_y = margin
        self.elements_height = 0

    def add_text_block(self, draw, text, font, color):
        lines = self._split_text(draw, text, font)
        for line in lines:
            draw.text((self.margin, self.current_y), line, font=font, fill=color)
            self.current_y += font.getbbox(line)[1] + 5
            self.elements_height += font.getbbox(line)[1] + 5

    def add_table(self, poster, df):
        table_image = df_to_image(df)
        poster.paste(table_image, (self.margin, self.current_y))
        self.current_y += table_image.height + 20
        self.elements_height += table_image.height + 20

    def add_plot(self, poster, plot_path):
        plot_image = Image.open(plot_path)
        poster.paste(plot_image, (self.margin, self.current_y))
        self.current_y += plot_image.height + 20
        self.elements_height += plot_image.height + 20

    def _split_text(self, draw, text, font):
        words = text.split(' ')
        lines = []
        line = ""
        for word in words:
            if draw.textbbox((0, 0), line + word, font=font)[2] <= self.poster_width - 2 * self.margin:
                line += word + " "
            else:
                lines.append(line)
                line = word + " "
        lines.append(line)
        return lines

    def calculate_total_height(self):
        return self.elements_height + 2 * self.margin

# Convert DataFrame to Image
def df_to_image(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return Image.open(buf)

# Function to create text boxes with dynamic font sizes
def add_text(draw, text, position, max_width, font_path, max_font_size=40):
    font_size = max_font_size
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)

    # Reduce font size until text fits within max_width
    while text_bbox[2] > max_width and font_size > 10:
        font_size -= 2
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)

    draw.text(position, text, font=font, fill="black")
    return draw

# Function to add a chart to the poster
def add_chart(img, chart_data, position, size):
    # Create a chart using matplotlib
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100))
    ax.plot(chart_data['x'], chart_data['y'])
    plt.tight_layout()

    # Save chart as an image and paste it onto the poster
    chart_path = 'chart_temp.png'
    fig.savefig(chart_path)
    chart_img = Image.open(chart_path)
    img.paste(chart_img, position)

    os.remove(chart_path)  # Clean up the chart image file
    return img

# Function to create the poster
def create_poster(title, summary_text, charts, tables, output_path):
    # Create a blank poster with white background
    poster_width = 1200
    poster_height = 1600
    img = Image.new('RGB', (poster_width, poster_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Add title
    title_position = (100, 50)
    draw = add_text(draw, title, title_position, max_width=img.width - 200, font_path="arial.ttf", max_font_size=60)

    # Add summary text
    summary_position = (100, 150)
    draw = add_text(draw, summary_text, summary_position, max_width=img.width - 200, font_path="arial.ttf",
                    max_font_size=40)

    # Add charts
    chart_start_position = (100, 300)
    chart_size = (400, 300)
    for i, chart_data in enumerate(charts):
        img = add_chart(img, chart_data, (chart_start_position[0], chart_start_position[1] + i * (chart_size[1] + 50)),
                        chart_size)

    # Add tables
    layout_manager = LayoutManager(img.width)
    layout_manager.current_y = chart_start_position[1] + len(charts) * (chart_size[1] + 50) + 50
    for df in tables:
        if not df.empty:
            layout_manager.add_table(img, df)
            layout_manager.current_y += 50  # Add space after table

    # Save the final poster
    img.save(output_path)
    print(f"Poster saved to {output_path}")

# Example usage
def main():
    pdf_file_path = "D:/Projects/Meta/23.pdf"
    output_path = "D:/Projects/Meta/output_poster_final.png"

    text = extract_text_from_pdf(pdf_file_path)
    summary, csv_tables = generate_summary_and_extract_tables(text)
    dataframes = convert_csv_to_dataframe(csv_tables)

    # Example chart data
    charts = [
        {'x': [1, 2, 3, 4, 5], 'y': [10, 20, 15, 25, 30]},
        {'x': [1, 2, 3, 4, 5], 'y': [5, 15, 10, 20, 25]},
    ]

    create_poster("Research Poster Title", summary, charts, dataframes, output_path)

if __name__ == "__main__":
    main()

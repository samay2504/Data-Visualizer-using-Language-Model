from pdfminer.high_level import extract_text
import re
from PIL import Image, ImageDraw, ImageFont
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from groq import Groq

# Initialize Groq client
client = Groq(
    api_key="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",
)

# Function to chunk text
def chunk_text(text, max_length=1000):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# Generate summary using Groq's LLaMA3-8B model
def generate_summary_llm(text):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Please summarize the following text:\n\n{chunk}",
                }
            ],
            model="llama3-8b-8192",
        )
        summaries.append(response.choices[0].message.content)

    return " ".join(summaries)


# Extract text from PDF using pdfminer
def extract_text_from_pdf(file_path):
    return extract_text(file_path)


# Extract bullet points from text
def extract_bullet_points(text):
    lines = text.split('\n')
    bullet_points = [line.strip() for line in lines if line.strip().startswith('-')]
    return bullet_points[:5]


# Extract numerical data for plotting
def extract_data_for_plotting(text):
    numbers = re.findall(r'\b\d+\b', text)
    numbers = list(map(int, numbers))
    return numbers[:10]


# Create customized Seaborn plot
def create_seaborn_plot(data, output_path):
    df = pd.DataFrame({'Values': data})
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.histplot(df['Values'], kde=True, color='skyblue', bins=8)
    plt.title("Histogram of Extracted Values")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Helper function to draw text with word wrapping
def draw_text(draw, text, position, font, max_width, fill):
    words = text.split()
    lines = []
    line = []

    for word in words:
        line_with_word = ' '.join(line + [word])
        width, _ = draw.textbbox((0, 0), line_with_word, font=font)[2:]
        if width <= max_width:
            line.append(word)
        else:
            lines.append(' '.join(line))
            line = [word]
    lines.append(' '.join(line))

    y = position[1]
    for line in lines:
        draw.text((position[0], y), line, font=font, fill=fill)
        _, height = draw.textbbox((0, 0), line, font=font)[2:]
        y += height

    return y


# Function to create the poster
def create_poster(text, bullet_points, charts, output_path):
    poster_width = 1200
    poster_height = 1600
    poster = Image.new('RGB', (poster_width, poster_height), 'white')
    draw = ImageDraw.Draw(poster)

    fonts = ["C:/Windows/Fonts/Arial.ttf", "C:/Windows/Fonts/Calibri.ttf"]
    for font in fonts:
        if not os.path.exists(font):
            print(f"Font file not found: {font}")
            fonts.remove(font)

    if not fonts:
        print("No font files found. Exiting.")
        return

    title_font_size = 50  # Consistent font sizes for better readability
    body_font_size = 30
    title_font = ImageFont.truetype(random.choice(fonts), title_font_size)
    body_font = ImageFont.truetype(random.choice(fonts), body_font_size)

    title_color = 'darkred'
    text_color = 'black'

    title = "Research Summary Poster"
    draw.text((20, 20), title, font=title_font, fill=title_color)

    text_max_width = poster_width - 40
    y_offset = draw_text(draw, text, (20, 100), body_font, text_max_width, text_color)

    y_offset += 40
    bullet_point_title = "Key Points:"
    draw.text((20, y_offset), bullet_point_title, font=title_font, fill=title_color)
    y_offset += title_font_size + 10

    for point in bullet_points:
        y_offset = draw_text(draw, f"- {point}", (40, y_offset), body_font, text_max_width - 40, text_color)
        y_offset += 10

    y_offset += 40
    for chart_path in charts:
        chart = Image.open(chart_path)
        chart.thumbnail((poster_width - 40, 400))
        poster.paste(chart, (20, y_offset))
        y_offset += chart.size[1] + 30

    poster.save(output_path)


def main():
    # Ask the user for the PDF file path
    pdf_file_path = input("Enter the path to your PDF file: ")

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_file_path)

    # Summarize the text
    summary = generate_summary_llm(text)

    # Extract bullet points from the text
    bullet_points = extract_bullet_points(text)

    # Extract data and generate plots
    data_for_plot1 = extract_data_for_plotting(text)
    chart_path1 = 'seaborn_plot1.png'
    create_seaborn_plot(data_for_plot1, chart_path1)

    data_for_plot2 = extract_data_for_plotting(text)
    chart_path2 = 'seaborn_plot2.png'
    create_seaborn_plot(data_for_plot2, chart_path2)

    # Create the poster
    charts = [chart_path1, chart_path2]
    output_path = 'research_poster3.png'
    create_poster(summary, bullet_points, charts, output_path)
    print(f"Poster created successfully! Saved at {output_path}")


if __name__ == "__main__":
    main()

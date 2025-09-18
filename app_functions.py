import http.client
import json
import os
from dotenv import load_dotenv
import re
import aiohttp
import fitz  # PyMuPDF
import asyncio
import markdownify
import readabilipy.simple_json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any
import asyncio
from agents.extensions.models.litellm_model import LitellmModel
from agents import OpenAIChatCompletionsModel, set_default_openai_client, set_tracing_disabled, set_default_openai_api, Agent, Runner, function_tool, ItemHelpers
from google import genai
import litellm
import logging
import math
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
import numpy as np
import mplfinance as mpf
import io

load_dotenv()

serp_api_key = os.getenv("SERP_API_KEY")
model_name = os.getenv("model_name")
gemini_api_key =os.getenv("GOOGLE_GEMINI_KEY")


def fetch_google_news_serpapi(query):
  conn = http.client.HTTPSConnection("google.serper.dev")
  payload = json.dumps({
    "q": query,
    "num": 5,
    "gl": "us",
    "hl": "en"
  })
  headers = {
    'X-API-KEY': serp_api_key,
    'Content-Type': 'application/json'
  }
  conn.request("POST", "/search", payload, headers)
  res = conn.getresponse()
  data = res.read()
  results = json.loads(data.decode("utf-8"))
  results = results.get("organic")

  final_results = []
  for web_result in results:
    result = {
        "title": web_result.get("title"),
        "snippet": web_result.get("snippet"),
        "link": web_result.get("link")
    }
    final_results.append(result)
  return final_results


def clean_text_basic(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _normalize_and_clean_url(href: str, base_url: str | None = None) -> str:
    if not href:
        return ""
    href = href.strip()
    # remove surrounding punctuation
    href = href.strip(" \n\t'\"<>.,;:()[]{}")
    # handle protocol-relative URLs
    if href.startswith("//"):
        href = "https:" + href
    # resolve relative URLs if base provided
    try:
        if base_url and not bool(urlparse(href).netloc):
            href = urljoin(base_url, href)
    except Exception:
        pass
    return href


async def _extract_content_with_inline_links(html: str, base_url: str | None = None) -> str:
    """
    Extracts readable content from HTML and ensures all anchor texts are followed
    by their link in parentheses, like: "Some text (https://example.com)".
    """
    soup = BeautifulSoup(html, "html.parser")

    # Replace <a> tags with "text (url)" format
    for a in soup.find_all("a"):
        href = _normalize_and_clean_url(a.get("href"), base_url)
        if href:
            # append link in parentheses if anchor has text
            if a.text.strip():
                a.replace_with(f"{a.text.strip()} ({href})")
            else:
                a.replace_with(href)

    # Extract text content with readability + markdownify fallback
    def sync_html():
        try:
            ret = readabilipy.simple_json.simple_json_from_html_string(
                str(soup), use_readability=True
            )
            if not ret.get("content"):
                return "<error>Failed to simplify HTML content</error>"
            return markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        except Exception:
            return soup.get_text(" ")

    return await asyncio.to_thread(sync_html)


async def extract_text_from_url(url: str) -> Dict[str, Any]:
    """
    Fetch URL and return a dict:
      {
        "text": "<cleaned markdown with inline links>"
      }
    Falls back to {"text": ""} on any error.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                content_type = resp.headers.get("Content-Type", "") or ""
                content_bytes = await resp.read()

        # PDF handling — we can only inline plain-text links
        if url.lower().endswith(".pdf") or "application/pdf" in content_type.lower():
            try:
                texts = []
                for page in fitz.open(stream=content_bytes, filetype="pdf"):
                    texts.append(page.get_text())
                raw_text = "\n".join(texts)
                clean_text = clean_text_basic(raw_text)
                # Inline links: replace with "(link)" next to them if found
                found_links = re.findall(r"https?://[^\s\"'<>)+,;]+", raw_text)
                for link in found_links:
                    clean_text = clean_text.replace(link, f"({link})")
                return clean_text
            except Exception as e:
                return None

        # Non-PDF: decode to string
        try:
            text = content_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = content_bytes.decode(errors="ignore")

        # If looks like HTML, process with readability and inline anchor links
        if ("<html" in text[:200].lower()) or ("text/html" in content_type.lower()) or not content_type:
            try:
                cleaned_markdown = await _extract_content_with_inline_links(text, base_url=url)
            except Exception:
                cleaned_markdown = ""
            return clean_text_basic(cleaned_markdown)

        # Fallback: treat raw response as plain text, inline any URLs
        cleaned = clean_text_basic(text)
        found_links = re.findall(r"https?://[^\s\"'<>)+,;]+", text)
        for link in found_links:
            cleaned = cleaned.replace(link, f"({link})")
        return cleaned

    except Exception as e:
        print(e)
        return None


@function_tool
async def web_search_tool(query):
    web_results = fetch_google_news_serpapi(query=query)
    if not web_results:
        return []

    async def fetch_content(result):
        link = result.get("link")
        result['content'] = await extract_text_from_url(link)
        return result

    # Run all fetches concurrently
    web_results_with_content = await asyncio.gather(*(fetch_content(r) for r in web_results))

    return web_results_with_content

# =============================
# Dummy Functions with Detailed Docstrings
# =============================
@function_tool
def plot_bar_chart(categories, values, title="Bar Chart", xlabel="Categories", ylabel="Values", color="skyblue"):
    """
    Create a vertical bar chart.

    Parameters:
    - categories (list[str]): Categories to display on the x-axis.
    - values (list[int|float]): Corresponding numerical values for each category.
    - title (str): Chart title.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.
    - color (str, optional): Color for bars. If not provided, default color is skyblue which is perfect

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_stacked_bar_chart(categories, data, title="Stacked Bar Chart", xlabel="Categories", ylabel="Values"):
    """
    Create a stacked bar chart.

    Parameters:
    - categories (list[str]): Categories on the x-axis.
    - data (dict): Dictionary where keys are subgroups and values are lists of numbers for each category.
    - title (str): Chart title.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool 
def plot_pie_chart(labels, values, title="Pie Chart"):
    """
    Create a pie chart.

    Parameters:
    - labels (list[str]): Categories represented in the pie.
    - values (list[int|float]): Numerical values for each category.
    - title (str): Chart title.
    - autopct (str): Percentage label format.
    - startangle (int): Rotation for the start of the pie chart.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_line_chart(x, y, title="Line Chart", xlabel="X", ylabel="Y", color="blue"):
    """
    Create a line chart.

    Parameters:
    - x (list[str|datetime]): X-axis values.
    - y (list[int|float]): Y-axis values.
    - title (str): Chart title.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.
    - color (str): Line color.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_area_chart(x, y, title="Area Chart", xlabel="X", ylabel="Y", color="lightblue"):
    """
    Create an area chart.

    Parameters:
    - x (list[str|datetime]): X-axis values.
    - y (list[int|float]): Y-axis values.
    - title (str): Chart title.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.
    - color (str): Fill color.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_histogram(values, bins=10, title="Histogram", xlabel="Value", ylabel="Frequency", color="purple"):
    """
    Create a histogram.

    Parameters:
    - values (list[int|float]): Data points to distribute into bins.
    - bins (int): Number of bins.
    - title (str): Chart title.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.
    - color (str): Bar color.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_scatter_chart(x, y, title="Scatter Plot", xlabel="X", ylabel="Y", color="green"):
    """
    Create a scatter plot.

    Parameters:
    - x (list[int|float]): X-axis values.
    - y (list[int|float]): Y-axis values.
    - title (str): Chart title.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.
    - color (str): Marker color.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_box_chart(values, title="Box Plot", ylabel="Values"):
    """
    Create a box plot.

    Parameters:
    - values (list[int|float]): Data values to summarize distribution.
    - title (str): Chart title.
    - ylabel (str): Label for y-axis.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_word_cloud(text, title="Word Cloud", max_words=100):
    """
    Generate a word cloud visualization.

    Parameters:
    - text (str): Input text to generate cloud.
    - title (str): Chart title.
    - max_words (int): Maximum number of words to display.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_heatmap(data, labels, title="Heatmap", cmap="Blues"):
    """
    Create a heatmap.

    Parameters:
    - data (list[list[float]]): 2D matrix of values.
    - labels (list[str]): Axis labels for rows and columns.
    - title (str): Chart title.
    - cmap (str): Color map scheme.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_candlestick_chart(dates, open_prices, high_prices, low_prices, close_prices, title="Candlestick Chart"):
    """
    Create a candlestick chart for stock market data.

    Parameters:
    - dates (list[str|datetime]): Dates for x-axis.
    - open_prices (list[float]): Opening prices.
    - high_prices (list[float]): Highest prices.
    - low_prices (list[float]): Lowest prices.
    - close_prices (list[float]): Closing prices.
    - title (str): Chart title.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_moving_average_chart(dates, values, window=7, title="Moving Average Chart", xlabel="Date", ylabel="Value"):
    """
    Create a moving average line chart.

    Parameters:
    - dates (list[str|datetime]): Dates for x-axis.
    - values (list[float]): Values to calculate moving average.
    - window (int): Rolling window size.
    - title (str): Chart title.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_stacked_area_chart(x, data, title="Stacked Area Chart", xlabel="X", ylabel="Value"):
    """
    Create a stacked area chart.

    Parameters:
    - x (list[str|datetime]): X-axis values.
    - data (dict): Dictionary of series to stack, with labels as keys and lists of values.
    - title (str): Chart title.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"
@function_tool
def plot_bubble_chart(x, y, sizes, title="Bubble Chart", xlabel="X", ylabel="Y", color="blue"):
    """
    Create a bubble chart.

    Parameters:
    - x (list[int|float]): X-axis values.
    - y (list[int|float]): Y-axis values.
    - sizes (list[int|float]): Sizes of bubbles.
    - title (str): Chart title.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.
    - color (str): Bubble color.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"

  
@function_tool
def plot_donut_chart(labels, values, title="Donut Chart"):
    """
    Create a donut chart.

    Parameters:
    - labels (list[str]): Categories represented in the donut chart.
    - values (list[int|float]): Numerical values for each category.
    - title (str): Chart title.

    Returns:
    - str: Dummy return "plot successful"
    """
    return "plot successful"

def plot_wordcloud(words):
    wc = WordCloud(width=600, height=400, background_color="white").generate(words)
    
    buf = io.BytesIO()
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    
    return buf


def plot_figure(plot_funcs):
  # Create a single figure with subplots
    n_plots = len(plot_funcs)
    if n_plots == 0:
        print("No plots generated by the agent.")
        return

    ncols = 2
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows))

    # axes might be 2D or 1D depending on nrows/ncols
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, (func, args) in zip(axes, plot_funcs):
      if isinstance(args, str):
        args = json.loads(args)

      args['ax'] = ax  # pass the subplot axes
      func(**args)

    # Remove empty subplots if any
    for i in range(len(plot_funcs), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


# =============================
# 1. Bar Chart
# =============================
def plot_bar_chart_main(categories: list, values: list, title: str = "Bar Chart", xlabel: str = "Category", ylabel: str = "Value", color: str = "skyblue", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.bar(categories, values, color=color)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

# =============================
# 2. Stacked Bar Chart
# =============================
def plot_stacked_bar_chart_main(categories: list, data: dict, title: str = "Stacked Bar Chart", xlabel: str = "Category", ylabel: str = "Value", ax=None):
    df = pd.DataFrame(data, index=categories)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    df.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

# =============================
# 3. Pie Chart
# =============================
def plot_pie_chart_main(labels: list, values: list, title: str = "Pie Chart", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig

# =============================
# 4. Line Chart
# =============================
def plot_line_chart_main(x: list, y: list, title: str = "Line Chart", xlabel: str = "X", ylabel: str = "Y", color: str = "blue", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig

# =============================
# 5. Area Chart
# =============================
def plot_area_chart_main(x: list, y: list, title: str = "Area Chart", xlabel: str = "X", ylabel: str = "Y", color: str = "lightblue", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.fill_between(x, y, color=color, alpha=0.6)
    ax.plot(x, y, color="blue")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

# =============================
# 6. Histogram
# =============================
def plot_histogram_main(values: list, bins: int = 10, title: str = "Histogram", xlabel: str = "Value", ylabel: str = "Frequency", color: str = "purple", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.hist(values, bins=bins, color=color, edgecolor="black", alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

# =============================
# 7. Scatter Plot
# =============================
def plot_scatter_chart_main(x: list, y: list, title: str = "Scatter Plot", xlabel: str = "X", ylabel: str = "Y", color: str = "green", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.scatter(x, y, color=color, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

# =============================
# 8. Box Plot
# =============================
def plot_box_chart_main(values: list, title: str = "Box Plot", ylabel: str = "Values", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.boxplot(values)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    return fig

# =============================
# 9. Word Cloud
# =============================
def plot_word_cloud_main(text: str, title: str = "Word Cloud", max_words: int = 100, ax=None):
    wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color="white").generate(text)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig

# =============================
# 10. Heatmap
# =============================
def plot_heatmap_main(data: list, labels: list, title: str = "Heatmap", cmap: str = "Blues", ax=None):
    df = pd.DataFrame(data, columns=labels, index=labels)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, cbar=True, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig

# =============================
# 11. Candlestick Chart
# =============================
def plot_candlestick_chart_main(dates: list, open_prices: list, high_prices: list, low_prices: list, close_prices: list, title: str = "Candlestick Chart", ax=None):
    data = pd.DataFrame({
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices
    }, index=pd.to_datetime(dates))
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    mpf.plot(data, type="candle", ax=ax, style="yahoo", show_nontrading=False)
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig

# =============================
# 12. Moving Average Chart
# =============================
def plot_moving_average_chart_main(dates: list, values: list, window: int = 7, title: str = "Moving Average Chart", xlabel: str = "Date", ylabel: str = "Value", ax=None):
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Value": values})
    df["MA"] = df["Value"].rolling(window=window).mean()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(df["Date"], df["Value"], label="Raw", color="blue", alpha=0.6)
    ax.plot(df["Date"], df["MA"], label=f"{window}-period MA", color="red", linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig

# =============================
# 13. Stacked Area Chart
# =============================
def plot_stacked_area_chart_main(x: list, data: dict, title: str = "Stacked Area Chart", xlabel: str = "X", ylabel: str = "Value", ax=None):
    df = pd.DataFrame(data, index=x)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.stackplot(df.index, df.T, labels=df.columns, alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    return fig

# =============================
# 14. Bubble Chart
# =============================
def plot_bubble_chart_main(x: list, y: list, sizes: list, title: str = "Bubble Chart", xlabel: str = "X", ylabel: str = "Y", color: str = "blue", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    scatter = ax.scatter(x, y, s=sizes, alpha=0.5, c=color)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

# =============================
# 15. Donut Chart
# =============================
def plot_donut_chart_main(labels: list, values: list, title: str = "Donut Chart", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.4))
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig


function_tool_map = {
    "plot_bar_chart": plot_bar_chart_main,
    "plot_stacked_bar_chart": plot_stacked_bar_chart_main,
    "plot_pie_chart": plot_pie_chart_main,
    "plot_line_chart": plot_line_chart_main,
    "plot_area_chart": plot_area_chart_main,
    "plot_histogram": plot_histogram_main,
    "plot_scatter_chart": plot_scatter_chart_main,
    "plot_box_chart": plot_box_chart_main,
    "plot_word_cloud": plot_word_cloud_main,
    "plot_heatmap": plot_heatmap_main,
    "plot_candlestick_chart": plot_candlestick_chart_main,
    "plot_moving_average_chart": plot_moving_average_chart_main,
    "plot_stacked_area_chart": plot_stacked_area_chart_main,
    "plot_bubble_chart": plot_bubble_chart_main,
    "plot_donut_chart": plot_donut_chart_main
}


class ProductResearcher():
  def __init__(self):
    self.tools = [web_search_tool, plot_pie_chart, plot_word_cloud, plot_line_chart, plot_bar_chart]
    self.agent = Agent(
        instructions="""
            You are a **Financial & Business Research Analyst AI**. Your role is to behave like a professional data analyst hired to provide actionable, data-driven insights. 
            Your workflow should follow these steps:

            1. **Web Research**  
              - Always begin with the `web_search_tool` to gather reliable, structured, and measurable data from the internet.  
              - Focus on sources that include financial indicators, market trends, business performance metrics, survey results, ratings, or other quantitative insights.  

            2. **Visualization & Data Analysis**  
              - After collecting data, you must use **at least five different visualization tools** to explain different aspects of your findings.  
              - Select the **most appropriate visualization tool** for each dataset. For example:
                  • Use line charts for time-series trends.  
                  • Use pie or donut charts for proportions.  
                  • Use bar or stacked bar charts for categorical comparisons.  
                  • Use heatmaps, scatter, or bubble charts for correlation patterns.  
                  • Use candlestick or moving averages for stock/financial markets.  
                  • Use word clouds for qualitative insights from text.  
              - Each visualization must reveal a unique perspective of the research — avoid redundancy.  

            3. **Insight Generation & Reporting**  
              - Present your results as if you are writing a **business analysis report**.  
              - For each visualization, provide a clear explanation of what it shows and why it matters in the context of the query.  
              - Correlate the plots with your narrative, highlighting trends, risks, opportunities, and key takeaways.  
              - Your explanations should sound like a financial consultant briefing executives: concise, professional, and actionable.  

            **Persona Notes**  
            - Be precise, analytical, and professional.  
            - Prioritize clarity and insights backed by data.  
            - Your goal is not just to show charts, but to deliver a **story with evidence** that supports better business decisions.  

            **Rules for Tool Usage**  
            - You must always call `web_search_tool` first to collect data.  
            - You must always call **at least five different visualization tools**.  
            - You must always choose the **best tool for the data type**, not randomly.  
            - Your final output must combine **executive-style commentary** with the generated visualizations.  
            """,
            name="ProductResearchAnalyst",
            model=LitellmModel(model_name, api_key=gemini_api_key),
            tools=self.tools
          )

  async def run(self, user_input):
    result = Runner.run_streamed(self.agent, user_input, max_turns=30)
    plot_funcs = []

    print("=== Run starting ===")
    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        print(event.type)
        if event.type == "raw_response_event":
            continue
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        elif event.type == "run_item_stream_event":
            print(f"-- Tool was called")
            if event.item.type == "tool_call_item":
              tool_args = event.item.raw_item.arguments
              tool_name = event.item.raw_item.name
              print(f"-- {tool_name} Tool was called")
              if tool_name != "web_search_tool":
                tool_args = event.item.raw_item.arguments  # dict from the agent
                tool_func = function_tool_map.get(tool_name)
                if tool_func:
                    # Dynamically call the function with the arguments
                    plot_funcs.append((tool_func, tool_args))
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                pass  # Ignore other event types

    print("=== Run complete ===")

    return result, plot_funcs

import base64
from openai import OpenAI
from io import BytesIO
from PIL import Image
import streamlit as st
import pandas as pd
import ast
import time
import yfinance as yf
import matplotlib.pyplot as plt
import os  # Add this import

image_path = "combined_pie_charts.jpg"
if os.path.exists(image_path):
    try:
        os.remove(image_path)
    except Exception as e:
        st.warning(f"⚠️ Could not remove the file '{image_path}': {e}")

# OpenAI setup: client

st.set_page_config(page_title="Modular MVP", layout="wide")
st.title("Modular MVP")

st.subheader(f"1. Portfolio Screenshot Upload")

# Upload images
uploaded_files = st.file_uploader("Upload portfolio screenshots", type=["jpg", "png"], accept_multiple_files=True)

# Helper to encode image
def encode_image(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Store all parsed holdings across images
all_holdings = []

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file)

        base64_img = encode_image(img)

        st.write(f"- Extracting portfolio data from {file.name} ...")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": '''Please extract a clean table of holdings in the format: Holding Name | Amount. Return the output as a Python list of dicts, like: [{{"holdingName": "BAYER AG NAMENS-AKTIEN O.N.", "amount": 6.846,88}}, ...]'''},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }},
                        ],
                    }
                ],
                max_tokens=1000,
            )

            result = response.choices[0].message.content.strip()

            time.sleep(2)

            # Extract list from LLM response
            list_str = result[result.index("["):result.rindex("]")+1]
            holdings = ast.literal_eval(list_str)

            # Add to total
            all_holdings.extend(holdings)

        except Exception as e:
            st.error(f"❌ Could not process {file.name}")
            st.exception(e)


# ...existing code...

# Initialize an empty DataFrame for df
df = pd.DataFrame()

# Show combined table
portfolioExtractComplete = False
if all_holdings:  # Ensure there are holdings to process
    df = pd.DataFrame(all_holdings)

    if "amount" in df.columns:  # Check if 'amount' column exists
        # Clean amount field safely
        def clean_amount(val):
            if isinstance(val, str):
                val = val.replace(".", "")       # remove thousand separators
                val = val.replace(",", ".")      # replace decimal comma with dot
                val = ''.join(c for c in val if c.isdigit() or c == '.')  # remove symbols
            try:
                return float(val)
            except:
                return None

        # Apply cleaning
        df["clean_amount"] = df["amount"].apply(clean_amount)

        # Drop rows where amount is completely invalid
        df = df.dropna(subset=["clean_amount"])

        # Keep the first valid row per holdingName
        df = df.sort_values("clean_amount", ascending=False)  # put valid ones first
        df = df.drop_duplicates(subset="holdingName", keep="first")

        # Replace old amount with clean one, round it
        df["amount"] = df["clean_amount"].round(2)
        df = df.drop(columns=["clean_amount"])

    else:
        st.warning("⚠️ No 'amount' column found in the extracted holdings.")
        df = pd.DataFrame()  # Create an empty DataFrame

    st.subheader("2. Extracted Holdings")
    st.dataframe(df, use_container_width=True)
    portfolioExtractComplete = True


if portfolioExtractComplete: 
    st.subheader("🧠 Enriching Holdings with Ticker, Sector & Country")

valid_rows = []

# Ensure df is not empty before iterating
if not df.empty:
    progress = st.progress(0)
    status_text = st.empty()

    for i, row in df.iterrows():
        holding_name = row["holdingName"]

        prompt = f"""
        What is the stock ticker symbol for the following holding name?
        Only respond with a valid ticker symbol (e.g. AAPL), or say "UNKNOWN" if not identifiable.

        Holding: "{holding_name}"
        """

        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            ticker = res.choices[0].message.content.strip().upper()

            if "UNKNOWN" in ticker or len(ticker) > 10:
                continue

            try:
                stock = yf.Ticker(ticker)
                time.sleep(2)  # wait before the next call
                info = stock.info
                sector = info.get("sector", "Unknown")
                country = info.get("country", "Unknown")
            except Exception as e:
                st.warning(f"⚠️ Failed to fetch data for '{ticker}': {e}")
                continue

            sector = info.get("sector", "Unknown")
            country = info.get("country", "Unknown")

            if sector == "Unknown" or country == "Unknown":
                continue

            enriched = {
                "holdingName": holding_name,
                "amount": row["amount"],
                "ticker": ticker,
                "sector": sector,
                "country": country
            }

            valid_rows.append(enriched)

        except Exception as e:
            st.warning(f"⚠️ Skipped '{holding_name}': {e}")
            continue

portfolioBreakDownComplete = False
if valid_rows:
    enriched_df = pd.DataFrame(valid_rows)
    st.dataframe(enriched_df, use_container_width=True)

    # Calculate weights
    total_value = enriched_df['amount'].sum()
    enriched_df['weight'] = enriched_df['amount'] / total_value

    # Sector breakdown
    sector_breakdown = (
        enriched_df.groupby('sector')['weight']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Country breakdown
    geo_breakdown = (
        enriched_df.groupby('country')['weight']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Plot pie charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].pie(sector_breakdown['weight'], labels=sector_breakdown['sector'], autopct='%1.1f%%')
    axes[0].set_title("Sector Breakdown by Weight")

    axes[1].pie(geo_breakdown['weight'], labels=geo_breakdown['country'], autopct='%1.1f%%')
    axes[1].set_title("Geographical Breakdown by Weight")

    plt.tight_layout()
    plt.savefig("combined_pie_charts.jpg", format="jpg", dpi=300, bbox_inches='tight')

    # Show in Streamlit
    st.subheader("📈 Portfolio Allocation Breakdown")
    st.pyplot(fig)

    portfolioBreakDownComplete = True

# IMAGE ANALYSIS SECTION

# 1. Define the encoding function
def encode_image(image_path):
    if not os.path.exists(image_path):
        st.warning(f"⚠️ The file '{image_path}' does not exist. Skipping image analysis.")
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 2. Encode and send the pie chart to OpenAI Vision
if portfolioBreakDownComplete:
    st.subheader("🧠 Portfolio Analysis (LLM-Generated Insight)")

# Check if the image file exists before proceeding
if os.path.exists("combined_pie_charts.jpg"):
    try:
        base64_image = encode_image("combined_pie_charts.jpg")
        if base64_image:  # Proceed only if the file was successfully encoded
            with st.spinner("🧠 LLM is generating response ..."):
                vision_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "This is an image of two pie charts that show a sector and geography allocation "
                                        "of a portfolio. Give a human-readable summary of the distribution and its implications "
                                        "for diversification."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                            ],
                        }
                    ],
                    max_tokens=500,
                )

            analysis_text = vision_response.choices[0].message.content.strip()
            st.markdown(f"📋 **Summary:**\n\n{analysis_text}")

            with st.spinner("🤖 Evaluating diversification level..."):
                try:
                    classification_prompt = f"""
                    Based on the following analysis of a portfolio's sector and geography allocation, rate the portfolio's diversification as either 'Good' or 'Bad'. Be strict in your assessment.

                    Analysis:
                    {analysis_text}

                    Respond only with 'Good' or 'Bad'.
                    """

                    judgment_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": classification_prompt}],
                        max_tokens=5
                    )

                    verdict = judgment_response.choices[0].message.content.strip().lower()
                    st.write(f"🔍 LLM Diversification Judgment: **{verdict.capitalize()}**")

                    if verdict == "bad":
                        st.warning("⚠️ Diversification may be problematic. We recommend speaking to an advisor.")
                        if st.button("📞 Call Advisor"):
                            st.success("✅ Advisor has been notified.")
                except Exception as e:
                    st.error("Could not evaluate diversification.")
                    st.exception(e)
        else:
            st.warning("⚠️ Skipping LLM-based analysis due to missing image.")
    except Exception as e:
        st.warning("⚠️ Could not generate LLM-based analysis.")
        st.exception(e)
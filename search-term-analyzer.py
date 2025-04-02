import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def google_ads_search_term_analyzer():
    st.header("Google Ads Search Term Analyzer")
    st.markdown(
        """
        Upload an Excel file (.xlsx) from your Google Ads search terms report and analyze it.
        This tool extracts n-grams which can be used to optimize your campaigns.
        Your paid search data can also inform your SEO content strategy if you have a big enough sample size.
        """
    )

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            # Read the Excel file, skipping the first two rows.
            df = pd.read_excel(uploaded_file, skiprows=2)

            # Check for required columns.
            required_columns = ["Search term", "Clicks", "Impressions", "Cost", "Conversions"]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"The following required columns are missing: {', '.join(missing_cols)}")
                return

            # Convert numeric columns.
            for col in ["Clicks", "Impressions", "Cost", "Conversions"]:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                except KeyError:
                    st.error(f"Column '{col}' not found in the uploaded Excel file.")
                    return

            st.subheader("N-gram Analysis")
            # Let the user choose the extraction method and parameters.
            extraction_method = st.radio("Select N-gram Extraction Method:", options=["Contiguous n-grams", "Skip-grams"], index=0)
            n_value = st.selectbox("Select N (number of words in phrase):", options=[1, 2, 3, 4], index=1)
            min_frequency = st.number_input("Minimum Frequency:", value=2, min_value=1)

            # Define extraction functions.
            def extract_ngrams(text, n):
                text = str(text).lower()
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
                ngrams_list = list(nltk.ngrams(tokens, n))
                return [" ".join(gram) for gram in ngrams_list]

            def extract_skipgrams(text, n):
                import itertools
                text = str(text).lower()
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
                if len(tokens) < n:
                    return []
                skipgrams_list = []
                for combo in itertools.combinations(range(len(tokens)), n):
                    skipgram = " ".join(tokens[i] for i in combo)
                    skipgrams_list.append(skipgram)
                return skipgrams_list

            # Initialize stopwords and lemmatizer.
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            # Extract n-grams or skip-grams from each search term.
            all_ngrams = []
            for term in df["Search term"]:
                if extraction_method == "Contiguous n-grams":
                    all_ngrams.extend(extract_ngrams(term, n_value))
                else:
                    all_ngrams.extend(extract_skipgrams(term, n_value))

            ngram_counts = Counter(all_ngrams)
            filtered_ngrams = {ngram: count for ngram, count in ngram_counts.items() if count >= min_frequency}

            if not filtered_ngrams:
                st.warning("No n-grams found with the specified minimum frequency.")
                return

            # Map each search term to its n-grams.
            search_term_to_ngrams = {}
            for term in df["Search term"]:
                if extraction_method == "Contiguous n-grams":
                    search_term_to_ngrams[term] = extract_ngrams(term, n_value)
                else:
                    search_term_to_ngrams[term] = extract_skipgrams(term, n_value)

            # Calculate performance metrics per n-gram.
            ngram_performance = {}
            for index, row in df.iterrows():
                search_term_text = row["Search term"]
                for ngram in search_term_to_ngrams[search_term_text]:
                    if ngram in filtered_ngrams:
                        if ngram not in ngram_performance:
                            ngram_performance[ngram] = {
                                "Clicks": 0,
                                "Impressions": 0,
                                "Cost": 0,
                                "Conversions": 0
                            }
                        ngram_performance[ngram]["Clicks"] += row["Clicks"]
                        ngram_performance[ngram]["Impressions"] += row["Impressions"]
                        ngram_performance[ngram]["Cost"] += row["Cost"]
                        ngram_performance[ngram]["Conversions"] += row["Conversions"]

            df_ngram_performance = pd.DataFrame.from_dict(ngram_performance, orient='index').reset_index().rename(columns={"index": "N-gram"})
            df_ngram_performance["CTR"] = (df_ngram_performance["Clicks"] / df_ngram_performance["Impressions"]) * 100
            df_ngram_performance["Conversion Rate"] = (df_ngram_performance["Conversions"] / df_ngram_performance["Clicks"]) * 100
            df_ngram_performance["Cost per Conversion"] = df_ngram_performance.apply(
                lambda row: "None" if row["Conversions"] == 0 else row["Cost"] / row["Conversions"], axis=1
            )
            df_ngram_performance['Cost per Conversion'] = df_ngram_performance['Cost per Conversion'].apply(lambda x: pd.NA if x == 'None' else x)
            df_ngram_performance['Cost per Conversion'] = pd.to_numeric(df_ngram_performance['Cost per Conversion'], errors='coerce')

            # Allow the user to sort the results.
            default_sort = "Conversions" if "Conversions" in df_ngram_performance.columns else df_ngram_performance.columns[0]
            sort_column = st.selectbox("Sort by Column:", options=df_ngram_performance.columns, index=list(df_ngram_performance.columns).index(default_sort))
            sort_ascending = st.checkbox("Sort Ascending", value=False)

            if sort_ascending:
                df_ngram_performance = df_ngram_performance.sort_values(by=sort_column, ascending=True, na_position='last')
            else:
                df_ngram_performance = df_ngram_performance.sort_values(by=sort_column, ascending=False, na_position='first')

            st.dataframe(df_ngram_performance.style.format({
                "Cost": "${:,.2f}",
                "Cost per Conversion": "${:,.2f}",
                "CTR": "{:,.2f}%",
                "Conversion Rate": "{:,.2f}%",
                "Conversions": "{:,.1f}"
            }))

        except Exception as e:
            st.error(f"An error occurred while processing the Excel file: {e}")

if __name__ == "__main__":
    google_ads_search_term_analyzer()
    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)", unsafe_allow_html=True)

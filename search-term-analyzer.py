import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def detect_product_categories(text):
    """Detect product categories for football jersey e-commerce"""
    text_lower = str(text).lower()
    categories = []
    
    # Product types
    product_types = {
        'jersey': r'\bjersey\b|\bshirt\b',
        'kit': r'\bkit\b',
        'training': r'\btraining\b',
        'shorts': r'\bshorts\b',
        'goalkeeper': r'\bgoalkeeper\b|\bgk\b|\bkeeper\b',
        'baby/kids': r'\bbaby\b|\bkids\b|\bchildren\b|\byouth\b|\bjunior\b|\btoddler\b'
    }
    
    # Product attributes
    attributes = {
        'home': r'\bhome\b',
        'away': r'\baway\b',
        'third': r'\bthird\b',
        'retro': r'\bretro\b|\bvintage\b|\bclassic\b',
        'long_sleeve': r'\blong sleeve\b|\bls\b'
    }
    
    # Detect product type
    for category, pattern in product_types.items():
        if re.search(pattern, text_lower):
            categories.append(category)
    
    # Detect attributes
    for attr, pattern in attributes.items():
        if re.search(pattern, text_lower):
            categories.append(attr)
    
    # Detect if it's a store/shop search
    if re.search(r'\bstore\b|\bshop\b|\bofficial\b', text_lower):
        categories.append('store_search')
    
    # Detect if it's a sale search
    if re.search(r'\bsale\b|\bdiscount\b|\bcheap\b|\bdeal\b', text_lower):
        categories.append('sale_search')
    
    return ', '.join(categories) if categories else 'generic'

def detect_famous_players(text):
    """Detect if search term contains famous player names"""
    text_lower = str(text).lower()
    famous_players = [
        'ronaldo', 'messi', 'neymar', 'mbappe', 'haaland', 'salah', 
        'benzema', 'lewandowski', 'modric', 'kane', 'son', 'de bruyne',
        'vinicius', 'grealish', 'foden', 'rashford', 'fernandes', 'casemiro'
    ]
    
    for player in famous_players:
        if player in text_lower:
            return player.title()
    return None

def google_ads_search_term_analyzer():
    st.set_page_config(page_title="E-commerce Google Ads Analyzer", layout="wide")
    st.header("🛒 E-commerce Google Ads Search Term Analyzer")
    st.markdown(
        """
        Upload a CSV file from your Google Ads search terms report and analyze it with e-commerce-specific metrics.
        This tool provides insights into ROAS, AOV, product categories, and keyword performance.
        """
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()

            # Check for required columns
            required_columns = ["Search term", "Clicks", "Impr.", "Cost"]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return

            # Convert numeric columns
            numeric_cols = ["Clicks", "Impr.", "Cost", "Conversions", "Conv. value", "Avg. CPC"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Calculate derived metrics
            df['CTR'] = (df['Clicks'] / df['Impr.'].replace(0, 1)) * 100
            
            if 'Conversions' in df.columns and 'Conv. value' in df.columns:
                df['CPA'] = df.apply(lambda row: row['Cost'] / row['Conversions'] if row['Conversions'] > 0 else 0, axis=1)
                df['AOV'] = df.apply(lambda row: row['Conv. value'] / row['Conversions'] if row['Conversions'] > 0 else 0, axis=1)
                df['ROAS'] = df.apply(lambda row: row['Conv. value'] / row['Cost'] if row['Cost'] > 0 else 0, axis=1)
                df['Conv. Rate'] = (df['Conversions'] / df['Clicks'].replace(0, 1)) * 100

            # Display overall metrics
            st.subheader("📊 Overall Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Clicks", f"{int(df['Clicks'].sum()):,}")
            with col2:
                st.metric("Total Cost", f"${df['Cost'].sum():,.2f}")
            with col3:
                if 'Conversions' in df.columns:
                    st.metric("Total Conversions", f"{int(df['Conversions'].sum()):,}")
            with col4:
                if 'Conv. value' in df.columns:
                    st.metric("Total Revenue", f"${df['Conv. value'].sum():,.2f}")
            with col5:
                if 'Conv. value' in df.columns and df['Cost'].sum() > 0:
                    overall_roas = df['Conv. value'].sum() / df['Cost'].sum()
                    st.metric("Overall ROAS", f"{overall_roas:.2f}x")

            # Add filters
            st.subheader("🔍 Filter Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_clicks = st.number_input("Min Clicks", value=0, min_value=0)
            with col2:
                min_conversions = st.number_input("Min Conversions", value=0, min_value=0)
            with col3:
                min_cost = st.number_input("Min Cost ($)", value=0.0, min_value=0.0, step=0.1)

            # Apply filters
            filtered_df = df[
                (df['Clicks'] >= min_clicks) & 
                (df.get('Conversions', 0) >= min_conversions) &
                (df['Cost'] >= min_cost)
            ].copy()

            st.info(f"Showing {len(filtered_df):,} of {len(df):,} search terms after filtering")

            # Product Category Analysis
            st.subheader("🏷️ Product Category Analysis")
            if st.checkbox("Analyze Product Categories"):
                filtered_df['Categories'] = filtered_df['Search term'].apply(detect_product_categories)
                filtered_df['Player Name'] = filtered_df['Search term'].apply(detect_famous_players)
                
                # Show category distribution
                category_counts = Counter()
                for cats in filtered_df['Categories']:
                    for cat in str(cats).split(', '):
                        if cat and cat != 'generic':
                            category_counts[cat] += 1
                
                if category_counts:
                    st.write("**Category Distribution:**")
                    cat_df = pd.DataFrame.from_dict(category_counts, orient='index', columns=['Count']).sort_values('Count', ascending=False)
                    st.dataframe(cat_df)

            # N-gram Analysis
            st.subheader("📝 N-gram Analysis")
            
            extraction_method = st.radio("Select N-gram Extraction Method:", 
                                        options=["Contiguous n-grams", "Skip-grams"], 
                                        index=0)
            
            col1, col2 = st.columns(2)
            with col1:
                n_value = st.selectbox("N-gram Size:", options=[1, 2, 3, 4], index=1)
            with col2:
                min_frequency = st.number_input("Minimum Frequency:", value=2, min_value=1)

            # Initialize stopwords and lemmatizer
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

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

            # Extract n-grams
            all_ngrams = []
            for term in filtered_df["Search term"]:
                if extraction_method == "Contiguous n-grams":
                    all_ngrams.extend(extract_ngrams(term, n_value))
                else:
                    all_ngrams.extend(extract_skipgrams(term, n_value))

            ngram_counts = Counter(all_ngrams)
            filtered_ngrams = {ngram: count for ngram, count in ngram_counts.items() if count >= min_frequency}

            if not filtered_ngrams:
                st.warning("No n-grams found with the specified minimum frequency.")
                return

            # Map search terms to n-grams
            search_term_to_ngrams = {}
            for term in filtered_df["Search term"]:
                if extraction_method == "Contiguous n-grams":
                    search_term_to_ngrams[term] = extract_ngrams(term, n_value)
                else:
                    search_term_to_ngrams[term] = extract_skipgrams(term, n_value)

            # Calculate performance metrics per n-gram
            ngram_performance = {}
            for index, row in filtered_df.iterrows():
                search_term_text = row["Search term"]
                for ngram in search_term_to_ngrams.get(search_term_text, []):
                    if ngram in filtered_ngrams:
                        if ngram not in ngram_performance:
                            ngram_performance[ngram] = {
                                "Clicks": 0, "Impr.": 0, "Cost": 0,
                                "Conversions": 0, "Conv. value": 0
                            }
                        ngram_performance[ngram]["Clicks"] += row["Clicks"]
                        ngram_performance[ngram]["Impr."] += row["Impr."]
                        ngram_performance[ngram]["Cost"] += row["Cost"]
                        if 'Conversions' in filtered_df.columns:
                            ngram_performance[ngram]["Conversions"] += row["Conversions"]
                        if 'Conv. value' in filtered_df.columns:
                            ngram_performance[ngram]["Conv. value"] += row["Conv. value"]

            df_ngram_performance = pd.DataFrame.from_dict(ngram_performance, orient='index').reset_index()
            df_ngram_performance.rename(columns={"index": "N-gram"}, inplace=True)
            
            # Calculate e-commerce metrics
            df_ngram_performance["CTR"] = (df_ngram_performance["Clicks"] / df_ngram_performance["Impr."].replace(0, 1)) * 100
            
            if 'Conversions' in df_ngram_performance.columns:
                df_ngram_performance["Conv. Rate"] = (df_ngram_performance["Conversions"] / df_ngram_performance["Clicks"].replace(0, 1)) * 100
                df_ngram_performance["CPA"] = df_ngram_performance.apply(
                    lambda row: row["Cost"] / row["Conversions"] if row["Conversions"] > 0 else None, axis=1)
            
            if 'Conv. value' in df_ngram_performance.columns:
                df_ngram_performance["AOV"] = df_ngram_performance.apply(
                    lambda row: row["Conv. value"] / row["Conversions"] if row["Conversions"] > 0 else None, axis=1)
                df_ngram_performance["ROAS"] = df_ngram_performance.apply(
                    lambda row: row["Conv. value"] / row["Cost"] if row["Cost"] > 0 else None, axis=1)
                df_ngram_performance["Revenue"] = df_ngram_performance["Conv. value"]

            # Sorting options
            default_sort = "Revenue" if "Revenue" in df_ngram_performance.columns else "Clicks"
            sort_column = st.selectbox("Sort by:", 
                                      options=df_ngram_performance.columns.tolist(),
                                      index=list(df_ngram_performance.columns).index(default_sort))
            sort_ascending = st.checkbox("Sort Ascending", value=False)

            df_sorted = df_ngram_performance.sort_values(
                by=sort_column, 
                ascending=sort_ascending, 
                na_position='last' if sort_ascending else 'first'
            )

            # Format and display
            format_dict = {
                "Cost": "${:,.2f}",
                "CPA": "${:,.2f}",
                "AOV": "${:,.2f}",
                "Revenue": "${:,.2f}",
                "CTR": "{:,.2f}%",
                "Conv. Rate": "{:,.2f}%",
                "ROAS": "{:,.2f}x",
                "Conversions": "{:,.1f}"
            }
            
            st.dataframe(df_sorted.style.format({k: v for k, v in format_dict.items() if k in df_sorted.columns}), 
                        use_container_width=True, height=600)

            # Export option
            st.subheader("💾 Export Results")
            csv = df_sorted.to_csv(index=False)
            st.download_button(
                label="Download N-gram Analysis as CSV",
                data=csv,
                file_name="ngram_analysis.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    google_ads_search_term_analyzer()
    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)", unsafe_allow_html=True)



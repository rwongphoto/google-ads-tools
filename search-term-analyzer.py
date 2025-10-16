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
        
        **Required columns:** Search term, Clicks, Impr., Cost  
        **Optional columns:** Conversions, Conv. value (enables ROAS/AOV metrics)
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

            # Check which optional columns are available
            has_conversions = 'Conversions' in df.columns
            has_conv_value = 'Conv. value' in df.columns
            
            # Show what metrics are available
            if has_conv_value:
                st.success("✅ Full e-commerce metrics available (ROAS, AOV, CPA)")
            elif has_conversions:
                st.info("ℹ️ Conversions tracked - showing CPA and Conv. Rate (add 'Conv. value' column for ROAS/AOV)")
            else:
                st.warning("⚠️ Basic metrics only - add 'Conversions' and 'Conv. value' columns for e-commerce metrics")
            
            # Calculate derived metrics
            df['CTR'] = (df['Clicks'] / df['Impr.'].replace(0, 1)) * 100
            
            if has_conversions and has_conv_value:
                df['CPA'] = df.apply(lambda row: row['Cost'] / row['Conversions'] if row['Conversions'] > 0 else 0, axis=1)
                df['AOV'] = df.apply(lambda row: row['Conv. value'] / row['Conversions'] if row['Conversions'] > 0 else 0, axis=1)
                df['ROAS'] = df.apply(lambda row: row['Conv. value'] / row['Cost'] if row['Cost'] > 0 else 0, axis=1)
                df['Conv. Rate'] = (df['Conversions'] / df['Clicks'].replace(0, 1)) * 100
            elif has_conversions:
                # Only conversions available, no revenue data
                df['CPA'] = df.apply(lambda row: row['Cost'] / row['Conversions'] if row['Conversions'] > 0 else 0, axis=1)
                df['Conv. Rate'] = (df['Conversions'] / df['Clicks'].replace(0, 1)) * 100

            # Display overall metrics
            st.subheader("📊 Overall Performance Metrics")
            
            if has_conv_value:
                # Full e-commerce metrics available
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Clicks", f"{int(df['Clicks'].sum()):,}")
                with col2:
                    st.metric("Total Cost", f"${df['Cost'].sum():,.2f}")
                with col3:
                    st.metric("Total Conversions", f"{int(df['Conversions'].sum()):,}")
                with col4:
                    st.metric("Total Revenue", f"${df['Conv. value'].sum():,.2f}")
                with col5:
                    overall_roas = df['Conv. value'].sum() / df['Cost'].sum() if df['Cost'].sum() > 0 else 0
                    st.metric("Overall ROAS", f"{overall_roas:.2f}x")
            elif has_conversions:
                # Only conversions available
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Clicks", f"{int(df['Clicks'].sum()):,}")
                with col2:
                    st.metric("Total Cost", f"${df['Cost'].sum():,.2f}")
                with col3:
                    st.metric("Total Conversions", f"{int(df['Conversions'].sum()):,}")
                with col4:
                    avg_cpa = df['Cost'].sum() / df['Conversions'].sum() if df['Conversions'].sum() > 0 else 0
                    st.metric("Avg CPA", f"${avg_cpa:.2f}")
            else:
                # Basic metrics only
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Clicks", f"{int(df['Clicks'].sum()):,}")
                with col2:
                    st.metric("Total Impressions", f"{int(df['Impr.'].sum()):,}")
                with col3:
                    st.metric("Total Cost", f"${df['Cost'].sum():,.2f}")

            # Add filters
            st.subheader("🔍 Filter Options")
            if has_conversions:
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_clicks = st.number_input("Min Clicks", value=0, min_value=0)
                with col2:
                    min_conversions = st.number_input("Min Conversions", value=0, min_value=0)
                with col3:
                    min_cost = st.number_input("Min Cost ($)", value=0.0, min_value=0.0, step=0.1)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    min_clicks = st.number_input("Min Clicks", value=0, min_value=0)
                with col2:
                    min_cost = st.number_input("Min Cost ($)", value=0.0, min_value=0.0, step=0.1)
                min_conversions = 0

            # Apply filters
            filter_conditions = (df['Clicks'] >= min_clicks) & (df['Cost'] >= min_cost)
            if has_conversions:
                filter_conditions = filter_conditions & (df['Conversions'] >= min_conversions)
            
            filtered_df = df[filter_conditions].copy()

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
            
            st.info("""
            **N-gram methods:**
            - **Contiguous n-grams**: Consecutive words (e.g., "manchester united jersey")
            - **Skip-grams**: Words with gaps (e.g., "manchester jersey", "united jersey") - slower but finds more patterns
            """)
            
            extraction_method = st.radio("Select N-gram Extraction Method:", 
                                        options=["Contiguous n-grams", "Skip-grams"], 
                                        index=0)
            
            if extraction_method == "Skip-grams":
                st.warning("⚠️ Skip-grams may take 30-60 seconds to process large datasets")
            
            col1, col2 = st.columns(2)
            with col1:
                n_value = st.selectbox("N-gram Size:", options=[1, 2, 3, 4], index=1)
            with col2:
                min_frequency = st.number_input("Minimum Frequency:", value=2, min_value=1)
            
            if extraction_method == "Skip-grams":
                max_skip = st.slider("Max Skip Distance:", min_value=1, max_value=2, value=1, 
                                    help="How many words can be skipped between n-gram terms. Higher = more patterns but slower.")
            else:
                max_skip = 0

            # Initialize stopwords and lemmatizer
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            def extract_ngrams(text, n):
                text = str(text).lower()
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
                ngrams_list = list(nltk.ngrams(tokens, n))
                return [" ".join(gram) for gram in ngrams_list]

            def extract_skipgrams(text, n, max_skip=1):
                """Extract skip-grams allowing gaps between words"""
                from itertools import combinations
                text = str(text).lower()
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
                
                if len(tokens) < n:
                    return []
                
                skipgrams_list = []
                # Slide a window and extract combinations within max_skip distance
                window_size = n + max_skip * (n - 1)  # Max span of n words with gaps
                
                for i in range(len(tokens) - n + 1):
                    window_end = min(i + window_size, len(tokens))
                    window_tokens = tokens[i:window_end]
                    
                    if len(window_tokens) >= n:
                        # Get all combinations of n words from window
                        for combo_indices in combinations(range(len(window_tokens)), n):
                            # Check if gaps are within max_skip
                            valid = True
                            for j in range(len(combo_indices) - 1):
                                gap = combo_indices[j+1] - combo_indices[j] - 1
                                if gap > max_skip:
                                    valid = False
                                    break
                            
                            if valid:
                                skipgram = " ".join(window_tokens[idx] for idx in combo_indices)
                                skipgrams_list.append(skipgram)
                
                return list(set(skipgrams_list))  # Remove duplicates

            # Extract n-grams
            try:
                all_ngrams = []
                
                # Show progress for large datasets
                if len(filtered_df) > 1000:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, term in enumerate(filtered_df["Search term"]):
                        if extraction_method == "Contiguous n-grams":
                            all_ngrams.extend(extract_ngrams(term, n_value))
                        else:
                            all_ngrams.extend(extract_skipgrams(term, n_value, max_skip))
                        
                        # Update progress every 100 rows
                        if idx % 100 == 0:
                            progress = (idx + 1) / len(filtered_df)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: {idx + 1:,} / {len(filtered_df):,} search terms")
                    
                    progress_bar.empty()
                    status_text.empty()
                else:
                    for term in filtered_df["Search term"]:
                        if extraction_method == "Contiguous n-grams":
                            all_ngrams.extend(extract_ngrams(term, n_value))
                        else:
                            all_ngrams.extend(extract_skipgrams(term, n_value, max_skip))
            except Exception as e:
                st.error(f"Error during n-gram extraction: {str(e)}")
                st.info("Try using 'Contiguous n-grams' instead, or reduce your dataset size with filters.")
                return

            ngram_counts = Counter(all_ngrams)
            filtered_ngrams = {ngram: count for ngram, count in ngram_counts.items() if count >= min_frequency}

            if not filtered_ngrams:
                st.warning("No n-grams found with the specified minimum frequency.")
                return
            
            # Show diagnostic info
            st.info(f"""
            **Extraction Results:**
            - Total n-grams extracted: {len(all_ngrams):,}
            - Unique n-grams: {len(ngram_counts):,}
            - N-grams meeting min frequency ({min_frequency}): {len(filtered_ngrams):,}
            """)
            
            # Show examples
            if extraction_method == "Skip-grams":
                st.write("**Skip-gram Examples:**")
                example_ngrams = list(filtered_ngrams.keys())[:10]
                for ng in example_ngrams:
                    st.code(ng)
                
                # Test on a sample search term to verify skip-grams work
                if len(filtered_df) > 0:
                    sample_term = filtered_df.iloc[0]["Search term"]
                    st.write(f"\n**Test on sample term:** '{sample_term}'")
                    contiguous_test = extract_ngrams(sample_term, n_value)
                    skipgram_test = extract_skipgrams(sample_term, n_value, max_skip)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Contiguous n-grams:**")
                        for ng in contiguous_test:
                            st.code(ng)
                    with col2:
                        st.write("**Skip-grams:**")
                        for ng in skipgram_test:
                            # Highlight if not in contiguous
                            if ng not in contiguous_test:
                                st.code(f"🔸 {ng}  ← NEW")
                            else:
                                st.code(ng)

            # Map search terms to n-grams
            search_term_to_ngrams = {}
            for term in filtered_df["Search term"]:
                if extraction_method == "Contiguous n-grams":
                    search_term_to_ngrams[term] = extract_ngrams(term, n_value)
                else:
                    search_term_to_ngrams[term] = extract_skipgrams(term, n_value, max_skip)

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
                        if has_conversions:
                            ngram_performance[ngram]["Conversions"] += row["Conversions"]
                        if has_conv_value:
                            ngram_performance[ngram]["Conv. value"] += row["Conv. value"]

            df_ngram_performance = pd.DataFrame.from_dict(ngram_performance, orient='index').reset_index()
            df_ngram_performance.rename(columns={"index": "N-gram"}, inplace=True)
            
            # Calculate e-commerce metrics
            df_ngram_performance["CTR"] = (df_ngram_performance["Clicks"] / df_ngram_performance["Impr."].replace(0, 1)) * 100
            
            if has_conversions:
                df_ngram_performance["Conv. Rate"] = (df_ngram_performance["Conversions"] / df_ngram_performance["Clicks"].replace(0, 1)) * 100
                df_ngram_performance["CPA"] = df_ngram_performance.apply(
                    lambda row: row["Cost"] / row["Conversions"] if row["Conversions"] > 0 else None, axis=1)
            
            if has_conv_value:
                df_ngram_performance["AOV"] = df_ngram_performance.apply(
                    lambda row: row["Conv. value"] / row["Conversions"] if row["Conversions"] > 0 else None, axis=1)
                df_ngram_performance["ROAS"] = df_ngram_performance.apply(
                    lambda row: row["Conv. value"] / row["Cost"] if row["Cost"] > 0 else None, axis=1)
                df_ngram_performance["Revenue"] = df_ngram_performance["Conv. value"]
            
            # Remove Conv. value column if we have Revenue
            if has_conv_value:
                df_ngram_performance = df_ngram_performance.drop(columns=['Conv. value'])
            
            # Remove Conversions column with 0 if no conversions tracked
            if not has_conversions:
                df_ngram_performance = df_ngram_performance.drop(columns=['Conversions'])

            # Sorting options
            if has_conv_value and "Revenue" in df_ngram_performance.columns:
                default_sort = "Revenue"
            elif has_conversions and "Conversions" in df_ngram_performance.columns:
                default_sort = "Conversions"
            else:
                default_sort = "Clicks"
            
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
                "Clicks": "{:,.0f}",
                "Impr.": "{:,.0f}",
                "Cost": "${:,.2f}",
                "CPA": "${:,.2f}",
                "AOV": "${:,.2f}",
                "Revenue": "${:,.2f}",
                "CTR": "{:.2f}%",
                "Conv. Rate": "{:.2f}%",
                "ROAS": "{:.2f}x",
                "Conversions": "{:.0f}"
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



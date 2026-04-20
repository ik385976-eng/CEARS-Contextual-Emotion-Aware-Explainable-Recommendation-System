import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

from context import apply_context_adjustment
from emotion import apply_emotion_adjustment
from hybrid import HybridLayer

st.set_page_config(page_title="CEARS — AI Shopping", layout="wide", page_icon="🛍️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #07090f; color: #e2e8f0; }
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1300px; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #07090f !important;
    border-right: 1px solid #131c2e !important;
}
[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: #38bdf8 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(14,165,233,0.4) !important;
}

.stSelectbox > div > div {
    background: #0d1424 !important;
    border: 1px solid #1a2744 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

.stImage img {
    border-radius: 12px;
    object-fit: cover;
    width: 100%;
    max-height: 200px;
}

[data-testid="stMetric"] {
    background: #0d1424;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #1a2744;
}

[data-testid="stExpander"] {
    background: #0d1424 !important;
    border: 1px solid #1a2744 !important;
    border-radius: 12px !important;
}

hr { border-color: #1a2744 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("amazon.csv")
    df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(4.0)
    df['rating_count'] = pd.to_numeric(
        df['rating_count'].astype(str).str.replace(',', ''), errors='coerce'
    ).fillna(0).astype(int)
    df['main_cat'] = df['category'].apply(
        lambda x: x.split('|')[0].strip() if isinstance(x, str) else "Other"
    )
    df['discount_pct'] = (
        (df['actual_price'] - df['discounted_price']) / df['actual_price'] * 100
    ).clip(0, 100).round(0).astype(int)
    df['img_link'] = df['img_link'].apply(clean_image_url)
    return df


def clean_image_url(url):
    if pd.isna(url) or not isinstance(url, str):
        return ""
    url = str(url).strip()
    if '?' in url:
        url = url.split('?')[0]
    if url.startswith('//'):
        url = 'https:' + url
    elif url.startswith('/'):
        url = 'https://m.media-amazon.com' + url
    elif url.startswith('http://'):
        url = url.replace('http://', 'https://')
    valid_ext = any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif'])
    if 'amazon' in url.lower() and valid_ext:
        return url
    return ""


def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def build_why(row, s_score, c_bonus, e_bonus, s_polarity, budget_lvl, mood_input):
    reasons = []

    if s_score >= 0.6:
        reasons.append(("🔗 Similarity Match",
            f"Very closely matches your selected product in features and category "
            f"(similarity score: {s_score:.0%})."))
    elif s_score >= 0.3:
        reasons.append(("🔗 Similarity Match",
            f"Shares key characteristics with your selected product "
            f"(similarity score: {s_score:.0%})."))
    else:
        reasons.append(("🔗 Similarity Match",
            "Complementary product from the same category that broadens your options."))

    rating = row['rating']
    count = int(row.get('rating_count', 0))
    if rating >= 4.3:
        reasons.append(("⭐ Customer Rating",
            f"Highly rated at {rating}/5 by {count:,} verified buyers — "
            f"customers consistently rate this product excellent."))
    elif rating >= 4.0:
        reasons.append(("⭐ Customer Rating",
            f"Strong rating of {rating}/5 from {count:,} reviews — "
            f"a reliable, well-trusted choice."))
    else:
        reasons.append(("⭐ Customer Rating",
            f"Rated {rating}/5 — decent quality at this price point."))

    disc = row['discount_pct']
    price = int(row['discounted_price'])
    if budget_lvl in ("Budget", "Balanced") and disc >= 30:
        reasons.append(("💰 Price & Value",
            f"{disc}% off the original price — excellent value aligned "
            f"with your {budget_lvl.lower()} budget."))
    elif budget_lvl in ("Premium", "Luxury"):
        reasons.append(("💰 Price & Value",
            f"Priced at ₹{price:,} — positioned for your {budget_lvl.lower()} "
            f"preference where quality is the priority."))
    else:
        reasons.append(("💰 Price & Value",
            f"{disc}% discount brings this to ₹{price:,} — "
            f"smart value for the quality on offer."))

    if s_polarity > 0.2:
        reasons.append(("🗣 Review Sentiment",
            f"Buyer reviews are strongly positive (sentiment score: +{s_polarity:.2f}) — "
            f"high satisfaction reported across purchases."))
    elif s_polarity > 0:
        reasons.append(("🗣 Review Sentiment",
            f"Review sentiment is generally positive (+{s_polarity:.2f}), "
            f"indicating a reliable customer experience."))
    else:
        reasons.append(("⚠️ Review Sentiment",
            f"Reviews show mixed sentiment ({s_polarity:.2f}) — "
            f"recommended to read reviews carefully before purchasing."))

    if e_bonus > 0.55:
        reasons.append(("🧠 Preference Match",
            f"Your stated intent — \"{mood_input[:60]}\" — signals high confidence, "
            f"which this product aligns with well."))
    elif e_bonus > 0.4:
        reasons.append(("🧠 Preference Match",
            f"Aligns with your stated shopping preferences based on "
            f"mood and context analysis (score: {e_bonus:.2f})."))

    return reasons


def section_header(title):
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:11px;font-weight:700;
                letter-spacing:3px;text-transform:uppercase;color:#38bdf8;
                margin:1.5rem 0 1rem 0;display:flex;align-items:center;gap:12px;">
        {title}
        <div style="flex:1;height:1px;background:linear-gradient(90deg,#1a2744,transparent);"></div>
    </div>
    """, unsafe_allow_html=True)


df = load_data()
total_products = df['product_id'].nunique()
avg_rating = df['rating'].mean()
avg_discount = df['discount_pct'].mean()
num_cats = df['main_cat'].nunique()

st.markdown(f"""
<div style="padding:2rem 0 1.5rem 0;">
    <div style="display:inline-block;
                background:linear-gradient(90deg,rgba(56,189,248,0.1),rgba(129,140,248,0.1));
                border:1px solid rgba(56,189,248,0.25);color:#38bdf8;
                font-family:'Syne',sans-serif;font-size:10px;font-weight:700;
                letter-spacing:3px;text-transform:uppercase;
                padding:6px 16px;border-radius:100px;margin-bottom:1.2rem;">
        Hybrid AI — Multi-Signal Recommendation Engine
    </div>
    <h1 style="font-family:'Syne',sans-serif;font-size:56px;font-weight:800;line-height:1.0;
               background:linear-gradient(135deg,#f1f5f9 0%,#38bdf8 50%,#818cf8 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;margin:0 0 0.5rem 0;">
        CEARS
    </h1>
    <p style="font-size:15px;color:#64748b;font-weight:300;margin-bottom:1.8rem;max-width:600px;line-height:1.6;">
        Context-aware e-commerce recommendation system powered by sentiment analysis,
        emotion mapping, and content similarity scoring.
    </p>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <div style="background:#0d1424;border:1px solid #1a2744;border-radius:12px;padding:14px 22px;">
            <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:700;color:#38bdf8;">{total_products:,}</div>
            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-top:2px;">Products</div>
        </div>
        <div style="background:#0d1424;border:1px solid #1a2744;border-radius:12px;padding:14px 22px;">
            <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:700;color:#38bdf8;">{num_cats}</div>
            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-top:2px;">Categories</div>
        </div>
        <div style="background:#0d1424;border:1px solid #1a2744;border-radius:12px;padding:14px 22px;">
            <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:700;color:#38bdf8;">⭐ {avg_rating:.2f}</div>
            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-top:2px;">Avg Rating</div>
        </div>
        <div style="background:#0d1424;border:1px solid #1a2744;border-radius:12px;padding:14px 22px;">
            <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:700;color:#38bdf8;">{avg_discount:.0f}%</div>
            <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-top:2px;">Avg Discount</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Recommendation Core")
    mood_input = st.text_input("Shopping intent", "I'm looking for quality and reliability.")
    budget_lvl = st.select_slider("Budget Level", options=["Budget", "Balanced", "Premium", "Luxury"])
    st.divider()
    st.markdown("### Engine Settings")
    use_sentiment = st.toggle("Live Sentiment Analysis", True)
    top_n = st.slider("Results to show", 3, 10, 5)
    st.divider()
    st.markdown("### Category Filter")
    rec_cat_filter = st.selectbox(
        "Filter by category",
        ["All Categories"] + sorted(df['main_cat'].unique().tolist())
    )
    st.divider()
    st.markdown("### Dataset Stats")
    vdf = df if rec_cat_filter == "All Categories" else df[df['main_cat'] == rec_cat_filter]
    st.metric("Total Products", f"{vdf['product_id'].nunique():,}")
    st.metric("Avg Rating", f"⭐ {vdf['rating'].mean():.2f}")
    st.metric("Avg Discount", f"{vdf['discount_pct'].mean():.0f}% off")

section_header("Market Insights")

ch1, ch2 = st.columns([2, 1])
with ch1:
    cat_stats = df.groupby('main_cat')['rating'].mean().sort_values(ascending=False).reset_index()
    fig_bar = px.bar(
        cat_stats, x='main_cat', y='rating',
        labels={'main_cat': '', 'rating': 'Avg Rating'},
        color='rating',
        color_continuous_scale=[[0, '#0d2240'], [0.5, '#0ea5e9'], [1, '#38bdf8']]
    )
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#64748b', size=11),
        height=240, margin=dict(l=0, r=0, t=8, b=0),
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-30, gridcolor='#131c2e', linecolor='#131c2e'),
        yaxis=dict(gridcolor='#131c2e', range=[3.5, 5]),
    )
    fig_bar.update_traces(marker_line_width=0)
    st.plotly_chart(fig_bar, use_container_width=True)

with ch2:
    fig_box = px.box(
        df, y='discounted_price', log_y=True,
        labels={'discounted_price': 'Price (₹)'},
        color_discrete_sequence=['#6366f1']
    )
    fig_box.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#64748b', size=11),
        height=240, margin=dict(l=0, r=0, t=8, b=0),
        yaxis=dict(gridcolor='#131c2e'),
    )
    st.plotly_chart(fig_box, use_container_width=True)

section_header("Product Search")

target_p = st.selectbox("Select a product", df['product_name'].unique(), label_visibility="collapsed")
p_id = df[df['product_name'] == target_p]['product_id'].values[0]
target_cat = df[df['product_id'] == p_id]['main_cat'].iloc[0]

info_col, btn_col = st.columns([3, 1])
with info_col:
    st.caption(f"Selected: {target_p[:90]}")
    st.caption(f"Category: {target_cat}")
with btn_col:
    run = st.button("Run Analysis", use_container_width=True)

if run:
    with st.spinner("Running hybrid recommendation engine..."):
        hybrid = HybridLayer(df[['product_id', 'product_name']])
        recs = []

        if rec_cat_filter == "All Categories":
            category_pool = df.copy()
        else:
            category_pool = df[df['main_cat'] == rec_cat_filter]

        for _, r in category_pool.iterrows():
            if r['product_id'] != p_id:
                score = hybrid.get_content_score(r['product_id'], p_id)
                recs.append((r['product_id'], score))

        if len(recs) == 0:
            st.warning("No products found in the selected category. Try a different filter.")
        else:
            recs.sort(key=lambda x: x[1], reverse=True)
            context_adj = apply_context_adjustment(recs[:50], {"budget": budget_lvl})
            emotion_adj = apply_emotion_adjustment(
                context_adj, mood_input, dict(zip(df.product_id, df.product_name))
            )
            final_recs = hybrid.apply_hybrid(emotion_adj, p_id)

    if len(recs) > 0:
        section_header(f"Top {top_n} Personalised Recommendations")

        for idx, (pid, score, c_bonus, e_bonus, s_score) in enumerate(final_recs[:top_n]):
            row = df[df['product_id'] == pid].iloc[0]

            review_text = row.get('review_content', '')
            if pd.isna(review_text) or not review_text:
                review_text = row.get('review_title', '')
            s_polarity = analyze_sentiment(str(review_text)) if use_sentiment else 0.0

            if s_polarity > 0.1:
                s_label = "Positive 😊"
            elif s_polarity < -0.1:
                s_label = "Negative 😞"
            else:
                s_label = "Neutral 😐"

            composite = round((s_score + c_bonus + e_bonus) / 3, 2)
            img_url = row.get('img_link', '')

            with st.container(border=True):
                col1, col2 = st.columns([1, 3])

                with col1:
                    if img_url and img_url.strip():
                        try:
                            st.image(img_url, use_container_width=True)
                        except Exception:
                            st.markdown(
                                '<div style="background:#0d1424;border:1px solid #1a2744;'
                                'border-radius:12px;padding:30px 20px;text-align:center;">'
                                '<p style="color:#475569;font-size:13px;margin:0;">No Image Available</p></div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            '<div style="background:#0d1424;border:1px solid #1a2744;'
                            'border-radius:12px;padding:30px 20px;text-align:center;">'
                            '<p style="color:#475569;font-size:13px;margin:0;">No Image Available</p></div>',
                            unsafe_allow_html=True
                        )

                    st.caption(f"⭐ {row['rating']} / 5  ·  {row['rating_count']:,} reviews")
                    if use_sentiment:
                        st.caption(f"Sentiment: {s_label}")

                with col2:
                    st.markdown(
                        f"**#{idx + 1} — {row['product_name'][:80]}"
                        + ("..." if len(row['product_name']) > 80 else "") + "**"
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Price", f"₹{int(row['discounted_price']):,}")
                    m2.metric("Discount", f"{row['discount_pct']}%")
                    m3.metric("Match", f"{composite:.0%}")
                    m4.metric("Similarity", f"{s_score:.2f}")

                    with st.expander("Why is this recommended for you?"):
                        reasons = build_why(
                            row, s_score, c_bonus, e_bonus,
                            s_polarity, budget_lvl, mood_input
                        )
                        for label, text in reasons:
                            st.markdown(f"**{label}** — {text}")

                    product_link = row.get('product_link', 'https://amazon.in')
                    st.link_button("View on Amazon →", product_link, use_container_width=True)

        st.markdown("""
        <div style="background:#0d1424;border:1px solid #1a2744;border-radius:12px;
                    padding:1rem;margin-top:1.5rem;">
            <p style="color:#475569;font-size:12px;margin:0;line-height:1.6;">
                <b style="color:#64748b;">Disclaimer:</b> Recommendations are generated using content
                similarity, sentiment analysis, and user preferences. Results are for demonstration
                purposes. Always verify product details before purchasing.
            </p>
        </div>
        """, unsafe_allow_html=True)
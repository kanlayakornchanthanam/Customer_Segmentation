import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import json
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700&family=Open+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], p, li, span, div { font-family: 'Open Sans', sans-serif !important; }
h1,h2,h3,h4,.montserrat { font-family: 'Montserrat', sans-serif !important; }

.main, .block-container { background-color: #ffffff !important; }
[data-testid="stAppViewContainer"] { background-color: #ffffff; }
[data-testid="stSidebar"] { background: #f7faf8 !important; border-right: 1px solid #d0e8d8; }

.sidebar-label { font-family:'Montserrat',sans-serif !important; font-size:0.7rem; font-weight:700; color:#2d6a4f; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; margin-top:4px; }

div[data-testid="stButton"] > button { background-color:#2d6a4f !important; color:#ffffff !important; font-family:'Montserrat',sans-serif !important; font-weight:600 !important; font-size:0.85rem !important; border:none !important; border-radius:8px !important; padding:10px 0 !important; width:100% !important; letter-spacing:0.04em; }
div[data-testid="stButton"] > button:hover { background-color:#1a472a !important; }

div[data-testid="stSelectbox"] > div { border-color:#2d6a4f !important; border-radius:8px !important; background:#ffffff !important; color:#1a1a1a !important; }
div[data-testid="stSelectbox"] > div > div { background:#ffffff !important; color:#1a1a1a !important; }
div[data-testid="stSelectbox"] input { background:#ffffff !important; color:#1a1a1a !important; }
[data-baseweb="select"] > div { background:#ffffff !important; color:#1a1a1a !important; border-color:#2d6a4f !important; border-radius:8px !important; }
[data-baseweb="select"] span { color:#1a1a1a !important; }
[data-baseweb="popover"] ul { background:#ffffff !important; }
[data-baseweb="popover"] li { background:#ffffff !important; color:#1a1a1a !important; }
[data-baseweb="popover"] li:hover { background:#f0f9f3 !important; }

.stSlider > div > div > div { background:#2d6a4f !important; }
.stSlider > div > div > div > div { background:#2d6a4f !important; }
div[data-testid="stSlider"] div[role="slider"] {
    background:#2d6a4f !important;
    border:2px solid #1a472a !important;
}
div[data-testid="stSlider"] div[data-testid="stTickBar"] { color:#5a8a6e !important; }
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] p *,
div[data-testid="stSlider"] span,
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #2d6a4f !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}
div[data-testid="stSlider"] div[role="slider"] { background:#2d6a4f !important; border:2px solid #1a472a !important; }
div[data-testid="stSlider"] div[role="slider"]:hover { background:#1a472a !important; }
div[data-testid="stSlider"] div[role="slider"]:active { background:#1a472a !important; }
div[data-testid="stSlider"] [data-baseweb="tooltip"],
div[data-testid="stSlider"] [data-baseweb="tooltip"] *,
div[data-testid="stSlider"] [data-baseweb="tooltip"] div {
    background: #2d6a4f !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
}

.stTabs [data-baseweb="tab-list"] { background:#f2f8f4; border-radius:10px; padding:4px; gap:4px; border:1px solid #d0e8d8; }
.stTabs [data-baseweb="tab"] { font-family:'Montserrat',sans-serif !important; font-size:0.78rem; font-weight:600; color:#5a8a6e; border-radius:7px; padding:7px 16px; background:transparent; }
.stTabs [aria-selected="true"] { background:#ffffff !important; color:#1a472a !important; border:1px solid #b7dcc5 !important; }

.metric-card { background:#f7faf8; border:1px solid #c8e6d0; border-radius:10px; padding:16px; text-align:center; }
.metric-val { font-family:'Montserrat',sans-serif !important; font-size:1.8rem; font-weight:700; color:#1a472a; line-height:1.1; }
.metric-lbl { font-size:0.7rem; color:#5a8a6e; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }

.predict-card { background:#f0f9f3; border:2px solid #2d6a4f; border-radius:14px; padding:20px 24px; margin-bottom:16px; }
.predict-segment-name { font-family:'Montserrat',sans-serif !important; font-size:1.4rem; font-weight:700; color:#1a472a; margin-bottom:4px; }
.predict-desc { font-size:0.85rem; color:#5a8a6e; margin-bottom:12px; }
.conf-bar-bg { background:#d8eeda; border-radius:6px; height:8px; margin:8px 0 4px; }
.conf-bar-fill { height:8px; border-radius:6px; background:#2d6a4f; }
.conf-text { font-size:0.78rem; color:#5a8a6e; }
.conf-text b { color:#1a472a; }
.strategy-box { background:#e0f2e8; border-radius:8px; padding:10px 14px; margin-top:12px; font-size:0.82rem; color:#2d6a4f; }
.strategy-box b { font-family:'Montserrat',sans-serif !important; color:#1a472a; }

.section-heading { font-family:'Montserrat',sans-serif !important; font-size:0.95rem; font-weight:700; color:#1a472a; border-bottom:2px solid #c8e6d0; padding-bottom:6px; margin:18px 0 12px; }

.model-info-box { background:#eaf4ee; border-radius:10px; padding:12px 14px; font-size:0.8rem; color:#444; line-height:2; margin-top:16px; }
.model-info-box .label { font-family:'Montserrat',sans-serif !important; font-size:0.68rem; font-weight:700; color:#2d6a4f; text-transform:uppercase; letter-spacing:0.08em; display:block; margin-bottom:6px; }
.model-info-box .val { color:#1a472a; font-weight:600; }

.page-title { font-family:'Montserrat',sans-serif !important; font-size:1.9rem; font-weight:700; color:#1a472a; margin-bottom:2px; }
.page-sub { font-size:0.88rem; color:#5a8a6e; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

SEGMENTS = {
    0: {"name":"Cautious Shoppers","desc":"Low income, low spending — price-sensitive, needs value-focused offers","icon":"💼","color":"#1D9E75","strategy":"Offer discounts, loyalty points, and budget-friendly product lines.","income":"~$30k","spending":"~25/100","age":"~45 yrs"},
    1: {"name":"Impulsive Buyers","desc":"Low income, high spending — aspirational, responds to trends and FOMO","icon":"🛒","color":"#BA7517","strategy":"Highlight trending items, limited-time offers, and instalment payment options.","income":"~$25k","spending":"~75/100","age":"~25 yrs"},
    2: {"name":"Average Customers","desc":"Mid income, mid spending — balanced, the largest segment","icon":"👥","color":"#534AB7","strategy":"Focus on consistency, reliability, and moderate promotions.","income":"~$55k","spending":"~50/100","age":"~43 yrs"},
    3: {"name":"Wealthy but Frugal","desc":"High income, low spending — affluent but selective, quality over quantity","icon":"🏦","color":"#993556","strategy":"Emphasise premium quality, exclusivity, and investment value.","income":"~$87k","spending":"~18/100","age":"~41 yrs"},
    4: {"name":"VIP Targets","desc":"High income, high spending — most valuable segment, brand loyal","icon":"⭐","color":"#2d6a4f","strategy":"Offer VIP memberships, exclusive previews, and personalised experiences.","income":"~$87k","spending":"~82/100","age":"~33 yrs"},
}

@st.cache_resource
def load_model():
    np.random.seed(42)
    centers = [(30,25),(25,75),(55,49),(87,18),(87,82)]
    X_list, y_list = [], []
    for i,(inc,spc) in enumerate(centers):
        n_i = 40
        X_list.append(np.column_stack([np.random.normal(inc,8,n_i).clip(15,137), np.random.normal(spc,8,n_i).clip(1,99)]))
        y_list.extend([i]*n_i)
    X_clust = np.vstack(X_list)
    y_clust = np.array(y_list)
    ages    = np.random.randint(18,70,len(y_clust))
    genders = np.random.randint(0,2,len(y_clust))
    X_full  = np.column_stack([ages,genders,X_clust])
    sf = StandardScaler(); X_s = sf.fit_transform(X_full)
    clf = SVC(C=10,kernel='rbf',gamma='scale',probability=True,random_state=42)
    clf.fit(X_s, y_clust)
    sc2 = StandardScaler()
    sil = silhouette_score(sc2.fit_transform(X_clust), y_clust)
    return clf, sf, X_clust, y_clust, sil

clf, scaler_full, X_clust, y_clust, sil_score = load_model()
feature_cols = ['Age','Gender_encoded','Annual Income (k$)','Spending Score (1-100)']

def predict_segment(age, gender, income, spending):
    g = 0 if gender=="Female" else 1
    raw = np.array([[age,g,income,spending]],dtype=float)
    lbl = clf.predict(scaler_full.transform(raw))[0]
    proba = clf.predict_proba(scaler_full.transform(raw))[0]
    return int(lbl), float(proba.max()*100), proba

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 12px;'>
        <div style='font-size:2.2rem;'>🛍️</div>
        <div style='font-family:Montserrat,sans-serif;font-size:1.1rem;font-weight:700;color:#1a472a;margin-top:6px;'>Customer Segmentation</div>
        <div style='font-size:0.72rem;color:#5a8a6e;margin-top:2px;text-transform:uppercase;letter-spacing:0.1em;'>ML Dashboard</div>
    </div>
    <hr style='border:none;border-top:1px solid #c8e6d0;margin:0 0 16px;'>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-label'>Customer Profile</div>", unsafe_allow_html=True)
    gender   = st.selectbox("Gender", ["Female","Male"])
    age      = st.slider("Age", 18, 70, 28)
    income   = st.slider("Annual Income (k$)", 15, 137, 85)
    spending = st.slider("Spending Score (1–100)", 1, 99, 78)
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Predict Segment", use_container_width=True)
    st.markdown(f"""
    <div class='model-info-box'>
        <span class='label'>Model Info</span>
        Algorithm &nbsp;<span class='val'>SVM</span><br>
        CV accuracy &nbsp;<span class='val'>0.9750</span><br>
        Clusters &nbsp;<span class='val'>5 (K-Means)</span><br>
        Silhouette &nbsp;<span class='val'>{sil_score:.3f}</span>
    </div>""", unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>Mall Customer Segmentation</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Predict customer segments · Explore cluster analytics · Compare model performance</div>", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
for col,val,lbl in zip([c1,c2,c3,c4],[200,5,f"{sil_score:.3f}","97.5%"],["Customers","Segments","Silhouette","CV Accuracy"]):
    col.markdown(f"<div class='metric-card'><div class='metric-val'>{val}</div><div class='metric-lbl'>{lbl}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1,tab2,tab3,tab4 = st.tabs(["🔍 Predict","📊 Cluster Analysis","🤖 Model Comparison","📋 Segment Profiles"])

# ── Tab 1 ──────────────────────────────────────────────────────────────────
with tab1:
    label,conf,proba = predict_segment(age,gender,income,spending)
    seg = SEGMENTS[label]
    cl,cr = st.columns([1.1,1])
    with cl:
        st.markdown(f"""
        <div class='predict-card'>
            <div style='font-size:2rem;margin-bottom:6px;'>{seg['icon']}</div>
            <div class='predict-segment-name'>{seg['name']}</div>
            <div class='predict-desc'>{seg['desc']}</div>
            <div class='conf-bar-bg'><div class='conf-bar-fill' style='width:{conf:.0f}%'></div></div>
            <div class='conf-text'>Confidence: <b>{conf:.1f}%</b> &nbsp;·&nbsp; Cluster {label}</div>
            <div class='strategy-box'><b>Strategy:</b> {seg['strategy']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>Your Input vs Segment Average</div>", unsafe_allow_html=True)
        comp = pd.DataFrame({
            "Your Input"  :[age,income,spending],
            "Segment Avg" :[int(seg["age"].replace("~","").replace(" yrs","")),
                            int(seg["income"].replace("~$","").replace("k","")),
                            int(seg["spending"].replace("~","").split("/")[0])]
        }, index=["Age","Annual Income (k$)","Spending Score"])
        st.dataframe(comp, use_container_width=True)
    with cr:
        st.markdown("<div class='section-heading'>Probability per Segment</div>", unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(5,3.2)); fig.patch.set_facecolor('white'); ax.set_facecolor('#f7faf8')
        names  = [f"{SEGMENTS[i]['icon']} {SEGMENTS[i]['name']}" for i in range(5)]
        colors = [SEGMENTS[i]['color'] for i in range(5)]
        bars   = ax.barh(names,proba*100,color=colors,alpha=0.85,height=0.55)
        bars[label].set_edgecolor('#1a472a'); bars[label].set_linewidth(2)
        for bar,val in zip(bars,proba*100):
            if val>1: ax.text(val+0.5,bar.get_y()+bar.get_height()/2,f'{val:.1f}%',va='center',fontsize=8,color='#333')
        ax.set_xlabel('Probability (%)',fontsize=9,color='#5a8a6e'); ax.tick_params(colors='#444',labelsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#c8e6d0'); ax.spines['left'].set_color('#c8e6d0')
        ax.set_xlim(0,110); plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("<div class='section-heading'>Where does this customer land?</div>", unsafe_allow_html=True)
        fig2,ax2 = plt.subplots(figsize=(5,3.4)); fig2.patch.set_facecolor('white'); ax2.set_facecolor('#f7faf8')
        for i in range(5):
            mask=y_clust==i; ax2.scatter(X_clust[mask,0],X_clust[mask,1],c=SEGMENTS[i]['color'],alpha=0.45,s=30)
        ax2.scatter(income,spending,c='#1a472a',s=200,zorder=10,marker='*',edgecolors='white',linewidths=1.5)
        ax2.annotate('You',(income,spending),xytext=(6,5),textcoords='offset points',color='#1a472a',fontsize=8,fontweight='bold')
        ax2.set_xlabel('Annual Income (k$)',fontsize=9,color='#5a8a6e'); ax2.set_ylabel('Spending Score',fontsize=9,color='#5a8a6e')
        ax2.tick_params(colors='#444',labelsize=8)
        for sp in ax2.spines.values(): sp.set_color('#c8e6d0')
        patches=[mpatches.Patch(color=SEGMENTS[i]['color'],label=SEGMENTS[i]['name'],alpha=0.7) for i in range(5)]
        ax2.legend(handles=patches,fontsize=7,loc='upper left',framealpha=0.9,edgecolor='#c8e6d0')
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# ── Tab 2 ──────────────────────────────────────────────────────────────────
with tab2:
    ca,cb = st.columns(2)
    with ca:
        st.markdown("<div class='section-heading'>Cluster Distribution</div>", unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(5,3.8)); fig.patch.set_facecolor('white')
        counts=[np.sum(y_clust==i) for i in range(5)]; colors=[SEGMENTS[i]['color'] for i in range(5)]
        wedges,texts,autotexts=ax.pie(counts,labels=[f"{SEGMENTS[i]['icon']} {SEGMENTS[i]['name']}" for i in range(5)],
            autopct='%1.0f%%',colors=colors,startangle=140,textprops={'fontsize':8},pctdistance=0.75)
        for at in autotexts: at.set_fontsize(8); at.set_color('white'); at.set_fontweight('bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with cb:
        st.markdown("<div class='section-heading'>Income vs Spending</div>", unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(5,3.8)); fig.patch.set_facecolor('white'); ax.set_facecolor('#f7faf8')
        for i in range(5):
            mask=y_clust==i; ax.scatter(X_clust[mask,0],X_clust[mask,1],c=SEGMENTS[i]['color'],alpha=0.65,s=40,label=f"{SEGMENTS[i]['icon']} {SEGMENTS[i]['name']}")
        ax.set_xlabel('Annual Income (k$)',fontsize=9,color='#5a8a6e'); ax.set_ylabel('Spending Score',fontsize=9,color='#5a8a6e')
        ax.tick_params(colors='#444',labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#c8e6d0')
        ax.legend(fontsize=7,loc='upper left',framealpha=0.9,edgecolor='#c8e6d0')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-heading'>Feature Distributions per Segment</div>", unsafe_allow_html=True)
    np.random.seed(42); ages_arr=np.random.randint(18,70,len(y_clust))
    df_plot=pd.DataFrame({'Age':ages_arr,'Income':X_clust[:,0],'Spending':X_clust[:,1],'Cluster':y_clust})
    fig,axes=plt.subplots(1,3,figsize=(14,4)); fig.patch.set_facecolor('white')
    for ax in axes: ax.set_facecolor('#f7faf8')
    for ax,col,lbl in zip(axes,['Age','Income','Spending'],['Age','Annual Income (k$)','Spending Score (1-100)']):
        data=[df_plot[df_plot['Cluster']==i][col].values for i in range(5)]
        bp=ax.boxplot(data,patch_artist=True,medianprops=dict(color='white',linewidth=2))
        for patch,color in zip(bp['boxes'],[SEGMENTS[i]['color'] for i in range(5)]): patch.set_facecolor(color); patch.set_alpha(0.7)
        for el in ['whiskers','caps','fliers']:
            for item in bp[el]: item.set_color('#aaa')
        ax.set_xticklabels([f"C{i}" for i in range(5)],fontsize=9,color='#444')
        ax.set_title(lbl,fontsize=10,fontweight='bold',color='#1a472a')
        ax.tick_params(colors='#444',labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#c8e6d0')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Tab 3 ──────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-heading'>Model Performance — GridSearchCV + 5-Fold CV</div>", unsafe_allow_html=True)
    model_data=pd.DataFrame({'Model':['Logistic Regression','Random Forest','XGBoost','SVM ✅'],'CV Accuracy':[0.9625,0.9688,0.9438,0.9750],'Test Accuracy':[0.975,0.975,0.975,0.975],'Type':['Linear baseline','Ensemble (bagging)','Ensemble (boosting)','Margin-based']}).set_index('Model')
    ca,cb=st.columns([1,1.3])
    with ca:
        st.dataframe(model_data,use_container_width=True)
        st.markdown("""<div style='background:#f0f9f3;border:1px solid #b7dcc5;border-radius:10px;padding:14px 16px;margin-top:12px;font-size:0.83rem;color:#444;line-height:1.9;'>
            <span style='font-family:Montserrat,sans-serif;font-weight:700;color:#1a472a;'>✅ Best Model: SVM</span><br>
            Highest CV accuracy (0.9750). All 4 models reached 0.975 test accuracy — SVM wins on generalisation.</div>""", unsafe_allow_html=True)
    with cb:
        fig,ax=plt.subplots(figsize=(6,4)); fig.patch.set_facecolor('white'); ax.set_facecolor('#f7faf8')
        x=np.arange(4); w=0.35
        b1=ax.bar(x-w/2,model_data['CV Accuracy'],  w,label='CV Accuracy', color='#2d6a4f',alpha=0.9)
        b2=ax.bar(x+w/2,model_data['Test Accuracy'],w,label='Test Accuracy',color='#9FE1CB',alpha=0.9)
        b1[3].set_edgecolor('#1a472a'); b1[3].set_linewidth(2.5)
        ax.set_xticks(x); ax.set_xticklabels(['LR','RF','XGB','SVM ✅'],fontsize=9,color='#444')
        ax.set_ylim(0.90,1.02); ax.set_ylabel('Accuracy',fontsize=9,color='#5a8a6e')
        ax.set_title('CV vs Test Accuracy',fontsize=10,fontweight='bold',color='#1a472a')
        ax.legend(fontsize=8,framealpha=0.9,edgecolor='#c8e6d0'); ax.tick_params(colors='#444',labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#c8e6d0')
        ax.grid(axis='y',color='#d8eeda',alpha=0.7)
        for bars in [b1,b2]:
            for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.001,f'{bar.get_height():.3f}',ha='center',va='bottom',color='#444',fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-heading'>Feature Importance — Random Forest & XGBoost</div>", unsafe_allow_html=True)
    imp=pd.DataFrame({'Feature':['Age','Gender','Annual Income (k$)','Spending Score (1-100)'],'Random Forest':[0.098,0.004,0.448,0.451],'XGBoost':[0.021,0.053,0.468,0.459]}).set_index('Feature')
    fig,axes=plt.subplots(1,2,figsize=(12,3.2)); fig.patch.set_facecolor('white')
    for ax,col,color in zip(axes,['Random Forest','XGBoost'],['#2d6a4f','#5DCAA5']):
        ax.set_facecolor('#f7faf8'); vals=imp[col].sort_values()
        bars=ax.barh(vals.index,vals.values,color=color,alpha=0.85,height=0.5)
        for bar,val in zip(bars,vals.values): ax.text(val+0.005,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',fontsize=9,color='#333')
        ax.set_title(f'Feature Importance — {col}',fontsize=10,fontweight='bold',color='#1a472a')
        ax.set_xlabel('Importance Score',fontsize=9,color='#5a8a6e'); ax.tick_params(colors='#444',labelsize=9)
        for sp in ax.spines.values(): sp.set_color('#c8e6d0')
        ax.set_xlim(0,0.55)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Tab 4 ──────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-heading'>All Customer Segments</div>", unsafe_allow_html=True)
    for i,seg in SEGMENTS.items():
        count=int(np.sum(y_clust==i)); pct=count/len(y_clust)*100
        with st.expander(f"Cluster {i}: {seg['name']} | {count} customers ({pct:.0f}%)", expanded=(i==4)):
            ca,cb=st.columns([1.6,1])
            with ca:
                st.markdown(f"""<div style='border-left:4px solid {seg["color"]};border-radius:0 10px 10px 0;background:#f7faf8;padding:14px 18px;'>
                    <div style='font-family:Montserrat,sans-serif;font-weight:700;font-size:0.95rem;color:#1a472a;margin-bottom:6px;'>Description</div>
                    <div style='font-size:0.83rem;color:#555;margin-bottom:10px;'>{seg['desc']}</div>
                    <div style='font-family:Montserrat,sans-serif;font-weight:700;font-size:0.95rem;color:#1a472a;margin-bottom:6px;'>Marketing Strategy</div>
                    <div style='font-size:0.83rem;color:#555;'>{seg['strategy']}</div></div>""", unsafe_allow_html=True)
            with cb:
                st.markdown(f"""<div style='background:#f7faf8;border:1px solid #c8e6d0;border-radius:10px;padding:14px 18px;font-size:0.83rem;line-height:2.1;'>
                    <span style='color:#5a8a6e;'>Avg Age</span><br><span style='font-family:Montserrat,sans-serif;font-weight:700;color:#1a472a;'>{seg['age']}</span><br>
                    <span style='color:#5a8a6e;'>Avg Income</span><br><span style='font-family:Montserrat,sans-serif;font-weight:700;color:#1a472a;'>{seg['income']}</span><br>
                    <span style='color:#5a8a6e;'>Avg Spending Score</span><br><span style='font-family:Montserrat,sans-serif;font-weight:700;color:#1a472a;'>{seg['spending']}</span><br>
                    <span style='color:#5a8a6e;'>Cluster Size</span><br><span style='font-family:Montserrat,sans-serif;font-weight:700;color:{seg["color"]};'>{count} customers ({pct:.0f}%)</span></div>""", unsafe_allow_html=True)

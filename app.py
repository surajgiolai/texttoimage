import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from PIL import Image
import torch

from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForTokenClassification, 
    AutoModelForMaskedLM,
    pipeline
)

# --- Title and Layout ---
st.set_page_config(page_title="Environmental NLP & Image App", layout="wide")
st.title("🌱 Environmental NLP & Image Generation Suite")

# --- Load Models ---
@st.cache_resource
def load_classification_model():
    model_name = "bhadresh-savani/bert-base-uncased-emotion"  # You can fine-tune your own for prod
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier

@st.cache_resource
def load_ner_model():
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner

@st.cache_resource
def load_mask_model():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    return fill_mask

classifier = load_classification_model()
ner = load_ner_model()
fill_mask = load_mask_model()

# --- Sidebar ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Sentence Classification", "Image Generation", "NER & Graph Map", "Fill the Blank"]
)

# --- 1. Sentence Classification ---
if page == "Sentence Classification":
    st.header("🔎 Sentence Classification (Environmental Departments)")
    st.markdown("""
    - *Categories:* Waste, Water, Air, Energy, Biodiversity
    - Enter a sentence to classify it into an environmental department.
    """)

    user_text = st.text_area("Enter a sentence about the environment:", "")
    if st.button("Classify"):
        if user_text.strip():
            result = classifier(user_text, top_k=3)
            st.write("*Top Predictions:*")
            for r in result:
                st.write(f"- {r['label']} ({r['score']:.2%})")
        else:
            st.warning("Please enter a sentence.")

# --- 2. Image Generation ---
elif page == "Image Generation":
    st.header("🖼 Image Generation")
    st.markdown("""
    - Enter an environmental prompt (e.g., "A clean river in a forest").
    - Generates an image using Stable Diffusion.
    """)
    prompt = st.text_input("Image Prompt", "A clean river in a forest")
    if st.button("Generate Image"):
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            @st.cache_resource
            def load_sd():
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
                )
                pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
                return pipe
            sd_pipe = load_sd()
            with st.spinner("Generating image..."):
                image = sd_pipe(prompt).images[0]
                st.image(image, caption=prompt)
        except Exception as e:
            st.error("Stable Diffusion not available. Please run with GPU and install diffusers.")

# --- 3. NER & Graph Map ---
elif page == "NER & Graph Map":
    st.header("🕸 Named Entity Recognition & Graph Mapping")
    st.markdown("""
    - Extracts entities (ORG, LOC, etc.) from your text.
    - Visualizes relationships as a graph.
    """)
    ner_text = st.text_area("Enter environmental text for NER:", 
        "The Ministry of Environment and Forests monitors air quality in Delhi and Mumbai.")
    if st.button("Extract Entities and Visualize"):
        entities = ner(ner_text)
        st.write("*Entities:*")
        ent_df = pd.DataFrame(entities)
        st.dataframe(ent_df[["entity_group", "word", "score"]])

        # Simple graph: connect entities in order of appearance
        G = nx.Graph()
        for ent in entities:
            G.add_node(ent["word"], label=ent["entity_group"])
        for i in range(len(entities)-1):
            G.add_edge(entities[i]["word"], entities[i+1]["word"])

        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node} ({G.nodes[node]['label']})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines'))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers+text', marker=dict(size=20, color='skyblue'),
            text=node_text, textposition='top center'
        ))
        fig.update_layout(showlegend=False, title="Entity Graph Map", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# --- 4. Fill the Blank ---
elif page == "Fill the Blank":
    st.header("📝 Fill the Blank (Masked Language Model)")
    st.markdown("""
    - Enter a sentence with [MASK] (e.g., "The [MASK] is polluted due to industrial waste.").
    - The model predicts the masked word.
    """)
    mask_text = st.text_input("Enter sentence with [MASK]:", "The [MASK] is polluted due to industrial waste.")
    if st.button("Predict Mask"):
        if "[MASK]" not in mask_text:
            st.warning("Please include [MASK] in your sentence.")
        else:
            results = fill_mask(mask_text)
            st.write("*Top Predictions:*")
            for r in results:
                st.write(f"- {r['sequence']} (score: {r['score']:.2%})")

import streamlit as st
from transformers import pipeline
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load models
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
ner = spacy.load("en_core_web_sm")

st.title("🌱 GreenAI: Environmental Intelligence Dashboard")
st.markdown("Empowering climate solutions with AI")

tabs = st.tabs(["Fill in the Blank", "Named Entity Graph"])

# Tab 1: Fill in the Blank
with tabs[0]:
    st.header("🔎 Fill in the Missing Word")
    sentence = st.text_input("Enter a masked sentence (e.g., 'Trees absorb <mask> from the atmosphere.')",
                             value="Trees absorb <mask> from the atmosphere.")
    if st.button("Predict"):
        result = fill_mask(sentence)
        for r in result:
            st.write(r['sequence'])

# Tab 2: Named Entity Graph
with tabs[1]:
    st.header("🌍 Named Entity Relationship Graph")
    text = st.text_area("Enter a sentence", value="NASA collaborated with ISRO to monitor deforestation in the Amazon rainforest.")
    if st.button("Generate Graph"):
        doc = ner(text)
        G = nx.Graph()
        for ent in doc.ents:
            G.add_node(ent.text + f" ({ent.label_})")
        for i in range(len(doc.ents)-1):
            G.add_edge(doc.ents[i].text + f" ({doc.ents[i].label_})", doc.ents[i+1].text + f" ({doc.ents[i+1].label_})")

        pos = nx.spring_layout(G)
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, ax=ax)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(Image.open(buf))


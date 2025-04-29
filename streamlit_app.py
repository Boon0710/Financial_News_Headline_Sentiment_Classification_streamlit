import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
from huggingface_hub import hf_hub_download

# === Load word segmentation model ===
@st.cache_resource
def load_segmenter():
    seg_tokenizer = AutoTokenizer.from_pretrained("NlpHUST/vi-word-segmentation")
    seg_model = AutoModelForTokenClassification.from_pretrained("NlpHUST/vi-word-segmentation")
    return pipeline("token-classification", model=seg_model, tokenizer=seg_tokenizer)

def segment_sentence(text, segmenter):
    tokens = segmenter(text)
    result = ""
    for e in tokens:
        if "##" in e["word"]:
            result += e["word"].replace("##", "")
        elif e["entity"] == "I":
            result += "_" + e["word"]
        else:
            result += " " + e["word"]
    return result.strip()

# === Define sentiment model architecture ===
class CustomPhoBERTClassifier(torch.nn.Module):
    def __init__(self, phobert):
        super().__init__()
        self.phobert = phobert
        self.dropout = torch.nn.Dropout(0.3)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(768 * 4, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 3)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-4:]
        cls_output = torch.cat([h[:, 0, :] for h in hidden_states], dim=-1)
        cls_output = self.dropout(cls_output)
        logits = self.mlp(cls_output)
        return logits

@st.cache_resource
def load_sentiment_model():
    model_name = "Boon0710/phoBert-based-financial-news-sentiment-classification"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    phobert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model = CustomPhoBERTClassifier(phobert)
    weight_path = hf_hub_download(repo_id="Boon0710/phoBert-based-financial-news-sentiment-classification-v2", filename="pytorch_model.bin")
    model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=False)
    model.eval()
    return tokenizer, model

def predict_sentiment(sentence, segmenter, tokenizer, model):
    segmented = segment_sentence(sentence, segmenter)
    inputs = tokenizer(segmented, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    label_map = {0: "neg", 1: "neu", 2: "pos"}
    return segmented, label_map[pred_idx], f"{confidence:.2%}"

# === Streamlit App ===
st.title("📊 Vietnamese Financial Sentiment Analyzer")

# Load models with spinner
with st.spinner('Loading models... please wait ⏳'):
    segmenter = load_segmenter()
    tokenizer, model = load_sentiment_model()

input_mode = st.radio("Choose input mode:", ["Single Sentence", "CSV Upload"])

if input_mode == "Single Sentence":
    sentence = st.text_area("Enter a Vietnamese financial news headline:")
    if st.button("Analyze"):
        if sentence.strip():
            segmented, sentiment, confidence = predict_sentiment(sentence, segmenter, tokenizer, model)
            st.write(f"**Segmented:** {segmented}")
            st.write(f"**🧠 Sentiment:** {sentiment.upper()}")
            st.write(f"**🔍 Confidence:** {confidence}")
        else:
            st.warning("Please enter a sentence to analyze.")

else:
    uploaded_file = st.file_uploader("Upload a CSV file with a `Title` column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Title" not in df.columns:
            st.error("The CSV file must contain a 'Title' column.")
        else:
            progress = st.progress(0)
            segmented_titles = []
            sentiments = []
            confidences = []

            for i, title in enumerate(df["Title"]):
                segmented, sentiment, confidence = predict_sentiment(title, segmenter, tokenizer, model)
                segmented_titles.append(segmented)
                sentiments.append(sentiment)
                confidences.append(confidence)
                progress.progress((i + 1) / len(df))

            df["Segmented_Title"] = segmented_titles
            df["Sentiment"] = sentiments
            df["Confidence Percent"] = confidences

            st.success("✅ Analysis complete!")

            st.subheader("🔎 Preview of First 5 Predictions")
            st.dataframe(df.head(5))

            st.subheader("📊 Sentiment Distribution")
            fig1, ax1 = plt.subplots()
            df["Sentiment"].value_counts().plot(kind="bar", ax=ax1)
            ax1.set_xlabel("Sentiment")
            ax1.set_ylabel("Count")
            ax1.set_title("Sentiment Label Distribution")
            st.pyplot(fig1)

            st.subheader("🎯 Confidence Score Distribution")
            confidence_values = df["Confidence Percent"].str.rstrip('%').astype(float)
            fig2, ax2 = plt.subplots()
            ax2.hist(confidence_values, bins=10, edgecolor='black')
            ax2.set_xlabel("Confidence (%)")
            ax2.set_ylabel("Number of Samples")
            ax2.set_title("Prediction Confidence Distribution")
            st.pyplot(fig2)

            st.subheader("📄 Full Results")
            st.dataframe(df)

            st.download_button(
                label="📥 Download Results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

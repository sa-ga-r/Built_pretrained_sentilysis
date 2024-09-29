import streamlit as st 
import torch
import torch.nn.functional as F 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class sentilysis():
    def __init__(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)

    def process_sentiment(self, user_input):
        input = self.tokenizer(user_input, return_tensors = 'pt')
        with torch.no_grad():
            logits = self.model(**input).logits
            probs = F.softmax(logits, dim=-1)
            predicted_class_id = logits.argmax().item()
            label = self.model.config.id2label[predicted_class_id]
            neg_prob = round(probs[0][0].item()*100)
            pos_prob = round(probs[0][1].item()*100)
            return {
                "Analysis":"Probability",
                "Positive":pos_prob,
                "Negetive":neg_prob,
            }
        
def main():
    analizer = sentilysis()
    st.title("Sentiment Analysis")
    user_input = st.text_input("Enter sentence...")
    if st.button("Analyse"):
        result = analizer.process_sentiment(user_input)
        st.data_editor(result, column_config={
            "Positive":st.column_config.ProgressColumn(
                "Analysis result", width = "large", format="%2f", min_value=0, max_value=100),    
            "Negetive":st.column_config.ProgressColumn(
                "Analysis result", width = "large", format="%2f", min_value=0, max_value=100),
        }, hide_index=False,)

if __name__ == '__main__':
    main()

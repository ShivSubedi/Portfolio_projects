import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt



# #Load vader model once 
vader_analyzer=SentimentIntensityAnalyzer()

#load tokenizer and model once
hf_3lcass_model_name= "cardiffnlp/twitter-roberta-base-sentiment"
hf_2class_model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

hf_tokenizer=AutoTokenizer.from_pretrained(hf_3lcass_model_name)
hf_model = AutoModelForSequenceClassification.from_pretrained(hf_3lcass_model_name)

# DistilBERT: Binary classffier(Positive, negative), a lightweight, faster version of BERT (~60% speedup, ~40% smaller) with ~97% of BERT's performance — ideal for real-time tasks, but may miss subtle context compared to full BERT or domain-specific models.
distilbert_pipeline=pipeline('sentiment-analysis', model=hf_2class_model_name)
cardiff_pipeline=pipeline('sentiment-analysis', model=hf_model, tokenizer=hf_tokenizer)

#Label mapping from cardiff model output w.r.t grount truth representation
label_map={
    "LABEL_0":"negative",
    "LABEL_1":"neutral",
    "LABEL_2":"positive"
}

#==============TextBlob====================
def extract_textblob_sentiment(text):
    """
    Analyze the sentiment of a text message using Textblob.
    Args: 
        text(str): a message or sentence to analyze
    Returns:
        tuple: polarity = [-1.0, 1.0], shows how negative (denotes negative sentiment) or positive (denotes positive sentiment) a text is. 
               subjectivity= [0.0, 1.0], shows how opinionated the text is, i.e. assigns 0.0 for very factual and 1.0 for very subjective/opinion-based
               sentiment label (positive, neutral, negative)
extract_vader_sentiment(text):    """

    # Create a TextBlob object from the input text
    blob=TextBlob(text)

    # Get the polarity (positive/negative sentiment score)
    polarity = blob.sentiment.polarity

    # Get the subjectivity (how opinionated the text is)
    subjectivity = blob.sentiment.subjectivity

    #classification based on polarity thresholds
    if polarity > 0.1:
        label="positive"
    elif polarity<-0.01:
        label="negative"
    else:
        label = "neutral"

    # Return both values as a tuple
    return polarity, subjectivity, label


def extract_vader_sentiment(text):
    """
    Analyze the sentiment of a message using VADER (rule-based sentiment analysis)

    Args: 
        text(str): a message or sentence to analyze
    Returns: tuple
        compound: [-1.0, 1.0] reflects the overall sentiment, >0.05=> (+)ve,< -0.05 => (-)ve, -0.05 to 0.05 => neutral
        pos: proportional of (+)ve words in a sentence
        neu: proportional of (-ve) words in a sentence
        neg: proportional of (-ve) words in a sentence
        label: sentiment label (positive, neutral, negative)
    """

    # Get sentiment scores for the text
    scores = vader_analyzer.polarity_scores(text)

    #Extract each score independently
    compound = scores['compound']
    pos=scores['pos']
    neu=scores['neu']
    neg=scores['neg']

    #classify sentiment label based on compound score
    if compound>=0.05:
        label='positive'
    elif compound<=-0.05:
        label='negative'
    else:
        label='neutral'

    #Return all four scores
    return compound, pos, neu, neg, label



#=============huggingface================
def extract_binary_sentiment_distilbert(text):
    """
    Analyze the sentiment of a message using a HuggingFace model.
    Args: 
        text (str): Input text message.
    Returns:
        label: sentiment category (i.e. 'positive' or 'negative')
        score: confidence score as a float
    """
    # Use the pre-loaded sentiment analysis pipeline on the input text
    result_list = distilbert_pipeline(text)

    # The result is a list with one dictionary, so we get the first item
    result = result_list[0]

    # Get the sentiment label (e.g., POSITIVE or NEGATIVE)
    label = result["label"]

    # Get the confidence score and convert it to a float
    score = float(result["score"])

    # Return both values
    return label, score

#=============huggingface================
def extract_3class_sentiment_cardiff(text):
    """
    Analyze the sentiment of a message using a HuggingFace model.
    Args: 
        text (str): Input text message.
    Returns: tuple i.e: (label, confidence score)
        label: is 'positive' or 'neutral, or 'negative'
    """
    # Use the pre-loaded sentiment analysis pipeline on the input text
    result_list = cardiff_pipeline(text)[0]
    label=label_map[result_list['label']]
    score=float(result_list['score'])

    # Return both values
    return label, score




# LOCAL TEST OF EACH OF ABOVE MODELS FOR SENTIMENT ANALYSIS (optional), 
if __name__=="__main__": #runs the following test if 'nlp_utils.py' is executed locally(directly) and not imported from another file
    print("The 'nlp_utils.py' script is running locally to test its output")
    
    #////////////////////////////////////////////////////////////////////////////////////////
    #Note: Use this section only when there is need to pair messages to telco dataset
    #=============================SECTION START=============================================
    # import sys
    # sys.path.append("../src") # Adding src/ to the module search path
    # from data_loader import load_data
    # from interaction_generator import generate_interaction_logs
    # #step1: load the telco base data set
    # df_telco=load_data(filepath="../data/telco_churn.csv")
    # customers_ids= df_telco["customerID"].tolist()

    # #step2: generate synthetic messages per customer (5 messages per customer)
    # df_logs= generate_interaction_logs(customers_ids, messages_per_customer=5)

    # print("=======Comparing sentiment models============")

    # # === Analyze First N Messages (or sample from a customer) ===
    # num_rows=3
    # messages_to_analyze = df_logs['message'].head(num_rows)
    # true_tone= df_logs['tone'].head(num_rows)
    # # print(true_tone)
    
    # print(df_logs.head(num_rows))
    # print('--------------------------------------------------------------------------')
    #=============================SECTION STOP=============================================
    #////////////////////////////////////////////////////////////////////////////////////////


    #----------------Step 1: load the messages from JSON file---------------------------
    def load_message(filepath="../data/sample_15message.json"):
        """
        Loads the json file for sample customer messages
        Args:
            filepath of the json data
        Return:
            dataFrame with sample message and corresponding ground truth tone
        """
        with open(filepath, "r") as file:
            data=json.load(file)
            #initialize lists to store each message adn its corresponding tone label
            messages =[]
            tones=[]

            for tone in ["positive", "neutral", "negative"]:
                for msg in data[tone]["examples"]:
                    messages.append(msg)
                    tones.append(tone)

            #create a dictionary from the lists
            message_dict={
                "message":messages,
                "Ground_truth_tone":tones
            }
        return  pd.DataFrame(message_dict)


    df_sample_messages=load_message()
    print(df_sample_messages.head(5))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    messages_to_analyze=df_sample_messages["message"]
    true_tone=df_sample_messages["Ground_truth_tone"]

    hf_label_pred=[]
    # hf_scores_pred=[]
    vader_label_pred=[]
    tb_label_pred=[]

    #loop through each row in the 'message' column of 'df_logs' data frame
    for i, message in enumerate (messages_to_analyze):
        # Textblob sentiment analysis
        tb_polarity, tb_subjectivity, tb_label= extract_textblob_sentiment(message)
        #Vader sentiment analysis
        vader_comp, vader_pos, vader_neu, vader_neg, vader_label= extract_vader_sentiment(message)
        #huggingface sentiment analysis
        hf_label, hf_score = extract_3class_sentiment_cardiff(message) #assign the function for hugging face

        # print(f"Message{i+1}: {message}, Grount truth tone: {true_tone.iloc[i]}")
        # print(f"TextBlob sentiment analysis: Polarity= {tb_polarity:.2f}, Subjectivity = {tb_subjectivity:.2f}, label={tb_label}")
        # print(f"Vader sentiment analysis: Compound = {vader_comp:.2f}, Positive = {vader_pos:.2f}, Neutral = {vader_neu:.2f}, Negative = {vader_neg}, label = {vader_label}")
        # print(f"Huggingface sentiment analysis: Label = {hf_label}, Score = {hf_score:.2f}")
        # print('//////////////////////////////////////////////////////////////////////')

        hf_label_pred.append(hf_label.lower()) #changes the prediction to lowercase
        # hf_scores_pred.append(hf_score)
        vader_label_pred.append(vader_label)
        tb_label_pred.append(tb_label)



    #Add predictions to the original dataFrame
    df_sample_messages["CardiffNLP_label_pred"]=hf_label_pred
    # df_sample_messages["hf_scores_pred"]=hf_scores_pred
    df_sample_messages['Vader_label_pred']=vader_label_pred
    df_sample_messages['TextBlob_label_pred']=tb_label_pred

    print(df_sample_messages)
    df_sample_messages.to_csv("../output/model_tone_prediction_summary.csv", index=False)


    #////////////Evaluate the models///////////////////////
    def plot_per_tone_confusion_matrix(df, tone, model_col):
        """
        Plot a binary confusion matrix (one-vs-rest) for a specific tone and model.
        """
        
        y_true = (df["Ground_truth_tone"] == tone).astype(int)
        y_pred = (df[model_col] == tone).astype(int)

        labels = [0, 1]  # 0 = "other", 1 = "target tone"
        disp_labels = ["Other", tone.capitalize()]

        conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=disp_labels)

        disp.plot(cmap="Blues")
        if model_col == 'hf_label_pred':
            fig_title='CardiffNLP prediction'
        elif model_col == 'vader_label_pred':
            fig_title='VADER prediction'
        else:
            fig_title='TextBlob prediction'

        plt.title(f"{fig_title} | Confusion Matrix for '{tone.capitalize()}'")
        # plt.show() #this doesn't work in codespace, since it doesn't support graphical interface.
        

        # Save instead of showing
        filename = f"../output/confusion_matrix_{fig_title}_{tone.capitalize()}.png"
        plt.savefig(filename)
        print(f"Confusion matrix saved to: {filename}")
        plt.close()


        #Evaluate the metrics
        tn, fp, fn, tp=conf_mat.ravel()

        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)        # Overall correctness of predictions
        precision = tp / (tp + fp) if (tp + fp) else 0    # How reliable positive predictions are
        recall = tp / (tp + fn) if (tp + fn) else 0       # How well actual positives were identified
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0 # Balances precision and recall
        fpr = fp / (fp + tn) if (fp + tn) else 0 # Rate of false alarms among actual negatives
        fnr = fn / (fn + tp) if (fn + tp) else 0 # Rate of missed detections among actual positives

        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(f"Metrics (Tone: {tone.capitalize()}, {fig_title})")
        # print(f"True Positives (TP): {tp}")
        # print(f"False Positives (FP): {fp}")
        # print(f"False Negatives (FN): {fn}")
        # print(f"True Negatives (TN): {tn}")
        # print(f"Accuracy (Pred. how Correct?):     {accuracy:.3f}")
        # print(f"Precision (Pred. how Reliabile?):    {precision:.3f}")
        # print(f"Recall (Pred. how Sensitive?):       {recall:.3f}")
        # print(f"F1 Score:     {f1:.3f}")
        # print(f"FPR (Type I Error i.e. False Alarms):  {fpr:.3f}")
        # print(f"FNR (Type II Error i.e. Missed Detections): {fnr:.3f}")
        print("////////////////////////////////////////////////////////////////////")


        return{
            "Model":fig_title,
            "Tone": tone.capitalize(),
            "TP":tp,
            "FP":fp,
            "FN":fn,
            "TN":tn,
            "Accuracy":round(accuracy, 3),
            "Precision":round(precision, 3),
            "Recall":round(recall, 3),
            "F1 score": round(f1,3),
            "FPR (Type I Error)": round(fpr,3),
            "FNR (Type II Error)":round(fnr,3)
        }
       
    
    # tones_message=['neutral']
    tones_message=['positive', 'neutral', 'negative']
    models_investigated={'CardiffNLP_label_pred', 'Vader_label_pred', 'TextBlob_label_pred'}
    # models_investigated= {
    #     'CardiffNLP': 'hf_label_pred', 
    #     'VADER':'vader_label_pred', 
    #     'TextBlob':'tb_label_pred'}

    results=[]
    for tone in tones_message:
        for model_name in models_investigated:
            print(f'Evaluating {model_name} of tone: "{tone}"')
            metrics = plot_per_tone_confusion_matrix(df_sample_messages, tone, model_name)
            results.append(metrics)
    
    df_metrics = pd.DataFrame(results)
    print(df_metrics)
    # df_metrics.to_csv("../output/model_tone_metrics_summary.csv", index=False)

    # plot_per_tone_confusion_matrix(df_sample_messages, "neutral", "hf_label_pred")
    # print([y_true, y_pred])
    



    



    








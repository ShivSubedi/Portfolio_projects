#This code creates a simulated dataset of customer support interactions
#It pulls the 'telco_dataset' and generates 'fake messages' with a (random) tone and timestamp for each customer, imitating real-world support logs.

#WHAT THE CODE DOES?
# 1. Takes a list of customer IDs.
# 2. For each customer:
#   - Generates messages_per_customer number of fake messages.
#   - Randomly assigns a tone (positive, neutral, or negative) using weighted probabilities.
#   - Picks a message corresponding to the tone.
#   - Assigns a random timestamp within the past N (defined in the function) months.
# 3. Stores all the interactions in a list of dictionaries.
# 4. Converts the list into a pandas DataFrame, sorts by customer and time, and returns it.

#HOW IS THIS CODE USEFUL?
#Prepares data for NLP (sentiment analysis)
#Adds behavioral context for time-series modeling
#Enables customer clustering based on message patterns
#Complements the Telco churn dataset for deeper feature engineering

import pandas as pd
import random 
from datetime import datetime, timedelta
import json ## Import the json module to work with JSON files

# # Sample messages for different tones
# POSITIVE_MESSAGES = [
#     "Thanks for your excellent service!",
#     "App is working great, no issues.",
#     "Very happy with my internet speed!",
#     "Billing was accurate, appreciated.",
#     "Customer support was helpful today."
# ]

# NEUTRAL_MESSAGES = [
#     "Had to reset my router today.",
#     "Need some help with my billing info.",
#     "Where can I find my plan details?",
#     "Please update me on my ticket status.",
#     "Checking for available upgrade options."
# ]

# NEGATIVE_MESSAGES = [
#     "Still no resolution to my issue!",
#     "Internet has been slow for days.",
#     "Your service is disappointing.",
#     "Billing error again this month!",
#     "Support was unhelpful and rude."
# ]



# Open the JSON file that contains the customer messages
with open("../data/sample_15message.json", "r") as file:
    message_data=json.load(file) # Load JSON from a file and convert it to a Python object (e.g., dict or list)


# Get the list of example positive, neutral, and negative messages
POSITIVE_MESSAGES=message_data["positive"]["examples"]
NEUTRAL_MESSAGES=message_data["neutral"]["examples"]
NEGATIVE_MESSAGES=message_data["negative"]["examples"]

# #Confirm that we have loaded the messages from json
# print("\nPositive Messages:")
# print(json.dumps(POSITIVE_MESSAGES, indent=1)) #Dump a Python object (dict, list, etc.) as a JSON-formatted string



def generate_interaction_logs(customer_ids, messages_per_customer=1, month=6, seed=42):
    """
    This function is designed to simulate interaction logs per customer.
    Args:
        customer_ids(list): list of the customerIDs
        messages_per_customer(int): #of messages simulated per customer. This implies each customer will have assigned #of messages, timestamps, and tone
        month(int): how far back (in months) to simulate the customer interaction data
        seed(int): random seed taken for reproducability of data 
    Return:
        df_logs: pd.dataFrame i.e. a data frame with log of customer interaction having customerID, timestamp, and simulated message

    """
    # Set the random seed
    random.seed(seed)

    # Create a list to store all the messages
    logs = []

    # Go through each customer
    for customer_id in customer_ids:
        for i in range(messages_per_customer):
            # Pick a tone: positive, neutral, or negative
            tone_list = ["positive", "neutral", "negative"]
            #randomly picks a tone based on the defined weight distribution and stores it in the tone variable.
            #weights define how likely each tone is to be picked, here 20%->+ve, 50%->neutral, and 30%->negative
            #if we run the code long enough, half of the results will be 'neutral', then 'negative', and 'positive'
            #random.choices() returns a list and [0] picks the first element from the list
            tone = random.choices(tone_list, weights=[0.2, 0.5, 0.3])[0]

            # Pick a message based on the tone
            if tone == "positive":
                message = random.choice(POSITIVE_MESSAGES)
            elif tone == "neutral":
                message = random.choice(NEUTRAL_MESSAGES)
            else:
                message = random.choice(NEGATIVE_MESSAGES)

            # Pick a random day within the last few months
            total_days = month * 30
            days_ago = random.randint(1, total_days)
            message_time = datetime.now() - timedelta(days=days_ago)
            timestamp = message_time.strftime("%Y-%m-%d %H:%M:%S")

            # Add this interaction to the list
            logs.append({
                "customerID": customer_id,
                "timestamp": timestamp,
                "message": message,
                "tone": tone #optional: for later NLP evaluation
            })

    # Turn the list into a DataFrame and sort it
    df_logs = pd.DataFrame(logs)
    df_logs = df_logs.sort_values(by=["customerID", "timestamp"])

    # df_logs = df.reset_index(drop=True)
    return df_logs


# #INDEPENDENT TESTING OF ABOVE CODE 
# #execute from Terminal via (once inside the src directory): python3 interaction_generator.py 
if __name__=="__main__":
    from data_loader import load_data
    #load the data
    df_telco=load_data()
    customers_ids=df_telco['customerID'].tolist()

    df_logfile=generate_interaction_logs(customers_ids, messages_per_customer=1)
    # Reset index for clean preview
    df_logfile = df_logfile.reset_index(drop=True)
    print(f"Total #of rows = {len(df_logfile)}")
    print(df_logfile.head(17))





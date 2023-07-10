# utilities
import os
import emoji
import re

####
# Project hyperparameters
####


# switch this variable to toggle between
# downsampling the training data or using
# all the training data. 
using_downsampled_train_dataset = True

# epsilon to add to the probabilities to avoid error calculatin logits
# (log(0) is not defined)
LOGIT_THRESHOLD = 1e-4

# number of components for PCA dimensionality reduction of the logits
N_GRADIENT_COMPONENTS = 20

# least probability needed to take a classification decision
CLASSIFICATION_THRESHOLD = 0.80

# p parameter for F latency
# Value taken from Parapar, Losada 2022 (Overview of eRisk at CLEF 2021: Early Risk Prediction on the Internet (Extended Overview))
P_PARAMETER = 0.0078


####
# Project hyperparameters - END
####


def basic_text_cleaning(text):
    """to lower case. delete urls, emojis, and user mentions"""

    text = emoji.replace_emoji(text, replace='')    # Remove emojis
    text = re.sub(r"http\S+", "", text, flags=re.I)             # Remove urls. Ignore case for this.
    text = re.sub(r"@\S+", "", text)                # Remove user mentions
    text = text.lower() # tweets include tokens in all caps

    # this removes punctuation too early. I'm not using it:
    #text = re.sub(r"[^\w\s]", " ", text)            # everything that is not a word character or a whitespace becomes a whitespace
    
    text = text.strip()
    return text



from numpy import log

def calculate_logits(
        probs # a numpy array of probabilities
):
    # avoid 0 probabilities
    probs += LOGIT_THRESHOLD
    return log(probs)

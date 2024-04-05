import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaConfig, MambaForCausalLM
import torch
from collections import Counter
import pickle, re, os, time, random
import json
import tarfile
import argparse

from utils import *

parser = argparse.ArgumentParser()

import warnings

warnings.filterwarnings('ignore')


PROMPT_TEMPLATE_1 = f"""
Classify the following sentence into three categories: [0: negative, 1: neutral, 2: positive]
Provide answer in a structured format WITHOUT additional comments, I just want the numerical label for each sentence.
"""

PROMPT_TEMPLATE_2 = f"""
Your objective is to categorize each of the sentences listed below into one of three predefined sentiment categories, with each category represented by a numerical label as follows:
- 0: Negative, indicating the sentence conveys a negative or pessimistic sentiment.
- 1: Neutral, indicating the sentence expresses a sentiment that is neither positive nor negative, suggesting impartiality or a factual stance.
- 2: Positive, indicating the sentence portrays a positive or optimistic sentiment.

For each sentence, you are required to carefully evaluate the sentiment it expresses and assign the appropriate numerical label that corresponds to the identified sentiment category. It is crucial that your response is formatted to include only the numerical label for the sentiment category of each sentence, omitting any additional commentary, explanations, or extraneous text.

Please ensure your responses are clear and concise, directly mapping each sentence to its sentiment category through the numerical label alone. This streamlined approach is designed to facilitate a straightforward analysis of the sentiments conveyed in the sentences provided.

Example:
Sentence: 'The weather today is exceptionally beautiful.'
Response: 2

Proceed with the sentiment classification:
"""

PROMPT_TEMPLATE = [PROMPT_TEMPLATE_1, PROMPT_TEMPLATE_2]


def main(model, tokenizer, prompts, training_data, args):
    path = os.path.join(args.save_path, str(int(args.current_time)))
    # Create new dir
    if not os.path.exists(path):
        os.makedirs(path)
    # Save configrations
    with open(path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Append to the saved path
    data = []
    print(PROMPT_TEMPLATE[args.prompt_id - 1])
    for index, prompt in enumerate(prompts):
        preds, entropies = uncertainty_calculation(model, tokenizer, prompt, training_data,
                                                   args.decoding_strategy, args.num_demos,
                                                   args.num_demos_per_class, args.sampling_strategy, 
                                                   args.iter_demos, myPrompt=PROMPT_TEMPLATE[args.prompt_id - 1])
        
        AU, EU = token_uncertainty_calculation_new(preds, entropies, num_classes=3)
        print("Aleatoric Uncertainty: {}\t Epistemic Uncertainty: {}".format(AU, EU))
        
        pred = answer_extraction(preds)
        try:
            pred = Counter(pred).most_common()[0][0]
        except:
            pred = None
        #save_res = {"Question": prompt, "Label": labels[index], "Predicted_Label": preds,
        #            "Entropies": entropies, "AU": AU, "EU": EU, "AU_new": AU_new, "EU_new": EU_new}
        save_res = {"Question": prompt, "Label": labels[index], "Predicted_Label": pred,
                    "AU": AU, "EU": EU}
        
        data.append(save_res)
        
        if not args.resume_from:
            print("{} of {}, Done".format(index, len(prompts)))
        else:
            print("{} of {}, Done".format(index + args.resume_from, len(prompts)))
    return data


def post_processing(data, save_path, epochtime, model, sampling_strategy):
    if not data:
        data = []
        with open(save_path + '{}/{}_financial.pkl'.format(int(epochtime), model), 'rb') as fr:
            try:
                while True:
                    data.append(pickle.load(fr))
            except EOFError:
                pass
    # Create Dataframe
    data = pd.DataFrame(data)
    AU, EU, AU_new, EU_new, preds = [], [], [], [], []
    answers, entropies = list(data['Predicted_Label']), list(data['Entropies'])
    for i in range(len(answers)):
        # ale, epi = token_uncertainty_calculation(answers[i], entropies[i])
        ale_new, epi_new = token_uncertainty_calculation_new(answers[i], entropies[i])
        pred = answer_extraction(answers[i])
        preds.append(Counter(pred).most_common()[0][0])

        AU_new.append(ale_new)
        EU_new.append(epi_new)

    data['AU_new'] = AU_new
    data['EU_new'] = EU_new
    data['Preds'] = preds
    data = data.drop(columns=['Predicted_Label', 'Entropies'])
    data.to_json('/root/autodl-tmp/results/' + '{}/{}_financial_{}.json'.format(int(epochtime), 
                model, sampling_strategy), orient="records")


if __name__ == '__main__':
    parser.add_argument('--save_path', type=str, default='/root/autodl-tmp/results/')
    parser.add_argument('--model', type=str, default='google/gemma-2b')
    parser.add_argument('--num_demos', type=int, default=5)
    parser.add_argument('--num_demos_per_class', type=int, default=1)
    parser.add_argument('--sampling_strategy', choices=['random', 'class'], default='random')
    parser.add_argument('--decoding_strategy', choices=['beam_search', 'constractive', 'greedy', 'top_p'],
                        default='beam_search')
    parser.add_argument('--iter_demos', type=int, default=4)
    parser.add_argument('--load8bits', default=False, help='load model with 8 bits')
    parser.add_argument('--current_time', type=str, default=time.time())
    parser.add_argument('--resume_from', type=int)
    parser.add_argument('--prompt_id', type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading Model
    model_path = args.model
    if args.load8bits:
        model = MambaForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype='auto', load_in_8bit=True)
    else:
        print('load8bits is false')
        model = MambaForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    print("Done! Loaded Model: {}".format(args.model))

    # Loading Data
    training_data, test_data = get_data(dataset_name='financial_phrasebank')
    prompts = [i['sentence'] for i in test_data]
    labels = [i['label'] for i in test_data]
    print("Done! Loaded Data")

    if not args.resume_from:
        data = main(model, tokenizer, prompts, training_data, args)
    else:
        data = main(model, tokenizer, prompts[args.resume_from + 1:], training_data, args)
    
    data = pd.DataFrame(data)
    data.to_json('/root/autodl-tmp/results/' + '{}/{}_financial_{}_prompt{}.json'.format(int(args.current_time), args.model.split("/")[-1], args.sampling_strategy, args.prompt_id), orient="records")
    tar = tarfile.open('/root/autodl-tmp/results/' + '{}/{}_financial_{}_prompt{}.tar'.format(int(args.current_time), args.model.split("/")[-1], args.sampling_strategy, args.prompt_id),'w')
    tar.add('/root/autodl-tmp/results/' + '{}/{}_financial_{}_{}.json'.format(int(args.current_time), args.model.split("/")[-1], args.sampling_strategy, args.prompt_id))
    tar.add('/root/autodl-tmp/results/' + '{}/args.txt'.format(int(args.current_time)))
    tar.close()
    # post_processing(data, args.save_path, args.current_time, args.model, args.sampling_strategy)

from datasets import load_dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import Counter
import pickle, re, os, time, random
import json
import argparse

from utils import *

parser = argparse.ArgumentParser()

import warnings

warnings.filterwarnings('ignore')


PROMPT_TEMPLATE_1 = f"""
Classify the following sentence into two categories: [0: negative, 1: positive]
Provide answer in a structured format WITHOUT additional comments, I just want the numerical label for each sentence.
"""

PROMPT_TEMPLATE_2 = f"""
Your assignment is to categorize the sentiment of the sentence provided below into one of two possible sentiment categories, each represented by a numerical label for simplicity and clarity:
- 0: Negative, signifying that the sentence expresses a negative sentiment, indicating criticism, pessimism, or dissatisfaction.
- 1: Positive, signifying that the sentence expresses a positive sentiment, indicating approval, optimism, or satisfaction.

You are required to read the sentence and determine which sentiment category it falls into based on the sentiment it conveys. After your evaluation, provide the numerical label corresponding to the identified sentiment category. It is essential that your response only includes this numerical label, with no additional commentary, rationale, or extraneous information included.

This structured approach is designed for efficiency and clarity, ensuring that the sentiment classification is straightforward and unambiguous.

Proceed with the sentiment classification by providing the numerical label for the sentiment of the sentence:
"""


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
    if args.use_prompt == 1:
        myprompt = PROMPT_TEMPLATE_1
    else:
        myprompt = PROMPT_TEMPLATE_2
        
    for index, prompt in enumerate(prompts):
        preds, entropies = uncertainty_calculation(model, tokenizer, prompt, training_data,
                                                   args.decoding_strategy, args.num_demos,
                                                   args.num_demos_per_class, args.sampling_strategy, 
                                                   args.iter_demos, myPrompt=myprompt)
        AU, EU = token_uncertainty_calculation_new(preds, entropies)
        print("Aleatoric Uncertainty: {}\t Epistemic Uncertainty: {}".format(AU, EU))
        # aleatoric, epistemic = uncertainty_decomposition(entropies)
        save_res = {"Question": prompt, "Label": labels[index], "Predicted_Label": preds,
                     "AU": AU, "EU": EU}#"Entropies": entropies,
        data.append(save_res)

        if not args.resume_from:
            print("{} of {}, Done".format(index, len(prompts)))
        else:
            print("{} of {}, Done".format(index + args.resume_from, len(prompts)))
    return data


def post_processing(data, save_path, epochtime, model, sampling_strategy):
    if not data:
        data = []
        with open(save_path + '{}/{}_sst.pkl'.format(int(epochtime), model), 'rb') as fr:
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
        ale_new, epi_new = token_uncertainty_calculation_new(answers[i], entropies[i])
        pred = answer_extraction(answers[i])
        preds.append(Counter(pred).most_common()[0][0])

        AU_new.append(ale_new)
        EU_new.append(epi_new)

    data['AU_new'] = AU_new
    data['EU_new'] = EU_new
    data['Preds'] = preds
    data = data.drop(columns=['Predicted_Label', 'Entropies'])
    data.to_json('./LLM_UQ/results/' + '{}/{}_sst_{}.json'.format(int(epochtime), 
                model, sampling_strategy), orient="records")


if __name__ == '__main__':
    parser.add_argument('--save_path', type=str, default='/home/j/jiyuyu/workspace/cs5340/CS5340-Project/LLM_UQ/results/')
    parser.add_argument('--model', type=str, default='google/gemma-7b')
    parser.add_argument('--num_demos', type=int, default=5)
    parser.add_argument('--num_demos_per_class', type=int, default=1)
    parser.add_argument('--sampling_strategy', choices=['random', 'class'], default='random')
    parser.add_argument('--decoding_strategy', 
                        choices=['beam_search', 'constractive', 'greedy', 'top_p'],
                        default='beam_search')
    parser.add_argument('--iter_demos', type=int, default=4)
    parser.add_argument('--load8bits', default=False, help='load model with 8 bits')
    parser.add_argument('--current_time', type=str, default=time.time())
    parser.add_argument('--resume_from', type=int)
    parser.add_argument('--use_prompt', default=1,type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading Model
    model_path = args.model
    if args.load8bits:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype='auto', load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Done! Loaded Model: {}".format(args.model))

    # Loading Data
    training_data, test_data = get_data(dataset_name='glue')
    prompts = [i['sentence'] for i in test_data]
    labels = [i['label'] for i in test_data]
    print("Done! Loaded Data")

    if not args.resume_from:
        data = main(model, tokenizer, prompts, training_data, args)
    else:
        data = main(model, tokenizer, prompts[args.resume_from + 1:], training_data, args)
    data = pd.DataFrame(data)
    data.to_json('/home/j/jiyuyu/workspace/cs5340/CS5340-Project/LLM_UQ/results/' + '{}/{}_sst_{}.json'.format(int(args.current_time), 
                args.model.split("/")[-1], args.sampling_strategy), orient="records")    
    # post_processing(data, args.save_path, args.current_time, args.model, args.sampling_strategy)

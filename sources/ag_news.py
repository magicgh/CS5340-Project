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
Classify the topic of the following sentence into four labels: [0: world, 1: sports, 2: business, 3: Sci/Tech]
Provide answer in a structured format WITHOUT additional comments, I just want the numerical label for each sentence.
"""

PROMPT_TEMPLATE_2 = f"""
    Classify the topic of the sentences provided below into four specific categories, each represented by a numerical label:
    - 0: World
    - 1: Sports
    - 2: Business
    - 3: Sci/Tech

    Your task is to read each sentence and assign the most appropriate category label based on its content. Please format your response to include only the numerical label corresponding to the identified category for each sentence. Do not include additional comments or explanations in your response.

    Example:
    Sentence: 'The global economy is expected to grow this year.'
    Response: 2

    Now, proceed with the classification:
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
    for index, prompt in enumerate(prompts):
        preds, entropies = uncertainty_calculation(model, tokenizer, prompt, training_data,
                                                   args.decoding_strategy, args.num_demos,
                                                   args.num_demos_per_class, args.sampling_strategy, 
                                                   args.iter_demos, myPrompt=PROMPT_TEMPLATE_1)
        AU, EU = token_uncertainty_calculation_new(preds, entropies)
        print("Aleatoric Uncertainty: {}\t Epistemic Uncertainty: {}".format(AU, EU))
        save_res = {"Question": prompt, "Label": labels[index], "Predicted_Label": preds,
                    "Entropies": entropies, "AU": AU, "EU": EU}
        data.append(save_res)
        with open(path + '/{}_agnews.pkl'.format(args.model.split("/")[-1]), 'ab+') as fp:
            pickle.dump(save_res, fp)
        if not args.resume_from:
            print("{} of {}, Done".format(index, len(prompts)))
        else:
            print("{} of {}, Done".format(index + args.resume_from, len(prompts)))
    return data


def post_processing(data, save_path, epochtime, model, sampling_strategy):
    if not data:
        data = []
        with open(save_path + '{}/{}_agnews.pkl'.format(int(epochtime), model), 'rb') as fr:
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
    data.to_json('/root/autodl-tmp/LLM_UQ/results/' + '{}/{}_agnews_{}.json'.format(int(epochtime), 
                model, sampling_strategy), orient="records")


if __name__ == '__main__':
    parser.add_argument('--save_path', type=str, default='/root/autodl-tmp/results/')
    parser.add_argument('--model', type=str, default='google/gemma-2b')
    parser.add_argument('--num_demos', type=int, default=5)
    parser.add_argument('--num_demos_per_class', type=int, default=1)
    parser.add_argument('--sampling_strategy', choices=['random', 'class'], default='random')
    parser.add_argument('--decoding_strategy', 
                        choices=['beam_search', 'constractive', 'greedy', 'top_p'],
                        default='beam_search')
    parser.add_argument('--iter_demos', type=int, default=4)
    parser.add_argument('--load8bits', default=True, help='load model with 8 bits')
    parser.add_argument('--current_time', type=str, default=time.time())
    parser.add_argument('--resume_from', type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading Model
    model_path = args.model
    if args.load8bits:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Done! Loaded Model: {}".format(args.model))

    # Loading Data
    training_data, test_data = get_data(dataset_name='ag_news')
    prompts = [i['text'] for i in test_data]
    labels = [i['label'] for i in test_data]
    print("Done! Loaded Data")

    if not args.resume_from:
        data = main(model, tokenizer, prompts, training_data, args)
    else:
        data = main(model, tokenizer, prompts[args.resume_from + 1:], training_data, args)

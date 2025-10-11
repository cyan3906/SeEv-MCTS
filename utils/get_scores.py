import json
import os
from tqdm import tqdm

from data.CharXiv.src.get_stats import get_reasoning_scores, get_stats

def scoring_charxiv(cfg):
    data_cfg = cfg.data

    extract_file = os.path.join(cfg.extract_result_dir, f'scores-{data_cfg.mode}_{data_cfg.split}.json')
    
    image_meta = json.load(open(os.path.join(data_cfg.data_dir, f'image_metadata_{data_cfg.split}.json')))
    reasoning_meta = json.load(open(os.path.join(data_cfg.data_dir, f'reasoning_{data_cfg.split}.json')))
    descriptive_meta = json.load(open(os.path.join(data_cfg.data_dir, f'descriptive_{data_cfg.split}.json')))
    
    os.makedirs(cfg.extract_result_dir, exist_ok=True)
    output_file = os.path.join(cfg.extract_result_dir, f'stats.json')

    if os.path.exists(extract_file):
        reasoning_scores = json.load(open(extract_file))
        for k, v in reasoning_scores.items():
            if data_cfg.score_label == "llm_score":
                reasoning_scores[k]["score"] = 1 if v[data_cfg.score_label] else 0
            else:
                reasoning_scores[k]["score"] = 1 if v["ground_truth"] == v["extracted_answer"] else 0
        reasoning_stats = get_reasoning_scores(reasoning_scores, descriptive_meta, 
                                               reasoning_meta, image_meta)
        reasoning_stats = get_stats(reasoning_stats)
        json.dump(reasoning_stats, open(output_file, "w"), indent=4)
        print("### Reasoning Stats ###")
        print(json.dumps(reasoning_stats, indent=4))
        print("Stats saved to results folder")
    else:
        print("No scores file found. Skipping scoring.")

from data.MathVista.evaluation.calculate_score import safe_equal, normalize_extracted_answer
def scoring_mathvista(cfg):
    data_cfg = cfg.data
    
    extract_file = os.path.join(cfg.extract_result_dir, f'extraction.json')

    # read json
    print(f"Reading {extract_file}...")
    results = json.load(open(extract_file))

    os.makedirs(cfg.extract_result_dir, exist_ok=True)
    output_file = os.path.join(cfg.extract_result_dir, f'scores.json')

    # full pids
    full_pids = list(results.keys())

    ## [1] Evaluate if the prediction is true or false
    print("\nEvaluating the predictions...")
    update_json_flag = False
    for pid in full_pids:
        problem = results[pid]
        # print(problem)

        choices = problem['choices']
        question_type = problem['question_type']
        answer_type = problem['answer_type']
        precision = problem['precision']
        extraction = problem['extracted_answer']

        if 'answer' in problem:
            answer = problem['answer']

        # normalize the extracted answer to match the answer type
        prediction = normalize_extracted_answer(extraction, choices, question_type, answer_type, precision)

        # verify the prediction is true or false
        true_false = safe_equal(prediction, answer)
        
        # update the problem
        if "true_false" not in problem:
            update_json_flag = True

        elif true_false != problem['true_false']:
            update_json_flag = True

        if "prediction" not in problem:
            update_json_flag = True

        elif prediction !=  problem['prediction']:
            update_json_flag = True
            
        problem['prediction'] = prediction
        problem['true_false'] = true_false

        if data_cfg.score_label == "llm_score":
            problem['true_false'] = problem[data_cfg.score_label]
            update_json_flag = True

    # save the updated json
    if update_json_flag:
        print("\n!!!Some problems are updated.!!!")
        print(f"\nSaving {output_file}...")
        json.dump(results, open(output_file, "w"), indent=4)

    ## [2] Calculate the average accuracy
    total = len(full_pids)
    correct = 0
    for pid in full_pids:
        if results[pid]['true_false']:
            correct += 1
    accuracy = str(round(correct / total * 100, 2))
    print(f"\nCorrect: {correct}, Total: {total}, Accuracy: {accuracy}%")

    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    # save the scores
    os.makedirs(cfg.extract_result_dir, exist_ok=True)
    scores_file = os.path.join(cfg.extract_result_dir, "stats.json")
    print(f"\nSaving {scores_file}...")
    json.dump(scores, open(scores_file, "w"), indent=4)
    print("\nDone!")

def scoring_mathvision(cfg):
    data_cfg = cfg.data

    extract_file = os.path.join(cfg.extract_result_dir, f'scores.json')

    # read json
    print(f"Reading {extract_file}...")
    results = json.load(open(extract_file))

    os.makedirs(cfg.extract_result_dir, exist_ok=True)
    output_file = os.path.join(cfg.extract_result_dir, f'stats.json')
    if data_cfg.score_label == "rule_score":
        output_file = os.path.join(cfg.extract_result_dir, f'stats_rule.json')
    else:
        output_file = os.path.join(cfg.extract_result_dir, f'stats.json')

    results_dict = {}
    for line in tqdm(results, desc='math_level_subject_acc'):
        line = results[line]
        correct = line[data_cfg.score_label]
        subject = line['subject']
        level = line['level']
        for key in [
            '-all', 
            f'-level{level}', 
            f'{subject}', 
            f'{subject}_level{level}', 
            f'-level{level}_{subject}'
            ]:
            if key not in results_dict:
                results_dict[key] = [0,0]
            results_dict[key][0] += 1 if correct else 0
            results_dict[key][1] += 1


    for key in results_dict.keys():
        if results_dict[key][1] == 0:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}=0'
        else:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0]/ max(results_dict[key][1], 1)*100, 2)}%'


    results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
    
    print(f"\nSaving {output_file}...")
    json.dump(results_dict, open(output_file, "w"), indent=4)
    print("\nDone!")

def scoring(cfg):
    if cfg.data.name == "charxiv":
        scoring_charxiv(cfg)
    elif cfg.data.name == "mathvista":
        scoring_mathvista(cfg)
    elif cfg.data.name == "mathvision":
        scoring_mathvision(cfg)
    else:
        raise ValueError("Dataset not supported")
import json,os,time,re,sys
from tools.color_print import *


def get_messages(system_prompt = None,prompt = None, image_path = None):
    messages=[
        {"role": "system", "content": system_prompt},
        
        {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": image_path,
            },
            {"type": "text", "text": prompt},
        ],
        },
    ]
    
    return messages
def is_answer_consistent(model, answer1, answer2,n = 11):
    user_prompt = is_answer_consistent_prompt.format(answer1=answer1, answer2=answer2)
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                
                "content": [
                    # {
                    #     "type": "image_url",
                    #     "image_url": "",
                    # },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
    list_response = model.get_response(messages,n = n)["response"]
    
    win = loss = 0
    
    bool_list = []
    for response in list_response:
        match = re.search(r'"is_consistent"\s*:\s*(true|false)', response)
        if match:
            is_consistent = bool(match.group(1))  # 提取匹配到的值
            bool_list.append(is_consistent)
            if is_consistent:
                win += 1
            else:
                loss += 1
    
    # print(f"list_response: {list_response}")
    printyellow(f"bool_list: {bool_list}")
    if win + loss == 0:
        return 1
    
    return win > (win + loss) / 2

cot_system_prompt = """You are an expert AI assistant that explains your reasoning step by step. Generate the final answer according to your thought chain.

    # Strict Output Format Requirements
    1.  You must output **only a valid JSON object** that strictly matches the following structure:
        {
            "chain_of_thought": "Your step-by-step reasoning process, written in natural language, with clear logical connections between each step",
            "final_answer": "The concise, accurate final result of your reasoning"
        }
    2.  Ensure the JSON is **syntax error-free** (no trailing commas, correct quotation marks, valid key names).
    3.  The `chain_of_thought` field must detail your thinking: what you analyzed first, what rules you applied, how you deduced intermediate conclusions, and why you rejected other possible answers (if applicable).
    4.  The `final_answer` field must be a **direct response** to the user's question, avoiding redundant explanations.

    Let us think step by step and generate the final answer that complies with the above format strictly."""

cot_prompt_template = """
    **Example**: 

    **Question**: 
    What is the result of 2 + 3 × 4?

    **Answer**:
    {{
        "chain_of_thought": "Step 1: Follow the order of mathematical operations (PEMDAS/BODMAS), which prioritizes multiplication over addition. First calculate the multiplication part: 3 × 4 = 12. Step 2: Add the result to 2, so 2 + 12 = 14. No other calculation order is valid here, as ignoring the operation priority would lead to wrong results like (2+3)×4=20.",
        "final_answer": "14"
    }}

    **Question**: 
    Evaluate the difficulty level (L1/L2/L3/L4) of the QA pair: Question - "What is 1+1?", Answer - "2".

    **Answer**:
    {{
        "chain_of_thought": "Step 1: Analyze the knowledge breadth: the question only involves basic arithmetic knowledge with no requirement for professional domain expertise. Step 2: Judge the reasoning depth: no logical deduction is needed, and the answer can be given directly. Step 3: Assess semantic complexity: the question is clearly stated without ambiguity, and the answer is concise and unique. Based on the above three points, it meets the criteria for L1 (easy) difficulty level.",
        "final_answer": "L1"
    }}

    **Question**: 
    {question}

    **Answer**:

    """

cot_output_template = """
    ### Instruction:
    There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

    ### Response:
    Let's think step by step.
    Step 1: Identify the initial number of trees. The problem states there are 15 trees in the grove.
    Step 2: Identify the final number of trees. The problem states there will be 21 trees after the workers are done planting.
    Step 3: Subtract the initial number of trees from the final number of trees to find out how many trees were planted.
    Step 4: Therefore, the grove workers planted 21 (final number of trees) - 15 (initial number of trees) = 6 trees today.
    Step 5: The answer is 6.

    ### Instruction:
    If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

    ### Response:
    Let's think step by step.
    Step 1: Start with the number of cars that are already in the parking lot, which is 3 cars.
    Step 2: Add the number of cars that arrive, which is 2 cars.
    Step 3: Add the numbers together. there are 3 cars + 2 cars = 5 cars in the parking lot.
    Step 4: The answer is 5.

    ### Instruction:
    Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

    ### Response:
    Let's think step by step.
    Step 1: Add the number of chocolates Leah and her sister had initially. Leah had 32 chocolates and her sister had 42 chocolates. So, they had 32 + 42 = 74 chocolates in total.
    Step 2: Subtract the number of chocolates they ate from the total number of chocolates they had. They ate 35 chocolates. So, they have 74 - 35 = 39 chocolates left.
    Step 3: The answer is 39.

    ### Instruction:
    Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

    ### Response:
    Let's think step by step.
    Step 1: Identify the initial number of lollipops. The problem states Jason had 20 lollipops.
    Step 2: Identify the final number of lollipops. The problem states Jason now has 12 lollipops.
    Step 3: Subtract the final number of lollipops from the initial number to find out how many lollipops Jason gave to Denny.
    Step 4: Therefore, Jason gave Denny 20 (initial number of lollipops) - 12 (final number of lollipops) = 8 lollipops.
    Step 5: The answer is 8.
    
    ### Instruction:
    {question}
    
    ### Response:
    Let's think step by step.
    {answer}
    """

cot_output_system_promot = "A chat between a curious user and an AI assistant. The assistant gives step-by-step solutions to the user's questions. In the end of assistant's response, a final answer is given in the format of \"The answer is: <ANSWER>.\".\n\n"

is_answer_consistent_prompt = """
    Act as an **Answer Consistency Evaluator** and strictly follow these requirements:
    1.  Task Goal: Judge whether the **core content of [Answer A] and [Answer B] is completely consistent**, ignoring any format differences.
    2.  Judgment Rules (Priority from High to Low):
        1.  Ignore format differences: Text/JSON/list format, word order, synonyms, punctuation, line breaks, uppercase/lowercase, redundant modifiers are all irrelevant.
        2.  Definition of core consistency: Full match of three elements—key objects (e.g., "different subplot number"), key conclusions (e.g., "split direction is different"), and key data (e.g., "vertical direction").
        3.  If any of the three key elements do not match, mark as "Inconsistent".
    3.  Output Requirements: **Must output strictly in JSON format, with no redundant text**. The JSON structure is fixed as follows:
    {{
        "is_consistent": true/false,
        "explanation": "Explanation for the judgment"
    }}

    [Answer A] {answer1}
    [Answer B] {answer2}
            
"""


mathvision_qa_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=qa.mathvision.ALL/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=qa/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
mathvision_cot_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=cot/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=cot/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"
mathvision_baseline_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/old-outcomes/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores.json"

mathvista_qa_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=qa.mathvista.ALL/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=qa/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/extraction.json"
mathvista_cot_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=cot/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=cot/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/extraction.json"
mathvista_baseline_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvista/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/extraction.json"

charxiv_qa_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=qa.charxiv.ALL/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=qa/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"
charxiv_cot_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=cot/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=cot/ext_llm=qwen25/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"
charxiv_cot_data_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=cot/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=cot/gen-reasoning_val.json"
charxiv_baseline_score_file = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=charxiv/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/old-outcomes/ext_llm=Qwen2.5-7B-Instruct/method=ans_and_score/field=max_vote_terminal/n=10/reward_method=self_consistency/scores-reasoning_val.json"

with open(mathvision_qa_score_file, "r") as f:
    mathvision_qa_score = json.load(f)
with open(mathvision_cot_score_file, "r") as f:
    mathvision_cot_score = json.load(f)
with open(mathvision_baseline_score_file, "r") as f:
    mathvision_baseline_score = json.load(f)

with open(mathvista_qa_score_file, "r") as f:
    mathvista_qa_score = json.load(f)
with open(mathvista_cot_score_file, "r") as f:
    mathvista_cot_score = json.load(f)
with open(mathvista_baseline_score_file, "r") as f:
    mathvista_baseline_score = json.load(f)

with open(charxiv_qa_score_file, "r") as f:
    charxiv_qa_score = json.load(f)
with open(charxiv_cot_score_file, "r") as f:
    charxiv_cot_score = json.load(f)
with open(charxiv_cot_data_file, "r") as f:
    charxiv_cot_data = json.load(f)
with open(charxiv_baseline_score_file, "r") as f:
    charxiv_baseline_score = json.load(f)

mathvision_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/outputs/desc=baseline/data=mathvision/llm=Qwen2-VL-7B-Instruct/prompt_method=mctsv8/ori_generation.json"
mathvista_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/testmini.json"
charxiv_data = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/CharXiv/data/reasoning_val.json"
with open(mathvision_data, "r") as f:
    mathvision_data = json.load(f)
with open(mathvista_data, "r") as f:
    mathvista_data = json.load(f)
with open(charxiv_data, "r") as f:
    charxiv_data = json.load(f)
    

IMAGE_DIR_CONFIG = {
    "mathvision": "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MATH-V/images",
    "mathvista": "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/images/testmini/ALL", 
    "charxiv": "/home/jncsnlp4/SSD2/ZJJ/VReST/data/CharXiv/images"
}


data = {
    "mathvision": mathvision_data,
    "mathvista": mathvista_data,
    "charxiv": charxiv_data
}


data_baseline_score = {
    "mathvision": mathvision_baseline_score,
    "mathvista": mathvista_baseline_score,
    "charxiv": charxiv_baseline_score
}

data_qa_score = {
    "mathvision": mathvision_qa_score,
    "mathvista": mathvista_qa_score,
    "charxiv": charxiv_qa_score
}
    

data_cot_score = {
    "mathvision": mathvision_cot_score,
    "mathvista": mathvista_cot_score,
    "charxiv": charxiv_cot_score
}


# 没筛过的数据

mathvision_cot_true_baseline_false = ['45', '49', '50', '52', '149', '159', '218', '231', '233', '259', '269', '270', '279', '286', '297', '300', '357', '823', '870', '1101', '1113', '1246', '1255', '1355', '1512', '1546', '1559', '2288', '2634', '2757', '2773', '2786', '2955']
mathvision_cot_false_baseline_true = ['20', '32', '55', '64', '65', '78', '86', '92', '95', '114', '123', '152', '181', '206', '242', '253', '254', '318', '336', '403', '441', '522', '537', '628', '739', '766', '767', '812', '819', '855', '923', '946', '1060', '1137', '1203', '1215', '1222', '1343', '1389', '1451', '1634', '1680', '1813', '1969', '2443', '2565', '2688', '3033']

mathvista_cot_true_baseline_false = ['16', '22', '25', '41', '43', '72', '73', '74', '78', '86', '87', '88', '100', '102', '107', '113', '127', '137', '142', '153', '161', '164', '189', '202', '210', '213', '224', '226', '228', '237', '239', '242', '248', '250', '257', '260', '266', '267', '271', '274', '284', '321', '323', '324', '338', '340', '342', '360', '365', '367', '368', '373', '382', '385', '388', '389', '393', '424', '439', '442', '450', '451', '457', '475', '486', '502', '524', '541', '570', '576', '582', '593', '595', '597', '602', '612', '614', '620', '638', '652', '665', '673', '675', '692', '693', '694', '699', '705', '708', '710', '715', '723', '727', '729', '730', '738', '740', '753', '756', '758', '761', '775', '785', '790', '794', '803', '814', '822', '823', '853', '871', '873', '908', '932', '936', '939', '942', '956', '959', '963', '973', '976', '984']
mathvista_cot_false_baseline_true = ['4', '6', '9', '10', '12', '13', '14', '15', '19', '20', '28', '48', '52', '57', '58', '61', '63', '64', '68', '69', '79', '97', '101', '104', '108', '111', '116', '117', '120', '121', '129', '132', '135', '141', '149', '150', '152', '166', '174', '177', '181', '183', '190', '191', '195', '197', '203', '209', '212', '214', '216', '219', '220', '222', '227', '229', '231', '234', '244', '254', '256', '259', '265', '269', '275', '276', '280', '285', '296', '306', '316', '325', '328', '333', '336', '339', '343', '344', '351', '355', '357', '358', '362', '364', '372', '381', '383', '386', '390', '394', '397', '399', '400', '402', '419', '426', '427', '430', '433', '446', '447', '456', '459', '463', '469', '470', '477', '478', '480', '485', '487', '488', '490', '497', '499', '501', '506', '511', '515', '516', '526', '527', '528', '536', '539', '540', '542', '546', '549', '556', '560', '565', '566', '567', '571', '575', '579', '587', '594', '603', '610', '622', '624', '625', '628', '629', '632', '634', '637', '642', '647', '658', '663', '670', '672', '674', '679', '681', '685', '689', '707', '720', '726', '731', '736', '748', '752', '757', '763', '766', '772', '781', '783', '786', '787', '789', '798', '800', '817', '818', '819', '821', '842', '846', '849', '860', '866', '868', '870', '878', '882', '887', '892', '896', '900', '903', '906', '907', '909', '910', '913', '924', '927', '931', '935', '938', '955', '961', '964', '965', '969', '970', '975', '980', '982', '983', '989', '991', '994', '996']

charxiv_cot_true_baseline_false = ['0', '26', '29', '75', '80', '87', '107', '144', '159', '181', '225', '269', '275', '286', '288', '291', '389', '394', '406', '416', '461', '463', '484', '488', '489', '562', '592', '602', '631', '636', '642', '658', '671', '694', '695', '705', '713', '736', '741', '745', '766', '802', '813', '860', '880', '882', '888', '890', '897', '901', '905', '925', '936', '967', '975', '997', '1015', '1030', '1031', '1035', '1039', '1041', '1048', '1067', '1130', '1142', '1147', '1170', '1172', '1183', '1186', '1212', '1224', '1246', '1307', '1318', '1334', '1352', '1391', '1397', '1417', '1425', '1431', '1449', '1464', '1473', '1481', '1496', '1514', '1531', '1532', '1566', '1569', '1584', '1590', '1594', '1603', '1609', '1611', '1628', '1649', '1664', '1686', '1694', '1695', '1704', '1720', '1726', '1789', '1793', '1794', '1797', '1838', '1853', '1865', '1900', '1932', '1939', '1948', '1969', '1976', '1986', '2010', '2082', '2093', '2122', '2133', '2147', '2177', '2188', '2194', '2198', '2224', '2236', '2242', '2257', '2270', '2279', '2330', '2336', '2372', '2375']
charxiv_cot_false_baseline_true = ['2', '8', '17', '30', '39', '42', '51', '60', '61', '65', '84', '92', '95', '108', '111', '149', '160', '167', '193', '215', '217', '228', '240', '243', '248', '252', '255', '281', '283', '285', '292', '298', '310', '319', '321', '328', '330', '336', '337', '362', '367', '384', '410', '412', '431', '433', '438', '440', '445', '450', '452', '458', '470', '473', '474', '475', '496', '514', '515', '517', '521', '523', '527', '542', '548', '550', '552', '560', '565', '588', '606', '611', '620', '640', '646', '649', '659', '673', '678', '684', '685', '701', '703', '716', '722', '732', '777', '794', '814', '823', '848', '856', '900', '911', '913', '923', '924', '988', '995', '1002', '1011', '1038', '1046', '1076', '1098', '1102', '1117', '1124', '1136', '1141', '1175', '1177', '1179', '1192', '1197', '1199', '1208', '1229', '1251', '1258', '1265', '1289', '1294', '1305', '1312', '1314', '1330', '1336', '1363', '1369', '1393', '1405', '1406', '1415', '1416', '1432', '1435', '1455', '1456', '1463', '1475', '1493', '1533', '1542', '1559', '1568', '1581', '1582', '1588', '1643', '1650', '1651', '1659', '1661', '1662', '1668', '1702', '1722', '1729', '1753', '1791', '1803', '1822', '1826', '1842', '1851', '1857', '1862', '1937', '1953', '1970', '1973', '1985', '1987', '1994', '2000', '2003', '2005', '2009', '2018', '2035', '2055', '2059', '2071', '2081', '2083', '2090', '2106', '2124', '2145', '2150', '2157', '2187', '2211', '2221', '2240', '2245', '2307', '2310', '2347', '2357', '2359', '2365', '2371', '2380', '2394']


"""
基于cot 过俩掉step只有1步的已经出现回答不了的

mathvision_cot_true_baseline_false = ['45', '49', '50', '52', '149', '159', '233', '259', '270', '279', '297', '300', '357', '823', '870', '1101', '1246', '1255', '1355', '1512', '2288', '2634', '2757', '2773', '2786', '2955']
mathvision_cot_false_baseline_true = ['20', '32', '55', '64', '65', '78', '86', '92', '95', '114', '123', '152', '181', '206', '242', '253', '254', '318', '336', '403', '441', '522', '537', '628', '739', '766', '767', '812', '819', '855', '923', '946', '1060', '1137', '1203', '1215', '1222', '1343', '1389', '1451', '1634', '1680', '1813', '1969', '2443', '2565', '2688', '3033']

mathvista_cot_true_baseline_false = ['16', '25', '43', '72', '73', '74', '78', '107', '113', '127', '153', '161', '164', '202', '210', '213', '237', '239', '248', '257', '260', '266', '267', '321', '342', '360', '365', '367', '368', '373', '382', '385', '439', '442', '450', '451', '502', '541', '570', '576', '582', '593', '595', '597', '602', '612', '614', '620', '652', '673', '692', '693', '705', '727', '740', '756', '758', '775', '785', '790', '794', '803', '823', '908', '932', '936', '939', '942', '956', '959', '963', '973', '976', '984']
mathvista_cot_false_baseline_true = ['4', '10', '12', '13', '15', '20', '52', '57', '58', '61', '63', '64', '68', '69', '97', '101', '104', '108', '111', '120', '121', '129', '132', '135', '141', '149', '152', '166', '174', '177', '181', '183', '195', '197', '203', '212', '216', '219', '220', '222', '227', '229', '234', '244', '254', '259', '276', '296', '306', '316', '325', '328', '339', '343', '344', '351', '357', '362', '364', '372', '381', '386', '394', '399', '400', '419', '426', '430', '433', '447', '459', '463', '469', '470', '477', '480', '487', '488', '511', '515', '527', '528', '536', '539', '542', '556', '560', '566', '587', '594', '603', '624', '625', '629', '634', '637', '672', '674', '681', '685', '707', '731', '736', '748', '763', '766', '772', '781', '783', '789', '800', '817', '819', '846', '868', '870', '878', '882', '887', '892', '896', '900', '903', '906', '907', '909', '910', '913', '924', '931', '938', '964', '969', '970', '975', '980', '982', '989', '991', '996']

charxiv_cot_true_baseline_false = ['0', '26', '29', '80', '107', '144', '159', '225', '269', '275', '286', '288', '291', '389', '394', '406', '416', '461', '484', '488', '489', '562', '592', '602', '636', '642', '671', '694', '695', '713', '745', '766', '813', '880', '882', '888', '890', '897', '901', '925', '967', '1015', '1030', '1031', '1041', '1048', '1067', '1147', '1172', '1183', '1186', '1212', '1224', '1307', '1318', '1334', '1391', '1417', '1431', '1481', '1496', '1514', '1531', '1532', '1566', '1569', '1584', '1590', '1611', '1628', '1649', '1664', '1686', '1695', '1704', '1726', '1793', '1794', '1797', '1853', '1865', '1932', '1939', '1948', '1969', '1976', '1986', '2010', '2082', '2093', '2122', '2133', '2177', '2188', '2198', '2224', '2236', '2242', '2257', '2270', '2279', '2330', '2336', '2372', '2375']
charxiv_cot_false_baseline_true = ['2', '8', '17', '30', '39', '51', '60', '65', '84', '92', '108', '111', '149', '167', '215', '217', '228', '240', '243', '248', '252', '255', '281', '285', '292', '298', '319', '330', '336', '337', '362', '367', '384', '412', '431', '433', '438', '440', '445', '450', '452', '458', '470', '473', '474', '475', '496', '514', '515', '517', '521', '523', '527', '542', '548', '550', '552', '560', '565', '588', '606', '611', '620', '640', '646', '649', '659', '673', '678', '684', '685', '701', '703', '716', '722', '732', '777', '814', '848', '856', '911', '913', '923', '924', '988', '995', '1002', '1011', '1038', '1076', '1098', '1117', '1124', '1136', '1141', '1175', '1177', '1179', '1192', '1197', '1199', '1208', '1229', '1258', '1265', '1294', '1305', '1312', '1330', '1336', '1363', '1369', '1393', '1405', '1406', '1415', '1416', '1432', '1435', '1455', '1456', '1463', '1475', '1493', '1533', '1542', '1568', '1582', '1643', '1650', '1651', '1659', '1661', '1668', '1729', '1791', '1803', '1822', '1826', '1842', '1851', '1857', '1862', '1937', '1953', '1970', '1973', '1987', '1994', '2003', '2009', '2035', '2059', '2071', '2083', '2106', '2124', '2145', '2150', '2157', '2187', '2211', '2221', '2240', '2245', '2307', '2310', '2347', '2357', '2365', '2371', '2380', '2394']

"""


# mathvision_cot_true_baseline_false = ['49', '50', '52', '149', '233', '270', '279', '297', '300', '357', '823', '870', '1101', '1246', '1255', '1355', '1512', '2288', '2634', '2757', '2773', '2786', '2955']
# mathvision_cot_false_baseline_true = ['20', '32', '55', '64', '65', '78', '86', '92', '95', '114', '123', '152', '181', '206', '242', '253', '318', '336', '403', '441', '522', '537', '628', '739', '766', '767', '812', '819', '855', '923', '946', '1060', '1137', '1203', '1215', '1222', '1343', '1389', '1634', '1680', '1813', '1969', '2443', '2565', '2688', '3033']

# mathvista_cot_true_baseline_false = ['16', '25', '43', '72', '74', '78', '107', '113', '127', '153', '161', '164', '202', '210', '213', '237', '239', '248', '257', '266', '267', '321', '342', '365', '367', '368', '373', '382', '385', '439', '442', '450', '451', '502', '541', '570', '576', '582', '593', '595', '597', '602', '612', '614', '620', '652', '673', '692', '693', '705', '727', '740', '756', '758', '775', '785', '790', '794', '803', '823', '908', '936', '939', '942', '956', '959', '963', '973', '976', '984']
# mathvista_cot_false_baseline_true = ['4', '10', '12', '13', '15', '20', '52', '57', '58', '61', '63', '64', '68', '69', '97', '101', '104', '108', '111', '120', '121', '129', '135', '141', '152', '166', '174', '177', '181', '183', '195', '197', '203', '212', '216', '219', '220', '222', '227', '229', '234', '244', '254', '259', '276', '296', '306', '316', '325', '328', '343', '344', '351', '357', '362', '372', '381', '386', '394', '399', '400', '419', '426', '430', '433', '447', '459', '463', '469', '470', '477', '480', '487', '488', '511', '515', '527', '528', '536', '539', '542', '556', '560', '587', '594', '603', '624', '625', '629', '634', '637', '672', '674', '681', '685', '707', '731', '736', '763', '766', '772', '781', '783', '789', '800', '817', '819', '846', '868', '870', '878', '882', '887', '892', '896', '900', '903', '906', '907', '909', '910', '913', '924', '938', '964', '969', '970', '975', '980', '982', '989', '991', '996']

# charxiv_cot_true_baseline_false = ['0', '26', '29', '80', '107', '144', '159', '225', '269', '275', '286', '288', '291', '389', '394', '406', '416', '461', '484', '488', '489', '562', '592', '636', '642', '671', '694', '695', '713', '745', '813', '880', '882', '888', '890', '897', '901', '925', '967', '1015', '1030', '1031', '1041', '1048', '1067', '1147', '1172', '1183', '1186', '1212', '1224', '1307', '1318', '1334', '1391', '1417', '1431', '1481', '1496', '1514', '1531', '1532', '1566', '1569', '1584', '1590', '1611', '1628', '1649', '1664', '1686', '1695', '1704', '1726', '1793', '1794', '1797', '1853', '1865', '1932', '1939', '1948', '1969', '1976', '1986', '2010', '2093', '2122', '2133', '2177', '2188', '2198', '2224', '2236', '2242', '2257', '2270', '2330', '2336', '2372', '2375']
# charxiv_cot_false_baseline_true = ['2', '8', '17', '30', '39', '51', '60', '65', '84', '92', '108', '111', '149', '167', '215', '217', '228', '240', '243', '248', '252', '255', '281', '285', '292', '298', '319', '330', '336', '337', '362', '367', '384', '412', '431', '433', '438', '440', '445', '450', '452', '458', '470', '474', '475', '496', '514', '515', '517', '521', '523', '527', '542', '548', '550', '552', '560', '565', '588', '606', '611', '620', '640', '646', '649', '659', '673', '678', '684', '685', '701', '703', '716', '722', '732', '777', '848', '856', '911', '913', '923', '924', '988', '995', '1002', '1011', '1038', '1076', '1098', '1117', '1124', '1136', '1141', '1175', '1177', '1179', '1192', '1197', '1199', '1208', '1229', '1258', '1294', '1305', '1312', '1330', '1336', '1363', '1369', '1393', '1405', '1406', '1415', '1416', '1432', '1435', '1455', '1456', '1463', '1475', '1493', '1533', '1542', '1568', '1582', '1643', '1650', '1651', '1659', '1661', '1668', '1729', '1791', '1803', '1822', '1826', '1842', '1851', '1857', '1862', '1937', '1953', '1970', '1973', '1994', '2003', '2009', '2035', '2059', '2071', '2083', '2106', '2124', '2145', '2150', '2157', '2187', '2211', '2221', '2240', '2245', '2307', '2310', '2347', '2357', '2365', '2371', '2380', '2394']

# print(len(mathvision_cot_true_baseline_false))
# print(len(mathvision_cot_false_baseline_true))
# print(len(mathvista_cot_true_baseline_false))
# print(len(mathvista_cot_false_baseline_true))
# print(len(charxiv_cot_true_baseline_false))
# print(len(charxiv_cot_false_baseline_true))

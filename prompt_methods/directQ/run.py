


def directQ(input_text, image_path, model, cfg):
    prompt = "Please answer directly and only output the result. For multiple-choice questions, output the option.Do not generate any other irrelevant content.\n"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt + input_text},
            ],
        }
    ]
    response = model.get_response(messages, n = 1)
    
    return response







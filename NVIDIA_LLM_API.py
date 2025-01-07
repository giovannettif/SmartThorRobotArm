from openai import OpenAI

def process_input_with_api(input_prompt: str):
    # Complex prompt not necessary as arm is currently only capable of picking up predefined objects:
    # base_prompt = "Respond in a CSV file format and from an input sentence, list the object, action, target object, and classification (of whether it is picking, pushing, or placing the object), depending on the action listed. It MUST be one of these three classifications. For example: {Pick up the apple on the table}, object is apple, action is Pick up, target is table, classification is picking. Respond with nothing other than the CSV format {object, action, target object, classification}, no other characters, words, or the CSV column title are allowed . If a value does not exist, place a '?' for the corresponding section. No {} or \" ' "
    # Simple prompt for testing
    base_prompt = "Listen to the instructions carefully. A sentence will be provided, return only the object that is being acted on in the sentence. For example: {Pick up the apple on the table}, object is apple. Return 'apple' with no other characters, words, or text at all."

    # Initialize OpenAI client
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
    )

    # Create the completion request
    completion = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct-v0.3",
        messages=[{"role": "user", "content": f"{base_prompt} Try with the following: {input_prompt}"}],
        temperature=0.0,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )

    # Process the response stream
    response_content = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_content += chunk.choices[0].delta.content

    # Return the full response content (CSV format)
    return response_content

def process_csv_output(input_string: str):
    # Split the string by commas
    segments = input_string.split(',')
    
    # Ensure there are exactly 4 segments, otherwise return placeholders
    if len(segments) == 4:
        obj, action, target_obj, classification = segments
    else:
        obj, action, target_obj, classification = '?', '?', '?', '?'
    
    # Return the four variables
    return obj, action, target_obj, classification

if __name__ == "__main__":
  print(process_csv_output("pencil,Pick up,table,picking"))
  input_prompt = "Pick up the pencil"
  result = process_input_with_api(input_prompt)
  print(result)
import json
from typing import List

# Data Generation Prompt Builder
def build_prompt(dialog_context: List[dict], revealed_facts: List[str], image_descriptions: List[str], reasoning_mode: bool = False) -> str:
    dialog_str = "\n".join([f"{turn['speaker'].capitalize()}: {turn['text']}" for turn in dialog_context])
    facts_str = "\n- ".join(revealed_facts)
    images_str = "\n".join(image_descriptions) if image_descriptions else "None"

    if not reasoning_mode:
        instruction = """
            You are an expert reviewing a real multi-turn conversation between a user and an agriculture expert.

            Your task is to simulate what a thoughtful expert would have said in response to the user — either by:
            1. Asking a helpful clarification question if essential details are still missing.
            2. Or providing a grounded, helpful expert response if the user's goal can already be addressed.
            
            First, think step-by-step:
            - What is the user's goal?
            - What is known about their context?
            - What seems to be missing that is essential?
            - Then choose between <Clarify> or <Respond>
            - And generate the appropriate follow-up: clarification question or expert answer
            - Provide your reasoning for your decision in the <Reasoning> block

            

            You must return:
            <Reasoning>
            Your reasoning for your decision here
            </Reasoning>
            <Decision>
            <Clarify> or <Respond>
            </Decision>
            <Utterance> your clarification question or expert response here </Utterance>
            """ 
    else:
        instruction = """
            You are an expert reviewing a real multi-turn conversation between a user and a gardening or agriculture expert.

            Your task is to simulate what a thoughtful expert would have said in response to the user — either by:
            1. Asking a helpful clarification question if essential details are still missing.
            2. Or providing a grounded, helpful expert response if the user's goal can already be addressed.

            You must return:
            <Decision>
            <Clarify> or <Respond>
            </Decision>
            <Utterance> your clarification question or expert response here </Utterance>
            """

    prompt = f"""
    {instruction}

    Dialog context:

    {dialog_str}


    Revealed fact(s) from the user later:
 
    {facts_str}

    Attached image description(s):

    {images_str}
 """

    return prompt




# Prompt for Running LLM Zero-Shot and Reasoning baselines
def build_eval_prompt(dialog_context: List[dict], image_descriptions: List[str], mode: str = "zero_shot") -> str:
    """
    Builds prompt for clarify-or-respond task in zero-shot or reasoning mode.
    
    Args:
        dialog_context: A list of {"speaker": ..., "text": ...} dicts.
        image_descriptions: A list of image captions or placeholder strings.
        mode: One of ["zero_shot", "reasoning"]
    
    Returns:
        A formatted prompt string.
    """
    assert mode in {"zero_shot", "reasoning"}, f"Invalid mode: {mode}"

    # Format dialog history
    dialog_str = "\n".join([f"{turn['speaker'].capitalize()}: {turn['text']}" for turn in dialog_context])
    # Handle both list-of-strings and plain string
    if isinstance(image_descriptions, str):
        images_str = image_descriptions
    elif image_descriptions:
        images_str = "\n".join(image_descriptions)
    else:
        images_str = "None"

    if mode == "zero_shot":
        prompt = """
            You are an expert reviewer analyzing a real multi-turn conversation between a user and an agriculture expert. Your task is to simulate what a thoughtful expert would have said in response to the user's latest query or statement.

            Here is the conversation so far:

            <conversation>
            {CONVERSATION}
            </conversation>

            Attached image description(s):

            {IMAGE_DESCRIPTIONS}

            After carefully reviewing the conversation, you must decide whether to:
            1. Ask a helpful clarification question if essential details are still missing to address the user's needs fully.
            2. Provide a grounded, helpful expert response if you have enough information to address the user's goal.

            Your output must follow this format:
            <Decision>
            <Clarify> or <Respond>
            </Decision>
            <Utterance>
            Your clarification question or expert response here
            </Utterance>

            Guidelines for clarification questions:
            - Only ask for information that is truly necessary and not already provided in the conversation.
            - Frame questions in a way that guides the user towards providing specific, relevant details.

            Guidelines for expert responses:
            - Provide detailed, actionable advice based on the information given.
            - Include relevant scientific or technical information when appropriate.
            - Address all aspects of the user's query or concern.
            - If recommending products or techniques, explain the reasoning behind your recommendations.

            Remember to stay in character as a knowledgeable agriculture expert throughout your response. Your goal is to provide the most helpful and accurate information possible to assist the user with their agricultural query or concern.
"""

    elif mode == "reasoning":
        prompt = """
            You are an expert reviewer tasked with analyzing a real multi-turn conversation between a user and an agriculture expert. Your goal is to simulate what a thoughtful expert would have said in response to the user's latest query or statement.

            First, carefully read the conversation history:

            <conversation_history>
            {CONVERSATION}
            </conversation_history>

            Attached image description(s):

            {IMAGE_DESCRIPTIONS}

            Now, think through the following steps:

            1. Identify the user's primary goal or question.
            2. Analyze the known context and information provided so far.
            3. Determine if any essential details are missing to address the user's goal effectively.
            4. Decide whether to ask a clarification question or provide an expert response.

            Based on your analysis, choose between two actions:

            1. <Clarify>: Ask a helpful clarification question if essential details are still missing.
            2. <Respond>: Provide a grounded, helpful expert response if the user's goal can already be addressed.

            After making your decision, you must provide your output in the following format:

            <Reasoning>
            Explain your thought process and justification for your decision here. Include key points from your step-by-step thinking.
            </Reasoning>

            <Decision>
            State either <Clarify> or <Respond> based on your analysis.
            </Decision>

            <Utterance>
            If you chose <Clarify>, write a clear and specific clarification question here.
            If you chose <Respond>, provide a comprehensive, expert-level response addressing the user's goal or question.
            </Utterance>

            Additional guidelines:
            - Ensure your response is grounded in agricultural expertise and best practices.
            - If clarifying, ask only one question at a time and focus on the most critical missing information.
            - If responding, provide detailed, actionable advice tailored to the user's specific situation.
            - Maintain a professional and helpful tone throughout your response.
            - Do not introduce information that hasn't been mentioned in the conversation history.
            - If the user's question or goal is unclear, always err on the side of clarification.

            Remember to stay in character as a knowledgeable agriculture expert throughout your response. Your goal is to provide the most helpful and accurate information possible to assist the user with their agricultural query or concern.
            """

    return prompt.format(CONVERSATION=dialog_str, IMAGE_DESCRIPTIONS=images_str)


# Prompt for "decomposed" tasks: infer goal_state, then decision and generation
def build_eval_prompt_decomp(dialog_context: List[dict], image_descriptions: List[str], mode: str = "zero_shot") -> str:
    """
    Builds prompt for decomposed evaluation: first infer goal_state, then decide and generate.
    Supports zero-shot or reasoning (chain-of-thought) modes.
    """
    assert mode in {"zero_shot", "reasoning"}, f"Invalid mode: {mode}"
    dialog_str = "\n".join([f"{turn['speaker'].capitalize()}: {turn['text']}" for turn in dialog_context])
    # Handle both list-of-strings and plain string
    if isinstance(image_descriptions, str):
        images_str = image_descriptions
    elif image_descriptions:
        images_str = "\n".join(image_descriptions)
    else:
        images_str = "None"

    if mode == "zero_shot":
        prompt = """
            You are an expert assistant in agriculture and gardening. Your task is to analyze a multi-turn conversation and optional image(s), infer the task's goal state, decide whether to ask for clarification or provide a full response, and produce a follow-up utterance.

            Here is the conversation and optional image(s):

            First, review the conversation history:
            <conversation_history>
            {CONVERSATION_HISTORY}
            </conversation_history>

            Attached image description(s):

            {IMAGE_DESCRIPTIONS}

            If provided, review the image descriptions:
            <image_descriptions>
            {IMAGE_DESCRIPTIONS}
            </image_descriptions>


            To complete this task, follow these steps:

            1. Infer the goal state:
            - Carefully read through the conversation and examine any provided images.
            - Identify the known information related to the agricultural or gardening topic being discussed.
            - List any missing information that would be crucial to fully address the user's needs or questions.

            2. Decide on clarification or response:
            - If there is significant missing information that prevents you from providing a comprehensive answer, choose to ask for clarification.
            - If you have enough information to provide a helpful and complete response, choose to respond fully.

            3. Formulate your utterance:
            - If clarifying, ask a specific question to obtain the most critical missing information.
            - If responding, provide a comprehensive answer that addresses the user's needs, incorporating your expert knowledge in agriculture and gardening.

            4. Format your output:
            Present your analysis and response in the following JSON format:

            <output>
            {{
            "goal_state": {{
                "known": {{  
                // List key-value pairs of known information
                }},
                "missing": [
                // List missing information as strings
                ]
            }},
            "decision": "<Clarify> or <Respond>",
            "utterance": "Your clarification question or full response here"
            }}
            </output>

            Ensure that your utterance is tailored to the specific agricultural or gardening topic discussed in the conversation, and that it demonstrates your expertise in the field.
            """
    else:
        prompt = """
            You are an expert assistant in agriculture and gardening. Your task is to analyze a multi-turn conversation and optional image descriptions, then provide a structured response that demonstrates your reasoning and decision-making process.

            First, review the conversation history:
            <conversation_history>
            {CONVERSATION_HISTORY}
            </conversation_history>

            If provided, review the image descriptions:
            <image_descriptions>
            {IMAGE_DESCRIPTIONS}
            </image_descriptions>

            Now, follow these steps to analyze the conversation and formulate your response:

            1. Identify the user's goal: Determine the main objective or question the user is trying to address.
            2. Summarize known facts: List the relevant information provided in the conversation and image descriptions.
            3. List missing critical information: Identify any important details that are not yet known but would be helpful in addressing the user's goal.
            4. Decide whether to clarify or respond: Based on the available information, determine if you need to ask for clarification or if you have enough information to provide a response.
            5. Generate the corresponding utterance: Craft an appropriate response or question based on your decision.

            Present your reasoning and final output in the following format:

            <Think>
            [Your step-by-step reasoning about the user's goal, known facts, missing information, and decision to clarify or respond]
            </Think>

            <Finish>
            {{"goal_state": {{"known": {{[List known facts as key-value pairs]}}, "missing": [List missing critical information]}}, "decision": "<Clarify> or <Respond>", "utterance": "[Your generated response or question]"}}
            </Finish>

            Here are two examples of how to structure your response:

            Example 1:
            <Think>
            1. User's goal: The user wants to know how to care for their tomato plants that have yellowing leaves.
            2. Known facts: 
            - User has tomato plants
            - The leaves are turning yellow
            - Plants are grown in containers
            3. Missing information:
            - Watering frequency
            - Sunlight exposure
            - Fertilizer usage
            - Age of the plants
            4. Decision: Clarify, as we need more information to provide an accurate diagnosis and solution.
            5. Generate clarifying questions to gather missing information.
            </Think>

            <Finish>
            {{"goal_state": {{"known": {{"plant_type": "tomato", "symptom": "yellowing leaves", "growing_method": "containers"}}, "missing": ["watering_frequency", "sunlight_exposure", "fertilizer_usage", "plant_age"]}}, "decision": "<Clarify>", "utterance": "To better understand your tomato plant issue, could you please provide some additional information? How often do you water the plants? How much sunlight do they receive daily? Have you been using any fertilizer? Lastly, how old are the plants?"}}
            </Finish>

            Example 2:
            <Think>
            1. User's goal: The user wants advice on how to protect their garden from an upcoming frost.
            2. Known facts:
            - User has a vegetable garden
            - Frost is expected in the next few days
            - Garden includes tomatoes, peppers, and squash
            3. Missing information: None critical for this advice
            4. Decision: Respond, as we have enough information to provide useful advice on frost protection.
            5. Generate a response with methods to protect the garden from frost.
            </Think>

            <Finish>
            {{"goal_state": {{"known": {{"garden_type": "vegetable", "weather_forecast": "frost", "plants": ["tomatoes", "peppers", "squash"]}}, "missing": []}}, "decision": "<Respond>", "utterance": "To protect your vegetable garden from the upcoming frost, you can take several steps: 1) Cover your plants with blankets, sheets, or frost cloths before nightfall. 2) Water the soil thoroughly before the frost, as moist soil retains heat better than dry soil. 3) Use plastic milk jugs or bottles filled with warm water around your plants to release heat overnight. 4) For smaller plants, you can use inverted pots or buckets as covers. 5) If possible, harvest any ripe vegetables before the frost hits. Remember to remove the covers during the day to allow sunlight and air circulation."}}
            </Finish>

            Remember to always provide your reasoning in the <Think> tags and your structured output in the <Finish> tags. Your final output should only include these two sections.
            """
    return prompt.format(CONVERSATION_HISTORY=dialog_str, IMAGE_DESCRIPTIONS=images_str)   

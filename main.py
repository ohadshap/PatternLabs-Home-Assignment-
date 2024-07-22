from langchain_google_genai import ChatGoogleGenerativeAI
import json
import random
import asyncio

my_api_key = 'ENTER-KEY'


###########
#  SELECT - what part of the home assignment you want to check
###########
# input: 1 for part 1 / anything else for part 2
part_to_test = 1 


###########
# Project Settings
###########

# number of questions to send
number_of_questions_to_ask = 50

# should use async calls or should use delayed mechanizm - to prevent reaching free tier rate limit
# should stay false for free tier (rate limiting issues)
send_asyncly = False 

# delay in seconds between calls to llm
# used only if send_asyncly = False
delay_time = 3 

# the number of itteration the models can conflict each other - if they did not agree return first option
back_and_forth_limit = 4 

# create model instances
llm_main = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=my_api_key,
)
llm_sub = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=my_api_key,
)


# templase for explenations for model - beginning of prompt, need to add at the end the actual question
prompt_templates = {
    "basic": "I will give you a question or sentence to complete and two possible answers. Please answer either A or B, depending on which answer is better. You may write down your reasoning but please write your final answer (either A or B) between the <answer> and </answer> tags. ",
    "with_explain": "I will give you a question or sentence to complete and two possible answers. Please answer either A or B, depending on which answer is better AND write down your reasoning but please write your final answer (either A or B) between the <answer> and </answer> tags and the reasoning between <reason> and </reason> tags. ",
    "converse": "I  will give you a question or sentence to complete, two possible choices and an answer given by a different model and its reasoning. Please answer either Correct or Incorrect depending if you think the other model gave a correct or incorrect choice AND write down your reasoning but please write your final answer (either Correct or Incorrect ) between the <answer> and </answer> tags and the reasoning between <reason> and </reason> tags. "
}


# get fields from question object (which we get from the file) and prepare question string
def convert_question_to_string(question_object: dict):
    question = question_object["goal"]
    sol1 = question_object["sol1"]
    sol2 = question_object["sol2"]
    return f"'question': {question} 'Solution A': {sol1} 'Solution B': {sol2}"


# get prompt for starting conversation
def get_prompt(question_object: dict, template: str):
    question_string = convert_question_to_string(question_object)
    prompt = f"{prompt_templates[template]} {question_string}"
    return prompt


# get prompt for conversation between 2 models
def get_conversation_prompt(question_object, model_answer, model_reason):
    question_string = convert_question_to_string(question_object)
    review_prompt = f"{prompt_templates['converse']} {question_string} 'Answer From Model': {model_answer} 'Reason From Model: {model_reason}' "
    return review_prompt


# extract text between 2 given tags
def extract_result(model_answer: str, starting_tag: str, ending_tag: str):
    try:
        starting_index = model_answer.find(starting_tag)
        ending_index = model_answer.find(ending_tag)
        
        # if one of the tags was not found
        if starting_index == -1 or ending_index == -1: 
            raise Exception('Could not find index of starting_tag or ending_tag')
        
        cut_str_from = starting_index + len(starting_tag)
        extracted_result = model_answer[cut_str_from: ending_index]

        return extracted_result
    except Exception as e:
        print('Error - extract_result', e)
        raise Exception('Could not get data between the tags', e)


# call a given model with a given prompt
async def call_llm(llm_instance: ChatGoogleGenerativeAI, prompt: str):
    try:
        return llm_instance.invoke(prompt)
    except:
        print('calling llm - ERROR')
        raise Exception('Error calling llm')


# verify the model's answer against the answer from the answer file
def verify_model_answer_with_file(file_answer: str, model_answer: str):
    if int(file_answer) == 0:
        return model_answer == 'A'
    return model_answer == 'B'


# verify the first model's answer using another instance
# itterate a number of times and ask a model to evaluate the answer of a second model
# switch models each itteratiom
async def verify_with_second_instance(llm: str, question: dict, model_answer: str, model_reason):
    prev_model_answer = model_answer
    prev_model_reason = model_reason
    i = 0
    llm_to_use = llm
    # review_answer
    while i < back_and_forth_limit:
        print('prev_model_answer', prev_model_answer)
        print('prev_model_reason', prev_model_reason)
        print('question', question)
        conversation_prompt = get_conversation_prompt(question, prev_model_answer, prev_model_reason)
        if llm_to_use == 'sub':
            review_answer = await call_llm(llm_sub, conversation_prompt)
            llm_to_use = 'main'
        else:
            review_answer = await call_llm(llm_main, conversation_prompt)
            llm_to_use = 'sub'
            
        extracted_answer = extract_result(review_answer.content, '<answer>', '</answer>')
        extracted_reason = extract_result(review_answer.content, '<reason>', '</reason>')
        
        if extracted_answer == 'Correct':
            print('both-instances-aggreed', prev_model_answer)
            return prev_model_answer
        else:
            print('instances-did-not-agree', {
                "extracted_reason": extracted_reason,
                "prev_model_answer": prev_model_answer,
                "prev_model_reason": prev_model_reason,
                "extracted_answer": extracted_answer,
                })
            
            # update values for next itteration
            if prev_model_answer == 'A':
                prev_model_answer = 'B'
            else:
                prev_model_answer = 'A'
            
            prev_model_reason = extracted_reason
            i += 1
            # wait before sending another request to prevent rate limiting
            await asyncio.sleep(delay_time)
    
    print(f"After: {back_and_forth_limit} conversations between the 2 models they did NOT agree - returning None")   
    return None


async def ask_models(answer: str, question: dict):
    try:
        # get prepared prompt    
        prompt = get_prompt(question, 'with_explain')
        model_answer = await call_llm(llm_main, prompt)
        
        # get result from <answer>{result}</answer> tags
        extracted_answer = extract_result(model_answer.content, '<answer>', '</answer>')
        extracted_reason = extract_result(model_answer.content, '<reason>', '</reason>')
        
        verified_answer = await verify_with_second_instance('sub', question, extracted_answer, extracted_reason)
        
        final_answer = verified_answer or extracted_answer
        were_models_correct = verify_model_answer_with_file(answer, final_answer)
        full_answer = {
                "model_result": final_answer,
                "expected_answer": answer,
                "was_model_correct": were_models_correct,
            }
        print('FINAL RESULT - ', full_answer)
        return full_answer
    except:
        print('ask_model method FAILED')
        return {
            "was_model_correct": False,
        }
        

async def ask_model(answer: str, question: dict):
    try:
        # get prepared prompt    
        prompt = get_prompt(question, 'basic')
        
        # invoke instance with a question
        model_answer = await call_llm(llm_main, prompt)

        # get result from <answer>{result}</answer> tags
        extracted_result = extract_result(model_answer.content, '<answer>', '</answer>')
        
        # check if model was correct - based on the answers file
        was_model_correct = verify_model_answer_with_file(answer, extracted_result)
        # add relevant data
        full_answer = {
            "model_result": extracted_result,
            "expected_answer": answer,
            "was_model_correct": was_model_correct,
        }
        print('FINAL RESULT - ', full_answer)
        return full_answer        
    except:
        print('ask_model method FAILED')
        return {
            "was_model_correct": False,
        }

# calculate success rate in percentages (%)
def calculate_success_rate(success: int):
    return (success / number_of_questions_to_ask) * 100


# use asincio task group to send all request asyncly
async def send_prompts_asyncly(question_lines: list[str], answers_lines: list[str], selected_indexes: list[int]):
    tasks = []
    if part_to_test == 1:
        print('Testing Part 1')
        method_to_trigger = ask_model
    else:
        print('Testing Part 2')
        method_to_trigger = ask_models
        
    async with asyncio.TaskGroup() as tg:
        for selected_index in selected_indexes:
            question_object = json.loads(question_lines[selected_index])
            answer = answers_lines[selected_index]
            task = tg.create_task(method_to_trigger(answer, question_object))
            tasks.append(task)

    # wait and resolve all tasks
    results = [task.result() for task in tasks]
    return results
    

# send requests with delay in between to aviod reaching rate limit for free account
async def send_prompts_with_delay(question_lines: list[str], answers_lines: list[str], selected_indexes: list[int]):
    results = []
    if part_to_test == 1:
        print('Testing Part 1')
        method_to_trigger = ask_model
    else:
        print('Testing Part 2')
        method_to_trigger = ask_models

    for i, selected_index in enumerate(selected_indexes):
        print('send_prompts_with_delay - current index', i)
        print('send_prompts_with_delay - selected_index', selected_index)
        question_object = json.loads(question_lines[selected_index])
        answer = answers_lines[selected_index]
        result = await method_to_trigger(answer, question_object)
        results.append(result)
        await asyncio.sleep(delay_time)
    
    return results

async def main():
    print('main - started running')
    # open answers file and seperate it to lines
    answers_file = open('pattern-labs-answers.txt', 'r')
    answers_lines = answers_file.readlines()

    # open questions file and seperate it to lines
    questions_file = open('pattern-labs-questions.txt', 'r')
    question_lines = questions_file.readlines()

    # get number of random indexes - to make sure we follow the indexes of the random questions and answers
    selected_indexes = random.choices(range(0, len(question_lines)), k=number_of_questions_to_ask)
    print('selected_indexes', selected_indexes)
    # send all questions to the model
    results: list
    if send_asyncly:
        # send each question as a separate task to do all questions asyncly
        results = await send_prompts_asyncly(question_lines, answers_lines, selected_indexes)
    else:
        # send each question one by one with delay
        results = await send_prompts_with_delay(question_lines, answers_lines, selected_indexes)
    
    # itterate over results to register success vs fail
    succeeded = 0
    failed = 0
    for result in results:
        was_correct = result['was_model_correct']
        if was_correct:
            succeeded += 1
        else:
            failed += 1

    # calculate success rate in %
    success_rate = calculate_success_rate(succeeded)
    print("SUCCESS_RATE", success_rate)


asyncio.run(main())


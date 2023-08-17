"""
author: Elena Lowery

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
You will need to provide your IBM Cloud API key and a watonx.ai project id (any project)
for accessing watsonx.ai
This example shows a simple generation or Q&A use case without comprehensive prompt tuning
"""

# Install the wml api your Python env prior to running this example:
# pip install ibm-watson-machine-learning

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# Important: hardcoding the API key in Python code is not a best practice. We are using
# this approach for the ease of demo setup. In a production application these variables
# can be stored in an .env or a properties file

# URL of the hosted LLMs is hardcoded because at this time all LLMs share the same endpoint
url = "https://us-south.ml.cloud.ibm.com"

# Replace with your watsonx project id (look up in the project Manage tab)
watsonx_project_id = ""
# Replace with your IBM Cloud key
api_key = ""

# The get_model function creates an LLM model object with the specified parameters

def get_model(model_type,max_tokens,min_tokens,decoding,temperature):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
        )

    return model

def get_prompt(prompt_text):

    # While we are not running any meaningful logic in this function, we are demonstrating
    # a best practice for organizing code. Later we can create more complicated prompts in this function
    final_prompt = prompt_text

    return final_prompt

def invoke_LLM():

    # Look up parameters in documentation:
    # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#

    # You can specify any prompt and change parameters for different runs

    # If you want the end user to have a choice of the number of tokens in the output as well as decoding
    # and temperature, you can parameterize these values

    final_prompt = get_prompt("Write a paragraph about the capital of France.")
    model_type = ModelTypes.FLAN_UL2
    max_tokens = 300
    min_tokens = 50
    decoding = DecodingMethods.SAMPLE
    temperature = 0.7

    # Instantiate the model
    model = get_model(model_type,max_tokens,min_tokens,decoding, temperature)
    # Invoke the model and print the results
    generated_response = model.generate(prompt=final_prompt)
    # WML API returns a dictionary object. Generated response is a list object that contains generated text
    # as well as several other items such as token count and seed
    # We recommmend that you put a breakpoint on this line and example the result object
    print("Answer: " + generated_response['results'][0]['generated_text'])

invoke_LLM()
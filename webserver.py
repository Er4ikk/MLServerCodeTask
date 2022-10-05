import os
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline,T5Tokenizer, T5Config, T5ForConditionalGeneration
from flask import Flask, json,request
import nlp
import torch


INPUT_FILE="code_abstraction/buggyCode.txt"
OUTPUT_FILE="code_abstraction/abstractCode.txt"
IDIOMS_FILE="code_abstraction/idioms/idioms.csv"
COMMENT_GENERATOR_MODEL_PATH="finetuned-model/comment_generator"
MULTI_TASK_MODEL_PATH="finetuned-model/Pytorch-Model"
ABSTRACT_CONVERTER="code_abstraction/src2abs.jar"
MAP_FILE="code_abstraction/abstractCode.txt.map"
MODEL_RESPONSE_FILE="code_abstraction/modelResponse.txt"
ABSTRACT_MODEL_RESPONSE="code_abstraction/abstractModelResponse.txt"

api = Flask(__name__)


#EXAMPLE OF ABSTRACT CODE
#private TYPE_1 getType ( TYPE_2 VAR_1 ) { TYPE_3 VAR_2 = new TYPE_3 ( STRING_1 ) ; return new TYPE_1 ( VAR_2 , VAR_2 ) ; }


#POSSIBLES INFERENCES
# 1. generate small patch --> little bug fix on abstract code

# 2. generate medium patch --> medium bug fix maxium 100 tokens on abstract code

# 3. generate raw assert --> generates an assertion from raw code

# 4. generate abt patch --> generates an assertion from abstract code

# 5. generate comment --> generates a summarization of raw code note: the code must be pre-processed


#this method receives a post request containg the following object
#  {"message":"System.out.println("Hello")"}
# the inference has the prefix "generate small patch"

@api.route('/bug_fix_small', methods=['POST'])   
def small_bug_fix():
  json_received=request.get_json()
  code = abstract_code(json_received['message'])
  item='generate small patch: ' + code    
   
  answer=generate_answer(item)
  answer_to_string = ' '.join(map(str,answer))
  answer_to_string = abstract_model_response(answer_to_string.replace('"',' '))
  answer_to_string =deabstract_code(answer_to_string)
  
  print(answer_to_string )
  return json.dumps(answer_to_string )
  
@api.route('/bug_fix_medium', methods=['POST'])   
def medium_bug_fix():
  json_received=request.get_json()
  code = abstract_code(json_received['message'])
  item='generate medium patch: ' + code    
  
  answer=generate_answer(item)
  answer_to_string = ' '.join(map(str,answer))
  answer_to_string = abstract_model_response(answer_to_string.replace('"',' '))
  answer_to_string =deabstract_code(answer_to_string)
  
 
  print(answer_to_string )
  return json.dumps(answer_to_string )
   
  
  
@api.route('/assertion_raw', methods=['POST'])   
def assertion_raw():
  json_received=request.get_json()
  code = json_received['message']
  item='generate raw assert: ' + code 
  answer=generate_answer(item)
  return json.dumps(answer)
  
  
@api.route('/comment_summary', methods=['POST'])   
def comment_summary():
  json_received=request.get_json()
  code = json_received['message']
  item='generate comment: ' + code     
  answer=generate_answer(item)
  return json.dumps(answer)
   
   
def abstract_code(code):
  code_fragment=os.path.abspath(INPUT_FILE)
  abstract_code_fragment=os.path.abspath(OUTPUT_FILE)
  
  example_file= open (INPUT_FILE,"w")
  example_file.write(code)
  example_file.close()

  os.system("java -jar "+ABSTRACT_CONVERTER+" single method "+code_fragment+" "+abstract_code_fragment+" "+IDIOMS_FILE)

  output_file = open (OUTPUT_FILE,"r")
  response = output_file.read()
  return response
  
  
def abstract_model_response(code):
  code_fragment=os.path.abspath(MODEL_RESPONSE_FILE)
  abstract_code_fragment=os.path.abspath(ABSTRACT_MODEL_RESPONSE)
  
  example_file= open (MODEL_RESPONSE_FILE,"w")
  example_file.write(code)
  example_file.close()

  os.system("java -jar "+ABSTRACT_CONVERTER+" single method "+code_fragment+" "+abstract_code_fragment+" "+IDIOMS_FILE)

  output_file = open (ABSTRACT_MODEL_RESPONSE,"r")
  response = output_file.read()
  return response

def deabstract_code(code):
  map_fragment=os.path.abspath(MAP_FILE)
  model_code_fragment=os.path.abspath(MODEL_RESPONSE_FILE)
  
  example_file= open (MODEL_RESPONSE_FILE,"w")
  example_file.write(code)
  example_file.close()

  os.system("java -jar code_abstraction/deabstractor.main.jar  "+ map_fragment+" "+ model_code_fragment+" idioms/idioms.csv")

  model_response_file = open (MODEL_RESPONSE_FILE,"r")
  response = model_response_file.read()
  
  return response
  
def generate_answer(item):
   # tokenizer
   spm_path = MULTI_TASK_MODEL_PATH+'/dl4se_vocab.model'
   
   config_file = MULTI_TASK_MODEL_PATH+'/config.json'
   config = T5Config.from_json_file(config_file)
   
   tokenizer = T5Tokenizer.from_pretrained(spm_path)
   finetuned_model_path = MULTI_TASK_MODEL_PATH+'/pytorch_model.bin'
   
   #model creation
   model = T5ForConditionalGeneration.from_pretrained(
          finetuned_model_path,
          config=config
          )
        
   model.eval()
   
   tokenized_code=tokenizer.encode(item,return_tensors='pt')

   beam_output = model.generate(
                  tokenized_code,
                  max_length=512, 
                  num_beams=25, 
                  early_stopping=True
                )
                
                
   result= [tokenizer.decode(ids, skip_special_tokens=True)  for ids in beam_output]
     
   return result
   
if __name__ == '__main__':
    from waitress import serve
    serve(api, host="0.0.0.0", port=5000)






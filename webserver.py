
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline,T5Tokenizer, T5Config, T5ForConditionalGeneration
from flask import Flask, json,request
import nlp
import torch

COMMENT_GENERATOR_MODEL_PATH="finetuned-model/comment_generator"
MULTI_TASK_MODEL_PATH="finetuned-model/Pytorch-Model"

api = Flask(__name__)



@api.route('/code', methods=['POST'])
def comment_code():
  json_received=request.get_json()
  code = json_received['message']
  
  comment_for_code=generate_comment_code(code)
  
  
  return json.dumps(comment_for_code)

@api.route('/bug_fix_small', methods=['POST'])   
def small_bug_fix():
  json_received=request.get_json()
  print(json_received)
  code = json_received['message']
  
  item='generate small patch: ' + code 
  
      
  answer=generate_answer(item)
  return json.dumps(answer)
  
  
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
   
   
   
def generate_comment_code(code):
   pipeline = SummarizationPipeline(
    model=AutoModelWithLMHead.from_pretrained(COMMENT_GENERATOR_MODEL_PATH),
    tokenizer=AutoTokenizer.from_pretrained(COMMENT_GENERATOR_MODEL_PATH, skip_special_tokens=True)
   	#device=2
    )

   tokenized_code =code
   return pipeline([tokenized_code])
   

   
   
   
def generate_answer(item):
   spm_path = MULTI_TASK_MODEL_PATH+'/dl4se_vocab.model'
   
   config_file = MULTI_TASK_MODEL_PATH+'/config.json'
   config = T5Config.from_json_file(config_file)
   
   tokenizer = T5Tokenizer.from_pretrained(spm_path)
   finetuned_model_path = MULTI_TASK_MODEL_PATH+'/pytorch_model.bin'

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
    api.run() 






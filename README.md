# MLServerCodeTask

This is a web server written in python that use a t5 model
specialized in several coding tasks:

- generate bug fix
- generate comment
- generate assertions
- generate comment summary

for using this web server you need to install :

-torch
-flask (for http requests)
-transformers (for running inferences)


command to start the server:
> python3 webserver.py


server will run on http://localhost:5000!

API:
  
generate assertion(POST)
  http://localhost:5000/assertion_raw  

generate bug fix(POST)
  http://localhost:5000/bug_fix_small
  
 generate comment summary(POST)
  http://localhost:5000/comment_summary
 
 
download models here:
 https://www.dropbox.com/s/7reltqulwyh1wnq/finetuned-model.zip?dl=0
  
multi-task model used:
  https://drive.google.com/drive/folders/167AS_TI7cCWKpGzowgRdCTfDF11m9zsU
  


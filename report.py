from mdutils.mdutils import MdUtils
from mdutils import Html

import json
 
# Opening JSON file
f = open('metrics.json',)
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
print(data['accuracy'])
 
# Closing file
f.close()

mdFile = MdUtils(file_name='Report', title='DVC ML Pipeline testing')
mdFile.new_header(level=1, title='Accuracy of model is:')
mdFile.new_paragraph("**Accuracy**: {} ".format(data['accuracy']))

mdFile.create_md_file()
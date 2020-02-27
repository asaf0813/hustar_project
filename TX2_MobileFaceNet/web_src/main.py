import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud import storage
import os
import subprocess
from PIL import Image

device_name = "test"

access = subprocess.Popen(["/bin/bash","-i","-c","export GOOGLE_APPLICATION_CREDENTIALS='/home/tx2/CODE/app_src/google-services.json'"])
access.communicate()
#export GOOGLE_APPLICATION_CREDENTIALS="/home/tx2/CODE/app_src/google-services.json"

cred = credentials.Certificate('./google-services.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
folder_list = list()
docs = db.collection(u'Group').where(u'id',u'==', device_name).stream()
for doc in docs:
    users = db.collection(u'Group').document(doc.id).collection(u'User').stream() 
    for user in users:
        folder_list.append(user.to_dict()['name'])

client  = storage.Client()
bucket = client.get_bucket('fams-83306.appspot.com')
for i in folder_list:
    if os.path.isdir("../faces/training/%s" %i) == False:
        os.mkdir(os.path.join("../faces/training/%s"%i))
    else:
        pass

for j in folder_list:
    for i in range(60):
        try:
            blob = storage.Blob('hustar/%s/%s/%s%s.png' %(j,j,i), bucket)
            with open('../faces/training/%s/%s.jpg' %(j,i), 'wb') as file_obj:
                client.download_blob_to_file(blob, file_obj)
                print("success save file %s/%s" %(j,i))
        except:
            print("there is no file %s/%s" %(j,i))
            pass
# sp1 = subprocess.Popen(["/bin/bash","-i","-c","chmod 777 ./new_person.sh"])
# sp2 = subprocess.Popen(["/bin/bash","-i","-c","./new_person.sh"])
# sp1.communicate()
# sp2.communicate()
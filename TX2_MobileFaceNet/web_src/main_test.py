import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud import storage
import os
import subprocess

device_name = "test"
# access = subprocess.Popen(["/bin/bash","-i","-c","export GOOGLE_APPLICATION_CREDENTIALS='/home/tx2/CODE/app_src/google-services.json'"])
# access.communicate()
#export GOOGLE_APPLICATION_CREDENTIALS="/home/tx2/CODE/app_src/google-services.json"
cred = credentials.Certificate('./google-services.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

group_dict = dict()
docs = db.collection(u'Group').where(u'id',u'==', device_name).stream()
for doc in docs:
    users = db.collection(u'Group').document(doc.id).get()
    folder_list = list()
    for user in users.to_dict()[u'user_list']:
        folder_list.append(user)
    group_dict[doc.id] = folder_list
print(group_dict)
person_list = list()

client  = storage.Client()
bucket = client.get_bucket('fams-83306.appspot.com')
for group_name in group_dict.keys():
    if os.path.isdir("../faces/training/%s" %group_name) == False:
        os.mkdir(os.path.join("../faces/training/%s"%group_name))
    for person_name in group_dict[group_name]:
        person_list.append(person_name)
        if os.path.isdir("../faces/training/%s/%s" %(group_name,person_name)) == False and os.path.isdir("../faces/training/%s" %group_name) == True:
            os.mkdir(os.path.join("../faces/training/%s/%s"%(group_name,person_name)))
        if os.path.isdir("../faces/training/faceimage/%s" %(person_name)) == False:
            os.mkdir(os.path.join("../faces/training/faceimage/%s"%(person_name)))
        for person_picture_count in range(60):
            try:
                blob = storage.Blob('hustar/%s/%s/ %s_%s%s.png' %(group_name,person_name,group_name,person_name,person_picture_count), bucket)
                with open('../faces/training/faceimage/%s/%s.jpg' %(person_name,person_picture_count), 'wb') as file_obj:
                    client.download_blob_to_file(blob, file_obj)
                    print("success save file %s%s" %(person_name,person_picture_count))
            except:
                print("there is no file %s%s" %(person_name,person_picture_count))
                pass                      
        else:
            pass
    else:
        pass

# for j in folder_list:
#     for i in range(60):
#         try:
#             blob = storage.Blob('%s/%s/%s%s.png' %(j,j,i), bucket)
#             with open('../faces/training/%s/%s.jpg' %(j,i), 'wb') as file_obj:
#                 client.download_blob_to_file(blob, file_obj)
#                 print("success save file %s/%s" %(j,i))
#         except:
#             print("there is no file %s/%s" %(j,i))
#             pass
# sp1 = subprocess.Popen(["/bin/bash","-i","-c","chmod 777 ./new_person.sh"])
# sp2 = subprocess.Popen(["/bin/bash","-i","-c","./new_person.sh"])
# sp1.communicate()
# sp2.communicate()
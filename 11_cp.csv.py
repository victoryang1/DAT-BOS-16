from __future__ import print_function
import os, sys, time, random, re, optparse, datetime, shutil
basepath=os.path.abspath(os.curdir)
print("base %s" % (basepath))
cnt=0

for root, dirs, files in os.walk(os.curdir):
  #  print("root %s, dirs %s, files %s" % (root, dirs, files))
    for file in files:
      file_path=os.path.join(root, file)
      suffix=str(file_path)
  #    print ("file_path %s" % file_path)    
      words = re.split(r"[//]", suffix) 
      if len(words) >=1:
        suffix=words[1]
      else:
        suffix=str(cnt)   
#      print ("suffix %s" % suffix)                       
      if os.path.isfile(file_path):
        (f,ext)=os.path.splitext(file) 	
    #    print ("%s --> %s" % (file_path,new_path))	          
        if (ext.lower() == '.csv'):
          f=f+ext        	
          new_path=os.path.join(basepath, f)	          
          print ("%s to %s" % (file_path,new_path))	
          if not os.path.isfile(new_path):	         
            shutil.copy(file_path,new_path)	  
            cnt += 1	
                            
print("Files copied %d" % cnt)
import re, math, os, sys, time, random, subprocess, optparse
usage = 'Usage: %s -p <path>' % sys.argv[0]
# python all.random.rename.dir.py -u y -p ./
# python all.random.rename.dir.py -p ./
# Zebonastic Art, Design and Style
# Art, Design and Style with a tasty and satisfying crunch.
parser=optparse.OptionParser()
parser.add_option('-p', '--path', help='base dir relative path')
parser.add_option('-u', '--uniq', help='uniq')
parser.add_option('-l', '--lower', help='lower')
(opts, args) = parser.parse_args()

rename="python random.rename.dir.py -n "
clear="python remove.uniq.size.walkdir.py -p "

basepath=os.curdir 
if opts.path is not None:
  basepath = os.path.join(os.curdir,opts.path)

  
uniq=False
if opts.uniq is not None:
  uniq=True
  
lower=False
if opts.lower is not None:
  lower=True  

print (basepath)
abspath=os.path.abspath(basepath)
os.chdir(basepath)
print (os.getcwd())
print (abspath)

cnt=1
rm=1
for name in os.listdir(os.getcwd()):
  if os.path.isdir(name):
    print (name)
    sz={}
    newpath = os.path.abspath(os.path.join(abspath,name))
    print (newpath)
    os.chdir(newpath)
    for file in os.listdir(os.getcwd()):	
      if os.path.isfile(file):
        (root, ext) = os.path.splitext(file)    
        if (ext.lower() == '.csv') or (ext.lower() == '.ipynb'):
          size=os.path.getsize(file)
      #    print (size)		  
          size/=10.0
          size=math.floor(size)
          size*=10.0		  
       #   print ('round ',size	)	  
          cnt+=1	  
          if size in sz.keys():
            print ("rm %s sz %d" % (file, size))
            os.remove(file)	
            rm+=1		
          else:
            sz[size]=size
    os.chdir(abspath)

	
print (os.getcwd())
p=input()
print ( rm, ' removed from ',cnt,' files')

import os
HOME = os.getenv("HOME")

vivaldi_path = os.path.dirname(os.path.abspath(__file__))

f = open(HOME+'/.bashrc','r')
a = f.read()

b = "\nvivaldi_path=%s\npath=%s/src/py-src\nexport vivaldi_path\nexport PATH=$path:$PATH\n"%(vivaldi_path,vivaldi_path)
c = a + b
f.close()

f = open(HOME+'/.bashrc','w')
f.write(c)
f.close()

os.system('source ~/.bashrc')

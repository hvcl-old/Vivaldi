import os
HOME = os.getenv("HOME")

import os
vivaldi_path = os.path.dirname(os.path.abspath(__file__))

print vivaldi_path

f = open(HOME+'/.bash_profile','r')
a = f.read()

b = "\nvivaldi_path=%s\npath=%s/src/py-src\nexport vivaldi_path\nexport PATH=$path:$PATH\n"%(vivaldi_path,vivaldi_path)
c = a + b
f.close()

f = open(HOME+'/.bash_profile','w')
f.write(c)
f.close()


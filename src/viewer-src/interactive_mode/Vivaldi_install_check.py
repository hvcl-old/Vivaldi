try:
	from PyQt4 import QtGui, QtCore, QtOpenGL, Qt
	from PyQt4.QtOpenGL import QGLWidget
except:
	print "Vivaldi install warning: Cannot import PyQt"
	
# Opengl will be replaced to matrix module
#try:
#	from OpenGL.GL import *
#except:
#	print "Vivaldi install warning: cannot import OpenGL"

try:
	# Edit by Anukura $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	import sys, os
	Vivaldi_path = os.environ.get('vivaldi_path')
	sys.path.append(Vivaldi_path + "/src/viewer-src")

	import Vivaldi_viewer
	from Vivaldi_viewer import enable_viewer
except:
	print "Vivaldi install warning: cannot import Vivaldi viewer"
	
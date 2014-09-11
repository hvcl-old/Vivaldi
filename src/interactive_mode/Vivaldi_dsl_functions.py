try:
	from OpenGL.GL import *
except:
	print "Vivaldi cannot import OpenGL.GL"
	
try:
	from OpenGL.GLU import *
except:
	print "Vivaldi cannot import OpenGL.GLU"
	
try:
	from OpenGL.GLUT import *
except:
	print "Vivaldi cannot import OpenGL.GLUT"
	
	
glutInit (sys.argv)
glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
glutInitWindowSize (1,1)
idx = glutCreateWindow ("For OpenGL matrix init")
glutDestroyWindow(idx)




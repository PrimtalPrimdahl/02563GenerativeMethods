from numpy import eye, array, zeros
from math import cos, sin, pi
# If running this in Blender, comment out line below and back in the line below the one below.
from pygel3d import hmesh, graph, gl_display as gl
#import bpy

def rotation_matrix(angle, axis0):
    R = eye(4,4, dtype=float)
    axis1 = (axis0+1)%3
    axis2 = (axis0+2)%3
    R[axis1,axis1] = cos(angle)
    R[axis1,axis2] = -sin(angle)
    R[axis2,axis1] = sin(angle)
    R[axis2,axis2] = cos(angle)
    return R

def translation_matrix(v):
    T = eye(4,4, dtype=float)
    T[0:3,3] = v
    return T 

# Below the constants that define the specific type of tree. You need to 
# vary these in Part 3. Note that the parameters below are quite close
# to 'G' in Prusinkiewicz et al. Table 1.
r1 = 0.80
r2 = 0.80
alpha1 = 30 * pi/180
alpha2 = -30 * pi/180
phi1 = 137 * pi/180
phi2 = 137 * pi/180
q = 0.50
e = 0.50

min = 5.0

def transform(sstr):
    '''This function accepts an input string, which in our implementation is a list of tuples.
    The string is rewritten according to the production rules and returned.'''
    new_sstr = []
    for sym in sstr:
        match sym:
            case 'A', s, w:
                # Part 1:
                # Implemenet the production rule in Sec. 6.3 of Prusinkiewicz et al.
                if s < min:
                    continue
                new_sstr += [('!',w), ('F',s)]

                new_sstr += [('['),('+',alpha1), ('/', phi1),('A',s*r1, w * pow(q,e)),(']')]

                new_sstr += [('['),('+',alpha2), ('/', phi2),('A',s*r2, w * pow(1.0-q,e)),(']')]
            case _:
                new_sstr.append(sym)
    return new_sstr

# Part 2: Implement the SpaceTurtle
class SpaceTurtle:
    '''The SpaceTurtle class implements a simple 3D drawing system. The turtle can move and rotate around the Y axis, which
    we think of as turning and around the Z axis (rolling). It can also move forward in the Z direction. At any time you can
    query its position and width. Finally, the state can be pushed and popped from a stack.'''

    def __init__(self):
        self.T = eye(4,4)
        self.w = 0
        self.Tw_stack = []

    def get_pos(self):
        ''' Get position of SpaceTurtle. returns a 3D vector. '''
        p = self.T @ array([0,0,0,1],dtype=float)
        return p[0:3]

    def forward(self, dz):
        ''' Move dz forward in Z direction. '''
        self.T = self.T @ translation_matrix(array([0,0,dz]))

    def turn(self, _a):
        ''' Turn around Y axis. '''
        self.T = self.T @ rotation_matrix(_a,1)
        # Implement this function
        # placeholder to ensure program runs

    def roll(self, _a):
        ''' Roll around Z axis. '''
        self.T = self.T @ rotation_matrix(_a,2)
        # Implement this function
        # placeholder to ensure program runs

    def push(self):
        ''' Push current transformation matrix and width onto stack. '''
        self.Tw_stack.append((self.T,self.w))
        # Implement this function
        # placeholder to ensure program runs

    def pop(self):
        ''' Pop current transformation matrix and width from stack. '''
        self.T, self.w = self.Tw_stack.pop()
        # Implement this function
        # placeholder to ensure program runs

    def set_width(self, w):
        ''' Set the width. '''
        self.w = w
        # Implement this function
        # placeholder to ensure program runs

def realize(sstr):
    ''' The code below takes in a generated string and produces a list 
    of vertices and associated thicknesses by interpreting the string 
    and producing SpaceTurtle commands for each symbol. '''
    turtle = SpaceTurtle()
    vertices = [turtle.get_pos()]
    edges = []
    v = 0
    radii = {v: turtle.w}
    vertex_stack = []
    for sym in sstr:
        match sym:
            case 'F', d:
                turtle.forward(d)
                vo = v
                v = len(vertices)
                vertices.append(turtle.get_pos())
                edges.append((v,vo))
                radii[v] = turtle.w
            # Part 3:
            # The code above handles the turtle "drawing"
            # Implement code that handles all the other cases.
            case '[':
                turtle.push()
                #Remember position 
                vertex_stack.append(v)
            case ']': 
                turtle.pop()
                v = vertex_stack.pop()
            case '+', a :
                turtle.turn(a)
            case '/', a:
                turtle.roll(a)
            case '!', w:
                turtle.set_width(w)

    R = zeros((len(vertices), 1), dtype=float)
    for k,v in radii.items():
        R[k] = v
    return array(vertices), edges, R.flatten()


############################################################
#
# Exercise Part 3 cont.: 
# Once the realize function works, you can generate the
# tree and visualize the structure as a graph using PyGEL.
#
# HAND-IN
# For the assignment create at least three different trees
# by varying the parameters. You can get inspiration from 
# Figure 8 and Table 1 in the paper by Prusinkiewicz et al. 
# It is fine to simply use the parameters in Table 1 or you
# can come up with your own. In either case, Please discuss 
# how the parameter choices affect your results.
#
# Upload only a single PDF with 
# - the code you wrote, 
# - screenshots of the results, and 
# - the discussion of how parameters affect output.
#
############################################################

axiom = [('A', 100,40)]
sstr = axiom
# Levels of recursion
N = 10
for _ in range(N):
    sstr = transform(sstr)

vertices, edges, radii = realize(sstr)

# The code block below converts the tree to a PyGEL3D graph
# you want to comment it out if running from within Blender
g = graph.Graph()
for v in vertices:
    g.add_node(v)
for v0, v1 in edges:
    g.connect_nodes(v0,v1)
v = gl.Viewer()
v.display(g, bg_col=[1,1,1])

############################################################
#
# Part 4: (Non-mandatory)
# 4a - create the tree in Blender. If you comment out the 
#      PyGEL code, you can run the script from inside 
#      Blender. There is code below which allows you to
#      generate a set of curves in Blender, so this step
#      should be easy.
# 4b - convert the curves to a mesh using geometry nodes.
#      This can be done in a number of ways. I recommend
#      using the 'Curve to Mesh' node with a circle as 
#      profile curve.
# 4c - add materials and leaves using shader nodes and 
#      geometry nodes.
# 4d - create a grove of trees by using Poisson Disk 
#      Sampling to distribute points in a terrain.
#      
# # Uncomment to create a Blender object
# curve = bpy.data.curves.new('curve', 'CURVE')
# for v0, v1 in edges:
#     spline = curve.splines.new('POLY')
#     spline.points.add(1)
#     spline.points[0].co[0:3] = vertices[v0]
#     spline.points[0].radius = radii[v0]
#     spline.points[1].co[0:3] = vertices[v1]
#     spline.points[1].radius = radii[v1]    
# new_object = bpy.data.objects.new('tree', curve)
# bpy.data.collections['Collection'].objects.link(new_object)
#
############################################################


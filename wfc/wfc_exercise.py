# Wave Function Collapse Exercise
#
# Fill in the propagate and observe functions
# Complete the tasks at the end of the file
# Hand in short lab journal with 
# - the code that you wrote, 
# - answers to questions, and
# - the images you have produced.
# You hand in as a single PDF generated
# however you see fit.

from os import listdir
from os.path import isfile, join
from PIL import Image
import matplotlib.pyplot as plt
from numpy import zeros, ndindex, ones, ndenumerate
from random import randint, shuffle, uniform
from queue import Queue, LifoQueue
from time import time
import traceback 
import time


# The East North West South vector contains index pairs for
# adjacent cells.
ENWS = [[1,0],[0,1],[-1,0],[0,-1]]

#Use this to choose the size of the grid 
GRID_SIZE = 20

#Flags for enabling and disabling different methods 
use_queue = True
use_set = True

chosen_start = [-1,-1]



def ingrid(g, i, j):
    '''Check if i,j is inside the grid g'''
    return 0 <= i < g.shape[0] and 0 <= j < g.shape[1]

class Tile:
    NO_TILES = 0
    def __init__(self, img, N) -> None:
        self.img = img
        self.connections = zeros((4,N), dtype=int)
        self.n = Tile.NO_TILES
        Tile.NO_TILES += 1

    def show(self):
        '''Show is only for debugging purposes'''
        self.img.show()
        print(self.img.width, self.img.height)
        print(self.connections)

    def compatible_adjacency(self, t2, t2_idx):
        ''' Figure out if two tiles are compatible by checking row
        of pixels along adjacent edges.'''
        W,H = self.img.width, self.img.height
        pixels1 = [self.img.getpixel((W-1,y)) for y in range(H)]
        pixels2 = [t2.img.getpixel((0,y)) for y in range(H)]
        if pixels1 == pixels2:
            self.connections[0, t2_idx] = 1
        pixels1 = [self.img.getpixel((x,H-1)) for x in range(W)]
        pixels2 = [t2.img.getpixel((x,0)) for x in range(W)]
        if pixels1 == pixels2:
            self.connections[1, t2_idx] = 1
        pixels1 = [self.img.getpixel((0,y)) for y in range(H)]
        pixels2 = [t2.img.getpixel((W-1,y)) for y in range(H)]
        if pixels1 == pixels2:
            self.connections[2, t2_idx] = 1
        pixels1 = [self.img.getpixel((x,0)) for x in range(W)]
        pixels2 = [t2.img.getpixel((x,H-1)) for x in range(W)]
        if pixels1 == pixels2:
            self.connections[3, t2_idx] = 1

def load_tiles(exemplar):
    '''Load tiles from the specified tileset. Expand by creating
    rotated versions of each tileset'''
    path = 'wfc/tilesets/' + exemplar + '/'
    tiles = []
    fnames = [ f for f in listdir(path) if isfile(join(path, f)) ] 
    N = 4*len(fnames)
    for f in fnames:
        print(f)
        img = Image.open(join(path, f))
        tiles.append(Tile(img, N))
        tiles.append(Tile(img.transpose(Image.Transpose.ROTATE_90), N))
        tiles.append(Tile(img.transpose(Image.Transpose.ROTATE_180), N))
        tiles.append(Tile(img.transpose(Image.Transpose.ROTATE_270), N))
    for t0 in tiles:
        for i, t1 in enumerate(tiles):
            t0.compatible_adjacency(t1,i)
    return tiles

# -------------------------------------
# Here the tile set is loaded, so change line below to try other
# tile set.
tiles = load_tiles("Castle")
# -------------------------------------

#Dictionary which stores uncollapsed cells
wave_set = set()

def create_wave_set(wave_grid):
    s = set()
    for i in range(wave_grid.shape[0]):
        for j in range(wave_grid.shape[1]):
            if(sum(wave_grid[i,j]) > 1):
                s.add((i,j))
    return s



def pick_tile(we):
    '''from an array of possible tiles (0's and 1's) choose an entry containing 1
    and produce a new vector where only this entry is 1 and the rest are zeros'''
    l = []
    for i, wec in enumerate(we):
        if wec==1:
            l.append(i)
    x = randint(0,len(l)-1)
    we_new = zeros(len(we), dtype=int)
    we_new[l[x]] = 1
    return we_new

def pick_specific_tile(we, index):
    we_new = zeros(len(we), dtype=int)
    we_new[index] = 1
    return we_new

def wave_grid_to_image(wave_grid):
    '''Produce a displayable image from a wavegrid - possibly with superpositions.'''
    W = tiles[0].img.width
    H = tiles[0].img.height
    I = Image.new('RGBA',size=(W*wave_grid.shape[0], H*wave_grid.shape[1]), color=(255,255,255,255))
    N = len(tiles)
    for i,j in ndindex(wave_grid.shape[0:2]):
        entropy =  min(255.0,max(1.0,sum(wave_grid[i,j,:])+1e-6))
        mask = Image.new('RGBA', size=(W,H), color=(255, 255, 255, int(255/entropy)))
        for t_idx in range(N):
          if wave_grid[i, j, t_idx] == 1:
            I.paste(tiles[t_idx].img,  (W*i, H*j), mask)
    return I


is_first = True

def observe_no_set(wave_grid):
    '''The observe function picks a tile of minimum entropy and collapses
    its part of the wave function. A tuple with the tile indices is returned.'''

    x = chosen_start[0]
    y = chosen_start[1]
    global is_first
    if(x != -1 and y!=-1 and is_first):
        wave_grid[x, y, :] = pick_tile(wave_grid[x, y, :])
        is_first  = False
        return x,y
    
    #Choose random tile in grid 
    i, j = randint(0,wave_grid.shape[0]-1), randint(0,wave_grid.shape[1]-1)

    # Student code begin ---------------
    min_entropy = 100000
    min_entropy_cells = []
    index = 0

    #Find lowest entropy cell in wave_grid
    for row in range(0, wave_grid.shape[0]):
        for col in range(0,wave_grid.shape[1]):
            # if this tile has lower entropy we should save those indecies 
            curr_entropy = sum(wave_grid[row, col])
            if(curr_entropy > 1):
                if(curr_entropy == min_entropy):
                    min_entropy_cells.append([row,col])
                elif(curr_entropy < min_entropy and curr_entropy != 0):
                    min_entropy = curr_entropy
                    min_entropy_cells.clear()
                    min_entropy_cells.append([row,col])
    
    # Randomly choose index in min_entropy_cells
    if(len(min_entropy_cells) == 0):
        return -1,-1
    elif(len(min_entropy_cells) > 1):
        index = randint(0,len(min_entropy_cells)-1)
        
    i,j = min_entropy_cells[index]

    # Student code end ---------------
    wave_grid[i, j, :] = pick_tile(wave_grid[i, j, :]) #Observe and collapse this tile and return the index of the newly collapsed tile

    return i, j

def observe_set(wave_grid):
    '''The observe function picks a tile of minimum entropy and collapses
    its part of the wave function. A tuple with the tile indices is returned.'''

    x = chosen_start[0]
    y = chosen_start[1]
    global is_first
    if(x != -1 and y!=-1 and is_first):
        wave_grid[x, y, :] = pick_tile(wave_grid[x, y, :])
        is_first = False
        return x,y

    i, j = -1,-1

    # Student code begin ---------------
    min_entropy = 100000
    min_entropy_cells = []
    index = 0

    for element in wave_set:
        curr_entropy = sum(wave_grid[element[0], element[1]])
        if (curr_entropy == min_entropy):
            min_entropy_cells.append([element[0],element[1]])
        elif(curr_entropy < min_entropy and curr_entropy != 0):
            min_entropy = curr_entropy
            min_entropy_cells.clear()
            min_entropy_cells.append([element[0],element[1]])
    
    # Randomly choose index in min_entropy_cells
    if(len(min_entropy_cells) == 0):
        return -1,-1
    elif(len(min_entropy_cells) > 1):
        index = randint(0,len(min_entropy_cells)-1)
        
    i,j = min_entropy_cells[index]

    # Student code end ---------------
    wave_grid[i, j, :] = pick_tile(wave_grid[i, j, :]) #Observe and collapse this tile and return the index of the newly collapsed tile

    wave_set.remove((i,j))

    return i, j


def propagate_queue(wave_grid, i, j):
    '''Propagates the changes when a cell has been observed.'''
    # Student code begin ---------------
    #print("Propergating")
    q = LifoQueue()
    #Add current grid point to LIFO queue
    q.put((i,j))

    #Go through the queue until it is empty
    while not q.empty():
        #print("Going through queue")
        i,j = q.get()

        #Iterate through neighboring offsets 
        for d, n_off in enumerate(ENWS):
            ni, nj = i + n_off[0], j + n_off[1] #Get neighbor index 

            if ingrid(wave_grid, ni, nj): # If the neighbor is inside the grid 

                #Get current superposition 
                currSuperpos = sum(wave_grid[ni, nj])

                #Go though tiles in neighbor
                for idx_n, tile in enumerate(wave_grid[ni,nj]):

                    if(tile == 0):
                        continue

                    hasConnection = False
                   
                    # Check if it is compatible with the chosen tile for i,j

                    # Go through center tile and check neighbor tiles 
                    for idx_c, t1 in enumerate(wave_grid[i,j]):
                        if(t1 == 0):
                            continue  

                        #See if we have connection to this tile in this direction
                        hasConnection = (tiles[idx_c].connections[d, idx_n] == 1)

                        #We don't have to keep going since there is a match
                        if(hasConnection):
                            break


                    #If we don't have a connection to this tile remove it 
                    if(not hasConnection and sum(wave_grid[i,j]) != 0):
                        wave_grid[ni,nj,idx_n] = 0

                #If we ran out of choices move on to next neighbor
                if(sum(wave_grid[ni,nj]) == 0):
                    continue

                #Add the neighbor to the queue if we have connections to it and its superpositions got reduced 
                if(sum(wave_grid[ni,nj]) < currSuperpos):
                    q.put((ni,nj))
                
    # Student code end ---------------
                    
def propagate_no_queue(wave_grid, i, j):
    '''Propagates the changes when a cell has been observed.'''
    # Student code begin ---------------
    
    #Iterate through neighboring offsets 
    for d, n_off in enumerate(ENWS):

        ni, nj = i + n_off[0], j + n_off[1] #Get neighbor index 
        if ingrid(wave_grid, ni, nj): # If the neighbor is inside the grid 

            #Go though tiles in neighbor
            for idx_n, tile in enumerate(wave_grid[ni,nj]):
                if(tile == 0):
                    continue

                hasConnection = False
                
                # Go through center tile and check neighbor tiles 
                for idx_c, t1 in enumerate(wave_grid[i,j]):
                    if(t1 == 0):
                        continue 
                    
                    #See if we have connection to this tile in this direction
                    hasConnection = (tiles[idx_c].connections[d, idx_n] == 1)

                    #We don't have to keep going since there is a match
                    if(hasConnection):
                        break


                #If we don't have a connection to this tile remove it 
                if(not hasConnection):
                    wave_grid[ni,nj,idx_n] = 0  

                #If we ran out of choices move on to next neighbor
                if(sum(wave_grid[ni,nj]) == 0):
                    print("NEIGHBOR IS 0")
                    return
    # Student code end ---------------


def WFC(wave_grid):
    try:
        if(use_set):
            i, j = observe_set(wave_grid)
        else:
            i,j = observe_no_set(wave_grid)

        if(use_queue):
            propagate_queue(wave_grid, i, j)
        else:
            propagate_no_queue(wave_grid, i, j)
        if(i == -1 and j == -1):
            print("NO MORE TILES")
            return False
        return True
    except Exception as e:
        traceback.print_stack
        print(e)
        return False


def run_interactive(wave_grid):
    '''This function runs WFC interactively, showing the result of each
    step. '''
    I = wave_grid_to_image(wave_grid)
    I.save("wfc/img0.png")
    fig = plt.figure()
    plt.ion()
    plt.imshow(I)
    plt.show(block=False)
    W = tiles[0].img.width
    H = tiles[0].img.height
    while WFC(wave_grid):
        fig.clear()
        I = wave_grid_to_image(wave_grid)
        plt.imshow(I)
        plt.show(block=False)
        plt.pause(0.00000001)
    I.save("wfc/img1.png")


def run(wave_grid):
    '''Run WFC non-interactively. Much faster since converting to 
    an image is the slowest part by far.'''
    I = wave_grid_to_image(wave_grid)
    I.save("img0.png")
    while WFC(wave_grid):
        pass
    I = wave_grid_to_image(wave_grid)
    I.save("img1.png")

# Part 1: When observe and propagate have been fixed, 
# you can run the code below to produce a texture.
# Try with a number of tilesets. The resulting images
# are submitted. Try at least "Knots" and "FloorPlan"
wave_grid = ones((GRID_SIZE,GRID_SIZE,len(tiles)), dtype=int)
wave_set = create_wave_set(wave_grid)

run_interactive(wave_grid)
run(wave_grid)

'''The resulting images made during this process can be seen in Circuit25.png, FloorPlan25.png, Knots20.png, Knots15.png, Rooms15.png'''

# Part 2: Introduce constraints by precollapsing one or more 
# cells to admit only one or more tiles in these cells. Discuss 
# what you are trying to achieve and submit the discussion along 
# with the resulting images.


tiles = load_tiles("Circuit")
wave_grid = ones((GRID_SIZE,GRID_SIZE,len(tiles)), dtype=int)
#First we choose a cell to precollapse
idx = int((GRID_SIZE-1)/2)

#We can either precollapse into a random tile or choose our own. Here we choose our own:
wave_grid[idx, idx] = pick_specific_tile(wave_grid[idx,idx],19)
wave_grid[idx+1, idx] = pick_specific_tile(wave_grid[idx+1,idx],18)
# wave_grid[idx, idx+1] = pick_specific_tile(wave_grid[idx,idx+1],5)
# wave_grid[idx+1, idx+1] = pick_specific_tile(wave_grid[idx+1,idx+1],5)

wave_set = create_wave_set(wave_grid)

#This variable makes it possible for us to choose the starting point of our propergation in the first round
chosen_start = [idx+1,idx]
#run_interactive(wave_grid)

'''Now by precollapsing specific cells to choose a tile, we are able to create a starting point which we like.
# E.g. for the tileset "Circuits" we wanted a component in the middle of the grid and thus we precollapse those cells into the specific tile we wanted. 
One downside of doing this is that depending on the number of precollapse tiles it makes it increasingly difficult for the algorithm to generate a texture with no 0 entropy cells. 
Meaning it has a higher probability of running into situations where no tile fits the given cell. 
Furthermore choosing specific tiles can cause the image which is generate to be very repetitive 
E.g. choosing the dskew tile for the circuts makes it so that there is a clear preference for that tile. This can be seen in image Circuts20dskew.png '''

# Part 3: Change your propagate function such that only adjacent
# cells are updated. Use this to produce a new texture based
# on FloorPlan. Does this change the result? If so how? Show
# images. 
tiles = load_tiles("FloorPlan")
wave_grid = ones((GRID_SIZE,GRID_SIZE,len(tiles)), dtype=int)
wave_set = create_wave_set(wave_grid)
use_queue = False
chosen_start = [-1,-1]

#start = time.time()
#run_interactive(wave_grid)
#end = time.time()
#print(end-start)

'''This has been implemented in the propagate_no_queue function. Setting the flag use_queue to False will enable usage of this function.
Using this approach we experience that the amount of 0 entropy cell occurences go up, which is probably explained by the fact the we don't propagate the entropy changes to the whole grid.
A version using the queue can be seen in FloorPlan20Queue.png
A version without the queue can be seen in FloorPlan20NoQueue.png
However, one upside of using this approach is that it is quite a bit faster that the queue implementation. With queue:2.135s Without queue: 0.766s '''

# Part 4 (NON-MANDATORY AND PROBABLY HARD)
# Input a single image and make the tileset from patches in this
# image. See if you can produce results similar to Marie's




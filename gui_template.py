# This file contains code for creating a simple GUI through which you can 
# have the user input data to train your network.
#
# The usage and/or manipulation of the contents of this file completely
# up to you: You can use this code as is, not use it at all, or manipulate 
# it anyway you would like.
#

import numpy as np
import pygame


# This function collects (binary, ie. black(-1) and white(1)) image data 
# from the user through a GUI. 
#
# The user is able to draw an image on the GUI by clicking on the 'squares'
# on the displayed black image. (There are DIMxDIM squares on the image.)
# Clicked square turns to white.
# Clicking it again turns it off and recolors it to black.
# The user can 'save' one input by pressing Enter.
# The GUI then starts waiting for new input.
# Inputting images phase can be stopped by pressing 'ESC'.
#
# The collected data will be of the type of a list of numpy.arrays,
# you can change both types if you wish.
#
def get_images():
    pygame.init()
    
    # CONSTANTS
    SIZE = 700       # Size of the GUI window (pixels)
    DIM = 20         # Number of clickable squares in one dimension of the image.
                     # The resulting training data will have dimensions DIMxDIM.

    white = [255,255,255]  #corresponds to 1 in the training data
    black = [0,0,0]        #corresponds to -1 in the training data

    # Size of the window:
    win_size = [SIZE, SIZE]

    # Create a pygame screen:
    screen = pygame.display.set_mode(win_size)
    pygame.display.set_caption('ASSOCIATIVE MEMORY::DATA COLLECTOR'      \
                               + ' -- press ESC/Q/q to quit')

    # Dimensions of the picture:
    dims = [DIM, DIM]

    # Initialize the training data to be stored
    data = []
    
    while True:
        # Allocate space for new datum, initialize it to -1's
        data.append(np.zeros((dims[0], dims[1])) - 1)

        # Re-initialize the GUI window
        screen.fill(black)
        guifont = pygame.font.Font(None,20)
        label   = guifont.render('Draw image #'            \
                                 +str(len(data))                \
                                 +' and press Enter', True, white, black)
        screen.blit(label,(win_size[0]*0.3,2))
        pygame.display.flip()

        is_input_complete = False
        draw_white = False
        draw_black = False
        while not is_input_complete:
            for event in pygame.event.get(): 
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    draw_white = True
                    draw_black = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    draw_white = False
                    draw_black = True                    
                elif event.type == pygame.MOUSEBUTTONUP: 
                    draw_white = False
                    draw_black = False
                
                elif event.type == pygame.KEYDOWN        \
                     and event.key == pygame.K_RETURN:
                        is_input_complete = True

                if event.type == pygame.QUIT             \
                   or (event.type == pygame.KEYDOWN      \
                       and (event.key == pygame.K_ESCAPE \
                            or event.key == pygame.K_q)):
                    # Discard final unused datum space
                    data.pop()
                    
                    # Function quits by returning collected data
                    print 'Successfully collected', len(data), 'data points'
                    return data

            if draw_white or draw_black:
                mouse_pos   = pygame.mouse.get_pos()
                coords      = [int(round(mouse_pos[0]*dims[0]/win_size[0])), \
                               int(round(mouse_pos[1]*dims[1]/win_size[1]))]
                image_patch = pygame.Rect(coords[0]*win_size[0]/dims[0],     \
                                          coords[1]*win_size[1]/dims[1],     \
                                          win_size[0]/dims[0],               \
                                          win_size[1]/dims[1])               
                                          
                if draw_white:               
                    data[-1][coords[0], coords[1]] = 1
                    pygame.draw.rect(screen,white,image_patch,0)     

                elif draw_black:
                    data[-1][coords[0], coords[1]] = -1
                    pygame.draw.rect(screen,black,image_patch,0)     
                pygame.display.flip()
       
    
            
            


# This function is for visualizing a single image
# Again the 'image' is assumed to be of numpy.array type,
# but you are free to change that.
def visualize_image(img):
    pygame.init()
    
    # CONSTANTS
    SIZE = 700       # Size of the GUI window (pixels)
    DIM = 20         # Number of clickable squares in one dimension of the image.
                     # The resulting training data will have dimensions DIMxDIM.

    white = [255,255,255]  #corresponds to 1 in the training data
    black = [0,0,0]        #corresponds to -1 in the training data
    
    # Size of the window:
    win_size = [SIZE, SIZE]

    # Create a pygame screen:
    screen = pygame.display.set_mode(win_size)
    pygame.display.set_caption('ASSOCIATIVE MEMORY::VISUALIZER'      \
                               + ' -- press ESC/Q/q to quit')

    # Dimensions of the picture:
    dims = [DIM, DIM]
 
    # Initialize the GUI window
    screen.fill(black)
    guifont = pygame.font.Font(None,20)
    label   = guifont.render('', True, white, black)
    screen.blit(label,(win_size[0]*0.4,2))
    pygame.display.flip()

    for x_coord in range(dims[0]):
        for y_coord in range(dims[0]):
            image_patch = pygame.Rect(x_coord*win_size[0]/dims[0],     \
                                      y_coord*win_size[1]/dims[1],     \
                                      win_size[0]/dims[0],             \
                                      win_size[1]/dims[1])                
            if img[x_coord, y_coord] == 1:                               
                pygame.draw.rect(screen,white,image_patch,0)     
            # if it was white, turn it to black
            else:                                               
                pygame.draw.rect(screen,black,image_patch,0)     
    pygame.display.flip()            

    while True:
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT             \
               or (event.type == pygame.KEYDOWN      \
                   and (event.key == pygame.K_ESCAPE \
                        or event.key == pygame.K_q)):
                return


if __name__=='__main__':
    data = get_images()
    if len(data) != 0:
        visualize_image(data[-1])

import pygame
import numpy as np
import math
from collections import deque




#initializing pygame
pygame.init()

screen = pygame.display.set_mode((800,600))
pygame.display.set_caption('Attack')
running = True
sship = pygame.image.load('spaceship.png')
alien = pygame.image.load('alien.png')
bullet = pygame.image.load('bullet (1).png')
space = pygame.image.load('6560061.jpg')



score = 0
store_pos = []
font = pygame.font.Font(None, 36)
text = font.render(f"Score : {score}", True, (0,255,0))

sship_x = 320
sship_y = 510

# enemy 1
random_X = np.random.randint(10,736)
random_y = np.random.randint(20,400)

# enemy 2
random_X1 = np.random.randint(10,736)
random_y1 = np.random.randint(20,400)

# enemy 3
random_X2 = np.random.randint(10,736)
random_y2 = np.random.randint(20,400)

# enemy 4
random_X3 = np.random.randint(10,736)
random_y3 = np.random.randint(20,400)

# enemy 5
random_X4 = np.random.randint(10,736)
random_y4 = np.random.randint(20,400)

# bullet 
bullet_y = 0
types = 'not_fire'

def bullet_(x,y):
    screen.blit(bullet,(x,y))

def our_sship(x,y):
    screen.blit(sship,(x,y))
    
def alien_(x,y):
    screen.blit(alien,(x,y))

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)



FPS = 60  # Desired frame rate
clock = pygame.time.Clock()

while running:
    screen.fill((0,0,0))
    screen.blit(space,(0,0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                sship_x -= 20
                
            if event.key == pygame.K_RIGHT:
                sship_x += 20
                
            if event.key == pygame.K_SPACE:
                types = 'Fire'
                
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_SPACE:
                pass
    
    if sship_x >= 736:
        sship_x = 736
    elif sship_x <= 10:
        sship_x = 10

    # enemy 1
    random_X  += 2.5
    if random_X >= 736:
        random_X = 10
        random_y += 45
    
    if random_y >= 550:
        random_X = 10
        random_y = 20  
    alien_(random_X,random_y)  
    

    # enemy 2
    random_X1  += 2
    if random_X1 >= 736:
        random_X1 = 10
        random_y1 += 45
    
    if random_y1 >= 550:
        random_X1 = 10
        random_y1 = 20     
    alien_(random_X1,random_y1)  
    
    
    
    # enemy 3
    random_X2  += 3.5
    if random_X2 >= 736:
        random_X2 = 10
        random_y2 += 45
    
    if random_y2 >= 550:
        random_X2 = 10
        random_y2 = 20   
    alien_(random_X2,random_y2)
    # store_pos.append((random_X2,random_y2))
    
    # enemy 4
    random_X3  += 4
    if random_X3 >= 736:
        random_X3 = 10
        random_y3 += 45
    
    if random_y3 >= 550:
        random_X3 = 10
        random_y3 = 20
    alien_(random_X3,random_y3)
   

    
    # bullet positions
    if types == 'Fire':
        
        pos1 = random_X,random_y,'alien1'
        pos2 = random_X1,random_y1,'alien2'
        pos3 = random_X2,random_y2,'alien3'
        pos4 = random_X3,random_y3,'alien4'
        
        print(pos1 , pos2 , pos3 , pos4) 

        bullet_y +=  7  
        bullet_(sship_x+20,sship_y-bullet_y) 
        
        for distance in (pos1,pos2,pos3,pos4):
            dis = calculate_distance(distance[0],distance[1],sship_x+20,sship_y-bullet_y)
            if 59 >= int(dis) <= 60:
                
                score += 1
                
                if distance[2] =='alien1':
                     random_X , random_y = 0,-610
                  
                     
                if distance[2] =='alien2':
                     random_X1 , random_y1 = 0,-610
                
                if distance[2] =='alien3':
                     random_X2 , random_y2 = 0,-610
                 
                    
                if distance[2] =='alien4':
                     random_X3 , random_y3 = 0,-610

             
                types = 'not_fire'
                bullet_y = 0
           

        if bullet_y > 500:
            types = 'not_fire'
            bullet_y = 0
            
    
    # ship position
    our_sship(sship_x,sship_y)
    
    text = font.render(f"Score : {score}", True, (0,255,0))
    screen.blit(text, (0,20))
    
    pygame.display.update()
    clock.tick(FPS)
    
    
    

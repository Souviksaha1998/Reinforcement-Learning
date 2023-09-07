import pygame
import numpy as np
import math
from collections import deque
import secrets
import itertools
import cv2

#initializing pygame
pygame.init()

class SpaceGame():
    def __init__(self,FPS) -> None:
        
        self.screen = pygame.display.set_mode((800,600))
        pygame.display.set_caption('Attack')
        self.running = True
        self.sship = pygame.image.load('spaceship.png')
        self.alien = pygame.image.load('alien.png')
        self.bullet = pygame.image.load('bullet (1).png')
        self.space = pygame.image.load('11.jpg')
        
        self.episode = 0
        self.score = 0
        self.enemys = []
        self.font = pygame.font.Font(None, 36)
        self.text = self.font.render(f"Score : {self.score}", True, (0,255,0))

        
        self.sship_x = np.random.randint(20,700)
        self.sship_y = 510
        
        self.bull = 0
        
        self.types = 'not_fire'
        self.bullet_y = 0
        self.bullet_x_constant =  0
        self.dis = None
        
        self.FPS = FPS
        self.clock = pygame.time.Clock()
        
        self.enemy1 = self.random_enemy_pos()
        self.enemy2 = self.random_enemy_pos()
        self.enemy3 = self.random_enemy_pos()
        self.enemy4 = self.random_enemy_pos()
        
        self.reward = 0
        self.bullet_no = 1
        self.dis = None
        
    
        
    def actions_(self):
        return ['left','right','fire']
    
    def random_action(self):
        return np.random.randint(3)
    
    def play_step(self,action):
        
                self.observation = []
                frame = pygame.surfarray.array3d(self.screen)
                frame = cv2.flip(cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE),1)
                
                # obs , reward , done , info
                self.done = False

                self.screen.fill((0,0,0))
                self.screen.blit(self.space,(0,0))
                

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        
                
                if action == 0:
            
                    self.sship_x -= 50
                    self.reward -= 0.03
                            
                if action == 1:
                    
                    
                    self.sship_x += 50
                    self.reward -= 0.03
                            
                if action == 2:
                    # self.reward -= 0.0002
                    if self.bullet_no == 1:
                            self.bullet_x_constant = self.sship_x
                            self.types = 'Fire'
                            self.bullet_no += 1
                            self.bull += 1
                         
                    # if self.dis is not None:
                    if  self.bullet_y == 0 or int(self.dis) <= 40 :
                                            self.bullet_x_constant = self.sship_x
                                            self.types = 'Fire'
                                            self.bull += 1
                                            
                    self.reward -= 0.03
                    
                    
                
                # enemy position and  their moving speed
                self.enemy1[0],self.enemy1[1] = self.alien_(self.enemy1[0],self.enemy1[1],2.5)
                self.enemy2[0],self.enemy2[1] = self.alien_(self.enemy2[0],self.enemy2[1],3.5)
                self.enemy3[0],self.enemy3[1] = self.alien_(self.enemy3[0],self.enemy3[1],4.5)
                self.enemy4[0],self.enemy4[1] = self.alien_(self.enemy4[0],self.enemy4[1],3)
                

                self.enemy_and_spaceship_distance()
                
                # bullet positions
                if self.types == 'Fire':
     
                    if self.bull >= 10:
                        self.reward -= 50
                        self.done = True

                    pos1 = self.enemy1[0],self.enemy1[1],'alien1'
                    pos2 = self.enemy2[0],self.enemy2[1],'alien2'
                    pos3 = self.enemy3[0],self.enemy3[1],'alien3'
                    pos4 = self.enemy4[0],self.enemy4[1],'alien4'
         
                    # bullet speed
                    self.bullet_y +=  12
                    self.bullet_(self.bullet_x_constant+20,self.sship_y-self.bullet_y)
         
                    for distance in (pos1,pos2,pos3,pos4):
                        self.dis = self.calculate_distance(distance[0],distance[1],self.bullet_x_constant+20,self.sship_y-self.bullet_y)
                        
                        
                        if  int(self.dis) <= 40 :
        
                            if distance[2] =='alien1':
                                self.score += 1
                                self.enemy1[0],self.enemy1[1] = 0,-0
                                self.reward += 10
                    
                            if distance[2] =='alien2':
                                self.score += 1
                                self.enemy2[0],self.enemy2[1] = 0,-0
                                self.reward += 10
                            
                            if distance[2] =='alien3':
                                self.score += 1
                                self.enemy3[0],self.enemy3[1] = 0,-0
                                self.reward += 10
                            
                                
                            if distance[2] =='alien4':
                                self.score += 1
                                self.enemy4[0],self.enemy4[1] = 0,-0
                                self.reward += 10

                        
                            self.types = 'not_fire'
                            self.bullet_y = 0
                          
                    
                     
                    if self.bullet_y > 500:
 
                        self.types = 'not_fire'
                        self.bullet_y = 0
                        self.reward -= 10
     
                self.our_sship(self.sship_x,self.sship_y)
            
                text = self.font.render(f"Score : {self.score}", True, (0,255,0))
                self.screen.blit(text, (0,20))

                pygame.display.update()
                self.clock.tick(self.FPS)
                
                if  self.score == 4:
                    print(f'Score 4')
                    self.reward += 100
                    self.done = True
                
                # self.observation.extend([frame])
                return frame , self.reward , self.done
            
        
    
    def bullet_(self,x,y):
        self.screen.blit(self.bullet,(x,y))

    def our_sship(self,x,y):
        if self.sship_x >= 736:
                self.sship_x = 736
        elif self.sship_x <= 10:
                self.sship_x = 10
        self.screen.blit(self.sship,(self.sship_x,self.sship_y))
        
    def random_enemy_pos(self):
        random_X = np.random.randint(10,736)
        random_y = np.random.randint(20,400)
        return [random_X,random_y]
    
    def alien_(self,x,y,move_speed):
        x += move_speed
        if x >= 736:
            x = 10
            y += 45
    
        if y >= 550:
            x = 10
            y = 20 

        self.screen.blit(self.alien,(x,y))
        return x , y
    
    def enemy_and_spaceship_distance(self):
        for distances in ((self.enemy1[0],self.enemy1[1]),(self.enemy2[0],self.enemy2[1]),(self.enemy3[0],self.enemy3[1]),(self.enemy4[0],self.enemy4[1])):
                dis = self.calculate_distance(distances[0],distances[1],self.sship_x,self.sship_y)
             
                if 59 >= int(dis) <= 60:
                    
                    self.reward -= 100
                    self.done = True
                    
                    

    def calculate_distance(self,x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def reset(self):
        frame = pygame.surfarray.array3d(self.screen)
        frame = cv2.flip(cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE),1)
        self.bull = 0
        self.done = False
        self.reward = 0
        self.observations = []
        self.score = 0
        self.bullet_y = 0
        self.bullet_x_constant =  0
        self.sship_x , self.sship_y = np.random.randint(20,700) , 510
        self.enemy1 = self.random_enemy_pos()
        self.enemy2 = self.random_enemy_pos()
        self.enemy3 = self.random_enemy_pos()
        self.enemy4 = self.random_enemy_pos()
        # self.observations.extend([frame])
        return frame , self.reward , self.done

    def play_game(self):
        
        
        self.enemy1 = self.random_enemy_pos()
        self.enemy2 = self.random_enemy_pos()
        self.enemy3 = self.random_enemy_pos()
        self.enemy4 = self.random_enemy_pos()
        
        

        while self.running:
            self.screen.fill((0,0,0))
            self.screen.blit(self.space,(0,0))
            
            
            # keyboard press
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.sship_x -= 50
                        
                    if event.key == pygame.K_RIGHT:
                        self.sship_x += 50
                    
                    if event.key == pygame.K_SPACE:
                        
                        if self.bullet_no == 1:
                            self.bullet_x_constant = self.sship_x
                            self.types = 'Fire'
                            self.bullet_no += 1
                            
                       
                        if  self.bullet_y == 0 or int(self.dis) <= 40 :
                                self.bullet_x_constant = self.sship_x
                                
                                print(f'ship : {self.sship_x}')
                  
                                self.types = 'Fire'
                         
                        
               
                            
                            
                
               
            # enemy position and  their moving speed
            self.enemy1[0],self.enemy1[1] = self.alien_(self.enemy1[0],self.enemy1[1],2.5)
            self.enemy2[0],self.enemy2[1] = self.alien_(self.enemy2[0],self.enemy2[1],3.5)
            self.enemy3[0],self.enemy3[1] = self.alien_(self.enemy3[0],self.enemy3[1],4.5)
            self.enemy4[0],self.enemy4[1] = self.alien_(self.enemy4[0],self.enemy4[1],3)
            
            
            
            self.enemy_and_spaceship_distance()
            
            # bullet positions
            if self.types == 'Fire' :


                pos1 = self.enemy1[0],self.enemy1[1],'alien1'
                pos2 = self.enemy2[0],self.enemy2[1],'alien2'
                pos3 = self.enemy3[0],self.enemy3[1],'alien3'
                pos4 = self.enemy4[0],self.enemy4[1],'alien4'

                # bullet speed
                self.bullet_y +=  13
                self.bullet_(self.bullet_x_constant+20,self.sship_y-self.bullet_y)

                
                for distance in (pos1,pos2,pos3,pos4):
                    self.dis = self.calculate_distance(distance[0],distance[1],self.bullet_x_constant+20,self.sship_y-self.bullet_y)
                  
                    if  int(self.dis) <= 40 :
  
                        self.score += 1
                        
                        if distance[2] =='alien1':
                            self.enemy1[0],self.enemy1[1] = 0,-610
                 
                        if distance[2] =='alien2':
                            self.enemy2[0],self.enemy2[1] = 0,-610
                        
                        if distance[2] =='alien3':
                            self.enemy3[0],self.enemy3[1] = 0,-610
                        
                            
                        if distance[2] =='alien4':
                            self.enemy4[0],self.enemy4[1] = 0,-610

                    
                        self.types = 'not_fire'
                        self.bullet_y = 0
                       
                        
                if self.bullet_y > 490:
                    self.types = 'not_fire'
                    self.bullet_y = 0
                    
                    

            self.our_sship(self.sship_x,self.sship_y)
        
            text = self.font.render(f"Score : {self.score}", True, (0,255,0))
            self.screen.blit(text, (0,20))


            pygame.display.update()
            self.clock.tick(self.FPS)
            
            if  self.score == 4:
                self.reset()
                print(f'Episode : {self.episode}')
                self.episode += 1
                




if __name__ == '__main__':
    game = SpaceGame(60)

    # game.play_game()
    obs , _ , _ = game.reset()
    episode = 1

    # osb = game.reset()
    for i in range(10000):
    
        action = game.random_action()
   
        new_obs , reward ,  done = game.play_step(action)
        new_obs = cv2.resize(new_obs,(160,160))
        cv2.imshow('im',new_obs)
        cv2.waitKey(10)
    #     obs = new_obs
    #     print(new_obs)
   
        
    #     break
    # print(obs)
    
        # if done:
        #     print(f'reward : {reward}')
        #     osb = game.reset()

        #     print(f'Episode : {episode}')
        #     episode += 1
    
    
    # print(action)
    # print(game.actions_()[action])

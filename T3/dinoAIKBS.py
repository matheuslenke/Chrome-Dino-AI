import pygame
import os
import random
import time
from sys import exit
from experta import *
from scipy.stats import ttest_rel, wilcoxon
import numpy as np
import pandas as pd
from multiprocessing import Pool
import seaborn as sns
from matplotlib import pyplot as plt

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(self, distance, obHeight, speed, obType):
        pass

    def updateState(self, state):
        pass

"""
def first(x):
    return x[0]
"""


class RuleBasedPlayer(KnowledgeEngine):

    def setAction(self, action):
        self.action = action

    def getAction(self):
        return self.action

    @Rule(AND(Fact(speed=P(lambda x: x < 15)),
              Fact(distance=P(lambda x: x < 300)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpSlow(self):
           self.retract(1)
           self.declare(Fact(action='K_UP'))
 
    @Rule(AND(Fact(speed=P(lambda x: x >= 15 and x < 17)),
              Fact(distance=P(lambda x: x < 400)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpFast(self):
           self.retract(1)
           self.declare(Fact(action='K_UP'))

    @Rule(AND(Fact(speed=P(lambda x: x >= 17)),
              Fact(distance=P(lambda x: x < 500)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpVeryFast(self):
           self.retract(1)
           self.declare(Fact(action='K_UP'))

    @Rule(AND(Fact(obType=P(lambda x: isinstance(x, Bird))),
              Fact(obHeight=P(lambda x: x > 50))))   
    def getDown(self): 
           self.retract(1)
           self.declare(Fact(action='K_DOWN'))

    @Rule(Fact(action=MATCH.action))
    def selectAction(self, action):
        self.setAction(action)

class RuleBasedKeyClassifier(KeyClassifier):
    def __init__(self):
        self.engine = RuleBasedPlayer()

    def keySelector(self, dist, obH, sp, obT):    
        self.engine.reset()
        self.engine.declare(Fact(action='K_NO'))
        self.engine.declare(Fact(distance=dist))
        self.engine.declare(Fact(obHeight=obH))
        self.engine.declare(Fact(speed=sp))
        self.engine.declare(Fact(obType=obT))
        self.engine.run()
        return self.engine.getAction()

class LenkeRuleBasedPlayer(KnowledgeEngine):
    def setAction(self, action):
        self.action = action

    def getAction(self):
        return self.action

    # Jump Rules
    @Rule(AND(Fact(speed=P(lambda x: x < 15)),
              Fact(distance=P(lambda x: x < 300 and x > 50)),
              Fact(obHeight=P(lambda x: x <= 58)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpSlow(self):
           self.retract(1)
           self.declare(Fact(action='K_UP'))
 
    @Rule(AND(Fact(speed=P(lambda x: x >= 15 and x < 17)),
              Fact(distance=P(lambda x: x < 350 and x > 50)),
              Fact(obHeight=P(lambda x: x <= 58)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpFast(self):
           self.retract(1)
           self.declare(Fact(action='K_UP'))

    @Rule(AND(Fact(speed=P(lambda x: x >= 17 and x < 23)),
              Fact(distance=P(lambda x: x < 450 and x > 50)),
              Fact(obHeight=P(lambda x: x <= 58)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpVeryFast(self):
           print("jumpVeryFast")
           self.retract(1)
           self.declare(Fact(action='K_UP'))
           
    @Rule(AND(Fact(speed=P(lambda x: x >= 23 and x < 25)),
              Fact(distance=P(lambda x: x < 500 and x > 50)),
              Fact(obHeight=P(lambda x: x <= 58)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpVeryFaster(self):
           print("jumpVeryFaster")
           self.retract(1)
           self.declare(Fact(action='K_UP'))

    @Rule(AND(Fact(speed=P(lambda x: x >= 25 and x < 29)),
              Fact(distance=P(lambda x: x < 600 and x > 50)),
              Fact(obHeight=P(lambda x: x <= 58)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpFasterThan25(self):
           self.retract(1)
           self.declare(Fact(action='K_UP'))

    @Rule(AND(Fact(speed=P(lambda x: x >= 29)),
              Fact(distance=P(lambda x: x < 700 and x > 50)),
              Fact(obHeight=P(lambda x: x <= 58)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpFasterThanEver(self):
           self.retract(1)
           self.declare(Fact(action='K_UP'))

    # Get Down when jumping Rules

    @Rule(AND(Fact(speed=P(lambda x: x >= 10 and x < 17)),
              Fact(distance=P(lambda x: x <= 0))))
    def getDownAfterJumpLower(self):
           self.retract(1)
           self.declare(Fact(action='K_DOWN'))

    @Rule(AND(Fact(speed=P(lambda x: x >= 17 and x < 25)),
              Fact(distance=P(lambda x: x <= 10)),
              Fact(obHeight=P(lambda x: x < 83))
              ))
    def getDownAfterJump(self):
           self.retract(1)
           self.declare(Fact(action='K_DOWN'))

    @Rule(AND(Fact(speed=P(lambda x: x >= 25)),
              Fact(distance=P(lambda x: x <= 30)),
              Fact(obHeight=P(lambda x: x < 83))
              ))
    def getDownAfterJumpFaster(self):
           self.retract(1)
           self.declare(Fact(action='K_DOWN'))



    # Get Down Rules

    @Rule(AND(Fact(speed=P(lambda x: x >= 17)),
              Fact(distance=P(lambda x: x < 500)),
              Fact(obHeight=P(lambda x: x >= 83)),
              NOT(Fact(action='K_UP'))))   
    def getDownWithSpeed(self):
           self.retract(1)
           self.declare(Fact(action='K_DOWN'))

    @Rule(AND(Fact(obHeight=P(lambda x: x >= 83)),
            NOT(Fact(action="K_DOWN"))))   
    def getDown(self): 
           self.retract(1)
           self.declare(Fact(action='K_DOWN'))

    @Rule(Fact(action=MATCH.action))
    def selectAction(self, action):
        self.setAction(action)

class LenkeRuleBasedKeyClassifier(KeyClassifier):
    def __init__(self):
        self.engine = LenkeRuleBasedPlayer()

    def keySelector(self, dist, obH, sp, obType):    
        self.engine.reset()
        self.engine.declare(Fact(action='K_NO'))
        self.engine.declare(Fact(distance=dist))
        self.engine.declare(Fact(obHeight=obH))
        self.engine.declare(Fact(speed=sp))
        self.engine.run()
        return self.engine.getAction()

def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    death_count = 0
    spawn_dist = 0

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        SCREEN.fill((255, 255, 255))

        distance = 1500
        obHeight = 0
        obType = 2
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            distance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
        else:
            # print(f"Dados: {distance}, {obHeight}, {game_speed}")
            userInput = aiPlayer.keySelector(distance, obHeight, game_speed, obType)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        player.update(userInput)
        player.draw(SCREEN)

        for obstacle in list(obstacles):
            obstacle.update()
            obstacle.draw(SCREEN)

        background()

        cloud.draw(SCREEN)
        cloud.update()

        score()

        clock.tick(60)
        pygame.display.update()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                ### print (game_speed, distance) 
                pygame.time.delay(2000)
                death_count += 1
                return points


def manyPlaysResults(rounds):
    print(f"Rodando {rounds} rounds")
    results = []
    for round in range(rounds):
        results += [playGame()]
    npResults = np.asarray(results)
    print(f"Resultados: {npResults}")
    return (results, npResults.mean() - npResults.std())

def save_final_results_to_file(results, npRes, filename):
    f = open(filename, "a")
    f.write(f"Resultados dos 30 rounds : {results} \n")
    f.write(f"Média: {npRes.mean()}\n")
    f.write(f"Std: {npRes.std()}\n")
    f.write(f"Valor: {npRes.mean() - npRes.std()}\n")
    f.close()

def compare_results_with_teacher(myResults, teacherResults):
    print("-------Resultados do professor-------\n", teacherResults)
    flavio_df = pd.DataFrame(teacherResults)
    print(flavio_df)
    flavioNpRes = np.asarray(teacherResults)
    print("Média: ", flavioNpRes.mean())
    print("Desvio padrão: ", flavioNpRes.std())

    print("\n\n-------Meus resultados-------\n", myResults)
    npRes = np.asarray(myResults)
    my_df = pd.DataFrame(myResults)
    print(my_df)

    print("Média: ", npRes.mean())
    print("Desvio padrão: ", npRes.std())

    print("Comparando SBC meu com professor")
    s,p = wilcoxon (myResults, teacherResults)
    print("Teste Wilcoxon",s, p)

    s2,p2 = ttest_rel(myResults, teacherResults)
    print("T test",s2, p2)

    t2_results = [1270.25, 862.5, 1429.0, 1342.25, 1119.75, 1299.25, 1111.0, 1542.5, 1553.25, 1370.5, 1273.0, 1745.25, 1703.5, 1410.75, 1105.75, 971.25, 1444.25, 1035.0, 1508.75, 1226.25, 621.75, 874.25, 610.0, 697.75, 1421.75, 784.25, 965.75, 1182.25, 1697.0, 1950.75] 

    t2_df = pd.DataFrame(t2_results)
    t2NpRes = np.asarray(t2_results)

    print("Comparando SBC meu com minha meta-heurística")
    s,p = wilcoxon (myResults, t2_results)
    print("Teste Wilcoxon",s, p)

    s2,p2 = ttest_rel(myResults, t2_results)
    print("T test",s2, p2)

    print("Comparando SBC do professor com minha meta-heurística")
    s,p = wilcoxon (teacherResults, t2_results)
    print("Teste Wilcoxon",s, p)

    s2,p2 = ttest_rel(teacherResults, t2_results)
    print("T test",s2, p2)

    scores = {
        "Matheus": myResults,
        "Professor": teacherResults,
        "Meta-heurística": t2_results
    }

    score_df = pd.DataFrame(scores)
    sns.boxplot(data=score_df)
    plt.show()


def main():
    global aiPlayer

    aiPlayer = LenkeRuleBasedKeyClassifier()
    print("Rodando IA do Lenke")
    lenkeResults, value = manyPlaysResults(30)
    npRes = np.asarray(lenkeResults)
    save_final_results_to_file(lenkeResults, npRes, "myResults.txt")

    aiPlayer = RuleBasedKeyClassifier()
    print("Rodando IA do professor")
    flavioResults, value = manyPlaysResults(30)
    npRes = np.asarray(flavioResults)
    save_final_results_to_file(flavioResults, npRes, "flavioResults.txt")
    
    # lenkeResults = [1855.5, 1961.0, 1455.5, 2099.5, 1409.5, 1780.25, 2031.0, 1893.5, 1852.0, 2217.75, 1266.5, 2009.75, 1607.0, 1443.75, 1802.5, 1971.0, 1807.25, 1538.75, 1449.25, 1997.25, 1538.75, 1396.75, 1299.75, 1350.25, 1518.5, 1534.75, 1447.25, 1336.25, 1806.75, 1850.25] 

    # flavioResults = [1056.75, 985.25, 1320.0, 124.75, 1160.25, 1292.5, 64.75, 37.0, 115.25, 912.75, 76.5, 1233.75, 158.75, 38.75, 1307.75, 168.5, 888.0, 59.5, 121.75, 74.5, 131.25, 36.5, 632.25, 1010.5, 1256.5, 924.0, 39.25, 1351.5, 1068.5, 58.25] 

    compare_results_with_teacher(lenkeResults, flavioResults)


main()


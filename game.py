import pygame
import os
import random
import time
import pygad
from sklearn.neighbors import KNeighborsClassifier
import datetime
from sys import exit

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"
CONTINUE_TRAINING = True
TRAINING_MODE = "MULTI_THREAD" # MULTI_THREAD or NORMAL

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
if TRAINING_MODE == "NORMAL":
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

START_TIME = time.time()
MAXIMUM_TRAINING_TIME = 5 * 60 # Tempo em minutos para limitar o tempo de treino

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
        if TRAINING_MODE == "NORMAL":
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
        if TRAINING_MODE == 'NORMAL':
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
        if TRAINING_MODE != 'MULTI_THREAD':
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
        if TRAINING_MODE != 'MULTI_THREAD':
            SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(self, distance, obHeight, speed, obType):
        pass

    def updateState(self, state):
        pass

def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"

def playGame(aiPlayer):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles, CONTINUE_TRAINING
    run = True
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0
    if TRAINING_MODE == "NORMAL":
        clock = pygame.time.Clock()
        font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    death_count = 0
    spawn_dist = 0

    userInputArray = pygame.key.get_pressed()
    if userInputArray[pygame.K_p]:
        CONTINUE_TRAINING = False
    # if TRAINING_MODE != "MULTI_THREAD" and CONTINUE_TRAINING == False:
    #     text = font.render("Treinamento irá terminar")
    #     textRect = text.get_rect()
    #     textRect.center = (500, 40)
    #     SCREEN.blit(text, textRect)

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        if TRAINING_MODE == "NORMAL":
            text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (1000, 40)
            SCREEN.blit(text, textRect)

    if TRAINING_MODE == "NORMAL":
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
        if TRAINING_MODE == "NORMAL":
            SCREEN.fill((255, 255, 255))

        distance = 1500
        obHeight = 0
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            distance = xy[0]
            obHeight = obstacles[0].getHeight()

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
        else:
            # print(f"Dados do jogo: {distance}, {game_speed}, {obHeight}")
            userInput = aiPlayer.keySelector(distance=distance, objectHeight=obHeight, speed=game_speed)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        player.update(userInput)
        if TRAINING_MODE == "NORMAL":
            player.draw(SCREEN)

        for obstacle in list(obstacles):
            obstacle.update()
            if TRAINING_MODE == "NORMAL":
                obstacle.draw(SCREEN)

        if TRAINING_MODE == "NORMAL":
            background()
            cloud.draw(SCREEN)
            cloud.update()

        score()

        if TRAINING_MODE == "NORMAL":
            clock.tick(60)
            pygame.display.update()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(20)
                death_count += 1
                return points

def playGameMultiThread(aiPlayer):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles

    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0
    obstacles = []
    death_count = 0
    spawn_dist = 0

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1


    while run:

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
            # userInput = aiPlayer.keySelector(game_speed, player, obType)
            # userInput = aiPlayer.keySelector(game_speed, obstacles, player)
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

        for obstacle in list(obstacles):
            obstacle.update()

        score()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                death_count += 1
                return points

# <---- SOLUÇÃO IMPLEMENTADA POR MATHEUS LENKE COUTINHO ---->

#  12 genes, em que em cada tupla temos: Distancia, Velocidade, Altura do Objeto
function_inputs = [ 783,   78,   58,  208,   68,   38, 1494,   51,  123,  598,   41,   38,
  281,   68,  123, 1126,   49,  123,  207,   49,   58,  830,   91,   38,
  267,   72,  123,  363,   44,   38,  541,   87,  123,  479,   35,  123,
  654,   64,   38,  233,   91,   58, 1270,   46,   38,  810,   24,   58,
 1317,   96,   38, 1283,   39,   58,  394,   91,   58, 1253,   70,   38,
 1258,   59,   58, 1096,   56,  123, 1061,   70,   38, 1086,   60,   58]

# 0 -> Agacha
# 1 -> Pula
# 2 -> Faz nada
input_classes = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2] # As classes da lista de genes
# Parâmetros do GA
num_generations = 20000 # Número alto para limitar por tempo
num_parents_mating = 2
sol_per_pop = 10
num_genes = len(function_inputs)
parent_selection_type = "sss"
keep_parents = 1
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

def getStateParameters(tuples):
    params = []
    for item in tuples:
        # Normalizando os parâmetros
        params.append(normalizeParameters(item))
    return params

def normalizeParameters(items):
    return (items[0] / 1500, items[1] / 100, items[2] / 123)

class KeyKNNClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state
        self.knn = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
        self.knn.fit(getStateParameters(state), input_classes)

    def keySelector(self, distance, speed, objectHeight):
        # Realizando Predict com os dados normalizados
        results = self.knn.predict([normalizeParameters([distance, speed, objectHeight])])
        if results[0] == 0:
            return "K_DOWN"
        elif results[0] == 1:
            return "K_UP"
        return "K_NO"

    def updateState(self, state):
        self.state = state

def genes_to_tuple(solution):
    items = []
    for i in range(0, len(solution), 3):
        items.append(solution[i:i+3])
    return items

# Função que roda o jogo e retorna a pontuação daquele jogo para o algoritmo genético
def fitness_function(solution, solution_idx):
    solution = genes_to_tuple(solution)
    save_actual_solution_to_file(solution)
    aiPlayer = KeyKNNClassifier(solution)
    return playGame(aiPlayer)

def run_genetic_algorithm():
    ga_instance = pygad.GA(num_generations=num_generations,
                        # initial_population=[function_inputs] * len(function_inputs),
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       on_generation=on_generation,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_space=[range(0,1500), range(10, 100), [38, 58, 123]] * len(input_classes),
                       )
    if TRAINING_MODE == "MULTI_THREAD":
        ga_instance.parallel_processing = ['thread', 10]
        print("Iniciando treinamento multi_thread!")
    else:
        print("Iniciando treinamento normal!")

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parâmetros da melhor solução: : {solution}".format(solution=solution))
    print("Valor Fitness da melhor solução = {solution_fitness}".format(solution_fitness=solution_fitness))
    save_solution_to_file(ga_instance, solution, solution_fitness)
    return solution, solution_fitness

def save_actual_solution_to_file(solution):
    f = open("temporary.txt", "a")
    f.write("\n")
    f.write(f"[{time.time()}]: ")
    f.write("{solution}".format(solution=solution))
    f.write("\n")
    f.close()

def save_solution_to_file(ga_instance, solution, solution_fitness):
    ga_instance.save("ga_instance.tmp")
    f = open("results.txt", "w")
    f.write("Parâmetros da melhor solução:{solution}".format(solution=solution))
    f.write("\nValor Fitness da melhor solução = {solution_fitness}".format(solution_fitness=solution_fitness))
    f.write("\n")
    f.close()

def on_generation(ga_instance):
    now = time.time() - START_TIME

    if now > MAXIMUM_TRAINING_TIME:
        print(f"Finalizando treinamento após {round(now/60, 0)} minutos")
        return 'stop'
    elif CONTINUE_TRAINING == False:
        print("Tecla P apertada, treinamento irá finalizar.")
        return 'stop'
    print(f"Continue a nadar! {round(now/60, 0)} minutos se passaram!")
    return 'continue'

from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import numpy as np
import pandas as pd
from multiprocessing import Pool

def manyPlaysResults(rounds):
    print(f"Rodando {rounds} rounds com parâmetros: {function_inputs}")
    aiPlayer = KeyKNNClassifier(genes_to_tuple(function_inputs))
    results = []
    if TRAINING_MODE == "NORMAL":
        for round in range(rounds):
            results += [playGame(aiPlayer)]
    elif TRAINING_MODE == "MULTI_THREAD":
        with Pool (os.cpu_count()-2) as p:
            results = p.starmap (playGame, zip ([aiPlayer]*rounds, range (rounds)))
    npResults = np.asarray(results)
    print(f"Resultados: {npResults}")
    return (results, npResults.mean() - npResults.std())



import pygame
from pygame.sprite import Sprite
import numpy as np
from a_1805106_train_1 import *
from time import sleep
import matplotlib.pyplot as plt

def show(image):
    plt.imshow(image)
    plt.show()
class Settings:
    def __init__(self):
        self.screen = (800, 600)
        self.bgcolor = (80, 80, 80)
        self.grid_size = (28, 28)
        self.cellWidth = 20
        self.cellHeight = 20
        self.gridOffsetW = 6
        self.gridOffsetH = 6
        self.mouseDown = False
        self.resetButton = (700, 100, 80, 40)
        self.submitButton = (700, 200, 80, 40)
        self.font = pygame.font.SysFont(None, 24)

        self.textArea = (620, 300, 160, 80)
        self.testFont = pygame.font.SysFont(None, 48)

class Test:
    def __init__(self, caption):
        self.model = FNN.load("model_1805106.pickle")
        self.text = ""
        pygame.init()
        self.settings = Settings()
        self.caption = caption
        self.screen = pygame.display.set_mode(self.settings.screen)
        pygame.display.set_caption(caption)
        self.grid = np.zeros(self.settings.grid_size)
        self.screen_rect = self.screen.get_rect()

    def handleDownKey(self, event):
        if event.key == pygame.K_q:
            exit(0)

    def handleUpKey(self, event):
        pass

    def handleMouseDown(self, event):
        if event.button == 1:
            self.settings.mouseDown = True
            self.handleMouseMotion(event)
    
    def handleMouseUp(self, event):
        if event.button == 1:
            self.settings.mouseDown = False
    
    def handleMouseMotion(self, event):
        if self.settings.mouseDown:
            x = event.pos[0]
            y = event.pos[1]
            if x >= self.settings.gridOffsetW and x < self.settings.gridOffsetW + self.settings.grid_size[0] * self.settings.cellWidth + self.settings.grid_size[0] - 1 and y >= self.settings.gridOffsetH and y < self.settings.gridOffsetH + self.settings.grid_size[1] * self.settings.cellHeight + self.settings.grid_size[1] - 1:
                x = x - self.settings.gridOffsetW
                y = y - self.settings.gridOffsetH
                x = x // (self.settings.cellWidth + 1)
                y = y // (self.settings.cellHeight + 1)
                self.grid[y][x] = 1
                # print(x, y)
            elif x >= self.settings.resetButton[0] and x < self.settings.resetButton[0] + self.settings.resetButton[2] and y >= self.settings.resetButton[1] and y < self.settings.resetButton[1] + self.settings.resetButton[3]:
                self.grid = np.zeros(self.settings.grid_size)
                self.text = ""
            elif x >= self.settings.submitButton[0] and x < self.settings.submitButton[0] + self.settings.submitButton[2] and y >= self.settings.submitButton[1] and y < self.settings.submitButton[1] + self.settings.submitButton[3]:
                # print(self.grid)
                # show(self.grid)
                char = self.model.predictClass([self.grid.flatten()])
                print(char)
                char = chr(char[0][0] - 1 + ord('A'))
                self.text += str(char)
                self.grid = np.zeros(self.settings.grid_size)
                sleep(0.5)

    def check_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)
            if event.type == pygame.KEYDOWN:
                self.handleDownKey(event)
            if event.type == pygame.KEYUP:
                self.handleUpKey(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.handleMouseDown(event)
            if event.type == pygame.MOUSEBUTTONUP:
                self.handleMouseUp(event)
            if event.type == pygame.MOUSEMOTION:
                self.handleMouseMotion(event)
    
    def drawCell(self, x, y, color):
        rect = pygame.Rect(x, y, self.settings.cellWidth, self.settings.cellHeight)
        pygame.draw.rect(self.screen, color, rect)
    
    def drawGrid(self):
        for i in range(self.settings.grid_size[0]):
            for j in range(self.settings.grid_size[1]):
                x = j * self.settings.cellWidth + self.settings.gridOffsetW + j
                y = i * self.settings.cellHeight + self.settings.gridOffsetH + i
                color = (0, 0, 0)
                if self.grid[i][j] == 0:
                    color = (255, 255, 255)
                self.drawCell(x, y, color)
    
    def drawButtons(self):
        rect = pygame.Rect(self.settings.resetButton)
        pygame.draw.rect(self.screen, (255, 0, 0), rect)
        rect = pygame.Rect(self.settings.submitButton)
        pygame.draw.rect(self.screen, (0, 255, 0), rect)

        #button text
        text = self.settings.font.render("Reset", True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (self.settings.resetButton[0] + self.settings.resetButton[2] // 2, self.settings.resetButton[1] + self.settings.resetButton[3] // 2)
        self.screen.blit(text, text_rect)

        text = self.settings.font.render("Submit", True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (self.settings.submitButton[0] + self.settings.submitButton[2] // 2, self.settings.submitButton[1] + self.settings.submitButton[3] // 2)
        self.screen.blit(text, text_rect)

        #text area
        rect = pygame.Rect(self.settings.textArea)
        pygame.draw.rect(self.screen, (0, 0, 0), rect)
        text = self.settings.testFont.render(self.text, True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (self.settings.textArea[0] + self.settings.textArea[2] // 2, self.settings.textArea[1] + self.settings.textArea[3] // 2)
        self.screen.blit(text, text_rect)




    def update_screen(self):
        self.screen.fill(self.settings.bgcolor)
        self.drawButtons()
        self.drawGrid()
        pygame.display.flip()
                      
    def run(self):
        while True:
            self.check_event()
            self.update_screen()

game = Test("Test")
game.run()

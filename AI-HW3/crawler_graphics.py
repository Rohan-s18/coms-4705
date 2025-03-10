# crawler_graphics.py
# -------------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).

import tkinter
import time
import threading
import sys
import crawler
from DP_Agent import DP_Agent
from RL_Agent import RL_Agent


class Application:

    def sigmoid(self, x):
        return 1.0 / (1.0 + 2.0 ** (-x))

    def incrementSpeed(self, inc):
        self.tickTime *= inc
        self.speed_label['text'] = 'Time per action (seconds): %.5f' % (self.tickTime)

    def incrementEpsilon(self, inc):
        self.ep += inc     
        self.epsilon = self.sigmoid(self.ep)   
        self.learner.setEpsilon(self.epsilon)
        self.epsilon_label['text'] = 'Epsilon: %.3f' % (self.epsilon)
                           
    def incrementGamma(self, inc):
        self.ga += inc     
        self.gamma = self.sigmoid(self.ga)   
        self.learner.setDiscount(self.gamma)
        self.gamma_label['text'] = 'Discount: %.3f' % (self.gamma)

        if not self.q:
            self.learner.policy_iteration(self.robotEnvironment.getPossibleActions, self.robotEnvironment.transition)
            self.robotEnvironment.reset()
                
    def incrementAlpha(self, inc):
        self.al += inc     
        self.alpha = self.sigmoid(self.al)   
        self.learner.setLearningRate(self.alpha)
        self.alpha_label['text'] = 'Learning Rate: %.3f' % (self.alpha)
        
    def __initGUI(self, win):
        ## Window ##
        self.win = win
        
        ## Initialize Frame ##    
        win.grid()
        self.dec = -.5
        self.inc = .5
        self.tickTime = 0.1

        ## Buttons + Labels ##
        self.setupSpeedButtonAndLabel(win)
        self.setupEpsilonButtonAndLabel(win)
        self.setUpGammaButtonAndLabel(win)
        self.setupAlphaButtonAndLabel(win)
        
        ## Exit Button ##
        self.exit_button = tkinter.Button(win, text='Quit', command=self.exit)
        self.exit_button.grid(row=0, column=9)
        
        ## Canvas ##
        self.canvas = tkinter.Canvas(root, height=200, width=1000)
        self.canvas.grid(row=2, columnspan=10)

    def setupAlphaButtonAndLabel(self, win):
        self.alpha_minus = tkinter.Button(win,
        text="-", command=(lambda: self.incrementAlpha(self.dec)))
        self.alpha_minus.grid(row=1, column=3, padx=10)
        
        self.alpha = self.sigmoid(self.al)
        self.alpha_label = tkinter.Label(win, text='Learning rate (alpha): %.3f' % (self.alpha))
        self.alpha_label.grid(row=1, column=4)
        
        self.alpha_plus = tkinter.Button(win,
        text="+", command=(lambda: self.incrementAlpha(self.inc)))
        self.alpha_plus.grid(row=1, column=5, padx=10)

    def setUpGammaButtonAndLabel(self, win):
        self.gamma_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementGamma(self.dec)))                
        self.gamma_minus.grid(row=1, column=0, padx=10)
        
        self.gamma = self.sigmoid(self.ga)   
        self.gamma_label = tkinter.Label(win, text='Discount factor (gamma): %.3f' % (self.gamma))
        self.gamma_label.grid(row=1, column=1)
        
        self.gamma_plus = tkinter.Button(win,
        text="+", command=(lambda: self.incrementGamma(self.inc)))
        self.gamma_plus.grid(row=1, column=2, padx=10)

    def setupEpsilonButtonAndLabel(self, win):
        self.epsilon_minus = tkinter.Button(win,
        text="-", command=(lambda: self.incrementEpsilon(self.dec)))
        self.epsilon_minus.grid(row=0, column=3)
        
        self.epsilon = self.sigmoid(self.ep)   
        self.epsilon_label = tkinter.Label(win, text='Exploration rate (epsilon): %.3f' % (self.epsilon))
        self.epsilon_label.grid(row=0, column=4)
        
        self.epsilon_plus = tkinter.Button(win,
        text="+", command=(lambda: self.incrementEpsilon(self.inc)))
        self.epsilon_plus.grid(row=0, column=5)

    def setupSpeedButtonAndLabel(self, win):
        self.speed_minus = tkinter.Button(win,
        text="-", command=(lambda: self.incrementSpeed(.5)))
        self.speed_minus.grid(row=0, column=0)
        
        self.speed_label = tkinter.Label(win, text='Time per action (seconds): %.5f' % (self.tickTime))
        self.speed_label.grid(row=0, column=1)
        
        self.speed_plus = tkinter.Button(win,
        text="+", command=(lambda: self.incrementSpeed(2)))
        self.speed_plus.grid(row=0, column=2)

    def skip5kSteps(self):
        self.stepsToSkip = 5000


    def __init__(self, win, q):
    
        self.ep = 0
        self.ga = 2
        self.al = 2
        self.stepCount = 0
        self.q = q

        ## Init GUI and environment
        self.__initGUI(win)
        self.robot = crawler.CrawlingRobot(self.canvas)
        self.robotEnvironment = crawler.CrawlingRobotEnvironment(self.robot)
  
        # Init Agent      
        parameters = {"alpha":self.alpha, "epsilon":self.epsilon, "gamma":self.gamma, "V0":0, "Q0":0}
        if self.q:
            self.learner = RL_Agent(self.robotEnvironment.getAllStates(), self.robotEnvironment.getPossibleActions, parameters)
        else:
            self.learner = DP_Agent(self.robotEnvironment.getAllStates(), parameters)
            self.learner.policy_iteration(self.robotEnvironment.getPossibleActions, self.robotEnvironment.transition)
            self.robotEnvironment.reset()

        # Start GUI
        self.running = True
        self.stopped = False
        self.stepsToSkip = 0
        self.thread = threading.Thread(target=self.run)
        self.thread.start()


    def exit(self):
        self.running = False
        for i in range(5):
            if not self.stopped:
                time.sleep(0.1)
        self.win.destroy()
        sys.exit(0)
      
    def step(self):
        self.stepCount += 1
        state = self.robotEnvironment.getCurrentState()
        action = self.learner.choose_action(state, self.robotEnvironment.getPossibleActions(state))
        successor, reward = self.robotEnvironment.transition(state, action)
        self.learner.update(state, action, reward, successor, self.robotEnvironment.getPossibleActions(successor))
        
    def run(self):
        self.stepCount = 0
        while True:
            minSleep = .01
            tm = max(minSleep, self.tickTime)
            time.sleep(tm)
            self.stepsToSkip = int(tm / self.tickTime) - 1

            if not self.running:
                self.stopped = True
                return
            for i in range(self.stepsToSkip):
                self.step()
            self.stepsToSkip = 0
            self.step()

    def start(self):
        self.win.mainloop()


def run(q=False):
    global root
    root = tkinter.Tk()
    root.title('Crawler GUI')
    root.resizable(0, 0)

    app = Application(root,q)
    def update_gui():
        app.robot.draw(app.stepCount, app.tickTime)
        root.after(10, update_gui)
    update_gui()

    root.protocol('WM_DELETE_WINDOW', app.exit)
    app.start()

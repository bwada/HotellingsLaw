from typing import List
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import random
import matplotlib.animation as animation
import copy
import pickle

SIM_WIDTH = 50
SIM_HEIGHT = 50

PERSON_MAX_ENERGY = 10
PERSON_STEP_SIZE = 20
PERSON_MOVE_COST = 1

VENDOR_STEP_SIZE = PERSON_STEP_SIZE/2
VENDOR_MAX_WAIT_TIME = 30
VENDOR_EXPLORE_SIZE = 5


@dataclass
class Position:
    coords: np.array

    def dist(self, other_position):
        return np.sqrt(np.sum(np.power(self.coords - other_position.coords, 2)))

    def vector(self, other_position):
        diff = other_position.coords - self.coords
        vec = np.array(diff)
        return vec/np.linalg.norm(vec)

    def move_to_target(self, goal, step_size):
        if self.dist(goal)<step_size:
            self.coords = np.copy(goal.coords)
            return True
        v = self.vector(goal)
        self.coords += v*step_size

    @staticmethod
    def rand_dir():
        xdir = random.random()-0.5
        ydir = random.random()-0.5
        vec = np.array((xdir,ydir))
        norm = np.linalg.norm((xdir,ydir))
        return(vec/norm)

    @staticmethod
    def check_boundary(pos):
        x_bound = True
        y_bound = True
        if pos.coords[0] < 0 or pos.coords[0] > SIM_WIDTH:
            x_bound = False
        if pos.coords[1] < 0 or pos.coords[1] > SIM_HEIGHT:
            y_bound = False
        return np.array((x_bound,y_bound))


@dataclass
class Vendor:
    pos: Position
    total_sales: int
    position_sales: int
    best_pos: Position
    best_sales: int
    wait_time: int
    moving: bool
    next_pos: Position

    step_size = VENDOR_STEP_SIZE
    max_wait_time = VENDOR_MAX_WAIT_TIME
    explore_size = VENDOR_EXPLORE_SIZE

    @classmethod
    def new_random_pos(cls, x_range, y_range):
        x = random.random()*(x_range[1]-x_range[0])+x_range[0]
        y = random.random()*(y_range[1]-y_range[0])+y_range[0]
        return cls(Position(np.array((x,y))), 0, 0, Position(np.array((x,y))), 0, 0, False, Position(np.array((x,y))))

    def sale(self):
        self.position_sales += 1
        self.total_sales += 1

    def pick_new_pos(self, pos, x_range=None, y_range=None):
        new_dir = self.explore_size * Position.rand_dir() * random.random()
        new_pos = Position(pos.coords + new_dir)
        in_bound = Position.check_boundary(new_pos)
        if not in_bound[0]:
            new_pos.coords[0] = pos.coords[0] - new_dir[0]
        if not in_bound[1]:
            new_pos.coords[1] = pos.coords[1] - new_dir[1]
        return new_pos 

    def timestep(self):
        if self.moving:
            self.move_to_target(self.next_pos)
        else:
            self.wait_time += 1
            if self.wait_time >= self.max_wait_time:
                if self.position_sales >= self.best_sales:
                    self.best_sales = self.position_sales
                    self.best_pos = copy.deepcopy(self.pos)
                    new_pos = self.pick_new_pos(self.best_pos)
                else:
                    new_pos = self.pick_new_pos(self.best_pos)
                self.moving = True
                self.next_pos = new_pos
                print(self.position_sales)
                print(self.best_pos)
    
    def move_to_target(self, goal):
        reached_goal = self.pos.move_to_target(goal, self.step_size)
        if reached_goal:
            self.pos = copy.deepcopy(goal)
            self.moving = False
            self.wait_time = 0
            self.position_sales = 0


@dataclass
class Person:
    pos: Position
    hungry: bool
    energy: float
    target: Position
    vend: Vendor

    max_energy = PERSON_MAX_ENERGY
    move_cost = PERSON_MOVE_COST
    step_size = PERSON_STEP_SIZE

    @classmethod
    def new_random_pos(cls, x_range, y_range):
        x = random.random()*(x_range[1]-x_range[0])+x_range[0]
        y = random.random()*(y_range[1]-y_range[0])+y_range[0]
        energy = random.random()*cls.max_energy
        return cls(Position((x,y)), False, energy, None, None)

    def move(self,x_range=None,y_range=None):
        if self.hungry:
            self.move_to_target(x_range=x_range,y_range=y_range)
        else:
            self.wander(x_range=x_range,y_range=y_range)
        self.energy -= self.move_cost
        if self.energy <= 0:
            self.hungry = True

    def eat(self):
        self.energy = self.max_energy
        self.target = None
        self.hungry = False
        self.vend.sale()
        self.vend = None
    
    def set_target(self, target_vendor):
        self.vend = target_vendor
        # We do want this to point to the vendor's position, and not a copy
        self.target = target_vendor.pos

    def wander(self,x_range=None,y_range=None):
        rand_dir = self.step_size * Position.rand_dir() * random.random()
        new_pos = Position(self.pos.coords + rand_dir)
        if Position.check_boundary(new_pos).all():
            self.pos = new_pos
    
    def move_to_target(self,x_range=None,y_range=None):
        reached_goal = self.pos.move_to_target(self.target,self.step_size)
        if reached_goal:
            self.eat()
    
    # def update_pos(self,x_vec,y_vec,x_range,y_range):
    #     new_x = self.pos.x + x_vec
    #     new_y = self.pos.y + y_vec
    #     if new_x < x_range[0] or new_x > x_range[1]:
    #         self.pos.x -= x_vec
    #     else:
    #         self.pos.x = new_x
    #     if new_y < y_range[0] or new_y > y_range[1]:
    #         self.pos.y -= y_vec
    #     else:
    #        self.pos.y = new_y

@dataclass
class environment:
    vendors: List[Vendor]
    people: List[Person]
    time: float
    width: float
    height: float

    @classmethod
    def create_new(cls, x_len, y_len, num_vendors, num_people):
        vendors = [Vendor.new_random_pos((0,x_len),(0,y_len)) for i in range(num_vendors)]
        people = [Person.new_random_pos((0,x_len),(0,y_len)) for i in range(num_people)]
        return cls(vendors, people, 0, x_len, y_len)

    def nearest_vendor(self, position):
        min_dist = np.inf
        closest_vendor = None
        for v in self.vendors:
            dist = v.pos.dist(position)
            if dist > min_dist:
                min_dist = dist
                closest_vendor = v
        return closest_vendor
    
    def pick_vendor(self, position):
        probs = np.zeros(len(self.vendors))
        for ind,v in enumerate(self.vendors):
            dist = v.pos.dist(position)
            probs[ind] = np.exp(-dist)
        probs = probs/np.sum(probs)
        vend = random.choices(self.vendors, weights=probs)[0]
        return vend 

    def timestep(self):
        for v in self.vendors:
            v.timestep()
        for p in self.people:
            p.move(x_range=(0,self.width),y_range=(0,self.height))
            if p.energy <= 0:
                vendor = self.pick_vendor(p.pos)
                p.set_target(vendor)
        self.time += 1
        print(f"time: {self.time}")

    def get_vendor_positions(self):
        x = [person.pos.coords[0] for person in self.vendors]
        y = [person.pos.coords[1] for person in self.vendors]
        return x,y
 
    def get_people_positions(self):
        x = [person.pos.coords[0] for person in self.people]
        y = [person.pos.coords[1] for person in self.people]
        return x,y
    
if __name__ == "__main__":
    random.seed(1234)
    sim_time = 10000
    num_people = 400
    num_vendors = 2

    print("running")
    width = SIM_WIDTH
    height = SIM_HEIGHT

    fig, ax = plt.subplots()
    beach = environment.create_new(width,height,num_vendors,num_people)
    def aniplot(i):
        beach.timestep()
        for i in beach.people:
            if not Position.check_boundary(i.pos).all():
                print("shit")
        people_x, people_y = beach.get_people_positions()
        vendor_x, vendor_y = beach.get_vendor_positions()
        
        ax.clear()
        ax.set_xlim(0,width)
        ax.set_ylim(0,height)
        people_artist = ax.scatter(people_x,people_y,marker='o',color="blue",s=5)
        vendor_artist = ax.scatter(vendor_x,vendor_y,marker='x',color="red")
        return [people_artist,vendor_artist] 
    
    ani = animation.FuncAnimation(fig, aniplot, np.arange(1, sim_time), interval=25, blit=True)
    ani.save('animation.gif', writer='ffmpeg', fps=10)
    with open("1_out","wb") as f:
        pickle.dump(beach,f)    

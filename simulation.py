from re import I
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import random
import matplotlib.animation as aniomation
import copy

@dataclass
class position:
    x: float
    y: float

    def dist(self, other_position):
        return np.sqrt((self.x-other_position.x)**2 + (self.y-other_position.y)**2)

    def vector(self, other_position):
        x_dir = other_position.x - self.x
        y_dir = other_position.y - self.y
        vec = np.array(x_dir,y_dir)
        return vec/np.linalg.norm(vec)

# @dataclass
# class renderer:
#     @staticmethod
#     def render(vender_pos, people_pos):
#         f,ax = plt.subplots()
#         v_x = [pos.x for pos in vender_pos]
#         v_y = [pos.y for pos in vender_pos]
#         p_x = [pos.x for pos in people_pos]
#         p_y = [pos.y for pos in people_pos]
#         ax.scatter(p_x,p_y)

@dataclass
class vendor:
    pos: position
    total_sales: int
    position_sales: int
    best_pos: position
    best_sales: int
    moving: bool
    next_pos: position

    step_size = 0.5

    @classmethod
    def new_random_pos(cls, x_range, y_range):
        x = random.random()*(x_range[1]-x_range[0])+x_range[0]
        y = random.random()*(y_range[1]-y_range[0])+y_range[0]
        return cls(position(x,y), 0, 0, position(x,y), 0, False, position(x,y))

    def sale(self):
        self.position_sales += 1
        self.total_sales += 1

    def move(self, goal):
        if self.pos.dist(goal)<self.pos.dist(goal):
            self.pos = copy.deepcopy(goal)
        v = self.pos.vector(goal)
        self.pos.x += v[0]*self.step_size
        self.pos.y += v[1]*self.step_size




@dataclass
class person:
    pos: position
    hungry: bool
    energy: float
    target: position

    max_energy = 10
    move_cost = 1
    step_size = 1

    @classmethod
    def new_random_pos(cls, x_range, y_range):
        x = random.random()*(x_range[1]-x_range[0])+x_range[0]
        y = random.random()*(y_range[1]-y_range[0])+y_range[0]
        return cls(position(x,y), False, cls.max_energy, None)

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
    
    def set_target(self, target_pos):
        self.target = target_pos

    def wander(self,x_range=None,y_range=None):
        xdir = random.random()-0.5
        ydir = random.random()-0.5
        norm = np.linalg.norm((xdir,ydir))
        x_update = xdir*self.step_size/norm 
        y_update = ydir*self.step_size/norm
        self.update_pos(x_update,y_update,x_range=x_range,y_range=y_range)
    
    def move_to_target(self,x_range=None,y_range=None):
        xdir = self.target.x-self.pos.x
        ydir = self.target.y-self.pos.y
        norm = np.linalg.norm((xdir,ydir))
        self.update_pos(xdir*self.step_size/norm,ydir*self.step_size/norm,x_range=x_range,y_range=y_range)
    
    def update_pos(self,x_vec,y_vec,x_range,y_range):
        new_x = self.pos.x + x_vec
        new_y = self.pos.y + y_vec
        if new_x < x_range[0] or new_x > x_range[1]:
            self.pos.x -= x_vec
        else:
            self.pos.x = new_x
        if new_y < y_range[0] or new_y > y_range[1]:
            self.pos.y -= y_vec
        else:
            self.pos.y = new_y

@dataclass
class environment:
    vendors: List[vendor]
    people: List[person]
    time: float
    width: float
    height: float

    @classmethod
    def create_new(cls, x_len, y_len, num_vendors, num_people):
        vendors = [vendor.new_random_pos((0,x_len),(0,y_len)) for i in range(num_vendors)]
        people = [person.new_random_pos((0,x_len),(0,y_len)) for i in range(num_people)]
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
        print(position)
        print(probs)
        probs = probs/np.sum(probs)
        vend = random.choices(self.vendors, weights=probs)[0]
        return vend 

    def timestep(self):
        for v in self.vendors:
            pass
        for p in self.people:
            p.move(x_range=(0,self.width),y_range=(0,self.height))
            if p.energy <= 0:
                vendor = self.pick_vendor(p.pos)
                p.set_target(vendor.pos)
        self.time += 1
        print(f"time: {self.time}")

    def get_vendor_positions(self):
        x = [person.pos.x for person in self.vendors]
        y = [person.pos.y for person in self.vendors]
        return x,y
 
    def get_people_positions(self):
        x = [person.pos.x for person in self.people]
        y = [person.pos.y for person in self.people]
        return x,y
    
if __name__ == "__main__":
    print("running")
    width = 100
    height = 10

    fig, ax = plt.subplots()
    beach = environment.create_new(width,height,2,10)
    def aniplot(i):
        beach.timestep()
        people_x, people_y = beach.get_people_positions()
        vendor_x, vendor_y = beach.get_vendor_positions()
        
        ax.clear()
        ax.set_xlim(0,width)
        ax.set_ylim(0,height)
        people_artist = ax.scatter(people_x,people_y,marker='o',color="blue")
        vendor_artist = ax.scatter(vendor_x,vendor_y,marker='x',color="red")
        return [people_artist,vendor_artist] 
    
    ani = animation.FuncAnimation(fig, aniplot, np.arange(1, 100), interval=25, blit=True)
    ani.save('animation.gif', writer='ffmpeg', fps=60)

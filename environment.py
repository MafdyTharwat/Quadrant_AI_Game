import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class ChainReactionEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(ChainReactionEnv, self).__init__()
        self.rows, self.cols = 5, 5
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(low=0, high=8, shape=(5, 5, 2), dtype=np.int32)
        
        self.render_mode = render_mode
        self.screen_size = 500
        self.cell_size = self.screen_size // 5
        self.screen = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board_owners = np.zeros((5, 5), dtype=np.int32)
        self.board_counts = np.zeros((5, 5), dtype=np.int32)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.stack([self.board_owners, self.board_counts], axis=-1)

    def step(self, action):
        r, c = divmod(action, 5)
        
        # Big penalty if plays a wrong step
        ai_has_cells = np.any(self.board_owners == 2)
        if (ai_has_cells and self.board_owners[r, c] != 2) or \
           (not ai_has_cells and self.board_owners[r, c] != 0):
            return self._get_obs(), -5, False, False, {}

        # running step
        if self.board_owners[r, c] == 0:
            self.board_counts[r, c] = 3
        else:
            self.board_counts[r, c] += 1
        
        self.board_owners[r, c] = 2
        self._explode(2)

        # small reward for each correct step
        reward = 0.5 
        
        terminated = False
        # Very big reward if it win
        if np.sum(self.board_counts) > 6 and not np.any(self.board_owners == 1):
            terminated = True
            reward = 100 
        
        # penalty if it loss
        if np.sum(self.board_counts) > 6 and not np.any(self.board_owners == 2):
            terminated = True
            reward = -100

        return self._get_obs(), reward, terminated, False, {}
    
    def _explode(self, pid):
        stable = False
        safety_net = 0 
        
        while not stable and safety_net < 1000:
            stable = True
            safety_net += 1
            
            to_explode = []
            for r in range(5):
                for c in range(5):
                    # explode squares when counts (no. balls) = 4
                    if self.board_counts[r, c] >= 4:
                        to_explode.append((r, c))
                        stable = False 
            
            for (r, c) in to_explode:
                # remove 4 balls from exploded cell
                self.board_counts[r, c] -= 4
                
                if self.board_counts[r, c] == 0:
                    self.board_owners[r, c] = 0
                else:
                    self.board_owners[r, c] = pid

                # disributing balls in the four directions
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5:
                        
                        if self.board_owners[nr, nc] != pid and self.board_owners[nr, nc] != 0:
                            self.board_owners[nr, nc] = pid
                            self.board_counts[nr, nc] = 1 
                        else:
                            self.board_owners[nr, nc] = pid
                            self.board_counts[nr, nc] += 1

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Color Wars AI")
        
        BG_COLOR = (124, 184, 209)
        self.screen.fill(BG_COLOR)
        
        for r in range(5):
            for c in range(5):
                rect = pygame.Rect(c*self.cell_size+5, r*self.cell_size+5, self.cell_size-10, self.cell_size-10)
                pygame.draw.rect(self.screen, (235, 235, 220), rect, border_radius=12)
                
                count = self.board_counts[r, c]
                if count > 0:
                    color = (255, 80, 80) if self.board_owners[r, c] == 1 else (0, 191, 255)
                    center = (c*self.cell_size + self.cell_size//2, r*self.cell_size + self.cell_size//2)
                    self._draw_dots(count, color, center)

    def _draw_dots(self, count, color, center):
        if count == 1:
            pygame.draw.circle(self.screen, color, center, 12)
        elif count == 2:
            pygame.draw.circle(self.screen, color, (center[0]-15, center[1]), 10)
            pygame.draw.circle(self.screen, color, (center[0]+15, center[1]), 10)
        elif count == 3:
            pygame.draw.circle(self.screen, color, (center[0], center[1]-15), 10)
            pygame.draw.circle(self.screen, color, (center[0]-15, center[1]+12), 10)
            pygame.draw.circle(self.screen, color, (center[0]+15, center[1]+12), 10)
        else:
            pygame.draw.circle(self.screen, color, (center[0]-15, center[1]-15), 9)
            pygame.draw.circle(self.screen, color, (center[0]+15, center[1]-15), 9)
            pygame.draw.circle(self.screen, color, (center[0]-15, center[1]+15), 9)
            pygame.draw.circle(self.screen, color, (center[0]+15, center[1]+15), 9)
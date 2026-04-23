import pygame
import numpy as np
import random
from environment import ChainReactionEnv
from stable_baselines3 import PPO

# Basic settings and lines
pygame.init()
env = ChainReactionEnv(render_mode="human")
clock = pygame.time.Clock()

FONT_BIG = pygame.font.SysFont('Arial', 64, bold=True)
FONT_SMALL = pygame.font.SysFont('Arial', 28, bold=True)

# loding model. If doesn't exist, it will play with heuristic only
try:
    model = PPO.load("models/chain_reaction_v1.zip")
    print("Model Loaded")
except:
    model = None
    print("Using Heuristics Logic only.")

def get_heuristic_score(env, action, player_id):
    r, c = divmod(action, 5)
    opponent_id = 1 if player_id == 2 else 2
    score = 0
    
    temp_counts = env.board_counts.copy()
    temp_owners = env.board_owners.copy()
    
    if temp_owners[r, c] == 0: temp_counts[r, c] = 3
    else: temp_counts[r, c] += 1
    temp_owners[r, c] = player_id
    
    if temp_counts[r, c] == 3: score += 20
        
    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < 5 and 0 <= nc < 5:
            if env.board_owners[nr, nc] == opponent_id and env.board_counts[nr, nc] == 3:
                score -= 30
            if temp_counts[r, c] >= 4 and env.board_owners[nr, nc] == opponent_id:
                score += 50
    if temp_counts[r, c] >= 4: score += 15
    return score

def reset_game():
    obs, _ = env.reset()
    return obs, False, True, "" # obs, game_over, player_turn, winner_text

obs, game_over, player_turn, winner_text = reset_game()
ai_delay_timer = 0
running = True

while running:
    clock.tick(60)

    env.render()

    if game_over:
        popup_rect = pygame.Rect(50, 150, 400, 200)
        pygame.draw.rect(env.screen, (40, 40, 40), popup_rect, border_radius=15)
        pygame.draw.rect(env.screen, (255, 255, 255), popup_rect, 3, border_radius=15)
        
        msg_color = (255, 100, 100) if "You" in winner_text else (100, 200, 255)
        text_surf = FONT_BIG.render(winner_text, True, msg_color)
        env.screen.blit(text_surf, text_surf.get_rect(center=(250, 210)))
        
        btn_rect = pygame.Rect(170, 280, 160, 45)
        pygame.draw.rect(env.screen, (34, 139, 34), btn_rect, border_radius=10)
        btn_txt = FONT_SMALL.render("Play Again", True, (255, 255, 255))
        env.screen.blit(btn_txt, btn_txt.get_rect(center=btn_rect.center))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            
            if game_over:
                btn_rect = pygame.Rect(170, 280, 160, 45)
                if btn_rect.collidepoint(x, y):
                    obs, game_over, player_turn, winner_text = reset_game()
            
            elif player_turn:
                row, col = y // env.cell_size, x // env.cell_size
                if 0 <= row < 5 and 0 <= col < 5:
                    human_has_cells = np.any(env.board_owners == 1)
                    valid = (human_has_cells and env.board_owners[row, col] == 1) or \
                            (not human_has_cells and env.board_owners[row, col] == 0)
                    
                    if valid:
                        if env.board_owners[row, col] == 0: env.board_counts[row, col] = 3
                        else: env.board_counts[row, col] += 1
                        env.board_owners[row, col] = 1
                        env._explode(1) 
                        player_turn = False
                        ai_delay_timer = pygame.time.get_ticks()
                        
                        if np.sum(env.board_counts) > 6 and not np.any(env.board_owners == 2):
                            winner_text = "YOU WIN!"
                            game_over = True

    if not player_turn and not game_over:
        current_time = pygame.time.get_ticks()
        if current_time - ai_delay_timer > 500:
            ai_has_cells = np.any(env.board_owners == 2)
            valid_ai_actions = [i for i in range(25) if (ai_has_cells and env.board_owners[divmod(i, 5)] == 2) or \
                                (not ai_has_cells and env.board_owners[divmod(i, 5)] == 0)]

            if valid_ai_actions:
                best_action = valid_ai_actions[0]
                best_score = -9999
                
                for act in valid_ai_actions:
                    h_score = get_heuristic_score(env, act, 2)
                    
                    if model:
                        if model.predict(obs, deterministic=True)[0] == act:
                            h_score += 10 
                    
                    if h_score > best_score:
                        best_score = h_score
                        best_action = act
                
                obs, _, terminated, _, _ = env.step(best_action)
                player_turn = True
                if terminated:
                    winner_text = "AI WINS!"
                    game_over = True
            else:
                winner_text = "YOU WIN!"
                game_over = True

pygame.quit()
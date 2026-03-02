"""Pygame visualization and interactive demo for Soulgue-Maze."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pygame

from agents import MapperQAgent, WalkerQAgent
from config import DEFAULT_CONFIG, ProjectConfig
from env import SoulgueMazeEnv, WALL
from utils import load_qtables


KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_RIGHT: 1,
    pygame.K_DOWN: 2,
    pygame.K_LEFT: 3,
}


class SoulgueUI:
    def __init__(self, cfg: ProjectConfig = DEFAULT_CONFIG) -> None:
        self.cfg = cfg
        self.env = SoulgueMazeEnv(cfg.maze, cfg.reward, cfg.runtime, seed=cfg.train.seed)

        self.walker = WalkerQAgent(
            height=cfg.maze.height,
            width=cfg.maze.width,
            alpha=cfg.train.alpha,
            gamma=cfg.train.gamma,
        )
        self.mapper = MapperQAgent(alpha=cfg.train.alpha, gamma=cfg.train.gamma)

        self._try_load_qtables()

        self.ai_mode = True
        self.paused = False
        self.show_truth_walls = cfg.ui.show_truth_walls_default
        self.show_truth_blackholes = cfg.ui.show_truth_blackholes_default

        self.manual_walker_action: int | None = None
        self.manual_mapper_reset = False

        self.episode = 1
        self.success_count = 0
        self.cum_walker_reward = 0.0
        self.cum_mapper_reward = 0.0
        self.last_step_info: dict = {}
        self.trace: list[tuple[int, int]] = []
        self.no_progress_steps = 0
        self.guard_reset_count = 0

        self.obs = self.env.reset()
        self.trace = [self.env.walker_pos]

        pygame.init()
        pygame.display.set_caption("Soulgue-Maze")

        self.grid_w = cfg.maze.width * cfg.ui.cell_px
        self.grid_h = cfg.maze.height * cfg.ui.cell_px
        block_w = self.grid_w + cfg.ui.margin_px * 2
        self.walker_origin = (cfg.ui.margin_px, cfg.ui.margin_px + 26)
        self.mapper_origin = (cfg.ui.margin_px + block_w + cfg.ui.view_gap_px, cfg.ui.margin_px + 26)
        panel_start_x = cfg.ui.margin_px + block_w * 2 + cfg.ui.view_gap_px
        self.panel_x = panel_start_x
        self.width_px = panel_start_x + cfg.ui.panel_width_px
        self.height_px = max(self.grid_h + cfg.ui.margin_px * 2 + 40, 620)

        self.screen = pygame.display.set_mode((self.width_px, self.height_px))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 18)
        self.font_small = pygame.font.SysFont("Menlo", 15)

    def _try_load_qtables(self) -> None:
        if Path("checkpoints/walker_q.npy").exists() and Path("checkpoints/mapper_q.npy").exists():
            w_q, m_q = load_qtables("checkpoints")
            if w_q.shape == self.walker.q.shape and m_q.shape == self.mapper.q.shape:
                self.walker.q[:] = w_q
                self.mapper.q[:] = m_q

    def _reset_episode(self) -> None:
        self.obs = self.env.reset()
        self.trace = [self.env.walker_pos]
        self.cum_walker_reward = 0.0
        self.cum_mapper_reward = 0.0
        self.no_progress_steps = 0

    def _select_guard_reset_action(self) -> int:
        assert self.env.truth is not None
        if len(self.env.truth.entries) <= 1:
            return 1
        wr, wc = self.env.walker_pos
        distances = [abs(wr - er) + abs(wc - ec) for er, ec in self.env.truth.entries]
        best_idx = int(np.argmax(distances))
        # Mapper action id = entry index + 1
        return min(2, best_idx + 1)

    def _is_two_cell_oscillation(self) -> bool:
        win = max(6, int(self.cfg.ui.oscillation_window))
        if len(self.trace) < win:
            return False
        tail = self.trace[-win:]
        uniq = list(dict.fromkeys(tail))
        if len(uniq) != 2:
            return False
        a, b = uniq[0], uniq[1]
        for i, p in enumerate(tail):
            if p != (a if i % 2 == 0 else b):
                return False
        return True

    def _advance(self) -> None:
        if self.ai_mode:
            s_w = self.walker.encode_state(self.obs.walker_obs)
            s_m = self.mapper.encode_state(self.obs.mapper_obs)
            a_w = self.walker.select_action(
                s_w,
                self.obs.walker_obs.valid_mask,
                epsilon=self.cfg.ui.ai_walker_epsilon,
                rng=self.env.rng,
            )
            a_m = self.mapper.select_action(s_m, epsilon=self.cfg.ui.ai_mapper_epsilon, rng=self.env.rng)

            stuck_guard = (
                self.obs.mapper_obs.stuck_bin == 2
                and self.no_progress_steps >= self.cfg.ui.stuck_reset_guard_steps
                and self.env.resets_left > 0
            )
            oscillation_guard = self._is_two_cell_oscillation() and self.env.resets_left > 0
            if stuck_guard or oscillation_guard:
                a_m = self._select_guard_reset_action()
                self.guard_reset_count += 1
        else:
            if self.manual_walker_action is None:
                return
            a_w = int(self.manual_walker_action)
            a_m = 1 if self.manual_mapper_reset else 0

        cov_before = self.env.compute_coverage()
        next_obs, rewards, done, info = self.env.step(a_w, a_m)
        self.obs = next_obs
        self.last_step_info = info
        self.cum_walker_reward += rewards.walker_reward
        self.cum_mapper_reward += rewards.mapper_reward
        self.trace.append(self.env.walker_pos)

        cov_after = float(info.get("coverage", cov_before))
        if cov_after > cov_before + 1e-12:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        self.manual_walker_action = None
        self.manual_mapper_reset = False

        timeout = self.env.step_count >= self.cfg.train.max_steps_per_episode
        if done or timeout:
            if done:
                self.success_count += 1
            self.episode += 1
            self._reset_episode()

    def _draw_cell_bg(self, x0: int, y0: int, visited: bool) -> None:
        color = (235, 239, 244) if visited else (200, 206, 214)
        pygame.draw.rect(self.screen, color, (x0, y0, self.cfg.ui.cell_px, self.cfg.ui.cell_px))

    def _draw_view_frame(self, origin: tuple[int, int], title: str) -> None:
        ox, oy = origin
        s = self.cfg.ui.cell_px
        w_px = self.cfg.maze.width * s
        h_px = self.cfg.maze.height * s
        pygame.draw.rect(self.screen, (17, 28, 45), (ox - 3, oy - 3, w_px + 6, h_px + 6), border_radius=4)
        title_txt = self.font_small.render(title, True, (219, 230, 246))
        self.screen.blit(title_txt, (ox, oy - 22))

    def _draw_walker_view(self) -> None:
        assert self.env.belief is not None and self.env.truth is not None

        m_x, m_y = self.walker_origin
        s = self.cfg.ui.cell_px

        self._draw_view_frame(self.walker_origin, "Walker View (only movement + fog)")

        # Walker-only layer + fog.
        for r in range(self.cfg.maze.height):
            for c in range(self.cfg.maze.width):
                x0 = m_x + c * s
                y0 = m_y + r * s
                visited = self.env.visits[r, c] > 0
                self._draw_cell_bg(x0, y0, visited=visited)

                if not visited:
                    fog = pygame.Surface((s, s), pygame.SRCALPHA)
                    fog.fill((45, 52, 66, self.cfg.ui.fog_alpha))
                    self.screen.blit(fog, (x0, y0))

        # Path trace.
        if len(self.trace) > 1:
            points = []
            for r, c in self.trace[-160:]:
                points.append((m_x + c * s + s // 2, m_y + r * s + s // 2))
            pygame.draw.lines(self.screen, (255, 165, 50), False, points, 2)

        # Walker.
        wr, wc = self.env.walker_pos
        cx, cy = m_x + wc * s + s // 2, m_y + wr * s + s // 2
        pygame.draw.circle(self.screen, (29, 40, 50), (cx + 2, cy + 2), s // 4)
        pygame.draw.circle(self.screen, (24, 175, 96), (cx, cy), s // 4)

    def _draw_mapper_view(self) -> None:
        assert self.env.belief is not None and self.env.truth is not None
        m_x, m_y = self.mapper_origin
        s = self.cfg.ui.cell_px

        self._draw_view_frame(self.mapper_origin, "Mapper View (reconstruction + blackholes)")

        for r in range(self.cfg.maze.height):
            for c in range(self.cfg.maze.width):
                x0 = m_x + c * s
                y0 = m_y + r * s
                pygame.draw.rect(self.screen, (225, 229, 236), (x0, y0, s, s))
                suspect = float(self.env.belief.hole_suspect[r, c])
                if suspect > 0.01:
                    alpha = int(min(170, 40 + suspect * 160))
                    surf = pygame.Surface((s, s), pygame.SRCALPHA)
                    surf.fill((210, 40, 40, alpha))
                    self.screen.blit(surf, (x0, y0))

                if (self.show_truth_blackholes or self.cfg.ui.mapper_show_true_blackholes) and self.env.truth.blackholes[r, c]:
                    pygame.draw.circle(self.screen, (40, 40, 40), (x0 + s // 2, y0 + s // 2), s // 7)

        # Estimated walls.
        for r in range(self.cfg.maze.height + 1):
            for c in range(self.cfg.maze.width):
                val = int(self.env.belief.h_est[r, c])
                if val == WALL:
                    x1, y1 = m_x + c * s, m_y + r * s
                    x2, y2 = x1 + s, y1
                    pygame.draw.line(self.screen, (12, 61, 138), (x1, y1), (x2, y2), 4)

        for r in range(self.cfg.maze.height):
            for c in range(self.cfg.maze.width + 1):
                val = int(self.env.belief.v_est[r, c])
                if val == WALL:
                    x1, y1 = m_x + c * s, m_y + r * s
                    x2, y2 = x1, y1 + s
                    pygame.draw.line(self.screen, (12, 61, 138), (x1, y1), (x2, y2), 4)

        if self.show_truth_walls:
            for r in range(self.cfg.maze.height + 1):
                for c in range(self.cfg.maze.width):
                    if self.env.truth.h_walls[r, c]:
                        x1, y1 = m_x + c * s, m_y + r * s
                        x2, y2 = x1 + s, y1
                        pygame.draw.line(self.screen, (220, 70, 70), (x1, y1), (x2, y2), 2)
            for r in range(self.cfg.maze.height):
                for c in range(self.cfg.maze.width + 1):
                    if self.env.truth.v_walls[r, c]:
                        x1, y1 = m_x + c * s, m_y + r * s
                        x2, y2 = x1, y1 + s
                        pygame.draw.line(self.screen, (220, 70, 70), (x1, y1), (x2, y2), 2)

        wr, wc = self.env.walker_pos
        cx, cy = m_x + wc * s + s // 2, m_y + wr * s + s // 2
        pygame.draw.circle(self.screen, (24, 175, 96), (cx, cy), s // 5)

        # Entry points.
        for idx, (er, ec) in enumerate(self.env.truth.entries):
            ex, ey = m_x + ec * s + 5, m_y + er * s + 5
            pygame.draw.rect(self.screen, (95, 160, 245), (ex, ey, 12, 12))
            txt = self.font_small.render(f"E{idx}", True, (20, 20, 20))
            self.screen.blit(txt, (ex + 13, ey - 2))

    def _draw_action_mask(self, x: int, y: int) -> None:
        labels = ["U", "R", "D", "L"]
        size = 34
        pad = 8
        mask = self.obs.walker_obs.valid_mask

        for i, label in enumerate(labels):
            px = x + i * (size + pad)
            color = (90, 90, 90) if mask[i] == 0 else (40, 130, 210)
            pygame.draw.rect(self.screen, color, (px, y, size, size), border_radius=6)
            txt = self.font.render(label, True, (250, 250, 250))
            self.screen.blit(txt, (px + 10, y + 6))

    def _draw_panel(self) -> None:
        panel_x = self.panel_x
        panel_rect = (panel_x, 0, self.cfg.ui.panel_width_px, self.height_px)
        pygame.draw.rect(self.screen, (23, 28, 34), panel_rect)

        lines = [
            "Soulgue-Maze",
            f"Mode: {'AI' if self.ai_mode else 'Manual'}",
            f"Paused: {self.paused}",
            f"Episode: {self.episode}",
            f"Step: {self.env.step_count}",
            f"Success Count: {self.success_count}",
            f"Walker Return: {self.cum_walker_reward:.2f}",
            f"Mapper Return: {self.cum_mapper_reward:.2f}",
            f"Coverage: {self.env.compute_coverage():.3f}",
            f"Accuracy: {self.env.compute_map_accuracy():.3f}",
            f"Resets Left: {self.env.resets_left}",
            f"No Progress Steps: {self.no_progress_steps}",
            f"Guard Resets: {self.guard_reset_count}",
            f"Stuck Bin: {self.obs.mapper_obs.stuck_bin}",
            f"Anomaly: {self.obs.walker_obs.anomaly_flag}",
            "",
            "Keys:",
            "M mode | Space pause",
            "R reset episode",
            "B blackhole debug",
            "W wall debug",
            "Arrow keys manual move",
            "E manual mapper reset",
        ]

        y = 20
        for idx, line in enumerate(lines):
            f = self.font if idx == 0 else self.font_small
            color = (230, 236, 244)
            txt = f.render(line, True, color)
            self.screen.blit(txt, (panel_x + 16, y))
            y += 28 if idx == 0 else 23

        self._draw_action_mask(panel_x + 18, self.height_px - 95)

    def _handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_m:
                self.ai_mode = not self.ai_mode
            elif event.key == pygame.K_r:
                self.episode += 1
                self._reset_episode()
            elif event.key == pygame.K_b:
                self.show_truth_blackholes = not self.show_truth_blackholes
            elif event.key == pygame.K_w:
                self.show_truth_walls = not self.show_truth_walls
            elif event.key == pygame.K_e:
                self.manual_mapper_reset = True
            elif event.key in KEY_TO_ACTION:
                self.manual_walker_action = KEY_TO_ACTION[event.key]
        return True

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                running = self._handle_event(event)
                if not running:
                    break
            if not running:
                break

            if not self.paused:
                self._advance()

            self.screen.fill((15, 18, 22))
            self._draw_walker_view()
            self._draw_mapper_view()
            self._draw_panel()
            pygame.display.flip()
            self.clock.tick(self.cfg.ui.fps)

        pygame.quit()


def main() -> None:
    # Pygame can emit noisy startup logs in some terminals.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    ui = SoulgueUI(DEFAULT_CONFIG)
    ui.run()


if __name__ == "__main__":
    main()

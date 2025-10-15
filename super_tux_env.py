import pystk
import numpy as np
import pygame
import cv2

from matplotlib.patches import FancyArrowPatch


class SuperTuxEnv:

    ACTIONS = {
        0: [0, 0, 0],
        1: [0, 0, 1],
        2: [0, 0, -1],
        3: [0, 0.05, 0],
        4: [0, 0.05, 1],
        5: [0, 0.05, -1],
        6: [1, 0, 0],
        7: [1, 0, 1],
        8: [1, 0, -1]
    }

    def __init__(self, track_name: str, num_karts: int = 1, num_laps: int = 3, max_episode_steps: int = 2500,
                 obs_size: tuple = (64, 64), reward_scheme: str = "nodes", node_reward_freq: int = 1,
                 use_time_reward: bool = True, render: bool = False, screen_width: int = 800, screen_height: int = 600,
                 show_map=False, render_data="image"):
        self._screen_width = screen_width
        self._screen_height = screen_height

        self._config = pystk.GraphicsConfig.hd()
        self._config.screen_width = obs_size[0]
        self._config.screen_height = obs_size[1]
        pystk.init(self._config)

        if render:
            pygame.display.init()
            self._screen = pygame.display.set_mode((self._screen_width, self._screen_height))
        else:
            self._screen = None

        self._config = pystk.RaceConfig()
        self._config.num_kart = num_karts
        self._config.track = track_name

        self._state = pystk.WorldState()
        self._track = pystk.Track()
        self._race = pystk.Race(self._config)

        # Other attributes
        assert len(obs_size) == 2, "obs_size must be (width, height)"
        self.obs_size = obs_size
        self._episode_steps = 0
        self._current_num_laps = 0
        self._position_metrics = []
        self._position_metrics_length = 150
        self._velocity_metrics = []
        self._velocity_metrics_length = 30
        self._num_laps = num_laps
        self._max_episode_steps = max_episode_steps
        self._num_nodes = 0

        self._next_reward_node = 0
        self._node_reward_freq = node_reward_freq
        self._closest_node_frequency_update = 250
        assert reward_scheme in ["super_sparse", "sparse", "nodes", "dense"], "Unknown reward scheme."
        self._reward_scheme = reward_scheme
        self._use_time_reward = use_time_reward
        self._rescue_countdown = 0

        assert render_data in ["image", "depth", "instance"], "Unknown render type."
        self._render_data = render_data

        # Advice: slow, use only for debugging
        """if render and show_map:
            self._fig, self._ax = plt.subplots()
        else:
            self._fig = None
            self._ax = None
            plt.ion()
            plt.show()"""

        # Initialize
        self._first_reset = True

    def reset(self):
        # Not pretty, but that's how pystk works
        if self._first_reset:
            self._race.start()
            self._state.update()
            self._track.update()
            # TODO: doesn't make sense, but avoids recomputing at each timestep
            self._track_nodes = np.mean(np.array(self._track.path_nodes), axis=1)
            self._num_nodes = self._track_nodes.shape[0]
            self._first_reset = False
        else:
            self._race.restart()

        obs = np.zeros(shape=(*self.obs_size, 3), dtype=np.float32)

        self._episode_steps = 0
        self._current_num_laps = 0
        self._next_reward_node = self._node_reward_freq
        self._reward_to_assign = 0.0
        self._position_metrics = []
        self._velocity_metrics = []
        self._rescue_countdown = 0
        self._stuck_detection_countdown = 0

        return obs, {}

    def step(self, action):
        self._episode_steps += 1
        has_progressed = False
        reward_multiplier = 1

        # TODO: change reward based on reward_scheme
        closest_node = self._get_closest_track_node()
        driving_backward = self._state.players[0].kart.finished_laps < 0 and self._episode_steps >= self._velocity_metrics_length and self._reward_scheme == "dense"
        if closest_node == self._next_reward_node:
            has_progressed = True
            self._next_reward_node = (self._next_reward_node + self._node_reward_freq) % self._num_nodes
        elif closest_node == self._next_reward_node + 1:
            # This takes into account a bug coming from PySTK where some nodes are skipped
            if self._next_reward_node > 2:
                has_progressed = True
                reward_multiplier = (closest_node - self._next_reward_node)
                self._next_reward_node = closest_node
                self._next_reward_node = ((self._next_reward_node + self._node_reward_freq) * reward_multiplier) % self._num_nodes
                reward_multiplier += 1

        if action is not None:
            _ = self._race.step(action)
        else:
            _ = self._race.step()

        if self._render_data == "image":
            obs = self._race.render_data[0].image
        elif self._render_data == "instance":
            obs = np.array(self._race.render_data[0].instance).astype(np.float32)
            obs = np.expand_dims(((obs - np.min(obs)) / np.ptp(obs) * 255.).astype(np.uint8), -1)
            obs = np.repeat(obs, 3, axis=-1)
        elif self._render_data == "depth":
            obs = np.array(self._race.render_data[0].depth).astype(np.float32)
            obs = np.expand_dims(((obs - np.min(obs)) / np.ptp(obs) * 255.).astype(np.uint8), -1)
            obs = np.repeat(obs, 3, axis=-1)

        self._state.update()

        # Update tracked metrics
        if len(self._position_metrics) > self._position_metrics_length:
            self._position_metrics.pop(0)
        self._position_metrics.append(self._state.players[0].kart.location)
        if len(self._velocity_metrics) > self._velocity_metrics_length:
            self._velocity_metrics.pop(0)
        self._velocity_metrics.append(self._state.players[0].kart.velocity)

        # Reward assignment
        reward, needs_rescue, info = self._assign_reward(driving_backward, has_progressed, reward_multiplier)

        # Terminal conditions
        terminated = self._current_num_laps == self._num_laps
        # Second condition tries to avoid exploiting bad polygons
        truncated = self._episode_steps >= self._max_episode_steps or driving_backward

        info["closest_node"] = closest_node
        if self._reward_scheme == "dense":
            info["driving_backward"] = driving_backward
        if self._reward_scheme == "super_sparse" and (terminated or truncated):
            reward -= 0.01 * self._episode_steps

        info["needs_rescue"] = needs_rescue
        info["next_reward_node"] = self._next_reward_node

        return obs, reward, terminated, truncated, info

    def _assign_reward(self, driving_backward: bool, has_progressed: bool, reward_multiplier: float):
        completed_laps = max(0, self._state.players[0].kart.finished_laps)
        needs_rescue = False
        info = {}

        # Time component
        time_reward = -0.01 if self._use_time_reward and self._reward_scheme != "super_sparse" else 0.0

        if self._reward_scheme == "nodes":
            if has_progressed:
                # Reward multiplier takes into account the skip node bug
                progress_reward = 0.1 * reward_multiplier
            else:
                progress_reward = 0.0
        elif self._reward_scheme in ["sparse", "super_sparse"]:
            progress_reward = 0.0
        else:
            current_lap_distance = self._state.players[0].kart.distance_down_track
            progress_reward = (completed_laps + 1) * current_lap_distance / self._track.length

        # Base reward
        reward = time_reward + progress_reward

        # Useful only if reward_scheme == "dense", to avoid exploitation of bad polygons
        if driving_backward and self._rescue_countdown == 0:
            reward -= 100

        if self._current_num_laps != completed_laps:
            reward += 10
            self._current_num_laps += 1
            self._best_node_reached = 0

        # The longer the run, the harder to get there
        if self._current_num_laps == self._num_laps:
            reward += self._num_laps * 10

        if self.is_player_falling() and self._rescue_countdown == 0:
            # TODO: -0.02 for time and -0.01 for falling?
            #reward -= .5
            self._velocity_metrics = []
            needs_rescue = True
            self._rescue_countdown = 30

        # TODO: add player stuck situation
        """if self.is_player_stuck() and self._stuck_detection_countdown == 0:
            reward -= 5
            self._position_metrics = []
            needs_rescue = True
            self._stuck_detection_countdown = 150"""

        # Timer to detect a new rescue condition (avoid applying negative reward to uncontrollable frames)
        if self._rescue_countdown > 0:
            self._rescue_countdown -= 1
        if self._stuck_detection_countdown > 0:
            self._stuck_detection_countdown -= 1

        info["completed_laps"] = self._current_num_laps

        return reward, needs_rescue, info

    # TODO: find an efficient method to detect this
    def is_player_stuck(self):
        if (self._stuck_detection_countdown > 0 or len(self._position_metrics) < self._position_metrics_length) or 5 <= self._get_closest_track_node() <= self._num_nodes - 5:
            return False

        mean_delta_location = np.linalg.norm(np.array(self._position_metrics[1:]) - np.array(self._position_metrics[:-1]), axis=0)
        return -2 <= np.mean(mean_delta_location, axis=0) <= 2

    def is_player_falling(self):
        if self._rescue_countdown > 0:
            return False

        return self._state.players[0].kart.jumping and np.mean(np.array(self._velocity_metrics)[:, 1]) <= 0

    def _get_closest_track_node(self):
        position = self._state.players[0].kart.location

        return np.argmin(np.linalg.norm(self._track_nodes - position, axis=1))

    def get_current_episode_steps(self):
        return self._episode_steps

    def get_kart_velocity(self):
        return self._state.players[0].kart.velocity

    def render(self, mode=None):
        if self._render_data == "image":
            obs = self._race.render_data[0].image
        elif self._render_data == "instance":
            obs = np.array(self._race.render_data[0].instance).astype(np.float32)
            obs = np.expand_dims(((obs - np.min(obs)) / np.ptp(obs) * 255.).astype(np.uint8), -1)
            obs = np.repeat(obs, 3, axis=-1)
        elif self._render_data == "depth":
            obs = np.array(self._race.render_data[0].depth).astype(np.float32)
            obs = np.expand_dims(((obs - np.min(obs)) / np.ptp(obs) * 255.).astype(np.uint8), -1)
            obs = np.repeat(obs, 3, axis=-1)

        #obs = cv2.resize(obs, (self._screen_width, self._screen_height))
        frame = pygame.surfarray.make_surface(np.swapaxes(obs, 0, 1))
        self._screen.blit(frame, (0, 0))
        pygame.display.flip()
        if mode == "human":
            pygame.time.delay(30)

    def _update_map(self):
        nodes = np.array(self._track.path_nodes)
        nodes = np.mean(nodes, axis=1)
        x_values = nodes[:, 0]
        y_values = nodes[:, 2]

        closest = self._get_closest_track_node()

        self._ax.plot(x_values[closest], y_values[closest], marker='o', color='red', markersize=8)  # First point in red
        for i in range(0, x_values.shape[0]):
            self._ax.add_patch(FancyArrowPatch((x_values[i - 1], y_values[i - 1]), (x_values[i], y_values[i]),
                                               mutation_scale=15, color='blue', arrowstyle='->'))

        self._ax.plot(x_values[1:closest], y_values[1:closest], marker='o', color='blue', markersize=8)
        self._ax.plot(x_values[closest + 1:], y_values[closest + 1:], marker='o', color='blue', markersize=8)

        #plt.draw()

    def close(self):
        #plt.ioff()
        self._race.stop()
        del self._race
        pystk.clean()

        if self._screen is not None:
            pygame.quit()
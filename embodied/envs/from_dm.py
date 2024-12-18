import functools
import os
import csv
import json
import atexit


import embodied
import numpy as np
from gymnasium import spaces


class FromDM(embodied.Env):

    def __init__(self, env, obs_key='observation', act_key='action'):
        self._env = env
        obs_spec = self._env.observation_spec()
        act_spec = self._env.action_spec()
        self._obs_dict = isinstance(obs_spec, dict)
        self._act_dict = isinstance(act_spec, dict)
        self._obs_key = not self._obs_dict and obs_key
        self._act_key = not self._act_dict and act_key
        self._obs_empty = []
        self._done = True

        # Define default confounder parameters directly within the class
        self.confounder_params = {
            'weight': {
                'scale_factor': 1.0,
                'body_parts': {
                    'torso': 1.0,  # Weight scaling
                    'left_leg': 1.0,  # Weight scaling
                    'right_leg': 1.0
                    # Add more body parts as needed
                }
            },
            'friction': {
                'scale_factor': 1.0  # Example: Increase friction by 50%
            },
            'gravity': {
                'scale_factor': 0.8  # Example: Decrease gravity by 10%
            },
            'external_force': {
                'enabled': False,
                'force_type': 'step',  # 'step' or 'swelling'
                'force_range': [10.0, 20.0],  # Range of force magnitudes
                'interval_mean': 100,  # Mean interval for force application
                'interval_std': 10,  # Std deviation for interval (if random)
                'random_chance': 0.5,  # Chance to apply random force
                'duration_min': 10,  # Min duration for force application
                'duration_max': 20,  # Max duration for force application
                'body_part': 'torso'  # Body part to apply force
            }
        }

        # Define default action masking parameters
        self.action_masking_params = {
            'enabled': False,
            'mode': 'whole_leg',  # 'whole_leg' or 'individual_joints'
            'cripple_mode': 'whole_leg',  # Redundant? Ensure consistency
            'cripple_targets': ['left_leg', 'right_leg'],  # Example targets
            'joints': []  # If 'individual_joints' mode is used
        }

        # Initialize attributes based on confounder_params
        self.cripple_part = self.action_masking_params.get('cripple_targets', None)
        self.force_type = self.confounder_params['external_force'].get('force_type', 'step')
        self.timing = self.confounder_params['external_force'].get('timing', 'random')
        self.body_part = self.confounder_params['external_force'].get('body_part', 'torso')
        self.force_range = self.confounder_params['external_force'].get('force_range', [10.0, 20.0])
        self.interval_mean = self.confounder_params['external_force'].get('interval_mean', 100)
        self.interval_std = self.confounder_params['external_force'].get('interval_std', 10)
        self.random_chance = self.confounder_params['external_force'].get('random_chance', 0.5)
        self.duration_min = self.confounder_params['external_force'].get('duration_min', 10)
        self.duration_max = self.confounder_params['external_force'].get('duration_max', 20)
        self.time_since_last_force = 0
        self.interval = self.interval_mean  # Initialize interval

        # Initialize lists and flags for confounders
        self.masked_joints_ids = []  # To store joint IDs for masking
        self.force_application_enabled = self.confounder_params['external_force'].get('enabled', False)

        # Apply confounders
        self.apply_confounders()

        # Apply action masking
        self.apply_action_masking()

        # Initialize logging
        self.logdir = 'transition_logs/Walker_RUN-gravity_0.8'
        os.makedirs(self.logdir, exist_ok=True)
        self.log_file_path = os.path.join(self.logdir, 'transitions.csv')
        self.buffer_size = 1024
        self.log_buffer = []
        self.step_number = 0

        # Define CSV headers
        self.csv_headers = [
            'step_number',
            'state',
            'action',
            'reward',
            'next_state',
            'done',
            'confounder_weight_scale_factor',
            'confounder_friction_scale_factor',
            'confounder_gravity_scale_factor',
            'external_force_enabled',
            'external_force_type',
            'external_force_range_min',
            'external_force_range_max',
            'external_force_interval_mean',
            'external_force_interval_std',
            'external_force_random_chance',
            'external_force_duration_min',
            'external_force_duration_max',
            'external_force_body_part',
            'action_mask'
        ]

        # Initialize the CSV file and write headers if the file is new
        file_exists = os.path.isfile(self.log_file_path)
        self.csv_file = open(self.log_file_path, mode='a', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_headers)
        if not file_exists:
            self.csv_writer.writeheader()

        # Register a cleanup function to ensure the file is closed properly
        atexit.register(self.close_logger)

        # Initialize previous observation
        self.prev_obs = None
        # Initialize the environment to set the first observation
        initial_time_step = self._env.reset()
        self.prev_obs = self._obs(initial_time_step)


    @functools.cached_property
    def obs_space(self):
        spec = self._env.observation_spec()
        spec = spec if self._obs_dict else {self._obs_key: spec}
        if 'reward' in spec:
            spec['obs_reward'] = spec.pop('reward')
        for key, value in spec.copy().items():
            if int(np.prod(value.shape)) == 0:
                self._obs_empty.append(key)
                del spec[key]
        spaces = {
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }
        for key, value in spec.items():
            key = key.replace('/', '_')
            spaces[key] = self._convert(value)
        return spaces

    @functools.cached_property
    def act_space(self):
        spec = self._env.action_spec()
        spec = spec if self._act_dict else {self._act_key: spec}
        return {
            'reset': embodied.Space(bool),
            **{k or self._act_key: self._convert(v) for k, v in spec.items()},
        }

    def step(self, action):
        action = action.copy()
        reset = action.pop('reset')
        if reset or self._done:
            time_step = self._env.reset()
            self.prev_obs = self._obs(time_step)  # Reset prev_obs on episode start
            # No transition to log yet, as this is a reset step
            return self.prev_obs, 0.0, False, False, {}

        # Apply external force if enabled
        if self.force_application_enabled:
            self.apply_force()

        # Apply action masking
        masked_action = self._action_mask(
            action if self._act_dict else action[self._act_key])

        # Step the environment with the (optionally) modified action
        time_step = self._env.step(masked_action)
        reward = time_step.reward or 0.0
        done = time_step.last()
        obs = self._obs(time_step)

        # Log the transition:
        # We now have (prev_obs, action, reward, obs, done).
        # Make sure prev_obs is defined (it should be after the first reset call).
        # Note: Use the original action (before masking) if you want to log that.
        self.log_transition(
            state=self.prev_obs,
            action=action,  # Log the original action or 'masked_action' if you prefer
            reward=reward,
            next_state=obs,
            done=done
        )

        # Update prev_obs for the next call
        self.prev_obs = obs
        self._done = done

        # No explicit 'truncated' logic here, adjust if needed
        truncated = False
        return obs, reward, done, truncated, {}

    def _obs(self, time_step):
        if not time_step.first():
            assert time_step.discount in (0, 1), time_step.discount
        obs = time_step.observation
        obs = dict(obs) if self._obs_dict else {self._obs_key: obs}
        if 'reward' in obs:
            obs['obs_reward'] = obs.pop('reward')
        for key in self._obs_empty:
            del obs[key]
        obs = {k.replace('/', '_'): v for k, v in obs.items()}
        return dict(
            reward=np.float32(0.0 if time_step.first() else time_step.reward),
            is_first=time_step.first(),
            is_last=time_step.last(),
            is_terminal=False if time_step.first() else time_step.discount == 0,
            **obs,
        )

    def _convert(self, space):
        if hasattr(space, 'num_values'):
            return embodied.Space(space.dtype, (), 0, space.num_values)
        elif hasattr(space, 'minimum'):
            assert np.isfinite(space.minimum).all(), space.minimum
            assert np.isfinite(space.maximum).all(), space.maximum
            return embodied.Space(
                space.dtype, space.shape, space.minimum, space.maximum)
        else:
            return embodied.Space(space.dtype, space.shape, None, None)

    def apply_confounders(self):
        """
        Apply various confounders such as weight scaling, friction scaling, and gravity scaling.
        """
        # Apply weight scaling
        weight_conf = self.confounder_params.get('weight', {})
        scale_factor = weight_conf.get('scale_factor', 1.0)
        body_parts = weight_conf.get('body_parts', {})

        for body_part, scale in body_parts.items():
            try:
                body_id = self._env.physics.model.name2id(body_part, 'body')
                original_mass = self._env.physics.model.body_mass[body_id]
                self._env.physics.model.body_mass[body_id] = original_mass * scale_factor * scale
            except KeyError:
                print(f"Warning: Body part '{body_part}' not found in the environment.")

        # Apply friction scaling
        friction_conf = self.confounder_params.get('friction', {})
        friction_scale = friction_conf.get('scale_factor', 1.0)
        try:
            geom_id = self._env.physics.model.name2id('floor', 'geom')
            self._env.physics.model.geom_friction[geom_id, :] *= friction_scale
        except KeyError:
            print("Warning: 'floor' geom not found in the environment.")

        # Apply gravity scaling
        gravity_conf = self.confounder_params.get('gravity', {})
        gravity_scale = gravity_conf.get('scale_factor', 1.0)
        self._env.physics.model.opt.gravity[:] *= gravity_scale

    def apply_action_masking(self):
        """
        Apply action masking based on the configured mode and targets.
        """
        if not self.action_masking_params.get('enabled', False):
            return  # No action masking to apply

        mode = self.action_masking_params.get('mode', 'whole_leg')
        cripple_mode = self.action_masking_params.get('cripple_mode', 'whole_leg')
        cripple_targets = self.action_masking_params.get('cripple_targets', [])
        cripple_joints = []

        if mode == 'whole_leg':
            # Define mapping from leg to joints
            leg_to_joints = {
                'left_leg': ['left_hip', 'left_knee', 'left_ankle'],
                'right_leg': ['right_hip', 'right_knee', 'right_ankle'],
                'front_leg': ['front_hip', 'front_knee', 'front_ankle'],
                'rear_leg': ['back_hip', 'back_knee', 'back_ankle'],
                'front_left_leg': ['front_left_hip', 'front_left_knee', 'front_left_ankle'],
                'front_right_leg': ['front_right_hip', 'front_right_knee', 'front_right_ankle'],
                'rear_left_leg': ['rear_left_hip', 'rear_left_knee', 'rear_left_ankle'],
                'rear_right_leg': ['rear_right_hip', 'rear_right_knee', 'rear_right_ankle']
            }
            for leg in cripple_targets:
                joints = leg_to_joints.get(leg, [])
                cripple_joints.extend(joints)
        elif mode == 'individual_joints':
            cripple_joints = self.action_masking_params.get('joints', [])

        # Add joints to mask
        for joint in cripple_joints:
            try:
                joint_id = self._env.physics.model.name2id(joint, 'joint')
                self.masked_joints_ids.append(joint_id)
            except KeyError:
                print(f"Warning: Joint '{joint}' not found in the environment.")

    def apply_force(self):
        """
        Apply external forces to the specified body part based on the configured parameters.
        """
        if self.timing == 'random':
            # Sample a new interval from a normal distribution
            self.interval = max(30, int(np.random.normal(self.interval_mean, self.interval_std)))
            if np.random.uniform() > self.random_chance:
                return  # Do not apply force this step

        # Update the timing
        self.time_since_last_force += 1
        if self.time_since_last_force < self.interval:
            return  # Not time to apply force yet

        # Reset timing for next force application
        self.time_since_last_force = 0

        # Sample the force magnitude from a normal distribution within the range
        force_magnitude = np.clip(
            np.random.normal(
                (self.force_range[0] + self.force_range[1]) / 2,
                (self.force_range[1] - self.force_range[0]) / 6
            ),
            self.force_range[0],
            self.force_range[1]
        )

        # Calculate the duration for the force application if 'swelling'
        duration = np.random.randint(self.duration_min, self.duration_max + 1)

        # Flip the direction for additional variability
        direction = np.random.choice([-1, 1])

        # Construct the force vector based on the force type
        if self.force_type == 'step':
            force = np.array([direction * force_magnitude, 0, 0, 0, 0, 0])
        elif self.force_type == 'swelling':
            # Calculate the time step where the force magnitude is at its peak
            peak_time = duration / 2
            # Calculate the standard deviation to control the width of the bell curve
            sigma = duration / 6  # Adjust as needed for the desired width
            # Calculate the force magnitude at the current time step using a Gaussian function
            time_step_normalized = (self.time_since_last_force - peak_time) / sigma
            magnitude = force_magnitude * np.exp(-0.5 * (time_step_normalized ** 2))
            force = np.array([direction * magnitude, 0, 0, 0, 0, 0])
        else:
            print(f"Warning: Unknown force type '{self.force_type}'. No force applied.")
            return

        try:
            body_id = self._env.physics.model.name2id(self.body_part, 'body')
            # Apply the force
            self._env.physics.data.xfrc_applied[body_id] = force
        except KeyError:
            print(f"Warning: Body part '{self.body_part}' not found in the environment.")

    def _action_mask(self, action):
        """
        Apply action masking by zeroing out actions for masked joints.
        Assumes that the action is a NumPy array or similar.
        """
        if not self.masked_joints_ids:
            return action  # No masking needed

        # Zero out actions for masked joints
        for joint_id in self.masked_joints_ids:
            if joint_id < len(action):
                action[joint_id] = 0.0  # Adjust indexing as per your action space
            else:
                print(f"Warning: Joint ID '{joint_id}' is out of action space bounds.")
        return action

    def log_transition(self, state, action, reward, next_state, done):
        """
        Log a single transition to the buffer.

        Args:
            state (dict): The observation before the action.
            action (np.ndarray or list): The action taken.
            reward (float): The reward received.
            next_state (dict): The observation after the action.
            done (bool): Whether the episode has terminated.
        """
        transition = {
            'step_number': self.step_number,
            'state': json.dumps(state),
            'action': json.dumps(action.tolist() if isinstance(action, np.ndarray) else action),
            'reward': reward,
            'next_state': json.dumps(next_state),
            'done': done,
            'confounder_weight_scale_factor': self.weight_scale_factor if self.confounder_active else None,
            'confounder_friction_scale_factor': self.friction_scale_factor if self.confounder_active else None,
            'confounder_gravity_scale_factor': self.gravity_scale_factor if self.confounder_active else None,
            'external_force_enabled': self.force_application_enabled if self.confounder_active else None,
            'external_force_type': self.force_type if self.confounder_active else None,
            'external_force_range_min': self.force_range[0] if self.confounder_active else None,
            'external_force_range_max': self.force_range[1] if self.confounder_active else None,
            'external_force_interval_mean': self.interval_mean if self.confounder_active else None,
            'external_force_interval_std': self.interval_std if self.confounder_active else None,
            'external_force_random_chance': self.random_chance if self.confounder_active else None,
            'external_force_duration_min': self.duration_min if self.confounder_active else None,
            'external_force_duration_max': self.duration_max if self.confounder_active else None,
            'external_force_body_part': self.body_part if self.confounder_active else None,
            'action_mask': json.dumps(self.action_mask.tolist()) if self.action_mask is not None else json.dumps([])
        }

        self.log_buffer.append(transition)
        self.step_number += 1

        # Flush the buffer if it reaches the buffer_size
        if len(self.log_buffer) >= self.buffer_size:
            self.flush_buffer()

    def flush_buffer(self):
        """
        Flush the buffered transitions to the CSV file.
        """
        if not self.log_buffer:
            return  # Nothing to flush

        self.csv_writer.writerows(self.log_buffer)
        self.csv_file.flush()
        self.log_buffer = []

    def close_logger(self):
        """
        Flush any remaining transitions and close the CSV file.
        """
        self.flush_buffer()
        self.csv_file.close()

    def close(self):
        """
        Close the environment and the logger.
        """
        self.close_logger()
        self._env.close()


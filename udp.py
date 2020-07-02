import time
import torch
import pickle
import platform

import sys
import datetime

import select, termios, tty

try:
  from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
  from .cassiemujoco.cassieUDP import *
  from .cassiemujoco.cassiemujoco_ctypes import *
except ImportError:
  from cassiemujoco.cassieUDP import *
  from cassiemujoco.cassiemujoco_ctypes import *

import numpy as np

import math
import numpy as np

def inverse_quaternion(quaternion):
	result = np.copy(quaternion)
	result[1:4] = -result[1:4]
	return result

def quaternion_product(q1, q2):
	result = np.zeros(4)
	result[0] = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
	result[1] = q1[0]*q2[1]+q2[0]*q1[1]+q1[2]*q2[3]-q1[3]*q2[2]
	result[2] = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
	result[3] = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
	return result

def rotate_by_quaternion(vector, quaternion):
	q1 = np.copy(quaternion)
	q2 = np.zeros(4)
	q2[1:4] = np.copy(vector)
	q3 = inverse_quaternion(quaternion)
	q = quaternion_product(q2, q3)
	q = quaternion_product(q1, q)
	result = q[1:4]
	return result

def quaternion2euler(quaternion):
	w = quaternion[0]
	x = quaternion[1]
	y = quaternion[2]
	z = quaternion[3]
	ysqr = y * y

	ti = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(ti, t1))

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))

	result = np.zeros(3)
	result[0] = X * np.pi / 180
	result[1] = Y * np.pi / 180
	result[2] = Z * np.pi / 180

	return result

def euler2quat(z=0, y=0, x=0):

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    result =  np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])
    if result[0] < 0:
    	result = -result
    return result

def check_stdin():
  return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def run_udp(policy_files):
  from util.env import env_factory

  policies   = [torch.load(p) for p in policy_files]
  m_policies = [torch.load(p) for p in policy_files]


  legacies = ['legacy' if not (hasattr(p, 'legacy') and p.legacy == False) else '' for p in policies]
  envs     = [env_factory(p.env_name + legacy)() for p, legacy in zip(policies, legacies)]

  for i, env in enumerate(envs):
    if not env.state_est:
      print("{} was not trained with state estimation and cannot be run on the robot.".format(policy_files[i]))
      raise RuntimeError
  
  for i, p in enumerate(policies):
    print("Policy {:3d} is a: {}".format(i, p.__class__.__name__))
  time.sleep(1)

  time_log   = [] # time stamp
  input_log  = [] # network inputs
  output_log = [] # network outputs 
  state_log  = [] # cassie state
  target_log = [] # PD target log

  env = envs[0]
  clock_based = env.clock
  no_delta = True

  u = pd_in_t()
  for i in range(5):
      u.leftLeg.motorPd.pGain[i]  = env.P[i]
      u.leftLeg.motorPd.dGain[i]  = env.D[i]
      u.rightLeg.motorPd.pGain[i] = env.P[i]
      u.rightLeg.motorPd.dGain[i] = env.D[i]

  if platform.node() == 'cassie':
      cassie = CassieUdp(remote_addr='10.10.10.3', remote_port='25010',
                         local_addr='10.10.10.100', local_port='25011')
  else:
      cassie = CassieUdp() # local testing

  print('Connecting...')
  y = None
  while y is None:
      cassie.send_pd(pd_in_t())
      time.sleep(0.001)
      y = cassie.recv_newest_pd()

  policy_idx = None

  print('Connected!\n')

  # Whether or not STO has been TOGGLED (i.e. it does not count the initial STO condition)
  # STO = True means that STO is ON (i.e. robot is not running) and STO = False means that STO is
  # OFF (i.e. robot *is* running)
  ESTOP = False

  # We have multiple modes of operation
  # 0: Normal operation, walking with policy
  # 1: Zero out memory of policy (if applicable)
  # 2: Stop, drop, and hopefully not roll, damping mode with no P gain
  operation_mode = 0

  # Command inputs
  speed        = 0
  side_speed   = 0
  orient_add   = 0
  phase_add    = 75
  phase        = 0

  D_mult       = 1
  actual_speed = 0
  delay        = 30
  counter      = 0
  pitch_bias   = 0
  ESTOP_count  = 0
  max_speed    =  2.00
  min_speed    = -0.30
  max_y_speed  =  0.25
  min_y_speed  = -0.25
  cmd_height   = 0.9
  cmd_foot_height = 0.05
  logged       = True
  mirror       = False
  last_torque  = None

  for policy in policies:
    if hasattr(policy, 'init_hidden_state'):
      policy.init_hidden_state()
  for policy in m_policies:
    if hasattr(policy, 'init_hidden_state'):
      policy.init_hidden_state()

  old_settings = termios.tcgetattr(sys.stdin)
  try:
    tty.setcbreak(sys.stdin.fileno())

    t  = time.monotonic()
    t0 = t
    with torch.no_grad():
      while True:
        t = time.monotonic()

        tt = time.monotonic() - t0

        # Get newest state
        state = None
        while state is None:
          state = cassie.recv_newest_pd()

        if platform.node() == 'cassie':
          """ 
           Control of the physical robot using a wireless handheld controller.
          """

          # Switch the operation mode based on the toggle next to STO
          if state.radio.channel[9] < -0.5: # towards operator means damping shutdown mode
              operation_mode = 2
          elif state.radio.channel[9] > 0.5: # away from the operator means zero hidden states
            operation_mode = 1
          else:                              # Middle means normal walking
            operation_mode = 0

          # Radio control
          orient_add -= state.radio.channel[3] / 60.0

          # Reset orientation on STO
          if state.radio.channel[8] < 0:
              orient_add = quaternion2euler(state.pelvis.orientation[:])[2]
              ESTOP = True
          else:
              ESTOP = False
              logged = False

          raw_spd = (state.radio.channel[0])
          speed = raw_spd * max_speed if raw_spd > 0 else -raw_spd * min_speed

          raw_side_spd = -state.radio.channel[1]
          side_speed = raw_side_spd * max_y_speed if raw_side_spd > 0 else -raw_side_spd * min_y_speed

          phase_add = env.simrate + env.simrate * (state.radio.channel[4] + 0.75)/2
          
          cmd_height = 0.8 + (state.radio.channel[6] + 1)/10

          cmd_foot_height = 0.03 + (state.radio.channel[7] + 1)/15

          pitch_bias      = state.radio.channel[5]/6
          policy_idx      = int(state.radio.channel[10])
          mirror = state.radio.channel[11] > 0
          if policy_idx == -1:
              policy_idx = None

        else:
          """ 
           Control of the robot in simulation using a keyboard.
          """
          tt = time.monotonic() - t0

          if check_stdin():
            c = sys.stdin.read(1)
            if c == 'w':
              speed += 0.1
              phase_add = np.clip(phase_add, (0.6 * speed) * 60, None)
            if c == 's':
              speed -= 0.1
            if c == 'q':
              orient_add -= 0.1
            if c == 'e':
              orient_add += 0.1
            if c == 'a':
              side_speed -= 0.05
            if c == 'd':
              side_speed += 0.05
            if c == 'r':
              speed = 0.5
              orient_add = 0
              side_speed = 0
            if c == 't':
              phase_add += 1
            if c == 'g':
              phase_add -= 1
            if c == 'm':
              mirror = not mirror
            if c == 'y':
              cmd_height += 0.01
            if c == 'h':
              cmd_height -= 0.01
            if c == 'c':
              cmd_foot_height += 0.01
            if c == 'v':
              cmd_foot_height -= 0.01
            if c.isdigit():
              if int(c) > len(policies) - 1:
                policy_idx = None
              else:
                policy_idx = int(c)
            if c == 'x':
              for policy in policies:
                policy.init_hidden_state()
              ESTOP = not ESTOP
              logged = False

          side_speed = max(min_y_speed, side_speed)
          side_speed = min(max_y_speed, side_speed)

        if ESTOP:
            # Save log files after STO toggle (skipping first STO)
            if not logged:
                logged = True
                
                #log(ESTOP_count)
                data = {"time": time_log,
                        "output": output_log,
                        "input": input_log,
                        "state": state_log,
                        "target": target_log}

                
                fname = 'log_' + \
                        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M') + \
                        '_' + str(datetime.timedelta(seconds=round(tt))) + \
                        '.pkl'
                print()
                print(fname)
                print()

                filep = open(fname, 'wb')
                pickle.dump(data, filep)
                filep.close()

                ESTOP_count += 1

                # Clear out logs
                time_log   = []
                input_log  = []
                output_log = []
                state_log  = []
                target_log = []
                t0 = time.monotonic()

            for policy in policies:
              if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

        #------------------------------- Normal Walking ---------------------------
        if operation_mode == 1:
          for policy in policies:
            if hasattr(policy, 'init_hidden_state'):
              policy.init_hidden_state()

        # Quat before bias modification
        quaternion = euler2quat(z=orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)
        new_orient = quaternion_product(iquaternion, state.pelvis.orientation[:])

        clock = [np.sin(2 * np.pi *  phase / env.phase_len), np.cos(2 * np.pi *  phase / env.phase_len)]

        motor_pos = state.motor.position[:]
        joint_pos = state.joint.position[:]

        motor_vel = state.motor.velocity[:]
        joint_vel = state.joint.velocity[:]

        joint_pos = joint_pos[:2] + joint_pos[3:5] # remove double-counted joint/motor positions
        joint_vel = joint_vel[:2] + joint_vel[3:5]

        if new_orient[0] < 0:
          new_orient = [-1 * x for x in new_orient]

        if env.clock:
          ext_state   = np.concatenate((clock, [speed, side_speed, cmd_height, cmd_foot_height]))
        else:
          ext_state   = np.concatenate(([speed], [side_speed], [cmd_height], [cmd_foot_height]))

        pelvis_vel   = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
        pelvis_rvel  = state.pelvis.rotationalVelocity[:]
        pelvis_hgt   = state.pelvis.position[2] - state.terrain.height

        #torque = np.asarray(state.motor.torque[:])
        #if last_torque is None:
        #  torque_penalty = 0
        #else:
        #  torque_penalty = sum(np.abs(last_torque - torque)) / len(torque) / 5
        #  print("torque penalty: {:4.3f}".format(np.exp(-torque_penalty)))
        #last_torque = torque

        robot_state = np.concatenate([
                new_orient,             # pelvis orientation
                motor_pos,
                pelvis_vel,             # pelvis translational velocity
                pelvis_rvel,
                motor_vel,              # actuated joint velocities
                joint_pos,
                joint_vel               # unactuated joint velocities
        ])
          
        if operation_mode == 2 or ESTOP:
          mode = 'DAMP'
        elif operation_mode == 1:
          mode = 'WIPE'
        elif operation_mode == 0:
          mode = 'WALK'

        if mirror:
          mode += ' (M) '

        actual_speed    = 0.9 * actual_speed + 0.1 * pelvis_vel[0]
        RL_state        = np.concatenate([robot_state, ext_state])
        mirror_RL_state = env.mirror_state(RL_state)

        # Construct input vector
        torch_states         = [torch.Tensor(RL_state) for _ in policies]
        mirror_torch_states  = [torch.Tensor(mirror_RL_state) for _ in policies]

        if no_delta:
          offset = env.offset
        else:
          offset = env.get_ref_state(phase=phase)[0][env.pos_idx]

        actions        = [policy(state).numpy() for state, policy in zip(torch_states, policies)]
        mirror_actions = [env.mirror_action(policy(state).numpy()) for state, policy in zip(mirror_torch_states, m_policies)]

        if mirror:
          env_action = [(a + m) / 2 for a, m in zip(actions, mirror_actions)]
        else:
          env_action = actions

        targets = [action[:10] + offset for action in env_action]

        p_gains = [action[10:20] if len(action) > 10 else np.zeros(10) for action in env_action]
        d_gains = [action[20:30] if len(action) > 20 else np.zeros(10) for action in env_action]

        if policy_idx is None:
          target = np.mean(targets, axis=0)
          p_gain = np.mean(p_gains, axis=0)
          d_gain = np.mean(d_gains, axis=0)
        else:
          target = targets[policy_idx]
          p_gain = p_gains[policy_idx]
          d_gain = d_gains[policy_idx]

        if ESTOP or operation_mode == 2:
          for i in range(5):
            u.leftLeg.motorPd.pGain[i] = 0.001
            u.leftLeg.motorPd.dGain[i] = D_mult*env.D[i]
            u.rightLeg.motorPd.pGain[i] = 0.001
            u.rightLeg.motorPd.dGain[i] = D_mult*env.D[i]
            u.leftLeg.motorPd.pTarget[i] = 0.001
            u.rightLeg.motorPd.pTarget[i] = 0.001
        else:
          # Send action
          for i in range(5):
            u.leftLeg.motorPd.pGain[i] = env.P[i] + p_gain[i]
            u.leftLeg.motorPd.dGain[i] = env.D[i] + d_gain[i]
            u.rightLeg.motorPd.pGain[i] = env.P[i] + p_gain[i+5]
            u.rightLeg.motorPd.dGain[i] = env.D[i] + d_gain[i+5]
            u.leftLeg.motorPd.pTarget[i] = target[i]
            u.rightLeg.motorPd.pTarget[i] = target[i+5]

          time_log.append(time.time())
          state_log.append(state)
          input_log.append(RL_state)
          output_log.append(env_action)
          target_log.append(target)
        cassie.send_pd(u)
        
        while time.monotonic() - t < 0.03:
            time.sleep(0.001)
        delay = (time.monotonic() - t) * 1000

        print("MODE {:10s} | IDX {} | Des. Spd. {:5.2f} | Speed {:5.1f} | Sidespeed {:4.1f} | Heading {:5.1f} | Freq. {:3d} | Delay {:6.3f} | Height {:6.4f} | Foot Apex {:6.5f} | {:20s}".format(mode, policy_idx, speed, actual_speed, side_speed, orient_add, int(phase_add), delay, cmd_height, cmd_foot_height, ''), end='\r')

        # Track phase
        phase += phase_add
        if phase >= env.phase_len:
          phase = phase % env.phase_len - 1
          counter += 1
  finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def logvis(filename):
  print('loading log file {}'.format(filename))

  sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
  vis = CassieVis(sim, "./cassie/cassiemujoco/cassie.xml")

  logs        = np.load(open(filename, 'rb'), allow_pickle=True)
  states      = logs['state'] 
  qpos        = np.zeros(35) #set initial values to zero
  inv_pelvis  = inverse_quaternion(states[0].pelvis.orientation[:])

  q_offset    = quaternion_product([1,0,0,0],inv_pelvis)
  curr_foot   = np.zeros(6)
  initial     = states[0].pelvis.position[0:3]
  initial[2] -= states[0].terrain.height

  if initial[2] < 1:
    initial[2] -= initial[2]
  while True:
    for s in states:
        
        qpos[0:3]   = [x1 - x2 for x1, x2 in zip(s.pelvis.position[0:3], initial)]
        qpos[2]     = qpos[2] - s.terrain.height
        qpos[3:7]   = quaternion_product(q_offset, s.pelvis.orientation[:]) # Pelvis Orientation

        #Left side
        qpos[7:10]  = s.motor.position[0:3] #double check this is correct!! (left hip roll, pitch, and yaw)
        theta1      = euler2quat(x=0, z=-s.joint.position[1], y=0)
        qpos[10:14] = theta1
        qpos[14]    = s.motor.position[3] #knee
        qpos[15:17] = s.joint.position[0:2] # shin and tarsus
        qpos[17]    = s.leftFoot.position[0]
        qpos[18]    = s.motor.position[4] + 0.11 #Set left foot crank w offset from foot
        qpos[19]    = -qpos[18] - 0.0184 #Set left plantar rod w offset from foot crank
        qpos[20]    = s.motor.position[4] # check if correct (Motor [4], Joint [2])

        #Right side
        qpos[21:24] = s.motor.position[5:8] #double check (right hip roll, pitch, and yaw)
        theta2      = euler2quat(x=0, z=-s.joint.position[4], y=0)
        qpos[24:28] = theta2
        qpos[28]    = s.motor.position[8] #right knee
        qpos[29:31] = s.joint.position[3:5] # Right shin and tarsus
        qpos[31]    = s.rightFoot.position[0] #######Still Do########
        qpos[32]    = s.motor.position[9] + 0.11 #Set right foot crank w offset from foot
        qpos[33]    = -qpos[32] - 0.0184 #Set right plantar rod w offset from foot crank
        qpos[34]    = s.motor.position[9]

        sim.set_qpos(qpos)
        sim.set_qvel(np.zeros(sim.nv))
        sim.step_pd(pd_in_t())
        sim.foot_pos(curr_foot)
        
        render_state = vis.draw(sim)
        time.sleep(0.0303)

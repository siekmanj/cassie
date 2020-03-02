import time
import torch
import pickle
import platform

import sys
import datetime

import select, termios, tty

bias_idx = 4

#from cassie.cassiemujoco.cassieUDP import *
#from cassie.cassiemujoco.cassiemujoco_ctypes import *
try:
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

	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))

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

def run_udp(args):
  from util.env import env_factory

  policy = torch.load(args.policy)
  #policy.eval()

  env = env_factory(policy.env_name)()
  if not env.state_est:
    print("This policy was not trained with state estimation and cannot be run on the robot.")
    raise RuntimeError

  print("This policy is: {}".format(policy.__class__.__name__))
  time.sleep(1)

  time_log   = [] # time stamp
  input_log  = [] # network inputs
  output_log = [] # network outputs 
  state_log  = [] # cassie state
  target_log = [] #PD target log

  clock_based = env.clock
  no_delta = env.no_delta

  u = pd_in_t()
  for i in range(5):
      u.leftLeg.motorPd.pGain[i] = env.P[i]
      u.leftLeg.motorPd.dGain[i] = env.D[i]
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

  received_data = True
  t = time.monotonic()
  t0 = t

  print('Connected!\n')

  action = 0
  # Whether or not STO has been TOGGLED (i.e. it does not count the initial STO condition)
  # STO = True means that STO is ON (i.e. robot is not running) and STO = False means that STO is
  # OFF (i.e. robot *is* running)
  sto = True
  sto_count = 0

  orient_add = 0

  # We have multiple modes of operation
  # 0: Normal operation, walking with policy
  # 1: Start up, Standing Pose with variable height (no balance)
  # 2: Stop Drop and hopefully not roll, Damping Mode with no P gain
  operation_mode = 0
  standing_height = 0.7
  MAX_HEIGHT = 0.8
  MIN_HEIGHT = 0.4
  D_mult = 1  # Reaaaaaally bad stability problems if this is pushed higher as a multiplier
                   # Might be worth tuning by joint but something else if probably needed
  phase = 0
  counter = 0
  phase_add = env.simrate
  speed = 0
  side_speed = 0

  max_speed = 2.0
  min_speed = -0.3
  max_y_speed = 0.25
  min_y_speed = -0.25

  joint_bias = 0.0
  joint_bias = 0.0

  pitch_bias = 0
  roll_bias = 0

  delay = 30

  old_settings = termios.tcgetattr(sys.stdin)

  try:
    tty.setcbreak(sys.stdin.fileno())

    while True:
      t = time.monotonic()

      tt = time.monotonic() - t0

      # Get newest state
      state = cassie.recv_newest_pd()

      if state is None:
          print('Missed a cycle!                ')
          continue	

      if platform.node() == 'cassie':

        # Radio control
        orient_add -= state.radio.channel[3] / 60.0

        # Reset orientation on STO
        if state.radio.channel[8] < 0:
            orient_add = quaternion2euler(state.pelvis.orientation[:])[2]

            # Save log files after STO toggle (skipping first STO)
            if sto is False:
                #log(sto_count)
                sto_count += 1
                sto = True
                # Clear out logs
                time_log   = [] # time stamp
                input_log  = [] # network inputs
                output_log = [] # network outputs
                state_log  = [] # cassie state
                target_log = [] #PD target log

            if hasattr(policy, 'init_hidden_state'):
              print("RESETTING HIDDEN STATES TO ZERO!")
              policy.init_hidden_state()

        else:
            sto = False

        if state.radio.channel[15] < 0 and hasattr(policy, 'init_hidden_state'):
            print("(TOGGLE SWITCH) RESETTING HIDDEN STATES TO ZERO!")
            policy.init_hidden_state()

        # Switch the operation mode based on the toggle next to STO
        if state.radio.channel[9] < -0.5: # towards operator means damping shutdown mode
            operation_mode = 2
        elif state.radio.channel[9] > 0.5: # away from the operator means zero hidden states
          operation_mode = 1
        else:                               # Middle means normal walking
          operation_mode = 0

        #raw_spd = (state.radio.channel[6] + 1)/2 + state.radio.channel[0]/5
        raw_spd = state.radio.channel[0]
        speed = raw_spd * max_speed if raw_spd > 0 else -raw_spd * min_speed

        raw_side_spd = -state.radio.channel[1]
        side_speed = raw_side_spd * max_y_speed if raw_side_spd > 0 else -raw_side_spd * min_y_speed

        phase_add = env.simrate #+ env.simrate * (state.radio.channel[7] + 0.75)/2

        joint_bias = state.radio.channel[6]/2
        joint_bias = state.radio.channel[7]/2

        pitch_bias = state.radio.channel[5]/6
      else:
        # Automatically change orientation and speed
        tt = time.monotonic() - t0

        if check_stdin():
          c = sys.stdin.read(1)
          if c == 'w':
            speed += 0.1
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
            phase_add += 5
          if c == 'g':
            phase_add -= 5
          if c == 'z':
            joint_bias += 0.01
            joint_bias += 0.01
          if c == 'x':
            joint_bias -= 0.01
            joint_bias -= 0.01
          if c == 'p':
            pitch_bias += 0.01
          if c == 'l':
            pitch_bias -= 0.01


        speed = max(min_speed, speed)
        speed = min(max_speed, speed)

        side_speed = max(min_y_speed, side_speed)
        side_speed = min(max_y_speed, side_speed)

        #env.calc_stepping_freq(speed)

        #phase_add = env.stepping_freq

      #------------------------------- Normal Walking ---------------------------
      if operation_mode == 0 or operation_mode == 1:
          if operation_mode == 1:
            if hasattr(policy, 'init_hidden_state'):
              print("RESETTING HIDDEN STATES TO ZERO!")
              policy.init_hidden_state()

          
          # Reassign because it might have been changed by the damping mode
          for i in range(5):
              u.leftLeg.motorPd.pGain[i] = env.P[i]
              u.leftLeg.motorPd.dGain[i] = env.D[i]
              u.rightLeg.motorPd.pGain[i] = env.P[i]
              u.rightLeg.motorPd.dGain[i] = env.D[i]

          clock = [np.sin(2 * np.pi *  phase / len(env.trajectory)), np.cos(2 * np.pi *  phase / len(env.trajectory))]

          # Quat before bias modification
          quaternion = euler2quat(z=orient_add, y=0, x=0)
          iquaternion = inverse_quaternion(quaternion)
          new_orient = quaternion_product(iquaternion, state.pelvis.orientation[:])
          
          # Adding bias to quat (ROLL PITCH YAW)
          euler_orient = quaternion2euler(new_orient) + [0, pitch_bias, 0]
          new_orient = euler2quat(z=euler_orient[2], y=euler_orient[1], x=euler_orient[0])

          left_foot_pos = state.motor.position[4] #(state.motor.position[4] + state.joint.position[2])/2
          right_foot_pos = state.motor.position[9] #(state.motor.position[9] + state.joint.position[5])/2

          """
          print("{:6s}".format(" "), end="")
          for y in range(6):
            print("{:6d}".format(y), end=", ")
          print()

          for x in range(10):
            print("{:6d}".format(x), end=": ")
            for y in range(6):
              print("{:6.2f}".format(state.motor.position[x] - state.joint.position[y]), end=", ")
            print()
          #print("ERR: {:5.4f}, {:5.4f}".format(state.motor.position[4] - state.joint.position[2], state.motor.position[9] - state.joint.position[5]))
          """

          motor_pos = state.motor.position[:]
          joint_pos = state.joint.position[:]
          joint_pos[2] = left_foot_pos
          motor_pos[4] = left_foot_pos
          joint_pos[5] = right_foot_pos
          motor_pos[9] = right_foot_pos

          joint_diffs = [x - motor_pos[bias_idx] for x in state.joint.position[:]]
          for x in joint_diffs[:3]:
            print("{:3.2f}".format(x), end=", ")
          print()

          motor_pos[bias_idx]   += joint_bias
          motor_pos[bias_idx+5] += joint_bias

          print("\tspeed: {:4.2f} | sidespeed {:4.2f} | orientation {:4.2f} | clock {:4.1f} {:4.1f} | step freq {:3d} | ljoint bias {:6.4f} | rjoint bias {:6.4f} | pitch bias {:4.3f} | delay {:6.3f}".format(speed, side_speed, orient_add, clock[0], clock[1], int(phase_add), joint_bias, joint_bias, pitch_bias, delay))
          if new_orient[0] < 0:
              new_orient[0] = -new_orient
          new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
              
          ext_state = np.concatenate((clock, [speed, side_speed]))
          robot_state = np.concatenate([
                  [state.pelvis.position[2] - state.terrain.height], # pelvis height
                  new_orient,                                     # pelvis orientation
                  #state.motor.position[:],                        # actuated joint positions
                  motor_pos,

                  new_translationalVelocity[:],                   # pelvis translational velocity
                  state.pelvis.rotationalVelocity[:],             # pelvis rotational velocity 
                  state.motor.velocity[:],                        # actuated joint velocities

                  state.pelvis.translationalAcceleration[:],      # pelvis translational acceleration
                  
                  joint_pos,
                  #state.joint.position[:],                        # unactuated joint positions
                  state.joint.velocity[:]                         # unactuated joint velocities
          ])
          RL_state = np.concatenate([robot_state, ext_state])
          
          # Construct input vector
          torch_state = torch.Tensor(RL_state)
          torch_state = policy.normalize_state(torch_state, update=False)

          if no_delta:
            offset = env.offset
          else:
            offset = env.get_ref_state(phase=phase)[0][env.pos_idx]

          action = policy(torch_state)
          env_action = action.data.numpy()
          target = env_action + offset

          target[bias_idx]   -= joint_bias
          target[bias_idx+5] -= joint_bias

          #target[4] -= joint_bias
          #target[9] -= joint_bias

          # Send action
          for i in range(5):
              u.leftLeg.motorPd.pTarget[i] = target[i]
              u.rightLeg.motorPd.pTarget[i] = target[i+5]
          cassie.send_pd(u)

          # Logging
          if sto == False:
              time_log.append(time.time())
              state_log.append(state)
              input_log.append(RL_state)
              output_log.append(env_action)
              target_log.append(target)
          """
          #------------------------------- Start Up Standing ---------------------------
          elif operation_mode == 1:
              print('Startup Standing. Height = ' + str(standing_height))
              #Do nothing
              # Reassign with new multiplier on damping
              for i in range(5):
                  u.leftLeg.motorPd.pGain[i] = 0.0
                  u.leftLeg.motorPd.dGain[i] = 0.0
                  u.rightLeg.motorPd.pGain[i] = 0.0
                  u.rightLeg.motorPd.dGain[i] = 0.0

              # Send action
              for i in range(5):
                  u.leftLeg.motorPd.pTarget[i] = 0.0
                  u.rightLeg.motorPd.pTarget[i] = 0.0
              cassie.send_pd(u)

          """
          #------------------------------- Shutdown Damping ---------------------------
      elif operation_mode == 2:

          print('Shutdown Damping. Multiplier = ' + str(D_mult))
          # Reassign with new multiplier on damping
          for i in range(5):
              u.leftLeg.motorPd.pGain[i] = 0.0
              u.leftLeg.motorPd.dGain[i] = D_mult*env.D[i]
              u.rightLeg.motorPd.pGain[i] = 0.0
              u.rightLeg.motorPd.dGain[i] = D_mult*env.D[i]

          # Send action
          for i in range(5):
              u.leftLeg.motorPd.pTarget[i] = 0.0
              u.rightLeg.motorPd.pTarget[i] = 0.0
          cassie.send_pd(u)

      #---------------------------- Other, should not happen -----------------------
      else:
          print('Error, In bad operation_mode with value: ' + str(operation_mode))
      
      # Measure delay
      # Wait until next cycle time
      while time.monotonic() - t < 0.03:
          time.sleep(0.001)
      delay = (time.monotonic() - t) * 1000
      #print('\tdelay: {:6.1f} ms'.format((time.monotonic() - t) * 1000))

      # Track phase
      phase += phase_add
      if phase >= len(env.trajectory):
        #print("RESET:", phase, phase % len(env.trajectory) - 1)
        phase = phase % len(env.trajectory) - 1
        counter += 1
      #if phase >= 28:
      #    phase = 0
      #    counter += 1
  finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

import modified_gym
import numpy as np

actions = []
observations = []
infos = []
render = 0

def main():
    env = modified_gym.make('Block-v1', pert_type='none')
    numItr = 25
    env.reset()
    env.seed(0)
    print("Reset!")
    
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs,numItr)
    
    for i in [4,6]:
        fileName = "block_demo_"
        fileName += str(numItr)
        if i == 6:    
            fileName += "_augmented"
            for j in range(numItr):
                for k in range(env._max_episode_steps):
                    observations[j][k]['observation'] = np.concatenate([observations[j][k]['observation'], [0.25, 0.25]])
                    actions[j][k] = np.concatenate([actions[j][k], [0.0, 0.0]])
        
        fileName += ".npz"
        np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file for

def goToGoal(env, lastObs,numItr):

    goal = lastObs['desired_goal'][:3]
    goal[2] += 0.004
    object_rel_pos = lastObs['observation'][3:6]
    desired_pos = -1
    commanding_pos = desired_pos - lastObs['observation'][6]
    
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object

    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)
    
    actiondim = 4

    while np.linalg.norm(object_oriented_goal) >= 0.01 and timeStep <= env._max_episode_steps:
        if render: env.render()
        action = []
        for i in range(actiondim): action += [0]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.05

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]*12
        
        action[3] = 1 #open

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        
        object_rel_pos = obsDataNew['observation'][3:6]
        commanding_pos = desired_pos - obsDataNew['observation'][6]
        if render: print(env.env.prev_lforce, env.env.prev_rforce)
        
    
    starttime = timeStep
    while np.linalg.norm(object_rel_pos) >= 0.01 and timeStep <= env._max_episode_steps :
        if render: env.render()
        action = []
        for i in range(actiondim): action += [0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]*6
        action[3] = commanding_pos * (timeStep - starttime)/15.0

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        object_rel_pos = obsDataNew['observation'][3:6]
        commanding_pos = desired_pos - obsDataNew['observation'][6]
        if render: print(env.env.prev_lforce, env.env.prev_rforce)

    while np.linalg.norm(obsDataNew['observation'][8:11]) >= 0.01 and timeStep <= env._max_episode_steps :
        if render: env.render()
        action = []
        for i in range(actiondim): action += [0]
        for i in range(len(obsDataNew['observation'][8:11])):
            action[i] = (obsDataNew['observation'][8:11])[i]*6

        action[3] = commanding_pos

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        object_rel_pos = obsDataNew['observation'][3:6]
        commanding_pos = desired_pos - obsDataNew['observation'][6]
        if render: print(env.env.prev_lforce, env.env.prev_rforce)
        
    while True: #limit the number of timesteps in the episode to a fixed duration
        if render: env.render()
        action = []
        for i in range(actiondim): action += [0]
        action[3] = commanding_pos # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        object_rel_pos = obsDataNew['observation'][3:6]
        commanding_pos = desired_pos - obsDataNew['observation'][6]
        if render: print(env.env.prev_lforce, env.env.prev_rforce)

        if timeStep >= env._max_episode_steps: break
    
    if np.linalg.norm(obsDataNew['observation'][8:11]) < env.env.distance_threshold:
        if goal[2] > 0.004 + env.env.height_offset and len(actions) < numItr/2 + 1:
            actions.append(episodeAcs)
            observations.append(episodeObs)
            infos.append(episodeInfo)
        elif len(actions) > numItr/2:
            actions.append(episodeAcs)
            observations.append(episodeObs)
            infos.append(episodeInfo)
    else: print("Goal was not reached")


if __name__ == "__main__":
    main()

import modified_gym
import numpy as np

actions = []
observations = []
infos = []
render = 0

def main():
    env = modified_gym.make('NuFingers-v1', pert_type='none')
    numItr = 25
    env.reset()
    env.seed(0)
    print("Reset!")
    
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs, numItr)
    
    for i in [2,4]:
        fileName = "nufingers_demo_"
        fileName += str(numItr)
        if i == 4:
            fileName += "_augmented"
            for j in range(numItr):
                for k in range(env._max_episode_steps):
                    observations[j][k]['observation'] = np.concatenate([observations[j][k]['observation'], [1.0, 1.0]])
                    actions[j][k] = np.concatenate([actions[j][k], [0.0, 0.0]])
    
        fileName += ".npz"
        np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file for

def goToGoal(env, lastObs, numItr):
    object_rel_pos = np.array([-0.05, 0.05]) - (lastObs['observation'][:2] + 1.0) / 10.0
    
    desired_pos = -1.0
    commanding_pos = desired_pos
    
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    
    object_oriented_goal = object_rel_pos.copy()

    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)
    
    actiondim = 2

    while env.env.prev_oforce <= 2 and timeStep <= env._max_episode_steps:
        if render: env.render()
        action = []
        for i in range(actiondim): action += [0]
        object_oriented_goal = object_rel_pos.copy()

        action =  np.array([-0.1, 0.])

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        
        commanding_pos = desired_pos
        if render: print(env.env.prev_lforce, env.env.prev_rforce)
        
    while np.linalg.norm(obsDataNew['observation'][6]) >= 0.01 and timeStep <= env._max_episode_steps :
        if render: env.render()
        action = []
        for i in range(actiondim): action += [0]

        action =  np.array([-0.8, 0.7*np.sign(obsDataNew['observation'][6])])
            
        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        object_rel_pos = np.array([commanding_pos, commanding_pos, 0., 0.]) - (obsDataNew['observation'][:4] + 1.0)/10.0
        commanding_pos = desired_pos
        if render: print(env.env.prev_lforce, env.env.prev_rforce)
    
    while True: #limit the number of timesteps in the episode to a fixed duration
        if render: env.render()
        action = []
        for i in range(actiondim): action += [0]
        action =  np.array([-1.0, 0.])

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)
        
        if render: print(env.env.prev_lforce, env.env.prev_rforce)
        if timeStep >= env._max_episode_steps: break
    
    if np.linalg.norm(obsDataNew['observation'][6]) < env.env.distance_threshold:
            actions.append(episodeAcs)
            observations.append(episodeObs)
            infos.append(episodeInfo)
    else: print("Goal was not reached")


if __name__ == "__main__":
    main()


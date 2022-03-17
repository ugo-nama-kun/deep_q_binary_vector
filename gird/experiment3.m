function experiment3()

% taskSpec = RL_init();
% RL_init();
global whichEpisode;
global agent_struct
global results

whichEpisode = 0;

N_experiment = 10;
N_episode = 500;
T = zeros(N_experiment, N_episode);
Steps = zeros(N_experiment, N_episode);
Reward = zeros(N_experiment, N_episode);

for exp = 1:N_experiment
    fprintf(1,'Experiment starting up!\n');
    taskSpec = RL_init();
    fprintf(1,'RL_init called, the environment sent task spec: %s\n',char(taskSpec));
    
    fprintf(1,'\n\n----------Running a few episodes----------\n');
    
    T_ = 0;
    for eps = 1:N_episode
        [R, t] = runEpisode(800);
        T_ = T_ + t;
        
        T(exp, eps) = t;
        Steps(exp, eps) = T_;
        Reward(exp, eps) = R;
        
        figure(2)
        hold on
        plot(1:eps, T(exp, 1:eps), 'r');
        axis([0, N_episode, 0, 500])
        drawnow

    end
    RL_cleanup();
end

results.T = T;
results.Steps = Steps;
results.Reward = Reward;

disconnectGlue();

figure
errorbar(1:20:N_episode,mean(results.T(:,1:20:N_episode)),std(results.T(:,1:20:N_episode)))
axis([0, N_episode, 0, 500])
end

%  Run One Episode of length maximum cutOff
function [totalReward, totalSteps] = runEpisode(stepLimit)
global whichEpisode;
terminal = RL_episode(stepLimit);

totalSteps = RL_num_steps();
totalReward = RL_return();

fprintf(1,'Episode %d\t %d steps \t %f total reward\t natural end %d\n',whichEpisode,totalSteps,totalReward,terminal);

whichEpisode=whichEpisode+1;
end



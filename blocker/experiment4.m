function experiment4()

% taskSpec = RL_init();
% RL_init();
global whichEpisode;
global agent_struct
global results

whichEpisode = 0;

N_experiment = 10;20;
TotalSteps = 1 * 10^5;4 * 10^5;
maxStep = 40;
evalSteps = 1000;%40000;
eval = evalSteps:evalSteps:TotalSteps;
T = eval;
avReward = zeros(N_experiment, length(eval));

% figure(2)
% subplot(2,1,1)
% plot([T(1),T(end)], [-0.8, -0.8],'-.k')

for exp = 1:N_experiment
    fprintf(1,'Experiment starting up!\n');
    taskSpec = RL_init();
    fprintf(1,'RL_init called, the environment sent task spec: %s\n',char(taskSpec));
    
    fprintf(1,'\n\n----------Running episodes----------\n');
    
    cumReward = 0;
    
    time = 1;
    while (time <= TotalSteps)
        RL_start();
        episode_step = 0;
        stepResponse.terminal = 0;
        
        while (stepResponse.terminal ~= 1) && (episode_step < maxStep) && (time <= TotalSteps)
            
            stepResponse = RL_step();
            cumReward = cumReward + stepResponse.r;
            
            if sum(time == eval)~=0
                disp(['Evaluation : ' num2str(cumReward/1000)])
                [~,indx] = max(time == eval);
                avReward(exp,indx) = cumReward/1000;
                cumReward = 0;
                
                figure(2)
                subplot(2,1,1)
                hold on
                plot(T(1:indx), avReward(exp,1:indx), 'r');
                ylim([-1,-0.6])
                xlim([1, TotalSteps])
                drawnow
            end
            episode_step = episode_step + 1;
            time = time + 1;
%              pause(0.5)
        end
        
%         if stepResponse.terminal==1
%             pause(0.5)
%         end
        
%         disp(agent_struct.epsilon)
        RL_cleanup();
    end
    
    
end

results.T = T;
results.avReward = avReward;

disconnectGlue();

figure

errorbar(T(1:10:end), mean(avReward(:,1:10:end)), std(avReward(:,1:10:end)))
% axis([0, N_episode, 0, 800])

end


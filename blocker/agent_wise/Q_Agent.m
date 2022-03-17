
function theAgent=Q_Agent()
theAgent.agent_init=@agent_init;
theAgent.agent_start=@agent_start;
theAgent.agent_step=@agent_step;
theAgent.agent_end=@agent_end;
theAgent.agent_cleanup=@agent_cleanup;
theAgent.agent_message=@agent_message;
end

function agent_init(taskSpec)
global agent_struct;
global EPISODE;

EPISODE = 1;
agent_struct.N_feature = 140;
agent_struct.N_action = 4 * 3;

N_hidden = 100;

agent_struct.Wh = sqrt(6)/sqrt(agent_struct.N_feature+1+N_hidden) * (rand(agent_struct.N_feature + 1, N_hidden) - 0.5)*2; %input

% value weight
agent_struct.Wc = 0.01 * (randn(N_hidden+1, 1) - 0.5)*2; % value function
agent_struct.Wa = 0.01 * (rand(N_hidden+1, agent_struct.N_action) - 0.5)*2; % action
    
agent_struct.alpha = 0.01;
agent_struct.epsilon = 0.1;
agent_struct.gamma = 0.95;
end

function theAction=agent_start(theObservation)
%This is a persistent struct we will use to store things
%that we want to keep around
global agent_struct;

theAction = org.rlcommunity.rlglue.codec.types.Action();
theAction.intArray = agent_policy(theObservation);

%Make copies (using Java methods) of the observation and action
%Store in our persistent struct
agent_struct.lastAction=theAction.duplicate();
agent_struct.lastObservation=theObservation.duplicate();
end

function theAction=agent_step(theReward, theObservation)
%This is a persistent struct we will use to store things
%that we want to keep around
global agent_struct;
global EPISODE;

theAction = org.rlcommunity.rlglue.codec.types.Action();
theAction.intArray = agent_policy(theObservation);

agent_learning(theReward, theObservation, theAction, 0);

% if EPISODE == 1
% figure(3)
% subplot(3,1,2)
% imagesc(theAction.intArray(1:4)')
% colormap gray
% pause(0.1)
% end
% if EPISODE > 290
% figure(3)
% subplot(3,1,2)
% imagesc(theAction.intArray(1:4)')
% colormap gray
% % pause(0.05)
% end

%Make copies (using Java methods) of the observation and action
%Store in our persistent struct
agent_struct.lastAction=theAction.duplicate();
agent_struct.lastObservation=theObservation.duplicate();
end

function agent_end(theReward)
global agent_struct;
global EPISODE


agent_learning(theReward, 0, 0, 1)

agent_struct.epsilon = (10^4/(EPISODE + 10^4)) * agent_struct.epsilon;

% showPolicies

% figure(3)
% subplot(3,1,3)
% imagesc(agent_struct.Wa)
% colormap gray

EPISODE = EPISODE + 1;
end

function returnMessage=agent_message(theMessageJavaObject)
%Java strings are objects, and we want a Matlab string
inMessage=char(theMessageJavaObject);

if strcmp(inMessage,'what is your name?')==1
    returnMessage='my name is acha_agent, Matlab edition!';
else
    returnMessage='I don\''t know how to respond to your message';
end
end

function agent_cleanup()
global agent_struct;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%           Description of the Agent
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function action = agent_policy(theObservation)
global agent_struct;

s = feature(theObservation.intArray);
linout = agent_struct.Wa' * [1;relu(agent_struct.Wh' * s)];
phi = [softmax(linout(1:4),1); softmax(linout(5:8),1); softmax(linout(9:12),1)];


% % discrete full e-greedy
% if rand < agent_struct.epsilon % random
%     action = [randi(4);randi(4);randi(4)];
% else % greedy
%     [~, a1] = max(phi(1:4));
%     [~, a2] = max(phi(5:8));
%     [~, a3] = max(phi(9:12));
%     action = [a1;a2;a3];
% end


% discrete agent_wise e-greedy
% greedy
[~, a1] = max(phi(1:4));
[~, a2] = max(phi(5:8));
[~, a3] = max(phi(9:12));
action = [a1;a2;a3];
for i=1:3
    if rand < agent_struct.epsilon % random
        action(i) = randi(4);
    end
end

% softmax/energy-base
% p=mu;
% rnd = rand;
% for i=1:agent_struct.N_action
%     if rnd < sum(p(1:i))
%         action = i;
%         break;
%     end
% end

if sum(isnan(phi))>0
    disp('NaN ERROR')
    pause
end
end


function agent_learning(theReward, theObservation, theAction, isTerminal)
global agent_struct;
global EPISODE;

s = double(feature(agent_struct.lastObservation.intArray));
% a = double(agent_struct.lastAction.intArray);
% a = double(agent_struct.lastAction.intArray == (1:agent_struct.N_action))';
a = [(1:4 == agent_struct.lastAction.intArray(1)), (1:4 == agent_struct.lastAction.intArray(2)), (1:4 == agent_struct.lastAction.intArray(3))]';

R = theReward;

% disp(a')

hidden = [1; relu(agent_struct.Wh' * s)];
V = agent_struct.Wc' * hidden;
A = a' * (agent_struct.Wa' * hidden);
Q = A+V;  

agent_struct.J = 0;
if isTerminal == 0
%     s_dash = feature(theObservation.doubleArray);
    s_dash = feature(theObservation.intArray);
    hidden_dash = [1; relu(agent_struct.Wh' * s_dash)];

    phi_dash = agent_struct.Wa' * hidden_dash;
    [~, a1] = max(phi_dash(1:4));
    [~, a2] = max(phi_dash(5:8));
    [~, a3] = max(phi_dash(9:12));
    a_dash = [double(a1==1:4)';double(a2==1:4)';double(a3==1:4)'];
    
    V = agent_struct.Wc' * hidden_dash;
    A = a_dash' * (agent_struct.Wa' * hidden_dash);
    Q_dash = A+V;
    
    delta = R + agent_struct.gamma * Q_dash - Q;
else
    delta = R - Q;
end

% Weight Update
error_3_c = delta;
error_3_a = delta * a;
z2 = [0; agent_struct.Wh' * s];
error_2_c = reluGradient(z2) .* (agent_struct.Wc * error_3_c);
error_2_a = reluGradient(z2) .* (agent_struct.Wa * error_3_a);

agent_struct.Wh = agent_struct.Wh + agent_struct.alpha * s * (error_2_c(2:end) + error_2_a(2:end))';
agent_struct.Wc = agent_struct.Wc + agent_struct.alpha * hidden * error_3_c';
agent_struct.Wa = agent_struct.Wa + agent_struct.alpha * hidden * error_3_a';



end



function out = feature(s_in)
global agent_struct
    out = [1; double(s_in)];
%     out = [1; double((1:agent_struct.N_feature) == s_in)'];
end

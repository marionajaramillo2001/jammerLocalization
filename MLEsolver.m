% MLE analysys
% Observation function:
%   Pn = P0 - gamma*log10*distance + wn
% wn:       gaussian noise
% theta:    parmeter to be estmated
% Pn:       obervation
% xn:       observation location

clearvars
close all

% % MSE solver ===============
% syms Pn
% %syms P0 gamma
% syms x y
% syms xn yn
% P0=10;
% gamma = 2;
% xn = 1; yn = 1; Pn = 1;
% func = Pn - P0 + 10*gamma*log10(sqrt((xn-x)^2+(yn-y)^2)) * (x-xn)/(sqrt((xn-x)^2+(yn-y)^2));
% 
% solve(func==0,x)


% MSE analysis

global normalize_obs
global normalize_function
normalize_obs = 1;
normalize_function = 1;

% settings
t0 = 1; % time instant evaluated
est_range1 = [200 +400]; % search space range
est_range2 = [200 +400]; % search space range

step = 30; % step of the search space

% load data
load('X.mat')
load('Y.mat')
load('trueJamLoc.mat');

[N,D,T] = size(XX);
xn_vec = XX(:,:,t0);
Y = YY(:,t0);
trueLoc = Jloc(t0,:);

% known parameters
P0 = 10;
gamma = 2;

% add noise
sigma = 1e-7;%0.01;
Pn_vec = Y + sqrt(sigma)*randn(N,1);

% % normalize
global Max_norm min_norm
Max_norm = max(Pn_vec); min_norm=min(Pn_vec);
if normalize_obs
    Pn_vec = (Pn_vec-min_norm)/(Max_norm-min_norm);
end

% init
theta_x = est_range1(1):step:est_range1(2);
theta_y = est_range2(1):step:est_range2(2);
grad_values_x = zeros(length(theta_x),length(theta_y));
grad_values_y = zeros(length(theta_x),length(theta_y));
llh_values = zeros(length(theta_x),length(theta_y));
llhmax = -inf;
epsilon_x = inf; epsilon_y = inf;

singularity_count = 0; % count the no. of times we test distances too short

for ii = 1:length(theta_x)
    for jj = 1:length(theta_y)
        theta = [theta_x(ii), theta_y(jj)];
        
        grad = grad_pl(theta,Pn_vec,xn_vec,P0,gamma,sigma);
        [llh, C] = llh_pl(theta,Pn_vec,xn_vec,P0,gamma,sigma);
        singularity_count = singularity_count+C;
        % select the maximum log-likelihood
        if llh > llhmax
            llhmax = llh;
            theta_llh = theta;
            idx_llh_x = ii;
            idx_llh_y = jj;
        end
        
        % select the gradient closest to zero
        if abs(grad(1)) < epsilon_x && abs(grad(2)) < epsilon_y 
            epsilon_x = abs(grad(1));
            epsilon_y = abs(grad(2));
            theta_grad = theta;
            idx_grad_x = ii;
            idx_grad_y = jj;
        end
        grad_values_x(ii,jj) = grad(1);
        grad_values_y(ii,jj) = grad(2);
        llh_values(ii,jj) = llh;
        
    end
end

% closest points to true loc
[mm,true_idx_x] = min(abs(theta_x-trueLoc(1)));
[mm,true_idx_y] = min(abs(theta_y-trueLoc(2)));


%% PLOTS
defaultAxesFontsize = 16;
defaultLegendFontsize = 16;
defaultLegendInterpreter = 'latex';
defaultLinelinewidth = 2;
defaultStemlinewidth = 2;
defaultAxesTickLabelInterpreter = 'latex';
defaultTextInterpreter = 'latex';
set(0,'defaultAxesFontsize',defaultAxesFontsize,'defaultLegendFontsize',defaultLegendFontsize,...
    'defaultLegendInterpreter',defaultLegendInterpreter,'defaultLinelinewidth',defaultLinelinewidth,...
    'defaultAxesTickLabelInterpreter',defaultAxesTickLabelInterpreter);
set(0,'defaultTextInterpreter',defaultTextInterpreter);
set(0,'defaultFigurePaperPositionMode','auto')
set(0,'defaultStemlinewidth',defaultStemlinewidth)

%plot3(X(idx),Y(idx),Z(idx),'.r','markersize',10)

figure
mesh(theta_y,theta_x,llh_values), hold on
ylabel('$\theta_1$')
xlabel('$\theta_2$')
zlabel('$\textrm{ln} p(\mathbf{P},\mathbf{\theta})$')
plot3(theta_y(idx_llh_y),theta_x(idx_llh_x),llh_values(idx_llh_x,idx_llh_y),'.r','markersize',20)
plot3(theta_y(true_idx_y),theta_x(true_idx_x),llh_values(true_idx_x,true_idx_y),'.k','markersize',20)

legend 'llh' 'optimal' 'true'
exportgraphics(gcf,'figs/llh.png','Resolution',300)

figure
mesh(theta_y,theta_x,grad_values_x), hold on
ylabel('$\theta_1$')
xlabel('$\theta_2$')
zlabel('$\nabla_1 \textrm{ln} p(\mathbf{P},\mathbf{\theta})$')
plot3(theta_y(idx_grad_y),theta_x(idx_grad_x),grad_values_x(idx_grad_x,idx_grad_y),'.r','markersize',20)
plot3(theta_y(true_idx_y),theta_x(true_idx_x),grad_values_x(true_idx_x,true_idx_y),'.k','markersize',20)
legend '$\nabla$ llh' 'optimal' 'true'
exportgraphics(gcf,'figs/grad1.png','Resolution',300)

figure
mesh(theta_y,theta_x,grad_values_y), hold on
ylabel('$\theta_1$')
xlabel('$\theta_2$')
zlabel('$\nabla_2 \textrm{ln} p(\mathbf{P},\mathbf{\theta})$')
plot3(theta_y(idx_grad_y),theta_x(idx_grad_x),grad_values_y(idx_grad_x,idx_grad_y),'.r','markersize',20)
plot3(theta_y(true_idx_y),theta_x(true_idx_x),grad_values_y(true_idx_x,true_idx_y),'.k','markersize',20)

% figure,
% A = 15;
% mesh(theta_x(true_idx_x-A:true_idx_x+A),theta_y(true_idx_y-A:true_idx_y+A),...
%     grad_values_y(true_idx_x-A:true_idx_x+A,true_idx_y-A:true_idx_y+A))
% % mesh(theta_x(:),theta_y(:),...
% %     grad_values_y(:,:))
% hold on
% plot3(theta_x(true_idx_x),theta_y(true_idx_y),grad_values_y(true_idx_x,true_idx_y),'.k','markersize',20)


legend '$\nabla$ llh' 'optimal' 'true'
exportgraphics(gcf,'figs/grad2.png','Resolution',300)


%% functions

function grad = grad_pl(theta,Pn_vec,xn_vec,P0,gamma,sigma)
% evaluate the function in theta
% Pn_vec is an array of N observations
% xn_vec is a NxD matrix of coorinates where observations are taken
N = length(Pn_vec);
const = -10*gamma/(sigma^2*log(10));
tot_x = 0; tot_y = 0;
for ii = 1:N
    obs_error = Pn_vec(ii) -obs_func(theta,xn_vec(ii,:),P0,gamma);
    d = dist(theta,xn_vec(ii,:));

    tot_x = tot_x+ (obs_error) * (theta(1)-xn_vec(ii,1))/d^2;
    tot_y = tot_y+ (obs_error) * (theta(2)-xn_vec(ii,2))/d^2;
end

grad_x = const*tot_x;
grad_y = const*tot_y;
grad = [grad_x, grad_y];

end

function [llh, counter] = llh_pl(theta,Pn_vec,xn_vec,P0,gamma,sigma)
N = length(Pn_vec);

%const = -N/2*log(2*pi*sigma^2);
const = 0;
tot = 0;
counter = 0;
for ii = 1:N
    if 0 %dist(theta,xn_vec(ii,:))<=pi
        %dif = Pn_vec(ii);
        dif = 0;
        tot = tot+dif^2;
        counter = counter+1;
    else
    tot = tot+ (Pn_vec(ii)-obs_func(theta,xn_vec(ii,:),P0,gamma))^2;
    end
end
llh = const -1/(2*sigma^2)*tot;
end

function obs = obs_func(theta,xn,P0,gamma)
% Note: even if gamma=2, FSPL is not perfectly modeled, because frequency 
% knowledge isnt provided. However, the MLE leads to the same solution

% obs = P0 -gamma*10*log10(dist(theta,xn));

f = 1575.42e6;
gamma = 2;
% obs = P0 -gamma*10*log10(4*pi*f*dist(theta,xn)/physconst('lightspeed'));
obs = P0 -gamma*10*log10(dist(theta,xn));


global normalize_function
if normalize_function
    global Max_norm min_norm
    obs = (obs-min_norm)/(Max_norm-min_norm);
end
end

function d = dist(theta,xn)
% distance
d = sqrt((xn-theta)*(xn-theta).');

end

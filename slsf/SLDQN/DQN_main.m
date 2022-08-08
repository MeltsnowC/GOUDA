clc
rng(0);
CACHDATA_BASE = dqncfg.CACHDATA_DIR;
NETREPORT_BASE = dqncfg.NETREPORT_DIR;
mkdir(CACHDATA_BASE);
mkdir(NETREPORT_BASE);
CACHDATA_FILE = dqncfg.CACHDATA_FILE;
CACHDATAFEATURE_FILE = dqncfg.CACHDATAFEATURE_FILE;
global totalState;
global totalFeature;
totalFeature = [];
totalState = [];
save(CACHDATAFEATURE_FILE,'totalFeature')
save(CACHDATA_FILE,'totalState')
global cfg_param;
cfg_param = [0.14 0.23 0.17 0.12 0.06 0.24 0.04];
actionsize = 3;
gamma = 0.9;
stateLen = dqncfg.stateLen;
trainFlag = 1;
totalNetWeight = {};

%% main
evn = SLDQNENV;
inputNum = stateLen;        %状态个数就是输入维度（宽度）
mainNet = fitnet(dqncfg.stateLen,'traingd');         %包含一个隐藏层，包含22个节点，梯度下降算法求解,返回一个函数拟合网络
mainNet.trainParam.lr = 0.1;        %神经网络步长为0.1（学习率）
mainNet.trainParam.epochs = 1;      %每个步长训练一个 
mainNet.trainParam.showWindow = false;
mainNet.trainParam.lr_dec = 0.9;   %alpha的衰减率decay     
mainNet = train(mainNet,rand(inputNum,dqncfg.NUM_TESTS),rand(actionsize,dqncfg.NUM_TESTS));       %搭建网络,输入为inputNum,输出为actionSize ,列向量
netWeight = getwb(mainNet);     %初始网络的权值
targetNet = mainNet;            %target action value function approximation
net2 = mainNet;
net3 = mainNet;
net4 = mainNet;
net5 = mainNet;
net6 = mainNet;
net7 = mainNet;
tar2 = net2;
tar3 = net3;
tar4 = net4;
tar5 = net5;
tar6 = net6;
tar7 = net7;
netWeight2 = getwb(net2);
netWeight3 = getwb(net3);
netWeight4 = getwb(net4);
netWeight5 = getwb(net5);
netWeight6 = getwb(net6);
netWeight7 = getwb(net7);
expReplayLen = dqncfg.NUM_TESTS*2;
batchSize = [32 1];
%S,A,R,S_,T
expRelay = [];

%% train
epsiodeNumForTrain = 2;     %训练次数（步长）
for ii = 1:epsiodeNumForTrain
    t = 1;
    disp(['current episode:' num2str(ii)]);
    totalpoint = [];
    points = 0;
    [~] = evn.reset();          %reset by random state，随机产生初始状态
    while 1     %step operation in one episode
        [actionIdx_1] = takeAction(evn,mainNet,trainFlag,ii);      %代理根据状态选择一个动作
        [actionIdx_2] = takeAction(evn,net2,trainFlag,ii);
        [actionIdx_3] = takeAction(evn,net3,trainFlag,ii);
        [actionIdx_4] = takeAction(evn,net4,trainFlag,ii);
        [actionIdx_5] = takeAction(evn,net5,trainFlag,ii);
        [actionIdx_6] = takeAction(evn,net6,trainFlag,ii);
        [actionIdx_7] = takeAction(evn,net7,trainFlag,ii);
        assert(actionIdx_1 >= 1 && actionIdx_1 <= 3, 'actionIdx erro!');
        disp('-------------------action------------------')
        action = [actionIdx_1 actionIdx_2 actionIdx_3 actionIdx_4 actionIdx_5 actionIdx_6 actionIdx_7];
        disp(action)
        [state, action, reward, nextState, terminateFlag] = evn.step(action,t);       %在这个函数中，进行一次slforge,并求特征。求出nextState
        
        points = points + reward;   %累计奖励
        totalpoint = [totalpoint,points];
        disp(['----------------points:' num2str(points) '------------------'])
        % record exp buffer
        if size(expRelay,1) < expReplayLen
            %                         
            [prestate,preaction,prereward,preterminateFlag] = predata(state,action,reward,terminateFlag);
            %                       100*23  100*7        100*1      100*23    100*1
            expRelay = [expRelay;[prestate,preaction,prereward,nextState,preterminateFlag]];       %applend to exp replay,直接行向量保存，防止不一致
        else
            expRelay(1:dqncfg.NUM_TESTS,:) = [];     %如果满了，清除前5行存储的
            expRelay = [expRelay;[state,preaction,prereward,nextState,preterminateFlag]];
        end
        
        % update state for evn
        %evn.State = nextState';
        evn.setState(nextState');          %注释上边那个
        if terminateFlag || evn.failedCode        %if (goal achieved or ) failed in one episode terminate it 
            break;
        end
        
        %采样
        %after collect experience replay,start to train net
        if size(expRelay,1) >= expReplayLen
            %batch sample from exp buffer 
            batchIdx = randi(expReplayLen,batchSize);   %采样32
            stateBatch = expRelay(batchIdx, 1:stateLen);
            actionBatch = expRelay(batchIdx, stateLen+1:stateLen+7);
            rewardBatch = expRelay(batchIdx, stateLen+8);
            nextStateBatch = expRelay(batchIdx, stateLen+9:stateLen*2+8);
            terminateFlagBatch = expRelay(batchIdx,2*stateLen+9);
            valueBatch = zeros(actionsize,batchSize(1));        %初始化Q值
            valueBatch2 = zeros(actionsize,batchSize(1));  
            valueBatch3 = zeros(actionsize,batchSize(1));  
            valueBatch4 = zeros(actionsize,batchSize(1));  
            valueBatch5 = zeros(actionsize,batchSize(1));  
            valueBatch6 = zeros(actionsize,batchSize(1));  
            valueBatch7 = zeros(actionsize,batchSize(1));  
            
            %Fix target Q for a while in order to agent to chase the       
            %chaging target
            if ~mod(ii,2)
                targetNet = mainNet;    %synchronize net
                tar2 = net2;
                tar3 = net3;
                tar4 = net4;
                tar5 = net5;
                tar6 = net6;
                tar7 = net7;              
            end
            
            for jj = 1:length(batchSize(1))
                if terminateFlagBatch(jj)
                    valueBatch(actionIdx_1,jj) = rewardBatch(jj);
                    valueBatch2(actionIdx_2,jj) = rewardBatch(jj);
                    valueBatch3(actionIdx_3,jj) = rewardBatch(jj);
                    valueBatch4(actionIdx_4,jj) = rewardBatch(jj);
                    valueBatch5(actionIdx_5,jj) = rewardBatch(jj);
                    valueBatch6(actionIdx_6,jj) = rewardBatch(jj);
                    valueBatch7(actionIdx_7,jj) = rewardBatch(jj);
                else
                    valueBatch(actionIdx_1,jj) = rewardBatch(jj) + gamma*max(targetNet(nextStateBatch(jj,:)'));  
                    valueBatch2(actionIdx_2,jj) = rewardBatch(jj) + gamma*max(tar2(nextStateBatch(jj,:)')); 
                    valueBatch3(actionIdx_3,jj) = rewardBatch(jj) + gamma*max(tar3(nextStateBatch(jj,:)')); 
                    valueBatch4(actionIdx_4,jj) = rewardBatch(jj) + gamma*max(tar4(nextStateBatch(jj,:)')); 
                    valueBatch5(actionIdx_5,jj) = rewardBatch(jj) + gamma*max(tar5(nextStateBatch(jj,:)')); 
                    valueBatch6(actionIdx_6,jj) = rewardBatch(jj) + gamma*max(tar6(nextStateBatch(jj,:)')); 
                    valueBatch7(actionIdx_7,jj) = rewardBatch(jj) + gamma*max(tar7(nextStateBatch(jj,:)')); 
                end
            end
            
            % train net
            setwb(mainNet,netWeight);
            setwb(net2,netWeight2);
            setwb(net3,netWeight3);
            setwb(net4,netWeight4);
            setwb(net5,netWeight5);
            setwb(net6,netWeight6);
            setwb(net7,netWeight7);
            mainNet = train(mainNet,stateBatch',valueBatch);    %训练目标，输入和网络目标
            net2 = train(net2,stateBatch',valueBatch2);
            net3 = train(net3,stateBatch',valueBatch3);
            net4 = train(net4,stateBatch',valueBatch4);
            net5 = train(net5,stateBatch',valueBatch5);
            net6 = train(net6,stateBatch',valueBatch6);
            net7 = train(net7,stateBatch',valueBatch7);
            netWeight = getwb(mainNet);
            netWeight2 = getwb(net2);
            netWeight3 = getwb(net3);
            netWeight4 = getwb(net4);
            netWeight5 = getwb(net5);
            netWeight6 = getwb(net6);
            netWeight7 = getwb(net7);
            
            t = t+1;
            disp('--------------------------t------------------------------')
            t
        end
        %totalNetWeight = {totalNetWeight,netWeight,netWeight2,netWeight3,netWeight4,netWeight5,netWeight6,netWeight7};
        save(fullfile('.','NetReport',[num2str(ii) '_' num2str(t) '_report.mat']),'action','nextState','reward','state','t','totalpoint')
        clear action nextState reward state 
    end
    
    if terminateFlag  
        disp('Agent achieved goal');
        save(fullfile('.','NetReport',[num2str(ii) '_' num2str(t) '_report.mat']),'action','nextState','reward','state','t','totalpoint')
        clear action nextState reward state 
    end
end

%% 验证部分
%{
env = SLDQNENV;
t = 1;
while 1
    %动作
    [action,~] = takeAction(evn,mainNet,0,1);
    %更新环境evn
    [state,action,reward,nextState,terminateFlag] = evn.step(action,t);
    evn.state = nextstate;
    if terminateFlag  
        break;
    end
    t = t+1;
end
    %}

            
            
            
                    
                
            
            
            
            
            
        
            
        
        
        
        
        
        

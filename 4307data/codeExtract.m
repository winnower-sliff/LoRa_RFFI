%% Title: Extract the signal data of the leading code
% Author: Sun Lifeng(1427035192@qq.com)
% Date  : 2025-03-05

%% Introduction
% 从lora报文信号中提取8段连续前导码，并附加上设备label，以下为实现步骤：
%
% # 对每个文件分别进行分析：
% ## 根据信号时域幅度划分阈值进行报文截取
% ## 对每个报文作STFT得到报文时频图
% ## 根据时频图峰值寻找前8段连续前导码索引
% ## 根据索引从原始信号中截取前导码信号段
% # 将每个文件得到的信号集与label合并保存为文件

%% 清除缓存
clc
clear
close all

%% 基本信息设置

% 500次报文采集
% fileName = ["devs6_t500_1M_sf7_rev2_indoor_4k8/DATA_device1_500T_433m_1M_15gain_RECE2",...
%     "devs6_t500_1M_sf7_rev2_indoor_4k8/DATA__device2_502T_433m_1M_15gain_RECE2",...
%     "devs6_t500_1M_sf7_rev2_indoor_4k8/DATA__decice3_501T_433m_1M__15gain_RECE2",...
%     "devs6_t500_1M_sf7_rev2_indoor_4k8/DATA__device4_500T_433m_1M_15gain_RECE2",...
%     "devs6_t500_1M_sf7_rev2_indoor_4k8/DATA_device5_501T_433m_1M_15gain_RECE2",...
%     "devs6_t500_1M_sf7_rev2_indoor_4k8/DATA_device6_502T_433m_1M_15gain_RECE2"];

% 300次报文采集
fileName = [
    % "devs6_t500_1M_sf7_rev2_indoor_4k8/DATA_device1_500T_433m_1M_15gain_RECE2",...
    "2025.3.5\DATA512_device7_433m_1M_3gain",...
    "2025.3.5\DATA500_dev8_433m_500K_70gain",...
    "2025.3.5\DATA500_dev9_433m_500K_70gain",...
    "2025.3.5\DATA500_dev10_433m_500K_70gain",...
    "2025.3.5\DATA500_dev11_433m_500K_70gain",...
    ];

% fileName = "data_20250113/DATA_dev4_200times_433125m_1m_3gain";
saveFileName = 'DATA_dev7-11_500times_433125m_1m_3gain.h5';
datasetName = '/data';
labelsetName = '/label';
complexDatas={};
pktNum=300;
pktLen=8192;
datas=[];
labels_numbers = (7:11)'; % 设备labels设置
labels = repmat(labels_numbers, 1, pktNum)'; % 复制行号以形成矩阵
labels=labels(:);
% 时域信号幅度阈值
threshold = 0.08; % 阈值 n

%% 提取前导码

for fn=1:1:length(fileName)
    disp(['设备号: ',num2str(labels_numbers(fn)),' 读取中...'])
    % 打开文件
    fileID = fopen(fileName(fn), 'rb');
    % 读取数据
    % 每个复数由两个 float32 组成，因此需要读取两倍的数据长度
    data = fread(fileID, Inf, 'float32');
    % 关闭文件
    fclose(fileID);
    
    % 将读取的数据转换为复数形式
    I = data(1:2:end); % 实部
    Q = data(2:2:end); % 虚部
    complexData = I + 1i * Q; % 复数信号
    signal=abs(complexData);
    
    % stiii=450e4;
    % lenn=8e4;
    % figure()
    % subplot(311)
    % plot(I(stiii:stiii+lenn)) %I路
    % subplot(312)
    % plot(Q(stiii:stiii+lenn)) %Q路
    % subplot(313)
    % plot(signal(stiii:stiii+lenn))
    
    % 初始化变量
    startIdx = []; % 开始截取的索引
    endIdx = []; % 结束截取的索引
    inSegment = false; % 标记是否在截取段内
    countBelowThreshold = 0; % 计数器，记录连续低于阈值的样本点数
    endlen=5000;
    
    % 遍历信号
    for i = 1:length(signal)
        if signal(i) >= threshold
            if ~inSegment
                % 开始新的截取段
                startIdx(end+1) = i;
                inSegment = true;
                
            end
            countBelowThreshold = 0; % 重置计数器
        else
            if inSegment
                % 计数连续低于阈值的样本点
                countBelowThreshold = countBelowThreshold + 1;
                if countBelowThreshold == endlen
                    % 结束当前截取段
                    endIdx(end+1) = i+1 - endlen;
                    inSegment = false;
                end
            end
        end
    end
    if length(startIdx)>length(endIdx)
        endIdx(end+1)=length(signal);
    end
    % 输出截取段的索引
    disp('截取段的起始和结束索引:');
    pktRealLen=[];
    data=[];

    %% 对每一个报文信号提取8段连续前导码

    for i = 1:1:pktNum+5
        sti=startIdx(i);
        edi=endIdx(i);
        
        
        % 绘制一个报文的I、Q路信号及其stft图
        % figure()
        % subplot(311)
        % plot(I(sti:edi)) %I路
        % subplot(312)
        % plot(Q(sti:edi)) %Q路
        % signal=abs(complexData);
        % subplot(313)
        % plot(signal(sti:edi)) %Q路
        % figure;
        % stft(complexData(sti:edi));
        
        [s,f,t]=stft(complexData(sti:edi));
        p=abs(s);
        [rowMax, rowMaxIdx] = max(abs(p), [], 1);
        
        % figure()
        % plot(rowMaxIdx)
        
        count=0;
        idx=[];
        for j = 1:length(rowMaxIdx)-1
            if count<9 && abs(rowMaxIdx(j)-rowMaxIdx(j+1))>24
                
                % disp([count,j,t(j)])
                count=count+1;
                idx(end+1)=t(j);
            end
            
        end
        % disp([sti,idx(1),idx(end),idx(end)-idx(1)])
        
        pktRealLen(end+1)=idx(end)-idx(1);
        if pktRealLen(end)==pktLen
            stii=sti+idx(1);
            iqs=[I(stii:stii+pktLen-1);Q(stii:stii+pktLen-1)];
            data=[data;iqs'];
        end
        
        
        % 绘制8个前导码的I、Q路信号及其stft图
        i1=I(stii:sti+idx(end));
        q1=Q(stii:sti+idx(end));
        cd1=i1+1i*q1;
        
        % figure()
        % subplot(311)
        % plot(i1) %I路
        % subplot(312)
        % plot(q1) %Q路
        % subplot(313)
        % plot(abs(cd1)) %Q路
        % figure()
        % stft(cd1);
        
        disp(['报文 ', num2str(i),...
            ': 起始索引 ', num2str(sti),...
            '，结束索引 ', num2str(edi),...
            '，总长度 ', num2str(edi-sti),...
            '，8前导码长度 ', num2str(pktRealLen(end)),...
            ]);
    end

    datas=[datas;data(1:pktNum,:)];
    % ans=char(fileName(1));
    % labels(:,fn)=str2double(ans(28))*ones(1,pktNum);
    
    % 使用 unique 函数获取数组中的唯一元素及其在原数组中的索引
    [uniqueElements, tmp, idx] = unique(pktRealLen);
    % 使用 accumarray 函数统计每个元素的出现次数
    counts = accumarray(idx, 1);
    % 显示结果
    disp('元素:');
    disp(uniqueElements);
    disp('出现次数:');
    disp(counts);
end

%% 数据保存

datas=datas';

% 创建数据集
% 确保数据集的大小与数据的大小匹配
h5create(saveFileName, datasetName, size(datas));
h5create(saveFileName, labelsetName, size(labels));
% 将数据写入 HDF5 文件
h5write(saveFileName, datasetName, datas);
h5write(saveFileName, labelsetName, labels);
disp("finish writing")


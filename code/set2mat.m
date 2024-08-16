clear;clc;
%%
eeglab
%cd afterica; % 获取独立成分分析之后的数据
experimes{1} = 'rest';
experimes{2} = 'conditionA';
experimes{3} = 'conditionB';
experimes{4} = 'conditionC';

condition{1} = 'hc';
condition{2} = 'mcs';
condition{3} = 'uws';

num(1) = 22;
num(2) = 24;
num(3) = 21;
% %% test
% % % filename是一个结构体，包含文件的名称、日期等，提取结构体中的name
% filename = dir(strcat(experimes{1},'\',condition{1},'\*.set')); % 数据保存在.set文件中
% filepath = strcat(experimes{1},'\',condition{1});
% eeg = pop_loadset(filename(1).name, filepath);
%%
% for i = 1:4
for i = 1 % rest data
    for j = 1:3 % 3 kinds of subjects
        count = 0;
        EEG = [];
        filename = dir(strcat('..\data\eegdata_origin\',experimes{i},'\',condition{j},'\*.set')); % 数据保存在.set文件中
        filepath = strcat('..\data\eegdata_origin\',experimes{i},'\',condition{j});
        disp(filepath)
        for k = 1:length(filename) % 读取3*4=12类数据到EEG数组中 % k表示文件名后面的数字
            eeg = pop_loadset(filename(k).name, filepath); % 读取.set和.fdt文件保存数据到eeg变量中
            fprintf('Import People %d  Success!\n', k); % 打印读取成功状态
            datas = eeg.data;
            % Trim the data to be divisible by 2400
            num_full_segments = floor(size(datas, 2) / 2400);
            trimmed_datas = datas(:, 1:(num_full_segments * 2400));
            % Reshape the data
            datas = reshape(trimmed_datas, [59, 2400, num_full_segments]);
            count = count + size(datas,3);
            [~, name, ext]=fileparts(filename(k).name);
            dataname = strcat('..\data\eegdata_mat\',experimes{i},'\',condition{j},'\',name,'.mat');
            save(dataname,'datas','-single');
        end
        disp(count);
    end
    
end

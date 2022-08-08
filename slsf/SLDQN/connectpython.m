clear
clc
pe = pyenv('Version','D:\Anaconda3\python.exe');
if pe.Status == "NotLoaded"
    [~,exepath] = system("where python");
    pe = pyenv('Version','D:\Anaconda3\python.exe');
end 


%pycal = py.getfeature('.\model', '.\data\word_dic.csv', '.\data\word_vector.csv',metric_path='.\data\word_vector.csv');
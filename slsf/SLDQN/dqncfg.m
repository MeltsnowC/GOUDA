classdef dqncfg
    properties(Constant = true)
        stateLen = 8;        %ģ����Ϣ�ĸ���
        NUM_TESTS = 100;      %��Ҫ��cfg�еı�������һ��
        MINSUCC_NUM = 15;   %���ٳɹ���ʮ�� ,����С�ڻ��������ģ������
        CACHDATA_DIR = 'cachdata';   %����totalState��totalfeature
        CACHDATA_BASE = 'cachdata';
        CACHDATA_FILE = ['cachdata' filesep 'totalstate']
        CACHDATAFEATURE_FILE = ['cachdata' filesep 'totalfeature']
        NETREPORT_DIR = 'NetReport'
    end


end
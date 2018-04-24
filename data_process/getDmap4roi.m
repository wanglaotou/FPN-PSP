clear;
clc;
mydir='/ssd/wangmaorui/data';

rectroiDIRS = fullfile(mydir,'RectRoi');
sceneDIRS = fullfile(mydir,'RoiImg');
DmapDIRS = fullfile(mydir,'Dmap');
dmapDIRS = fullfile(DmapDIRS,'Dmap4');     %get Dmap4 info
% dmapDIRS = fullfile(DmapDIRS,'Dmap8');   %get Dmap8 info
rectroipath = fullfile(rectroiDIRS,'rectroiall.txt');
scale = 4;  %dmap4
% scale = 8;  %dmap8
kscale = 1.0; %ksize scale
flid = fopen(rectroipath,'r');
while feof(flid) == 0
	line = fgetl(flid);		
	S = regexp(line,' ','split');
%     disp(length(S));
	imgpath = char(S(1));
    head = char(S(2));
    head = str2num(head);
    rects = [];
    for i = 3:5:length(S)
        pos_x = str2num(char(S(i+1)));
        pos_y = str2num(char(S(i+2)));
        wid = str2num(char(S(i+3)));
        hei = str2num(char(S(i+4)));
        rects = [rects;pos_x];
        rects = [rects;pos_y];
        rects = [rects;wid];
        rects = [rects;hei];
    end
%     disp(length(rects));
%     rect = char(S(2));
	Sl = regexp(imgpath,'/','split');
	scenename = char(Sl(6));
	jpgname = char(Sl(7));
    Sj = regexp(jpgname,'.jpg','split');
    dmapfo = char(Sj(1));
    dmapname = strcat(dmapfo,'.txt');
	dmapp = fullfile(dmapDIRS,scenename);
    if ~exist(dmapp)
        mkdir(dmapp);
    end
	DmapPath = fullfile(dmapp,dmapname);	%get DmapPath
    DmapPath = char(DmapPath);
%     disp(DmapPath);     %/ssd/wangmaorui/data/Dmap/Dmap4/scene21/20170808_frame_02350.txt     
    
    %show img
    img = imread(imgpath);
%     imshow(img);
%     hold on;
    [rwidth,rheight,chan] = size(img);

    %get dmap
%     make width,height divisible by 16
    width = ceil(rwidth/16)*16;
    height = ceil(rheight/16)*16;
    m=width/scale;n=height/scale;
    d_map = zeros(m,n);

    for k=1:4:length(rects)
        rect_x = rects(k);
        rect_y = rects(k+1);
        pvalue = rects(k+2);
%         pwid = rect(k+2);
%         phei = rects(k+3);
        rect_x = rect_x/scale;
        rect_y = rect_y/scale;
        x_ = max(1,floor(rect_x));
        y_ = max(1,floor(rect_y));

        ksize = floor(pvalue/kscale);
        if(mod(ksize,2) == 0)
            ksize = ksize + 1;
        end
        ksize = max(9,ksize);
        sigma = ksize*0.12;
%         ksize =25;
        sigma = 1.5;
        radius = (ksize-1)/2;

        h = fspecial('gaussian',ksize,sigma);

        if (x_-radius+1<1)  % if out of boundary 
            for ra = 0:radius-x_-1
                h(:,end-ra) = h(:,end-ra)+h(:,1);
                h(:,1)=[];
            end
        end
        if (y_-radius+1<1)
            for ra = 0:radius-y_-1
                h(end-ra,:) = h(end-ra,:)+h(1,:);
                h(1,:)=[];
            end
        end
        if (x_+ksize-radius>n)
            for ra = 0:x_+ksize-radius-n-1
                h (:,1+ra) = h(:,1+ra)+h(:,end);
                h(:,end) = [];
            end
        end
        if(y_+ksize-radius>m)
            for ra = 0:y_+ksize-radius-m-1
                h (1+ra,:) = h(1+ra,:)+h(end,:);
                h(end,:) = [];
            end
        end
        d_map(max(y_-radius+1,1):min(y_+ksize-radius,m),max(x_-radius+1,1):min(x_+ksize-radius,n))...
            = d_map(max(y_-radius+1,1):min(y_+ksize-radius,m),max(x_-radius+1,1):min(x_+ksize-radius,n))...
            + h;
    end
    
    fdid=fopen(DmapPath,'w');
    for h=1:m
        for w=1:n
            fprintf(fdid,'%d',d_map(h,w));
            fprintf(fdid,' ');
        end
        fprintf(fdid,'\n');
    end
    fclose(fdid);
    s1=sum(d_map(:));
%     imagesc(d_map);
    close all;

end
fclose(flid);

% List all 64 electrode labels in order
fid = fopen('64-channels.loc', 'r');
locData = textscan(fid, '%d %f %f %s');
fclose(fid);

labels = locData{4};
for i = 1:64
    fprintf('%d: %s\n', i, labels{i});
end
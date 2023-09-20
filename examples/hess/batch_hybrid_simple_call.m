function data = batch_hybrid_simple_call(datapath, toolboxpath, outpath)

if nargin < 1
    datapath = '/home/sg/estss/examples/hess/data.mat';
end
if nargin < 2
    toolboxpath = '/home/sg/work/hybrid/';
end
if nargin < 3
    outpath = 'hess_data.mat';
end
path(path, genpath(toolboxpath));

inputdata = load(datapath);
names = fieldnames(inputdata);
data = struct();

for ii = 1:length(names)
    name = names{ii};
    fprintf('---- %s ----\n', name)
    in_d = inputdata.(name);
    out_d = loop_hybrid_simple_call(in_d);
    data.(name) = out_d;
end

save(outpath, 'data')

end%function


%%
%% LOCAL FUNCTIONS

function out_array = loop_hybrid_simple_call(ts_array)
    n_ts = size(ts_array, 2);
    out_array = zeros(n_ts, 21);
    for ii = 1:n_ts
        if rem(ii, 64) == 0 || ii == 1 || ii == n_ts
            fprintf('Run %i/%i\n', ii, n_ts)
        end
        ts = ts_array(:, ii)';
        out_vec = hybrid_simple_call(ts);
        out_array(ii, :) = out_vec;
    end%for
end%function

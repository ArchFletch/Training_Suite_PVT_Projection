function [pred, meta] = predict_ctle(onnxFile, metaFile, X)
%PREDICT_CTLE Run a surrogate model exported from the training suite in MATLAB.
%
%   [PRED, META] = PREDICT_CTLE(ONNXFILE, METAFILE, X) imports the ONNX network
%   produced by `python -m xfmr_v2.export_onnx ...` and predicts the output
%   curves for the design parameters in X.
%
%   Because normalization is baked into the exported graph, the network maps
%   raw design parameters straight to physical-unit curves -- no pre/post
%   scaling is needed on the MATLAB side.
%
%   Inputs
%     onnxFile : path to the exported .onnx file
%     metaFile : path to the matching <name>.meta.json sidecar
%     X        : nSamples-by-nFeatures matrix of design parameters, given in the
%                feature ORDER and physical UNITS listed in
%                meta.input_feature_names (for CTLE: [CS_fF, LD_pH, M, RD, RS, MN]).
%
%   Outputs
%     pred : nSamples-by-nChannels-by-nFrequencies array, in physical units
%            (e.g. gain in dB, phase in deg).
%     meta : decoded metadata struct (channel names/units, frequency axis, ...).
%
%   Example
%     [pred, meta] = predict_ctle("ctle.onnx", "ctle.meta.json", [180 95 4 1200 800 3]);
%
%   Requires the Deep Learning Toolbox and the "Deep Learning Toolbox Converter
%   for ONNX Model Format" support package.

    meta = jsondecode(fileread(metaFile));

    % Normalize JSON string arrays/cells to MATLAB string arrays for easy indexing.
    featureNames = string(meta.input_feature_names);
    nFeat = numel(featureNames);
    if size(X, 2) ~= nFeat
        error("predict_ctle:badInputWidth", ...
            "X must have %d columns (features, in order: %s).", ...
            nFeat, strjoin(featureNames, ", "));
    end
    X = single(X);
    nSamples = size(X, 1);

    % --- Import the ONNX network (handles both old and new MATLAB releases) ---
    if exist("importNetworkFromONNX", "file") == 2          % R2023b and newer
        net = importNetworkFromONNX(onnxFile);
    elseif exist("importONNXNetwork", "file") == 2          % older releases
        net = importONNXNetwork(onnxFile, ...
            "InputDataFormats", "BC", "OutputDataFormats", "BC");
    else
        error("predict_ctle:noImporter", ...
            ["No ONNX importer found. Install the " ...
             "'Deep Learning Toolbox Converter for ONNX Model Format' support package."]);
    end
    if ~isa(net, "dlnetwork")
        net = dlnetwork(net);          % older importers may return a DAGNetwork
    end

    % --- Predict. Input is feature data: Batch (rows) x Channel (features). ---
    Y = predict(net, dlarray(X, "BC"));
    Y = extractdata(gather(Y));
    if size(Y, 1) ~= nSamples          % guard against a transposed output layout
        Y = Y.';
    end

    % --- Reshape the flat output back to (sample, channel, frequency). ---
    % The graph emits each sample as channel-major: [ch1_f1..ch1_fF, ch2_f1..].
    C = double(meta.num_channels);
    F = double(meta.num_frequencies);
    pred = zeros(nSamples, C, F);
    for i = 1:nSamples
        pred(i, :, :) = reshape(Y(i, :), [F, C]).';
    end
end

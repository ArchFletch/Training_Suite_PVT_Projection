% DEMO_PREDICT_CTLE  Load an exported CTLE surrogate, predict, and plot.
%
% Edit the two paths and the design point X below, then run this script.
% See predict_ctle.m for details.

onnxFile = "ctle.onnx";            % <-- path to the exported .onnx
metaFile = "ctle.meta.json";       % <-- path to the matching .meta.json sidecar

% --- Inspect what the model expects/produces --------------------------------
meta = jsondecode(fileread(metaFile));
fprintf("Model type : %s\n", meta.model_type);
fprintf("Inputs     : %s\n", strjoin(string(meta.input_feature_names), ", "));
fprintf("Outputs    : %s\n", strjoin(string(meta.channel_names), ", "));

% --- One design point -------------------------------------------------------
% Values MUST be in the same order and physical units as meta.input_feature_names.
% For CTLE that is [CS_fF, LD_pH, M, RD, RS, MN].  EDIT THESE:
X = [180, 95, 4, 1200, 800, 3];

pred = predict_ctle(onnxFile, metaFile, X);   % 1 x nChannels x nFrequencies

% --- Plot each channel against the frequency axis ---------------------------
fGHz   = meta.frequency_hz / 1e9;
names  = string(meta.channel_names);
units  = string(meta.channel_units);

figure;
tiledlayout(meta.num_channels, 1);
for c = 1:meta.num_channels
    nexttile;
    plot(fGHz, squeeze(pred(1, c, :)), "LineWidth", 1.5);
    grid on;
    xlabel(meta.sweep_label);
    if c <= numel(units) && strlength(units(c)) > 0
        ylabel(sprintf("%s (%s)", names(c), units(c)));
    else
        ylabel(names(c));
    end
    title(names(c));
end
sgtitle("CTLE surrogate prediction");

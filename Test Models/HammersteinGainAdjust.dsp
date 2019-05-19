fi = library("filters.lib");
an = library("analyzers.lib");

gainAdjust(x, n) = x <: *(_, an.amp_follower(rel) : pow(_, 1 - n) : min(_, lim))
with{
    rel = hslider("Release [unit:ms][style:knob]", 5, 1, 1000, 0.1) * 0.001;
    lim = 10.0;
};

process = _ : gainAdjust(_, 4);

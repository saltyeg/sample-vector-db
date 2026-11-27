#pragma once

#include "SampleDatabase.h"
#include "core/Types.h"

#include <functional>
#include <vector>

namespace svdb::db
{

// Pure C++ 2D projection: PCA initialization + parallel kNN cosine graph + UMAP SGD.
// Outputs LayoutUpdate entries normalized into [-1, 1]^2.
class Layout2D
{
public:
    static std::vector<LayoutUpdate> compute (
        const std::vector<SampleRecord>& records,
        float clapWeight = 0.65f,
        int kNeighbors = 15,
        int epochs = 140,
        std::function<void (int done, int total)> onProgress = nullptr);

    // Computes 3 Procrustes-aligned UMAP keyframes:
    //   0.0 (pure DSP), 0.5 (Hybrid), 1.0 (pure CLAP)
    // for instant 60fps morphing when dragging the DSP/AI slider.
    static std::vector<LayoutUpdate> computeMultiBlend (
        const std::vector<SampleRecord>& records,
        int kNeighbors = 15,
        int epochs = 130,
        std::function<void (int done, int total)> onProgress = nullptr);
};

} // namespace svdb::db

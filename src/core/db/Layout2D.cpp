#include "Layout2D.h"
#include "VectorIndex.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <thread>

namespace svdb::db
{
namespace
{
struct Edge
{
    int u = 0;
    int v = 0;
    float w = 1.0f;
};

// Fast xorshift32 RNG for SGD negative sampling.
struct FastRng
{
    uint32_t state = 0x9E3779B9u;
    explicit FastRng (uint32_t seed) : state (seed ? seed : 0x9E3779B9u) {}
    uint32_t next()
    {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
    int nextInt (int bound) { return static_cast<int> (next() % static_cast<uint32_t> (bound)); }
    float nextFloat() { return static_cast<float> (next() & 0x00FFFFFFu) / 16777216.0f; }
};

void normalizeCoords (std::vector<float>& xy)
{
    const size_t n = xy.size() / 2;
    if (n == 0) return;

    std::vector<float> xs (n), ys (n);
    for (size_t i = 0; i < n; ++i)
    {
        if (! std::isfinite (xy[i * 2]))     xy[i * 2] = 0.0f;
        if (! std::isfinite (xy[i * 2 + 1])) xy[i * 2 + 1] = 0.0f;
        xs[i] = xy[i * 2];
        ys[i] = xy[i * 2 + 1];
    }

    // Use true min/max bounds, only excluding extreme isolated outliers (> 4.5 * IQR past Q1/Q3)
    // so genuine clusters are NEVER crushed into a flat boundary wall.
    std::sort (xs.begin(), xs.end());
    std::sort (ys.begin(), ys.end());

    auto robustBounds = [n] (const std::vector<float>& sorted, float& loOut, float& hiOut) {
        if (n < 16)
        {
            loOut = sorted.front();
            hiOut = sorted.back();
            return;
        }
        const float q1 = sorted[n / 4];
        const float q3 = sorted[(3 * n) / 4];
        const float iqr = std::max (q3 - q1, 1e-4f);
        const float fenceLo = q1 - 4.5f * iqr;
        const float fenceHi = q3 + 4.5f * iqr;
        loOut = std::max (sorted.front(), fenceLo);
        hiOut = std::min (sorted.back(),  fenceHi);
    };

    float minX = 0.0f, maxX = 0.0f, minY = 0.0f, maxY = 0.0f;
    robustBounds (xs, minX, maxX);
    robustBounds (ys, minY, maxY);

    const float cx = 0.5f * (minX + maxX);
    const float cy = 0.5f * (minY + maxY);
    const float halfSpan = std::max (0.5f * std::max (maxX - minX, maxY - minY), 1e-4f);
    const float inv = 0.94f / halfSpan;

    for (size_t i = 0; i < n; ++i)
    {
        xy[i * 2]     = std::clamp ((xy[i * 2]     - cx) * inv, -0.99f, 0.99f);
        xy[i * 2 + 1] = std::clamp ((xy[i * 2 + 1] - cy) * inv, -0.99f, 0.99f);
    }
}

// Gentle micro anti-stacking pass: separates only pixel-on-pixel duplicates
// while keeping every point tightly anchored to its true UMAP neighborhood.
void relaxOverlaps (std::vector<float>& xy, int iterations = 10)
{
    const int n = static_cast<int> (xy.size() / 2);
    if (n < 16) return;

    const float minDist = std::clamp (0.58f / std::sqrt (static_cast<float> (n)), 0.0055f, 0.025f);
    const float minDistSq = minDist * minDist;
    const int gridRes = std::clamp (static_cast<int> (std::ceil (2.0f / minDist)), 32, 200);
    const float invCell = static_cast<float> (gridRes) / 2.2f;

    const std::vector<float> anchor = xy;
    std::vector<int> cellHead (static_cast<size_t> (gridRes * gridRes), -1);
    std::vector<int> nextNode (static_cast<size_t> (n), -1);
    std::vector<float> disp (static_cast<size_t> (n) * 2, 0.0f);

    for (int it = 0; it < iterations; ++it)
    {
        std::fill (cellHead.begin(), cellHead.end(), -1);
        std::fill (disp.begin(), disp.end(), 0.0f);

        for (int i = 0; i < n; ++i)
        {
            const int gx = std::clamp (static_cast<int> ((xy[static_cast<size_t> (i) * 2] + 1.1f) * invCell), 0, gridRes - 1);
            const int gy = std::clamp (static_cast<int> ((xy[static_cast<size_t> (i) * 2 + 1] + 1.1f) * invCell), 0, gridRes - 1);
            const int cell = gy * gridRes + gx;
            nextNode[static_cast<size_t> (i)] = cellHead[static_cast<size_t> (cell)];
            cellHead[static_cast<size_t> (cell)] = i;
        }

        const float stepScale = 0.30f * (1.0f - 0.50f * static_cast<float> (it) / static_cast<float> (iterations));

        for (int i = 0; i < n; ++i)
        {
            const float ix = xy[static_cast<size_t> (i) * 2];
            const float iy = xy[static_cast<size_t> (i) * 2 + 1];
            const int gx = std::clamp (static_cast<int> ((ix + 1.1f) * invCell), 0, gridRes - 1);
            const int gy = std::clamp (static_cast<int> ((iy + 1.1f) * invCell), 0, gridRes - 1);

            for (int ny = std::max (0, gy - 1); ny <= std::min (gridRes - 1, gy + 1); ++ny)
            {
                for (int nx = std::max (0, gx - 1); nx <= std::min (gridRes - 1, gx + 1); ++nx)
                {
                    for (int j = cellHead[static_cast<size_t> (ny * gridRes + nx)]; j != -1; j = nextNode[static_cast<size_t> (j)])
                    {
                        if (j <= i) continue;
                        float dx = ix - xy[static_cast<size_t> (j) * 2];
                        float dy = iy - xy[static_cast<size_t> (j) * 2 + 1];
                        float d2 = dx * dx + dy * dy;
                        if (d2 < minDistSq)
                        {
                            if (d2 < 1e-8f)
                            {
                                // Deterministic tiny break-tie angle for exact duplicates.
                                const float ang = static_cast<float> ((i * 73 + j * 19) % 360) * 0.0174533f;
                                dx = std::cos (ang) * 1e-4f;
                                dy = std::sin (ang) * 1e-4f;
                                d2 = 1e-8f;
                            }
                            const float d = std::sqrt (d2);
                            const float push = (minDist - d) * 0.5f * stepScale / d;
                            const float px = dx * push;
                            const float py = dy * push;
                            disp[static_cast<size_t> (i) * 2]     += px;
                            disp[static_cast<size_t> (i) * 2 + 1] += py;
                            disp[static_cast<size_t> (j) * 2]     -= px;
                            disp[static_cast<size_t> (j) * 2 + 1] -= py;
                        }
                    }
                }
            }
        }

        // Strong anchor spring preserves true UMAP proximity while eliminating exact overlaps.
        for (int i = 0; i < n; ++i)
        {
            const size_t k = static_cast<size_t> (i) * 2;
            const float dx = std::clamp (disp[k],     -minDist, minDist);
            const float dy = std::clamp (disp[k + 1], -minDist, minDist);
            xy[k]     += dx + 0.14f * (anchor[k]     - xy[k]);
            xy[k + 1] += dy + 0.14f * (anchor[k + 1] - xy[k + 1]);
        }
    }

    normalizeCoords (xy);
}
} // namespace

std::vector<LayoutUpdate> Layout2D::compute (
    const std::vector<SampleRecord>& records,
    float clapWeight,
    int kNeighbors,
    int epochs,
    std::function<void (int, int)> onProgress)
{
    const int n = static_cast<int> (records.size());
    std::vector<LayoutUpdate> out (static_cast<size_t> (n));
    if (n == 0)
        return out;

    if (n == 1)
    {
        out[0] = { records[0].meta.id, 0.0f, 0.0f };
        return out;
    }

    // 1. Build fused unit vectors: if majority have CLAP, use [sqrt(1-w)*DSP, sqrt(w)*CLAP] (576D),
    //    else pure DSP (64D).
    int clapCount = 0;
    for (const auto& r : records)
        if (r.meta.hasClap) ++clapCount;
    const bool useClap = (clapCount > n / 2) && (clapWeight > 0.01f);
    const int dim = useClap ? (kDspDim + kClapDim) : kDspDim;
    const float wClap = useClap ? std::sqrt (std::clamp (clapWeight, 0.0f, 1.0f)) : 0.0f;
    const float wDsp  = useClap ? std::sqrt (std::clamp (1.0f - clapWeight, 0.0f, 1.0f)) : 1.0f;

    std::vector<float> vecs (static_cast<size_t> (n) * static_cast<size_t> (dim), 0.0f);
    for (int i = 0; i < n; ++i)
    {
        float* row = vecs.data() + static_cast<size_t> (i) * static_cast<size_t> (dim);
        for (int d = 0; d < kDspDim; ++d)
            row[d] = wDsp * records[static_cast<size_t> (i)].dspVec[static_cast<size_t> (d)];
        if (useClap && records[static_cast<size_t> (i)].meta.hasClap)
        {
            for (int d = 0; d < kClapDim; ++d)
                row[kDspDim + d] = wClap * records[static_cast<size_t> (i)].clapVec[static_cast<size_t> (d)];
        }
        // L2-normalize row.
        const float norm = std::sqrt (VectorIndex::dotProduct (row, row, dim)) + 1e-8f;
        const float inv = 1.0f / norm;
        for (int d = 0; d < dim; ++d)
            row[d] *= inv;
    }

    // 2. PCA initialization (top 2 principal components via power iteration).
    std::vector<float> mean (static_cast<size_t> (dim), 0.0f);
    for (int i = 0; i < n; ++i)
    {
        const float* row = vecs.data() + static_cast<size_t> (i) * static_cast<size_t> (dim);
        for (int d = 0; d < dim; ++d)
            mean[static_cast<size_t> (d)] += row[d];
    }
    for (float& m : mean) m /= static_cast<float> (n);

    std::vector<float> pc1 (static_cast<size_t> (dim), 0.0f);
    std::vector<float> pc2 (static_cast<size_t> (dim), 0.0f);
    FastRng rng (42);
    for (int d = 0; d < dim; ++d)
    {
        pc1[static_cast<size_t> (d)] = rng.nextFloat() - 0.5f;
        pc2[static_cast<size_t> (d)] = rng.nextFloat() - 0.5f;
    }

    auto powerIter = [&] (std::vector<float>& pc, const std::vector<float>* ortho) {
        std::vector<float> next (static_cast<size_t> (dim), 0.0f);
        for (int iter = 0; iter < 14; ++iter)
        {
            std::fill (next.begin(), next.end(), 0.0f);
            for (int i = 0; i < n; ++i)
            {
                const float* row = vecs.data() + static_cast<size_t> (i) * static_cast<size_t> (dim);
                float proj = 0.0f;
                for (int d = 0; d < dim; ++d)
                    proj += (row[d] - mean[static_cast<size_t> (d)]) * pc[static_cast<size_t> (d)];
                for (int d = 0; d < dim; ++d)
                    next[static_cast<size_t> (d)] += proj * (row[d] - mean[static_cast<size_t> (d)]);
            }
            if (ortho)
            {
                const float dot = VectorIndex::dotProduct (next.data(), ortho->data(), dim);
                for (int d = 0; d < dim; ++d)
                    next[static_cast<size_t> (d)] -= dot * (*ortho)[static_cast<size_t> (d)];
            }
            const float norm = std::sqrt (VectorIndex::dotProduct (next.data(), next.data(), dim)) + 1e-8f;
            for (int d = 0; d < dim; ++d)
                pc[static_cast<size_t> (d)] = next[static_cast<size_t> (d)] / norm;
        }
    };
    powerIter (pc1, nullptr);
    powerIter (pc2, &pc1);

    std::vector<float> xy (static_cast<size_t> (n) * 2, 0.0f);
    for (int i = 0; i < n; ++i)
    {
        const float* row = vecs.data() + static_cast<size_t> (i) * static_cast<size_t> (dim);
        float x = 0.0f, y = 0.0f;
        for (int d = 0; d < dim; ++d)
        {
            const float c = row[d] - mean[static_cast<size_t> (d)];
            x += c * pc1[static_cast<size_t> (d)];
            y += c * pc2[static_cast<size_t> (d)];
        }
        xy[static_cast<size_t> (i) * 2]     = x;
        xy[static_cast<size_t> (i) * 2 + 1] = y;
    }
    normalizeCoords (xy);
    // Scale initial embedding gently to [-4, 4] for UMAP dynamics.
    for (float& v : xy)
        v *= 4.0f;

    // For tiny libraries (< 12 samples), PCA is already clean and stable.
    if (n < 12)
    {
        normalizeCoords (xy);
        for (int i = 0; i < n; ++i)
        {
            const float x = xy[static_cast<size_t> (i) * 2];
            const float y = xy[static_cast<size_t> (i) * 2 + 1];
            out[static_cast<size_t> (i)] = { records[static_cast<size_t> (i)].meta.id, x, y, x, y, x, y };
        }
        return out;
    }

    // 3. Parallel exact kNN graph (cosine distance = 1 - dot).
    const int k = std::min (kNeighbors, n - 1);
    const int numThreads = std::max (1, static_cast<int> (std::thread::hardware_concurrency()));
    std::vector<std::vector<Edge>> threadEdges (static_cast<size_t> (numThreads));

    auto knnWorker = [&] (int tid) {
        auto& localEdges = threadEdges[static_cast<size_t> (tid)];
        std::vector<std::pair<float, int>> topK;
        topK.reserve (static_cast<size_t> (k + 1));

        for (int i = tid; i < n; i += numThreads)
        {
            const float* vi = vecs.data() + static_cast<size_t> (i) * static_cast<size_t> (dim);
            topK.clear();

            for (int j = 0; j < n; ++j)
            {
                if (i == j) continue;
                const float sim = VectorIndex::dotProduct (vi, vecs.data() + static_cast<size_t> (j) * static_cast<size_t> (dim), dim);
                if (static_cast<int> (topK.size()) < k)
                {
                    topK.emplace_back (sim, j);
                    if (static_cast<int> (topK.size()) == k)
                        std::make_heap (topK.begin(), topK.end(), [] (auto& a, auto& b) { return a.first > b.first; });
                }
                else if (sim > topK.front().first)
                {
                    std::pop_heap (topK.begin(), topK.end(), [] (auto& a, auto& b) { return a.first > b.first; });
                    topK.back() = { sim, j };
                    std::push_heap (topK.begin(), topK.end(), [] (auto& a, auto& b) { return a.first > b.first; });
                }
            }

            std::sort (topK.begin(), topK.end(), [] (auto& a, auto& b) { return a.first > b.first; });
            const float rho = std::max (0.0f, 1.0f - topK[0].first);
            // Smooth local scale sigma via quick binary search targeting log2(k).
            const float target = std::log2 (static_cast<float> (k));
            float lo = 1e-4f, hi = 2.0f, sigma = 0.25f;
            for (int it = 0; it < 16; ++it)
            {
                sigma = 0.5f * (lo + hi);
                float sum = 0.0f;
                for (const auto& nb : topK)
                {
                    const float dist = std::max (0.0f, 1.0f - nb.first);
                    const float d = std::max (0.0f, dist - rho);
                    sum += std::exp (-d / sigma);
                }
                if (sum > target) hi = sigma;
                else              lo = sigma;
            }

            for (const auto& nb : topK)
            {
                const float dist = std::max (0.0f, 1.0f - nb.first);
                const float w = std::exp (-std::max (0.0f, dist - rho) / std::max (sigma, 1e-4f));
                if (w > 0.05f)
                    localEdges.push_back ({ i, nb.second, w });
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve (static_cast<size_t> (numThreads));
    for (int t = 0; t < numThreads; ++t)
        workers.emplace_back (knnWorker, t);
    for (auto& w : workers)
        w.join();

    std::vector<Edge> edges;
    size_t totalEdges = 0;
    for (const auto& te : threadEdges) totalEdges += te.size();
    edges.reserve (totalEdges);
    for (auto& te : threadEdges)
    {
        edges.insert (edges.end(), te.begin(), te.end());
        te.clear();
    }

    // 4. Tight, proximity-faithful UMAP SGD layout (min_dist ~ 0.15 -> a=1.35, b=0.88).
    constexpr float kA = 1.35f;
    constexpr float kB = 0.88f;
    constexpr int kNegSamples = 5;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        const float lr = 1.0f * (1.0f - static_cast<float> (epoch) / static_cast<float> (epochs));

        for (const auto& e : edges)
        {
            // Sample edge probabilistically according to its fuzzy weight.
            if (e.w < 0.99f && rng.nextFloat() > e.w)
                continue;

            float& ux = xy[static_cast<size_t> (e.u) * 2];
            float& uy = xy[static_cast<size_t> (e.u) * 2 + 1];
            float& vx = xy[static_cast<size_t> (e.v) * 2];
            float& vy = xy[static_cast<size_t> (e.v) * 2 + 1];

            const float dx = ux - vx;
            const float dy = uy - vy;
            const float distSq = dx * dx + dy * dy + 1e-4f;

            // Attractive force.
            const float distPow = std::pow (distSq, kB - 1.0f);
            const float gradCoeff = (-2.0f * kA * kB * distPow) / (1.0f + kA * distPow * distSq);
            const float gx = std::clamp (gradCoeff * dx, -4.0f, 4.0f) * lr;
            const float gy = std::clamp (gradCoeff * dy, -4.0f, 4.0f) * lr;
            ux += gx;
            uy += gy;
            vx -= gx;
            vy -= gy;

            // Repulsive negative samples.
            for (int ns = 0; ns < kNegSamples; ++ns)
            {
                const int neg = rng.nextInt (n);
                if (neg == e.u) continue;
                const float nx = xy[static_cast<size_t> (neg) * 2];
                const float ny = xy[static_cast<size_t> (neg) * 2 + 1];
                const float ndx = ux - nx;
                const float ndy = uy - ny;
                const float nDistSq = ndx * ndx + ndy * ndy + 1e-3f;
                const float repCoeff = (2.0f * kB) / ((0.001f + nDistSq) * (1.0f + kA * std::pow (nDistSq, kB)));
                ux += std::clamp (repCoeff * ndx, -4.0f, 4.0f) * lr;
                uy += std::clamp (repCoeff * ndy, -4.0f, 4.0f) * lr;
            }
        }

        if (onProgress && (epoch % 10 == 0 || epoch == epochs - 1))
            onProgress (epoch + 1, epochs);
    }

    normalizeCoords (xy);
    relaxOverlaps (xy, 10);
    for (int i = 0; i < n; ++i)
    {
        out[static_cast<size_t> (i)] = {
            records[static_cast<size_t> (i)].meta.id,
            xy[static_cast<size_t> (i) * 2],
            xy[static_cast<size_t> (i) * 2 + 1],
            xy[static_cast<size_t> (i) * 2],
            xy[static_cast<size_t> (i) * 2 + 1],
            xy[static_cast<size_t> (i) * 2],
            xy[static_cast<size_t> (i) * 2 + 1]
        };
    }
    return out;
}

namespace
{
// Exact 2D orthogonal Procrustes alignment (rotation + optional reflection)
// minimizing sum_i ||R * src_i - ref_i||^2, keeping coordinates in [-1, 1]^2.
void alignProcrustes2D (std::vector<LayoutUpdate>& src,
                        const std::vector<LayoutUpdate>& ref,
                        bool writeToDsp)
{
    const size_t n = std::min (src.size(), ref.size());
    if (n == 0) return;

    auto evalFit = [&] (bool flipX, float& outCos, float& outSin) -> double {
        double a = 0.0, b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double sx = flipX ? -static_cast<double> (src[i].x) : static_cast<double> (src[i].x);
            const double sy = static_cast<double> (src[i].y);
            const double rx = static_cast<double> (ref[i].x);
            const double ry = static_cast<double> (ref[i].y);
            a += sx * rx + sy * ry;
            b += sx * ry - sy * rx;
        }
        const double theta = std::atan2 (b, a);
        const double c = std::cos (theta);
        const double s = std::sin (theta);
        outCos = static_cast<float> (c);
        outSin = static_cast<float> (s);

        double err = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double sx = flipX ? -static_cast<double> (src[i].x) : static_cast<double> (src[i].x);
            const double sy = static_cast<double> (src[i].y);
            const double tx = c * sx - s * sy;
            const double ty = s * sx + c * sy;
            const double dx = tx - ref[i].x;
            const double dy = ty - ref[i].y;
            err += dx * dx + dy * dy;
        }
        return err;
    };

    float c0 = 1.0f, s0 = 0.0f, c1 = 1.0f, s1 = 0.0f;
    const double errRot  = evalFit (false, c0, s0);
    const double errFlip = evalFit (true,  c1, s1);
    const bool useFlip = (errFlip < errRot);
    const float c = useFlip ? c1 : c0;
    const float s = useFlip ? s1 : s0;

    std::vector<float> xy (n * 2);
    for (size_t i = 0; i < n; ++i)
    {
        const float sx = useFlip ? -src[i].x : src[i].x;
        const float sy = src[i].y;
        xy[i * 2]     = c * sx - s * sy;
        xy[i * 2 + 1] = s * sx + c * sy;
    }
    normalizeCoords (xy);

    for (size_t i = 0; i < n; ++i)
    {
        if (writeToDsp)
        {
            src[i].dspX = xy[i * 2];
            src[i].dspY = xy[i * 2 + 1];
        }
        else
        {
            src[i].clapX = xy[i * 2];
            src[i].clapY = xy[i * 2 + 1];
        }
    }
}
} // namespace

std::vector<LayoutUpdate> Layout2D::computeMultiBlend (
    const std::vector<SampleRecord>& records,
    int kNeighbors,
    int epochs,
    std::function<void (int, int)> onProgress)
{
    const int n = static_cast<int> (records.size());
    if (n == 0)
        return {};

    int clapCount = 0;
    for (const auto& r : records)
        if (r.meta.hasClap) ++clapCount;
    const bool hasClap = (clapCount > n / 2);

    if (! hasClap)
    {
        return compute (records, 0.0f, kNeighbors, epochs, onProgress);
    }

    const int totalSteps = epochs * 3;
    // 1. Hybrid (0.5) reference anchor
    auto hybrid = compute (records, 0.50f, kNeighbors, epochs, [&] (int d, int) {
        if (onProgress) onProgress (d, totalSteps);
    });

    // 2. Pure DSP (0.0) aligned to Hybrid
    auto dsp = compute (records, 0.0f, kNeighbors, epochs, [&] (int d, int) {
        if (onProgress) onProgress (epochs + d, totalSteps);
    });
    alignProcrustes2D (dsp, hybrid, true);

    // 3. Pure CLAP (1.0) aligned to Hybrid
    auto clap = compute (records, 1.0f, kNeighbors, epochs, [&] (int d, int) {
        if (onProgress) onProgress (2 * epochs + d, totalSteps);
    });
    alignProcrustes2D (clap, hybrid, false);

    for (size_t i = 0; i < hybrid.size(); ++i)
    {
        hybrid[i].dspX  = dsp[i].dspX;
        hybrid[i].dspY  = dsp[i].dspY;
        hybrid[i].clapX = clap[i].clapX;
        hybrid[i].clapY = clap[i].clapY;
    }
    return hybrid;
}

} // namespace svdb::db

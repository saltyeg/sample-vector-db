#include "SampleMap.h"
#include "Palette.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace svdb::ui
{
namespace
{
constexpr int kNebulaRes = 72;
constexpr double kRippleLifeSec = 0.42;
constexpr double kTrailLifeSec  = 0.48;

double nowSec()
{
    return juce::Time::getMillisecondCounterHiRes() * 0.001;
}

juce::String formatMsShort (float ms)
{
    char buf[24] {};
    if (ms < 1000.0f) std::snprintf (buf, sizeof (buf), "%.0fms", ms);
    else              std::snprintf (buf, sizeof (buf), "%.2fs", ms / 1000.0f);
    return juce::String (buf);
}

void boxBlurRgba (std::vector<float>& buf, int w, int h, int radius)
{
    std::vector<float> tmp (buf.size(), 0.0f);
    const float inv = 1.0f / static_cast<float> (2 * radius + 1);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
            for (int k = -radius; k <= radius; ++k)
            {
                const int sx = std::clamp (x + k, 0, w - 1);
                const size_t idx = static_cast<size_t> ((y * w + sx) * 4);
                r += buf[idx];
                g += buf[idx + 1];
                b += buf[idx + 2];
                a += buf[idx + 3];
            }
            const size_t dst = static_cast<size_t> ((y * w + x) * 4);
            tmp[dst]     = r * inv;
            tmp[dst + 1] = g * inv;
            tmp[dst + 2] = b * inv;
            tmp[dst + 3] = a * inv;
        }
    }

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
            for (int k = -radius; k <= radius; ++k)
            {
                const int sy = std::clamp (y + k, 0, h - 1);
                const size_t idx = static_cast<size_t> ((sy * w + x) * 4);
                r += tmp[idx];
                g += tmp[idx + 1];
                b += tmp[idx + 2];
                a += tmp[idx + 3];
            }
            const size_t dst = static_cast<size_t> ((y * w + x) * 4);
            buf[dst]     = r * inv;
            buf[dst + 1] = g * inv;
            buf[dst + 2] = b * inv;
            buf[dst + 3] = a * inv;
        }
    }
}
} // namespace

SampleMap::SampleMap()
{
    setWantsKeyboardFocus (true);
}

void SampleMap::setData (std::shared_ptr<const db::IndexSnapshot> snapshot,
                         const std::vector<SearchResult>& visibleResults,
                         bool isRankedQuery)
{
    snap = std::move (snapshot);
    results = visibleResults;
    ranked = isRankedQuery;
    scoreById.clear();
    scoreById.reserve (results.size());
    for (const auto& r : results)
        scoreById[r.id] = r.score;
    hoveredResultIdx = -1;
    nebulaDirty = true;
    repaint();
}

void SampleMap::setSelectedSampleId (int64_t id)
{
    if (selectedId != id)
    {
        selectedId = id;
        spawnRipple (id, false);
        repaint();
    }
}

void SampleMap::setBlend (float clapBlend)
{
    const float clamped = std::clamp (clapBlend, 0.0f, 1.0f);
    if (std::abs (mapBlend - clamped) > 1e-4f)
    {
        mapBlend = clamped;
        nebulaDirty = true;
        repaint();
    }
}

void SampleMap::resetView()
{
    viewCx = 0.0f;
    viewCy = 0.0f;
    zoom = 1.0f;
    nebulaDirty = true;
    repaint();
}

void SampleMap::spawnRipple (int64_t sampleId, bool addTrail)
{
    if (! snap || sampleId < 0)
        return;
    auto it = snap->idToIndex.find (sampleId);
    if (it == snap->idToIndex.end())
        return;

    const auto& m = snap->meta[static_cast<size_t> (it->second)];
    float wx = 0.0f, wy = 0.0f;
    m.mapPosForBlend (mapBlend, wx, wy);
    const double t = nowSec();
    const auto c = categoryColour (m.category);

    ripples.push_back ({ wx, wy, c, t });
    if (ripples.size() > 10)
        ripples.erase (ripples.begin());

    if (addTrail)
    {
        scrubTrail.push_back ({ wx, wy, c, t });
        if (scrubTrail.size() > 16)
            scrubTrail.erase (scrubTrail.begin());
    }

    if (! isTimerRunning())
        startTimerHz (60);
}

void SampleMap::timerCallback()
{
    const double t = nowSec();
    ripples.erase (std::remove_if (ripples.begin(), ripples.end(),
                                   [t] (const Ripple& r) { return (t - r.startTimeSec) > kRippleLifeSec; }),
                   ripples.end());
    scrubTrail.erase (std::remove_if (scrubTrail.begin(), scrubTrail.end(),
                                      [t] (const TrailNode& n) { return (t - n.timeSec) > kTrailLifeSec; }),
                      scrubTrail.end());

    if (ripples.empty() && scrubTrail.empty())
        stopTimer();

    repaint();
}

void SampleMap::rebuildNebula (int w, int h)
{
    nebulaDirty = false;
    lastNebulaW = w;
    lastNebulaH = h;

    if (! snap || results.empty() || w <= 0 || h <= 0)
    {
        nebulaImg = {};
        return;
    }

    const int gw = kNebulaRes;
    const int gh = std::max (24, static_cast<int> (std::lround (static_cast<float> (kNebulaRes) * static_cast<float> (h) / static_cast<float> (w))));

    std::vector<float> grid (static_cast<size_t> (gw * gh * 4), 0.0f);
    const float sx = static_cast<float> (gw) / static_cast<float> (w);
    const float sy = static_cast<float> (gh) / static_cast<float> (h);

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& m = snap->meta[static_cast<size_t> (results[i].index)];
        float wx = 0.0f, wy = 0.0f;
        m.mapPosForBlend (mapBlend, wx, wy);
        const auto sp = worldToScreen (wx, wy);
        const int gx = static_cast<int> (sp.x * sx);
        const int gy = static_cast<int> (sp.y * sy);
        if (gx < 0 || gx >= gw || gy < 0 || gy >= gh)
            continue;

        const auto c = categoryColour (m.category);
        const float weight = ranked ? ((i < 30) ? 1.5f : (i < 120 ? 0.6f : 0.15f)) : 1.0f;
        const size_t idx = static_cast<size_t> ((gy * gw + gx) * 4);
        grid[idx]     += c.getFloatRed()   * weight;
        grid[idx + 1] += c.getFloatGreen() * weight;
        grid[idx + 2] += c.getFloatBlue()  * weight;
        grid[idx + 3] += weight;
    }

    boxBlurRgba (grid, gw, gh, 5);
    boxBlurRgba (grid, gw, gh, 4);

    nebulaImg = juce::Image (juce::Image::ARGB, gw, gh, true);
    juce::Image::BitmapData bmp (nebulaImg, juce::Image::BitmapData::writeOnly);

    float maxDens = 0.01f;
    for (int i = 0; i < gw * gh; ++i)
        maxDens = std::max (maxDens, grid[static_cast<size_t> (i * 4 + 3)]);
    const float exposure = 1.0f / std::max (0.15f, std::sqrt (maxDens));

    // Whisper-subtle atmospheric color tint (max alpha 0.075) so dots remain ultra-crisp.
    for (int y = 0; y < gh; ++y)
    {
        for (int x = 0; x < gw; ++x)
        {
            const size_t idx = static_cast<size_t> ((y * gw + x) * 4);
            const float d = grid[idx + 3];
            if (d < 1e-4f) continue;

            const float invD = 1.0f / d;
            const float r = std::clamp (grid[idx]     * invD, 0.0f, 1.0f);
            const float g = std::clamp (grid[idx + 1] * invD, 0.0f, 1.0f);
            const float b = std::clamp (grid[idx + 2] * invD, 0.0f, 1.0f);
            const float a = std::clamp (std::sqrt (d) * exposure * 0.070f, 0.0f, 0.075f);

            bmp.setPixelColour (x, y, juce::Colour::fromFloatRGBA (r, g, b, a));
        }
    }
}

juce::Point<float> SampleMap::worldToScreen (float wx, float wy) const
{
    const float w = static_cast<float> (getWidth());
    const float h = static_cast<float> (getHeight());
    const float scaleX = 0.46f * w * zoom;
    const float scaleY = 0.46f * h * zoom;
    // Blend isotropic + aspect-fit scale so the cloud breathes naturally across the pane.
    const float iso = 0.46f * std::min (w, h) * zoom;
    const float sx = 0.55f * iso + 0.45f * scaleX;
    const float sy = 0.55f * iso + 0.45f * scaleY;
    return {
        0.5f * w + (wx - viewCx) * sx,
        0.5f * h - (wy - viewCy) * sy
    };
}

juce::Point<float> SampleMap::screenToWorld (juce::Point<float> pt) const
{
    const float w = static_cast<float> (getWidth());
    const float h = static_cast<float> (getHeight());
    const float iso = 0.46f * std::min (w, h) * zoom;
    const float sx = std::max (1.0f, 0.55f * iso + 0.45f * (0.46f * w * zoom));
    const float sy = std::max (1.0f, 0.55f * iso + 0.45f * (0.46f * h * zoom));
    return {
        viewCx + (pt.x - 0.5f * w) / sx,
        viewCy - (pt.y - 0.5f * h) / sy
    };
}

int SampleMap::findNearestResult (juce::Point<float> screenPt, float maxDistPx) const
{
    if (! snap || results.empty())
        return -1;

    const float maxSq = maxDistPx * maxDistPx;
    float bestSq = maxSq;
    int bestIdx = -1;

    for (int i = 0; i < static_cast<int> (results.size()); ++i)
    {
        const auto& m = snap->meta[static_cast<size_t> (results[static_cast<size_t> (i)].index)];
        float mx = 0.0f, my = 0.0f;
        m.mapPosForBlend (mapBlend, mx, my);
        const auto p = worldToScreen (mx, my);
        const float dx = p.x - screenPt.x;
        const float dy = p.y - screenPt.y;
        const float d2 = dx * dx + dy * dy;
        if (d2 < bestSq)
        {
            bestSq = d2;
            bestIdx = i;
        }
    }
    return bestIdx;
}

void SampleMap::paint (juce::Graphics& g)
{
    const int wi = getWidth();
    const int hi = getHeight();
    const float w = static_cast<float> (wi);
    const float h = static_cast<float> (hi);

    // 1. Clean XO-style matte dark slate canvas with subtle vignette.
    juce::ColourGradient bgGrad (juce::Colour (0xff1c2026), w * 0.5f, h * 0.5f,
                                 juce::Colour (0xff14171c), 0.0f, 0.0f, true);
    g.setGradientFill (bgGrad);
    g.fillRect (getLocalBounds());

    if (! snap || snap->count() == 0)
    {
        g.setColour (col::muted);
        g.setFont (juce::FontOptions (12.0f));
        g.drawText ("No samples indexed. Click [Index] to scan a folder.",
                    getLocalBounds(), juce::Justification::centred);
        g.setColour (col::border);
        g.drawRect (getLocalBounds(), 1);
        return;
    }

    // 2. Whisper-subtle atmospheric region glow (never obscures dots).
    if (nebulaDirty || lastNebulaW != wi || lastNebulaH != hi)
        rebuildNebula (wi, hi);
    if (nebulaImg.isValid())
    {
        g.setImageResamplingQuality (juce::Graphics::highResamplingQuality);
        g.drawImage (nebulaImg, getLocalBounds().toFloat());
    }

    // Crisp, XO-sized individual dots: ~1.2px at zoom 1.0, gently growing as you zoom in.
    const float dotR = std::clamp (1.20f + 0.42f * std::log2 (std::max (1.0f, zoom)), 1.15f, 3.6f);

    // 3. Very subtle dim dust for filtered-out points so overall map silhouette stays legible.
    if (static_cast<int> (results.size()) < snap->count())
    {
        g.setColour (juce::Colour (0x26657082));
        for (int i = 0; i < snap->count(); ++i)
        {
            const auto& m = snap->meta[static_cast<size_t> (i)];
            if (scoreById.find (m.id) != scoreById.end())
                continue;
            float mx = 0.0f, my = 0.0f;
            m.mapPosForBlend (mapBlend, mx, my);
            const auto p = worldToScreen (mx, my);
            if (p.x < -3.0f || p.x > w + 3.0f || p.y < -3.0f || p.y > h + 3.0f)
                continue;
            g.fillEllipse (p.x - 0.85f, p.y - 0.85f, 1.7f, 1.7f);
        }
    }

    // Locate selected point for subtle neighbor tethers (ranked queries only, keeping browse super clean).
    juce::Point<float> selectedPt {};
    juce::Colour selectedCol = col::accent;
    bool hasSelectedPt = false;
    int selectedResultIdx = -1;

    for (int i = 0; i < static_cast<int> (results.size()); ++i)
    {
        if (results[static_cast<size_t> (i)].id == selectedId)
        {
            selectedResultIdx = i;
            const auto& m = snap->meta[static_cast<size_t> (results[static_cast<size_t> (i)].index)];
            float mx = 0.0f, my = 0.0f;
            m.mapPosForBlend (mapBlend, mx, my);
            selectedPt = worldToScreen (mx, my);
            selectedCol = categoryColour (m.category);
            hasSelectedPt = true;
            break;
        }
    }

    if (hasSelectedPt && ranked)
    {
        const int lim = std::min<int> (10, static_cast<int> (results.size()));
        for (int i = 0; i < lim; ++i)
        {
            if (i == selectedResultIdx) continue;
            const auto& m = snap->meta[static_cast<size_t> (results[static_cast<size_t> (i)].index)];
            float mx = 0.0f, my = 0.0f;
            m.mapPosForBlend (mapBlend, mx, my);
            const auto p = worldToScreen (mx, my);
            const float alpha = 0.12f * (1.0f - static_cast<float> (i) / static_cast<float> (lim + 2));
            g.setColour (selectedCol.withAlpha (alpha));
            g.drawLine (selectedPt.x, selectedPt.y, p.x, p.y, 0.8f);
        }
    }

    // 4. Crisp individual starfield dots (zero clumpy white glare, clean negative space).
    const SampleMeta* hoverMeta = nullptr;
    juce::Point<float> hoverPt {};

    for (int i = static_cast<int> (results.size()) - 1; i >= 0; --i)
    {
        const auto& r = results[static_cast<size_t> (i)];
        const auto& m = snap->meta[static_cast<size_t> (r.index)];
        float mx = 0.0f, my = 0.0f;
        m.mapPosForBlend (mapBlend, mx, my);
        const auto p = worldToScreen (mx, my);
        if (p.x < -6.0f || p.x > w + 6.0f || p.y < -6.0f || p.y > h + 6.0f)
            continue;

        // Subtle brightness tint per sample so neighboring dots have organic variation.
        const float brightShift = (m.brightness - 0.5f) * 0.18f;
        auto c = categoryColour (m.category);
        if (brightShift > 0.0f) c = c.brighter (brightShift);
        else                    c = c.darker (-brightShift * 0.65f);

        float radius = dotR * (0.88f + 0.24f * std::clamp (m.punch, 0.0f, 1.0f));
        float alpha = 0.80f;

        if (ranked)
        {
            if (i < 16)
            {
                alpha = 0.96f;
                radius = dotR + 0.85f;
                // Subtle individual halo only for top ranked matches.
                const float hr = radius * 2.1f;
                g.setColour (c.withAlpha (0.14f));
                g.fillEllipse (p.x - hr, p.y - hr, hr * 2.0f, hr * 2.0f);
            }
            else if (i < 80)
            {
                alpha = 0.72f;
                radius = dotR + 0.15f;
            }
            else
            {
                alpha = 0.28f;
                radius = dotR * 0.85f;
            }
        }

        g.setColour (c.withAlpha (alpha));
        g.fillEllipse (p.x - radius, p.y - radius, radius * 2.0f, radius * 2.0f);

        if (i == hoveredResultIdx)
        {
            hoverPt = p;
            hoverMeta = &m;
        }
    }

    // 5. Subtle fading scrub trail (when right-dragging across the cloud).
    const double tNow = nowSec();
    if (scrubTrail.size() >= 2)
    {
        for (size_t i = 1; i < scrubTrail.size(); ++i)
        {
            const auto& a = scrubTrail[i - 1];
            const auto& b = scrubTrail[i];
            const float age = static_cast<float> ((tNow - b.timeSec) / kTrailLifeSec);
            if (age >= 1.0f) continue;
            const float alpha = (1.0f - age) * (1.0f - age);
            const auto pa = worldToScreen (a.wx, a.wy);
            const auto pb = worldToScreen (b.wx, b.wy);
            g.setColour (b.colour.withAlpha (alpha * 0.30f));
            g.drawLine (pa.x, pa.y, pb.x, pb.y, 1.2f);
        }
    }

    // 6. Delicate single-ring ripple on trigger.
    for (const auto& rip : ripples)
    {
        const float prog = static_cast<float> ((tNow - rip.startTimeSec) / kRippleLifeSec);
        if (prog < 0.0f || prog >= 1.0f) continue;
        const float ease = 1.0f - (1.0f - prog) * (1.0f - prog);
        const float alpha = (1.0f - prog) * (1.0f - prog);
        const auto p = worldToScreen (rip.wx, rip.wy);
        const float rOuter = dotR + 3.0f + ease * 16.0f;

        g.setColour (rip.colour.withAlpha (alpha * 0.42f));
        g.drawEllipse (p.x - rOuter, p.y - rOuter, rOuter * 2.0f, rOuter * 2.0f, 1.0f);
    }

    // 7. XO-style crisp active dot (larger solid colored dot + subtle halo + thin white ring).
    if (hasSelectedPt)
    {
        const float activeR = std::max (3.4f, dotR + 1.8f);
        const float haloR = activeR + 5.0f;
        g.setColour (selectedCol.withAlpha (0.18f));
        g.fillEllipse (selectedPt.x - haloR, selectedPt.y - haloR, haloR * 2.0f, haloR * 2.0f);

        g.setColour (selectedCol.brighter (0.22f));
        g.fillEllipse (selectedPt.x - activeR, selectedPt.y - activeR, activeR * 2.0f, activeR * 2.0f);

        g.setColour (juce::Colours::white.withAlpha (0.85f));
        g.drawEllipse (selectedPt.x - activeR - 1.0f, selectedPt.y - activeR - 1.0f,
                       (activeR + 1.0f) * 2.0f, (activeR + 1.0f) * 2.0f, 1.0f);
    }

    // 8. Minimal clean hover highlight + compact floating tooltip with mini-waveform.
    if (hoverMeta != nullptr)
    {
        const auto hc = categoryColour (hoverMeta->category);
        const float hr = std::max (3.0f, dotR + 1.5f);
        g.setColour (hc.brighter (0.25f));
        g.fillEllipse (hoverPt.x - hr, hoverPt.y - hr, hr * 2.0f, hr * 2.0f);
        g.setColour (juce::Colours::white.withAlpha (0.55f));
        g.drawEllipse (hoverPt.x - hr - 1.5f, hoverPt.y - hr - 1.5f, (hr + 1.5f) * 2.0f, (hr + 1.5f) * 2.0f, 1.0f);

        juce::String title = juce::String::fromUTF8 (hoverMeta->filename.c_str());
        juce::String folderLine = juce::String::fromUTF8 (sampleFolderLabel (hoverMeta->path, hoverMeta->folder).c_str());
        juce::String sub = juce::String (categoryName (hoverMeta->category))
                         + " | " + formatMsShort (hoverMeta->durationMs);
        if (hoverMeta->midiNote >= 0)
            sub += " | " + juce::String (midiNoteToName (hoverMeta->midiNote));

        g.setFont (juce::FontOptions (11.0f));
        const int titleW = static_cast<int> (std::ceil (juce::GlyphArrangement::getStringWidth (g.getCurrentFont(), title)));
        g.setFont (juce::FontOptions (9.5f));
        const int folderW = static_cast<int> (std::ceil (juce::GlyphArrangement::getStringWidth (g.getCurrentFont(), folderLine)));
        const int cardW = std::clamp (std::max (titleW + 24, folderW + 18), 160, 300);
        const int cardH = 54;

        int tx = static_cast<int> (hoverPt.x) + 12;
        int ty = static_cast<int> (hoverPt.y) - cardH - 8;
        if (tx + cardW > wi - 6) tx = static_cast<int> (hoverPt.x) - cardW - 12;
        if (tx < 6) tx = 6;
        if (ty < 6) ty = static_cast<int> (hoverPt.y) + 12;

        const juce::Rectangle<float> card (static_cast<float> (tx), static_cast<float> (ty),
                                           static_cast<float> (cardW), static_cast<float> (cardH));
        g.setColour (juce::Colour (0xec171a20));
        g.fillRoundedRectangle (card, 3.0f);
        g.setColour (col::border);
        g.drawRoundedRectangle (card, 3.0f, 1.0f);

        g.setColour (hc);
        g.fillEllipse (static_cast<float> (tx + 7), static_cast<float> (ty + 7), 5.0f, 5.0f);
        g.setFont (juce::FontOptions (11.0f));
        g.setColour (col::text);
        g.drawText (title, tx + 16, ty + 2, cardW - 20, 15, juce::Justification::centredLeft, true);

        g.setFont (juce::FontOptions (9.5f));
        g.setColour (col::accent.withAlpha (0.85f));
        g.drawText (folderLine, tx + 7, ty + 18, cardW - 14, 14, juce::Justification::centredLeft, true);

        g.setColour (col::muted);
        g.drawText (sub, tx + 7, ty + 34, cardW - 72, 15, juce::Justification::centredLeft, true);

        const int wfX = tx + cardW - 62;
        const float wfMidY = static_cast<float> (ty + 41);
        g.setColour (hc.withAlpha (0.75f));
        for (int b = 0; b < 48; ++b)
        {
            const int src = b * 2;
            const float mn = static_cast<float> (hoverMeta->peaks[static_cast<size_t> (src * 2)]) / 127.0f;
            const float mx = static_cast<float> (hoverMeta->peaks[static_cast<size_t> (src * 2 + 1)]) / 127.0f;
            const float px = static_cast<float> (wfX) + static_cast<float> (b) * 1.15f;
            g.drawVerticalLine (static_cast<int> (px), wfMidY - mx * 7.0f, std::max (wfMidY - mx * 7.0f + 1.0f, wfMidY - mn * 7.0f));
        }
    }

    // 9. Minimal corner hint.
    g.setFont (juce::FontOptions (10.0f));
    g.setColour (col::muted.withAlpha (0.65f));
    const juce::String corner = juce::String (static_cast<int> (results.size())) + " shown | wheel=zoom mdrag=pan rdrag=scrub space=clear";
    g.drawText (corner, 8, hi - 18, wi - 16, 14, juce::Justification::centredLeft);

    g.setColour (col::border);
    g.drawRect (getLocalBounds(), 1);
}

void SampleMap::scrubAt (juce::Point<float> screenPt)
{
    const int idx = findNearestResult (screenPt, 24.0f);
    hoveredResultIdx = idx;
    if (idx >= 0 && snap)
    {
        const int64_t id = results[static_cast<size_t> (idx)].id;
        if (id != lastScrubbedId)
        {
            lastScrubbedId = id;
            selectedId = id;
            spawnRipple (id, true);
            if (onSelectSample)
                onSelectSample (id);
        }
    }
    repaint();
}

void SampleMap::mouseMove (const juce::MouseEvent& e)
{
    const int idx = findNearestResult (e.position, 14.0f);
    if (idx != hoveredResultIdx)
    {
        hoveredResultIdx = idx;
        repaint();
    }
}

void SampleMap::mouseExit (const juce::MouseEvent&)
{
    if (hoveredResultIdx != -1)
    {
        hoveredResultIdx = -1;
        repaint();
    }
}

void SampleMap::mouseDown (const juce::MouseEvent& e)
{
    grabKeyboardFocus();
    if (e.mods.isMiddleButtonDown())
    {
        panning = true;
        panStartMouse = e.position;
        panStartCx = viewCx;
        panStartCy = viewCy;
        return;
    }

    if (e.mods.isRightButtonDown())
    {
        scrubAuditioning = true;
        lastScrubbedId = -1;
        scrubAt (e.position);
        return;
    }

    if (e.mods.isLeftButtonDown())
    {
        const int idx = findNearestResult (e.position, 16.0f);
        if (idx >= 0 && snap)
        {
            const int64_t id = results[static_cast<size_t> (idx)].id;
            selectedId = id;
            potentialFileDrag = true;
            externalDragStarted = false;
            dragSampleId = id;
            dragStartPos = e.position;
            spawnRipple (id, false);
            repaint();
            if (onSelectSample)
                onSelectSample (id);
        }
        else
        {
            panning = true;
            panStartMouse = e.position;
            panStartCx = viewCx;
            panStartCy = viewCy;
        }
    }
}

void SampleMap::mouseDrag (const juce::MouseEvent& e)
{
    if (panning)
    {
        const float w = static_cast<float> (getWidth());
        const float h = static_cast<float> (getHeight());
        const float iso = 0.46f * std::min (w, h) * zoom;
        const float sx = std::max (1.0f, 0.55f * iso + 0.45f * (0.46f * w * zoom));
        const float sy = std::max (1.0f, 0.55f * iso + 0.45f * (0.46f * h * zoom));
        const auto delta = e.position - panStartMouse;
        viewCx = panStartCx - delta.x / sx;
        viewCy = panStartCy + delta.y / sy;
        nebulaDirty = true;
        repaint();
        return;
    }

    if (scrubAuditioning)
    {
        scrubAt (e.position);
        return;
    }

    if (potentialFileDrag && ! externalDragStarted && dragSampleId >= 0 && snap)
    {
        if (e.position.getDistanceFrom (dragStartPos) > 6.0f)
        {
            externalDragStarted = true;
            auto it = snap->idToIndex.find (dragSampleId);
            if (it != snap->idToIndex.end())
            {
                const auto& path = snap->meta[static_cast<size_t> (it->second)].path;
                juce::StringArray files;
                files.add (juce::String::fromUTF8 (path.c_str()));
                juce::DragAndDropContainer::performExternalDragDropOfFiles (files, false, this);
            }
        }
    }
}

void SampleMap::mouseUp (const juce::MouseEvent&)
{
    panning = false;
    scrubAuditioning = false;
    lastScrubbedId = -1;
    potentialFileDrag = false;
    externalDragStarted = false;
}

void SampleMap::mouseDoubleClick (const juce::MouseEvent& e)
{
    if (! e.mods.isLeftButtonDown())
        return;
    const int idx = findNearestResult (e.position, 16.0f);
    if (idx >= 0 && onFindSimilar)
        onFindSimilar (results[static_cast<size_t> (idx)].id);
}

void SampleMap::mouseWheelMove (const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel)
{
    if (std::abs (wheel.deltaY) < 1e-4f)
        return;

    const auto worldBefore = screenToWorld (e.position);
    const float factor = (wheel.deltaY > 0.0f) ? 1.25f : 0.80f;
    zoom = std::clamp (zoom * factor, 0.6f, 40.0f);
    const auto worldAfter = screenToWorld (e.position);
    viewCx += (worldBefore.x - worldAfter.x);
    viewCy += (worldBefore.y - worldAfter.y);
    nebulaDirty = true;
    repaint();
}

bool SampleMap::keyPressed (const juce::KeyPress& key)
{
    if (key.getTextCharacter() == 'r' || key.getTextCharacter() == 'R')
    {
        resetView();
        return true;
    }
    if (key == juce::KeyPress::spaceKey)
    {
        if (onResetSimilar)
            onResetSimilar();
        return true;
    }
    return false;
}

} // namespace svdb::ui

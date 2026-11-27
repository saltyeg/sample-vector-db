#pragma once

#include "core/Types.h"
#include "core/db/VectorIndex.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace svdb::ui
{

// Barebones interactive 2D similarity scatter map.
//   Left-click:       select + audition
//   Left-drag (>6px): OS drag-and-drop sample file into FL Studio
//   Double-click:     find similar to clicked sample
//   Right-drag:       scrub-audition each sample hovered over
//   Middle-drag:      pan
//   Wheel:            zoom around cursor
//   'R' key:          reset view
//   Spacebar:         clear similar-sample filter
class SampleMap final : public juce::Component,
                        private juce::Timer
{
public:
    SampleMap();

    void setData (std::shared_ptr<const db::IndexSnapshot> snapshot,
                  const std::vector<SearchResult>& visibleResults,
                  bool isRankedQuery);

    void setSelectedSampleId (int64_t id);
    void setBlend (float clapBlend);
    void resetView();

    std::function<void (int64_t sampleId)> onSelectSample;
    std::function<void (int64_t sampleId)> onFindSimilar;
    std::function<void()> onResetSimilar;

    void paint (juce::Graphics& g) override;
    void mouseMove (const juce::MouseEvent& e) override;
    void mouseExit (const juce::MouseEvent& e) override;
    void mouseDown (const juce::MouseEvent& e) override;
    void mouseDrag (const juce::MouseEvent& e) override;
    void mouseUp (const juce::MouseEvent& e) override;
    void mouseDoubleClick (const juce::MouseEvent& e) override;
    void mouseWheelMove (const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override;
    bool keyPressed (const juce::KeyPress& key) override;

private:
    struct Ripple
    {
        float wx = 0.0f;
        float wy = 0.0f;
        juce::Colour colour;
        double startTimeSec = 0.0;
    };

    struct TrailNode
    {
        float wx = 0.0f;
        float wy = 0.0f;
        juce::Colour colour;
        double timeSec = 0.0;
    };

    std::shared_ptr<const db::IndexSnapshot> snap;
    std::vector<SearchResult> results;
    std::unordered_map<int64_t, float> scoreById;
    bool ranked = false;

    int64_t selectedId = -1;
    int hoveredResultIdx = -1;
    float mapBlend = 0.55f;

    // Camera in normalized map space [-1, 1]^2.
    float viewCx = 0.0f;
    float viewCy = 0.0f;
    float zoom = 1.0f;

    bool panning = false;
    juce::Point<float> panStartMouse;
    float panStartCx = 0.0f;
    float panStartCy = 0.0f;

    bool scrubAuditioning = false;
    int64_t lastScrubbedId = -1;

    bool potentialFileDrag = false;
    bool externalDragStarted = false;
    int64_t dragSampleId = -1;
    juce::Point<float> dragStartPos;

    // Animated sonic ripples & scrub trail
    std::vector<Ripple> ripples;
    std::vector<TrailNode> scrubTrail;

    // Offscreen low-res nebula glow buffer (rebuilt lazily when view/data/blend changes)
    juce::Image nebulaImg;
    bool nebulaDirty = true;
    int lastNebulaW = 0;
    int lastNebulaH = 0;

    void timerCallback() override;
    void spawnRipple (int64_t sampleId, bool addTrail);
    void rebuildNebula (int w, int h);

    juce::Point<float> worldToScreen (float wx, float wy) const;
    juce::Point<float> screenToWorld (juce::Point<float> pt) const;
    int findNearestResult (juce::Point<float> screenPt, float maxDistPx = 14.0f) const;
    void scrubAt (juce::Point<float> screenPt);
};

} // namespace svdb::ui

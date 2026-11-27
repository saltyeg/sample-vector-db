#pragma once

#include "core/Types.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>

namespace svdb::ui
{

// Barebones bottom strip: waveform peaks + playhead, click to replay, drag to FL Studio,
// Pitch toggle (MIDI C5=unity), Gain slider, Pin toggle (Standalone), file path.
class AuditionBar final : public juce::Component
{
public:
    explicit AuditionBar (bool showPinButton);

    void setSample (const SampleMeta* meta);
    void setPlayhead (float normPos);
    void setPitchMode (bool chromatic);
    void setGain (float gain);

    std::function<void (float normPos)> onReplayAt;
    std::function<void (bool chromatic)> onPitchToggle;
    std::function<void (float gain)> onGainChange;
    std::function<void (bool pinned)> onPinToggle;
    std::function<void()> onFindSimilarCurrent;

    void paint (juce::Graphics& g) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent& e) override;
    void mouseDrag (const juce::MouseEvent& e) override;

private:
    bool hasSample = false;
    SampleMeta currentMeta;
    float playhead = -1.0f;

    juce::TextButton simBtn { "~" };
    juce::ToggleButton pitchBtn { "Pitch" };
    juce::ToggleButton pinBtn { "Pin" };
    juce::Slider gainSlider;
    bool showPin = false;

    juce::Rectangle<int> waveArea;
    juce::Point<float> dragStart;
    bool dragStarted = false;
};

} // namespace svdb::ui

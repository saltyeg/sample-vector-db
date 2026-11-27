#include "AuditionBar.h"
#include "Palette.h"

#include <algorithm>
#include <cmath>

namespace svdb::ui
{

AuditionBar::AuditionBar (bool showPinButton)
    : showPin (showPinButton)
{
    simBtn.setTooltip ("Find similar to selected sample");
    simBtn.setColour (juce::TextButton::buttonColourId, col::controlBg);
    simBtn.setColour (juce::TextButton::textColourOffId, col::text);
    simBtn.onClick = [this] { if (onFindSimilarCurrent) onFindSimilarCurrent(); };
    addAndMakeVisible (simBtn);

    pitchBtn.setTooltip ("Chromatic MIDI pitch (FL Studio C5 = original pitch)");
    pitchBtn.setColour (juce::ToggleButton::textColourId, col::text);
    pitchBtn.onClick = [this] { if (onPitchToggle) onPitchToggle (pitchBtn.getToggleState()); };
    addAndMakeVisible (pitchBtn);

    gainSlider.setSliderStyle (juce::Slider::LinearHorizontal);
    gainSlider.setTextBoxStyle (juce::Slider::NoTextBox, true, 0, 0);
    gainSlider.setRange (0.0, 1.2, 0.01);
    gainSlider.setValue (0.85, juce::dontSendNotification);
    gainSlider.setColour (juce::Slider::trackColourId, col::accent);
    gainSlider.setColour (juce::Slider::backgroundColourId, col::controlBg);
    gainSlider.setColour (juce::Slider::thumbColourId, col::text);
    gainSlider.onValueChange = [this] { if (onGainChange) onGainChange (static_cast<float> (gainSlider.getValue())); };
    addAndMakeVisible (gainSlider);

    if (showPin)
    {
        pinBtn.setTooltip ("Keep standalone window on top of FL Studio");
        pinBtn.setColour (juce::ToggleButton::textColourId, col::text);
        pinBtn.onClick = [this] { if (onPinToggle) onPinToggle (pinBtn.getToggleState()); };
        addAndMakeVisible (pinBtn);
    }
}

void AuditionBar::setSample (const SampleMeta* meta)
{
    if (meta)
    {
        hasSample = true;
        currentMeta = *meta;
    }
    else
    {
        hasSample = false;
    }
    repaint();
}

void AuditionBar::setPlayhead (float normPos)
{
    if (std::abs (playhead - normPos) > 0.004f)
    {
        playhead = normPos;
        repaint (waveArea);
    }
}

void AuditionBar::setPitchMode (bool chromatic)
{
    pitchBtn.setToggleState (chromatic, juce::dontSendNotification);
}

void AuditionBar::setGain (float gain)
{
    gainSlider.setValue (gain, juce::dontSendNotification);
}

void AuditionBar::resized()
{
    auto r = getLocalBounds().reduced (6, 6);
    if (showPin)
    {
        pinBtn.setBounds (r.removeFromRight (46));
        r.removeFromRight (4);
    }
    gainSlider.setBounds (r.removeFromRight (76));
    r.removeFromRight (6);
    pitchBtn.setBounds (r.removeFromRight (56));
    r.removeFromRight (4);
    simBtn.setBounds (r.removeFromRight (28));
    r.removeFromRight (8);

    const int waveW = std::min (260, std::max (120, r.getWidth() / 2));
    waveArea = r.removeFromLeft (waveW);
}

void AuditionBar::paint (juce::Graphics& g)
{
    g.fillAll (col::panel);
    g.setColour (col::border);
    g.drawHorizontalLine (0, 0.0f, static_cast<float> (getWidth()));

    // Waveform box.
    g.setColour (col::bg);
    g.fillRect (waveArea);
    g.setColour (col::border);
    g.drawRect (waveArea, 1);

    if (hasSample)
    {
        const float midY = static_cast<float> (waveArea.getCentreY());
        const float halfH = 0.44f * static_cast<float> (waveArea.getHeight());
        const float x0 = static_cast<float> (waveArea.getX() + 2);
        const float w  = static_cast<float> (waveArea.getWidth() - 4);

        g.setColour (col::waveform);
        for (int i = 0; i < kWaveformPeaks; ++i)
        {
            const float px = x0 + (static_cast<float> (i) / static_cast<float> (kWaveformPeaks)) * w;
            const float mn = static_cast<float> (currentMeta.peaks[static_cast<size_t> (i * 2)]) / 127.0f;
            const float mx = static_cast<float> (currentMeta.peaks[static_cast<size_t> (i * 2 + 1)]) / 127.0f;
            const float yTop = midY - mx * halfH;
            const float yBot = midY - mn * halfH;
            g.drawVerticalLine (static_cast<int> (px), yTop, std::max (yTop + 1.0f, yBot));
        }

        if (playhead >= 0.0f && playhead <= 1.0f)
        {
            const float px = x0 + playhead * w;
            g.setColour (col::playhead);
            g.drawVerticalLine (static_cast<int> (px), static_cast<float> (waveArea.getY() + 1),
                                static_cast<float> (waveArea.getBottom() - 1));
        }
    }

    // Selected sample info / path text between waveform and controls.
    const int textX = waveArea.getRight() + 8;
    const int textRight = simBtn.getX() - 8;
    if (textRight > textX + 20)
    {
        g.setFont (juce::FontOptions (11.5f));
        if (hasSample)
        {
            g.setColour (col::text);
            juce::String info = juce::String::fromUTF8 (currentMeta.filename.c_str())
                              + "  [" + categoryName (currentMeta.category) + "]";
            if (currentMeta.midiNote >= 0)
                info += "  " + juce::String (midiNoteToName (currentMeta.midiNote));
            g.drawText (info, textX, 4, textRight - textX, 16, juce::Justification::centredLeft, true);

            g.setFont (juce::FontOptions (10.5f));
            g.setColour (col::muted);
            g.drawText (juce::String::fromUTF8 (currentMeta.path.c_str()),
                        textX, 20, textRight - textX, 16, juce::Justification::centredLeft, true);
        }
        else
        {
            g.setColour (col::muted);
            g.drawText ("Click sample to audition | Drag to FL Studio | Drop WAV to match",
                        textX, 0, textRight - textX, getHeight(), juce::Justification::centredLeft, true);
        }
    }
}

void AuditionBar::mouseDown (const juce::MouseEvent& e)
{
    dragStart = e.position;
    dragStarted = false;
    if (hasSample && waveArea.contains (e.getPosition()))
    {
        const float x0 = static_cast<float> (waveArea.getX() + 2);
        const float w  = std::max (1.0f, static_cast<float> (waveArea.getWidth() - 4));
        const float normPos = std::clamp ((e.position.x - x0) / w, 0.0f, 0.99f);
        if (onReplayAt)
            onReplayAt (normPos);
    }
}

void AuditionBar::mouseDrag (const juce::MouseEvent& e)
{
    if (hasSample && ! dragStarted && e.position.getDistanceFrom (dragStart) > 8.0f
        && waveArea.contains (dragStart.toInt())
        && ! waveArea.expanded (0, 6).contains (e.getPosition()))
    {
        dragStarted = true;
        juce::StringArray files;
        files.add (juce::String::fromUTF8 (currentMeta.path.c_str()));
        juce::DragAndDropContainer::performExternalDragDropOfFiles (files, false, this);
    }
}

} // namespace svdb::ui

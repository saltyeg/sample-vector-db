#pragma once

#include "PluginProcessor.h"
#include "ui/AuditionBar.h"
#include "ui/SampleList.h"
#include "ui/SampleMap.h"

#include <juce_audio_processors/juce_audio_processors.h>

#include <memory>
#include <vector>

namespace svdb
{

class SampleVectorDBEditor final : public juce::AudioProcessorEditor,
                                   public juce::FileDragAndDropTarget,
                                   private juce::Timer
{
public:
    explicit SampleVectorDBEditor (SampleVectorDBProcessor& proc);
    ~SampleVectorDBEditor() override;

    void paint (juce::Graphics& g) override;
    void resized() override;
    bool keyPressed (const juce::KeyPress& key) override;

    // External WAV drop onto window -> Find Similar!
    bool isInterestedInFileDrag (const juce::StringArray& files) override;
    void filesDropped (const juce::StringArray& files, int x, int y) override;

    // Programmatic hooks for svdb-ui offscreen verification tool
    void setSearchTextAndRun (const juce::String& text);
    void setLengthFilter (LengthFilter f);
    void selectFirstResult();

private:
    SampleVectorDBProcessor& processor;

    // Barebones Top Bar controls
    juce::TextEditor searchBox;
    juce::TextButton clearRefBtn { "x" };
    juce::TextButton lenAllBtn { "All" };
    juce::TextButton lenOneBtn { "One" };
    juce::TextButton lenLoopBtn { "Loop" };
    juce::ComboBox catCombo;
    juce::Slider blendSlider;
    juce::Label blendLabel;
    juce::TextButton indexBtn { "Index" };
    juce::Label statusLabel;

    // Main split panes + bottom bar
    ui::SampleMap sampleMap;
    ui::SampleList sampleList;
    ui::AuditionBar auditionBar;

    // Active query state
    int64_t activeReferenceId = -1;
    std::vector<float> externalRefDsp;
    std::vector<float> externalRefClap;
    std::string externalRefName;

    uint64_t lastSeenRevision = 0;
    int64_t selectedSampleId = -1;
    std::unique_ptr<juce::FileChooser> folderChooser;

    void timerCallback() override;
    void refreshQuery();
    void selectAndAudition (int64_t sampleId);
    void setReferenceSample (int64_t sampleId);
    void clearReference();
    void updateLengthButtonStyles();
    void promptAndIndexFolder();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SampleVectorDBEditor)
};

} // namespace svdb

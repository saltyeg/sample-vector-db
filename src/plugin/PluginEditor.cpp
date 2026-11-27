#include "PluginEditor.h"
#include "ui/Palette.h"

namespace svdb
{

namespace
{
void styleFlatButton (juce::TextButton& b, bool active = false)
{
    b.setColour (juce::TextButton::buttonColourId, active ? ui::col::controlOn : ui::col::controlBg);
    b.setColour (juce::TextButton::buttonOnColourId, ui::col::controlOn);
    b.setColour (juce::TextButton::textColourOffId, ui::col::text);
    b.setColour (juce::TextButton::textColourOnId, ui::col::text);
}
} // namespace

SampleVectorDBEditor::SampleVectorDBEditor (SampleVectorDBProcessor& proc)
    : AudioProcessorEditor (&proc),
      processor (proc),
      auditionBar (proc.wrapperType == juce::AudioProcessor::wrapperType_Standalone)
{
    // Default to 2x original dimensions (1960x1160), clamped cleanly to the primary display work area.
    int initW = 1960;
    int initH = 1160;
    if (const auto* disp = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay())
    {
        const auto work = disp->userBounds.toNearestInt();
        if (work.getWidth() > 800 && work.getHeight() > 500)
        {
            initW = std::min (initW, work.getWidth() - 48);
            initH = std::min (initH, work.getHeight() - 72);
        }
    }
    setSize (initW, initH);
    setResizable (true, true);
    setResizeLimits (720, 420, 3840, 2160);

    // Search box
    searchBox.setTextToShowWhenEmpty ("Search (text / CLAP semantic, Enter)...", ui::col::muted);
    searchBox.setColour (juce::TextEditor::backgroundColourId, ui::col::controlBg);
    searchBox.setColour (juce::TextEditor::textColourId, ui::col::text);
    searchBox.setColour (juce::TextEditor::outlineColourId, ui::col::border);
    searchBox.setColour (juce::TextEditor::focusedOutlineColourId, ui::col::accent);
    searchBox.onReturnKey = [this] {
        clearReference();
        refreshQuery();
    };
    searchBox.onTextChange = [this] {
        // Instant lexical refresh while typing; when CLAP text model is ready, Return runs full semantic query,
        // or if text is cleared, reset immediately.
        if (searchBox.getText().isEmpty())
            refreshQuery();
    };
    addAndMakeVisible (searchBox);

    styleFlatButton (clearRefBtn);
    clearRefBtn.setTooltip ("Clear search / reference sample");
    clearRefBtn.onClick = [this] {
        searchBox.clear();
        clearReference();
        refreshQuery();
    };
    addAndMakeVisible (clearRefBtn);

    // Length toggles: All | One | Loop
    lenAllBtn.onClick  = [this] { setLengthFilter (LengthFilter::All); };
    lenOneBtn.onClick  = [this] { setLengthFilter (LengthFilter::Oneshots); };
    lenLoopBtn.onClick = [this] { setLengthFilter (LengthFilter::Loops); };
    addAndMakeVisible (lenAllBtn);
    addAndMakeVisible (lenOneBtn);
    addAndMakeVisible (lenLoopBtn);
    updateLengthButtonStyles();

    // Category filter dropdown
    catCombo.setColour (juce::ComboBox::backgroundColourId, ui::col::controlBg);
    catCombo.setColour (juce::ComboBox::textColourId, ui::col::text);
    catCombo.setColour (juce::ComboBox::outlineColourId, ui::col::border);
    catCombo.setColour (juce::ComboBox::arrowColourId, ui::col::muted);
    catCombo.addItem ("All Cats", 1);
    for (int c = 1; c < kNumCategories; ++c)
        catCombo.addItem (categoryName (static_cast<Category> (c)), c + 1);
    catCombo.addItem ("Other", kNumCategories + 1);
    const int initialCatId = (processor.uiCategoryFilter < 0)
        ? 1
        : (processor.uiCategoryFilter == 0 ? (kNumCategories + 1) : (processor.uiCategoryFilter + 1));
    catCombo.setSelectedId (initialCatId, juce::dontSendNotification);
    catCombo.onChange = [this] {
        const int id = catCombo.getSelectedId();
        if (id <= 1) processor.uiCategoryFilter = -1;
        else if (id == kNumCategories + 1) processor.uiCategoryFilter = 0;
        else processor.uiCategoryFilter = id - 1;
        refreshQuery();
    };
    addAndMakeVisible (catCombo);

    // DSP <-> CLAP blend slider
    blendLabel.setText ("DSP/AI", juce::dontSendNotification);
    blendLabel.setFont (juce::FontOptions (11.0f));
    blendLabel.setColour (juce::Label::textColourId, ui::col::muted);
    blendLabel.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (blendLabel);

    blendSlider.setSliderStyle (juce::Slider::LinearHorizontal);
    blendSlider.setTextBoxStyle (juce::Slider::NoTextBox, true, 0, 0);
    blendSlider.setRange (0.0, 1.0, 0.01);
    blendSlider.setValue (processor.uiClapBlend, juce::dontSendNotification);
    blendSlider.setTooltip ("0 = Acoustic DSP, 1 = CLAP AI (morphs 2D map live + Find Similar blend)");
    blendSlider.setColour (juce::Slider::trackColourId, ui::col::accent);
    blendSlider.setColour (juce::Slider::backgroundColourId, ui::col::controlBg);
    blendSlider.setColour (juce::Slider::thumbColourId, ui::col::text);
    blendSlider.onValueChange = [this] {
        processor.uiClapBlend = static_cast<float> (blendSlider.getValue());
        sampleMap.setBlend (processor.uiClapBlend);
        if (activeReferenceId >= 0 || ! externalRefDsp.empty())
            refreshQuery();
    };
    addAndMakeVisible (blendSlider);
    sampleMap.setBlend (processor.uiClapBlend);

    // Index button + status
    styleFlatButton (indexBtn);
    indexBtn.setTooltip ("Choose folder and index audio samples");
    indexBtn.onClick = [this] { promptAndIndexFolder(); };
    addAndMakeVisible (indexBtn);

    statusLabel.setFont (juce::FontOptions (11.0f));
    statusLabel.setColour (juce::Label::textColourId, ui::col::muted);
    statusLabel.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (statusLabel);

    // Main panes
    auto resetSimilarAction = [this] {
        if (activeReferenceId >= 0 || ! externalRefDsp.empty() || searchBox.getText().startsWithChar ('~'))
        {
            clearReference();
            if (searchBox.getText().startsWithChar ('~'))
                searchBox.clear();
            refreshQuery();
        }
    };

    sampleMap.onSelectSample = [this] (int64_t id) { selectAndAudition (id); };
    sampleMap.onFindSimilar  = [this] (int64_t id) { setReferenceSample (id); };
    sampleMap.onResetSimilar = resetSimilarAction;
    addAndMakeVisible (sampleMap);

    sampleList.onSelectSample = [this] (int64_t id) { selectAndAudition (id); };
    sampleList.onFindSimilar  = [this] (int64_t id) { setReferenceSample (id); };
    sampleList.onResetSimilar = resetSimilarAction;
    addAndMakeVisible (sampleList);

    // Bottom audition bar
    auditionBar.setPitchMode (processor.getAuditionEngine().isChromaticPitch());
    auditionBar.setGain (processor.getAuditionEngine().getGain());
    auditionBar.onReplayAt = [this] (float normPos) { processor.getAuditionEngine().triggerPreviewAt (normPos); };
    auditionBar.onPitchToggle = [this] (bool c) { processor.getAuditionEngine().setChromaticPitch (c); };
    auditionBar.onGainChange  = [this] (float g) { processor.getAuditionEngine().setGain (g); };
    auditionBar.onFindSimilarCurrent = [this] {
        if (selectedSampleId >= 0)
            setReferenceSample (selectedSampleId);
    };
    auditionBar.onPinToggle = [this] (bool pinned) {
        if (auto* top = getTopLevelComponent())
            top->setAlwaysOnTop (pinned);
    };
    addAndMakeVisible (auditionBar);

    lastSeenRevision = processor.getIndexRevision();
    refreshQuery();
    startTimerHz (24);
}

SampleVectorDBEditor::~SampleVectorDBEditor()
{
    stopTimer();
}

void SampleVectorDBEditor::paint (juce::Graphics& g)
{
    g.fillAll (ui::col::bg);
}

bool SampleVectorDBEditor::keyPressed (const juce::KeyPress& key)
{
    if (key == juce::KeyPress::spaceKey && ! searchBox.hasKeyboardFocus (true))
    {
        if (activeReferenceId >= 0 || ! externalRefDsp.empty() || searchBox.getText().startsWithChar ('~'))
        {
            clearReference();
            if (searchBox.getText().startsWithChar ('~'))
                searchBox.clear();
            refreshQuery();
            return true;
        }
    }
    return false;
}

void SampleVectorDBEditor::resized()
{
    auto area = getLocalBounds();

    // Top bar (34px)
    auto top = area.removeFromTop (34).reduced (6, 4);
    statusLabel.setBounds (top.removeFromRight (138));
    top.removeFromRight (4);
    indexBtn.setBounds (top.removeFromRight (52));
    top.removeFromRight (6);
    blendSlider.setBounds (top.removeFromRight (66));
    blendLabel.setBounds (top.removeFromRight (42));
    top.removeFromRight (6);
    catCombo.setBounds (top.removeFromRight (96));
    top.removeFromRight (6);
    lenLoopBtn.setBounds (top.removeFromRight (40));
    lenOneBtn.setBounds (top.removeFromRight (38));
    lenAllBtn.setBounds (top.removeFromRight (32));
    top.removeFromRight (6);
    clearRefBtn.setBounds (top.removeFromRight (24));
    top.removeFromRight (4);
    searchBox.setBounds (top);

    // Bottom audition bar (42px)
    auditionBar.setBounds (area.removeFromBottom (42));

    // Main split (left 46% map, right 54% list)
    auto main = area.reduced (6, 2);
    const int mapW = static_cast<int> (main.getWidth() * 0.46f);
    sampleMap.setBounds (main.removeFromLeft (mapW));
    main.removeFromLeft (6);
    sampleList.setBounds (main);
}

void SampleVectorDBEditor::timerCallback()
{
    auditionBar.setPlayhead (processor.getAuditionEngine().getPlayheadNormalized());

    const uint64_t rev = processor.getIndexRevision();
    if (rev != lastSeenRevision)
    {
        lastSeenRevision = rev;
        refreshQuery();
    }

    statusLabel.setText (juce::String::fromUTF8 (processor.getStatusSummary().c_str()), juce::dontSendNotification);
}

void SampleVectorDBEditor::updateLengthButtonStyles()
{
    styleFlatButton (lenAllBtn,  processor.uiLengthFilter == LengthFilter::All);
    styleFlatButton (lenOneBtn,  processor.uiLengthFilter == LengthFilter::Oneshots);
    styleFlatButton (lenLoopBtn, processor.uiLengthFilter == LengthFilter::Loops);
}

void SampleVectorDBEditor::setLengthFilter (LengthFilter f)
{
    processor.uiLengthFilter = f;
    updateLengthButtonStyles();
    refreshQuery();
}

void SampleVectorDBEditor::setSearchTextAndRun (const juce::String& text)
{
    clearReference();
    searchBox.setText (text, false);
    refreshQuery();
}

void SampleVectorDBEditor::selectFirstResult()
{
    const auto snap = processor.getVectorIndex().getSnapshot();
    SearchQuery q;
    q.lengthFilter   = processor.uiLengthFilter;
    q.categoryFilter = processor.uiCategoryFilter;
    q.clapBlend      = processor.uiClapBlend;
    q.maxResults     = 1;
    const juce::String txt = searchBox.getText().trim();
    if (txt.isNotEmpty() && ! txt.startsWithChar ('~'))
        q.text = txt.toStdString();
    const auto res = processor.runSearch (std::move (q));
    if (! res.empty())
        selectAndAudition (res[0].id);
}

void SampleVectorDBEditor::clearReference()
{
    activeReferenceId = -1;
    externalRefDsp.clear();
    externalRefClap.clear();
    externalRefName.clear();
}

void SampleVectorDBEditor::setReferenceSample (int64_t sampleId)
{
    activeReferenceId = sampleId;
    externalRefDsp.clear();
    externalRefClap.clear();
    externalRefName.clear();

    const auto snap = processor.getVectorIndex().getSnapshot();
    auto it = snap->idToIndex.find (sampleId);
    if (it != snap->idToIndex.end())
    {
        const auto& name = snap->meta[static_cast<size_t> (it->second)].filename;
        searchBox.setText ("~" + juce::String::fromUTF8 (name.c_str()), false);
    }
    refreshQuery();
}

void SampleVectorDBEditor::refreshQuery()
{
    const auto snap = processor.getVectorIndex().getSnapshot();

    SearchQuery q;
    q.lengthFilter   = processor.uiLengthFilter;
    q.categoryFilter = processor.uiCategoryFilter;
    q.clapBlend      = processor.uiClapBlend;

    bool isRanked = false;
    if (activeReferenceId >= 0)
    {
        q.referenceId = activeReferenceId;
        isRanked = true;
    }
    else if (! externalRefDsp.empty())
    {
        q.refDspVec  = externalRefDsp;
        q.refClapVec = externalRefClap;
        isRanked = true;
    }
    else
    {
        const juce::String txt = searchBox.getText().trim();
        if (txt.isNotEmpty() && ! txt.startsWithChar ('~'))
        {
            q.text = txt.toStdString();
            isRanked = true;
        }
    }

    q.maxResults = isRanked ? 500 : 0;

    const auto results = processor.runSearch (std::move (q));
    sampleMap.setData (snap, results, isRanked);
    sampleList.setData (snap, results, isRanked);
    statusLabel.setText (juce::String::fromUTF8 (processor.getStatusSummary().c_str()), juce::dontSendNotification);
}

void SampleVectorDBEditor::selectAndAudition (int64_t sampleId)
{
    selectedSampleId = sampleId;
    sampleMap.setSelectedSampleId (sampleId);
    sampleList.selectSampleId (sampleId, false);

    const auto snap = processor.getVectorIndex().getSnapshot();
    auto it = snap->idToIndex.find (sampleId);
    if (it != snap->idToIndex.end())
    {
        const auto& meta = snap->meta[static_cast<size_t> (it->second)];
        auditionBar.setSample (&meta);
        processor.getAuditionEngine().loadSample (meta.path, true);
    }
}

void SampleVectorDBEditor::promptAndIndexFolder()
{
    const juce::File startDir (juce::String::fromUTF8 (processor.getDefaultSampleFolder().c_str()));
    folderChooser = std::make_unique<juce::FileChooser> ("Select sample folder to index", startDir);
    folderChooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
                                [this] (const juce::FileChooser& fc) {
                                    const auto dir = fc.getResult();
                                    if (dir.isDirectory())
                                        processor.startIndexingFolder (dir.getFullPathName().toStdString());
                                });
}

bool SampleVectorDBEditor::isInterestedInFileDrag (const juce::StringArray& files)
{
    for (const auto& f : files)
        if (isSupportedAudioExtension (juce::File (f).getFileExtension().toLowerCase()))
            return true;
    return false;
}

void SampleVectorDBEditor::filesDropped (const juce::StringArray& files, int, int)
{
    for (const auto& fStr : files)
    {
        const juce::File f (fStr);
        if (! isSupportedAudioExtension (f.getFileExtension().toLowerCase()))
            continue;

        std::vector<float> dspVec, clapVec;
        if (processor.encodeExternalAudio (f, dspVec, clapVec))
        {
            activeReferenceId = -1;
            externalRefDsp  = std::move (dspVec);
            externalRefClap = std::move (clapVec);
            externalRefName = f.getFileName().toStdString();
            searchBox.setText ("~" + f.getFileName(), false);
            refreshQuery();
        }
        break;
    }
}

} // namespace svdb

#pragma once

#include "core/Types.h"
#include "core/db/VectorIndex.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <memory>
#include <vector>

namespace svdb::ui
{

// Barebones virtualized sample list with arrow-key audition, double-click/S find-similar,
// and native Windows drag-and-drop into FL Studio.
class SampleList final : public juce::Component,
                         private juce::ListBoxModel
{
public:
    SampleList();

    void setData (std::shared_ptr<const db::IndexSnapshot> snapshot,
                  const std::vector<SearchResult>& visibleResults,
                  bool isRankedQuery);

    void selectSampleId (int64_t id, bool scrollToRow = true);
    int64_t getSelectedSampleId() const { return selectedId; }

    std::function<void (int64_t sampleId)> onSelectSample;
    std::function<void (int64_t sampleId)> onFindSimilar;
    std::function<void()> onResetSimilar;

    void paint (juce::Graphics& g) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent& e) override;
    void mouseMove (const juce::MouseEvent& e) override;
    bool keyPressed (const juce::KeyPress& key) override;

private:
    class RowComponent;

    enum class SortColumn { None = 0, Name, Pack, Cat, Note, Len, Sim };
    struct ColumnLayout
    {
        int nameX = 0, nameW = 0;
        int packX = 0, packW = 0;
        int catX  = 0, catW  = 0;
        int noteX = 0, noteW = 0;
        int lenX  = 0, lenW  = 0;
        int simX  = 0, simW  = 0;
    };

    juce::ListBox listBox;
    std::shared_ptr<const db::IndexSnapshot> snap;
    std::vector<SearchResult> results;
    bool ranked = false;
    int64_t selectedId = -1;
    bool suppressSelectionCallback = false;

    SortColumn sortCol = SortColumn::Cat;
    bool sortAsc = true;

    ColumnLayout computeColumns (int totalW) const;
    SortColumn hitTestHeader (juce::Point<int> pt) const;
    void applySort();

    int getNumRows() override;
    void paintListBoxItem (int rowNumber, juce::Graphics& g, int width, int height, bool rowIsSelected) override;
    juce::Component* refreshComponentForRow (int rowNumber, bool isRowSelected, juce::Component* existing) override;
    void selectedRowsChanged (int lastRowSelected) override;
    void listBoxItemDoubleClicked (int row, const juce::MouseEvent&) override;
};

} // namespace svdb::ui

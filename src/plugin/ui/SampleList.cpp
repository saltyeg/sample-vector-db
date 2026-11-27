#include "SampleList.h"
#include "Palette.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace svdb::ui
{

namespace
{
constexpr int kHeaderH = 22;
constexpr int kRowH = 22;

juce::String formatDuration (float ms)
{
    char buf[32] {};
    if (ms < 1000.0f)
        std::snprintf (buf, sizeof (buf), "%.0fms", ms);
    else
        std::snprintf (buf, sizeof (buf), "%.2fs", ms / 1000.0f);
    return juce::String (buf);
}

int compareIgnoreCase (const std::string& a, const std::string& b)
{
    const size_t n = std::min (a.size(), b.size());
    for (size_t i = 0; i < n; ++i)
    {
        const unsigned char ca = static_cast<unsigned char> (std::tolower (static_cast<unsigned char> (a[i])));
        const unsigned char cb = static_cast<unsigned char> (std::tolower (static_cast<unsigned char> (b[i])));
        if (ca != cb) return (ca < cb) ? -1 : 1;
    }
    if (a.size() == b.size()) return 0;
    return (a.size() < b.size()) ? -1 : 1;
}
} // namespace

class SampleList::RowComponent final : public juce::Component
{
public:
    explicit RowComponent (SampleList& ownerList) : owner (ownerList)
    {
        setInterceptsMouseClicks (true, false);
    }

    void update (int rowIdx, bool selected)
    {
        row = rowIdx;
        isSelected = selected;
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        if (! owner.snap || row < 0 || row >= static_cast<int> (owner.results.size()))
            return;

        const auto& res = owner.results[static_cast<size_t> (row)];
        const auto& m = owner.snap->meta[static_cast<size_t> (res.index)];

        g.fillAll (isSelected ? col::rowSelected : ((row & 1) ? col::rowAlt : col::bg));

        const int h = getHeight();
        const auto cols = owner.computeColumns (getWidth());

        // Tiny category color dot.
        g.setColour (categoryColour (m.category));
        g.fillEllipse (5.0f, 0.5f * static_cast<float> (h) - 2.5f, 5.0f, 5.0f);

        g.setFont (juce::FontOptions (12.0f));
        g.setColour (col::text);
        g.drawText (juce::String::fromUTF8 (m.filename.c_str()), cols.nameX, 0, cols.nameW, h, juce::Justification::centredLeft, true);

        g.setFont (juce::FontOptions (11.0f));
        g.setColour (col::muted);
        const auto pack = samplePackName (m.path, m.folder);
        g.drawText (juce::String::fromUTF8 (pack.c_str()), cols.packX, 0, cols.packW - 4, h, juce::Justification::centredLeft, true);

        g.setColour (categoryColour (m.category));
        g.drawText (categoryShort (m.category), cols.catX, 0, cols.catW, h, juce::Justification::centredLeft);

        g.setColour (col::muted);
        g.drawText (midiNoteToName (m.midiNote), cols.noteX, 0, cols.noteW, h, juce::Justification::centredLeft);

        g.drawText (formatDuration (m.durationMs), cols.lenX, 0, cols.lenW, h, juce::Justification::centredRight);

        if (cols.simW > 0)
        {
            const int pct = std::clamp (static_cast<int> (std::lround (res.score * 100.0f)), 0, 100);
            g.setColour (col::accent);
            g.drawText (juce::String (pct) + "%", cols.simX, 0, cols.simW - 4, h, juce::Justification::centredRight);
        }
    }

    void mouseDown (const juce::MouseEvent& e) override
    {
        dragStart = e.position;
        dragStarted = false;
        owner.listBox.grabKeyboardFocus();
        owner.listBox.selectRow (row, false, true);
        if (owner.onSelectSample && row >= 0 && row < static_cast<int> (owner.results.size()))
        {
            owner.selectedId = owner.results[static_cast<size_t> (row)].id;
            owner.onSelectSample (owner.selectedId);
        }
    }

    void mouseDrag (const juce::MouseEvent& e) override
    {
        if (! dragStarted && e.position.getDistanceFrom (dragStart) > 6.0f
            && owner.snap && row >= 0 && row < static_cast<int> (owner.results.size()))
        {
            dragStarted = true;
            const auto& m = owner.snap->meta[static_cast<size_t> (owner.results[static_cast<size_t> (row)].index)];
            juce::StringArray files;
            files.add (juce::String::fromUTF8 (m.path.c_str()));
            juce::DragAndDropContainer::performExternalDragDropOfFiles (files, false, this);
        }
    }

    void mouseDoubleClick (const juce::MouseEvent&) override
    {
        if (owner.onFindSimilar && row >= 0 && row < static_cast<int> (owner.results.size()))
            owner.onFindSimilar (owner.results[static_cast<size_t> (row)].id);
    }

private:
    SampleList& owner;
    int row = -1;
    bool isSelected = false;
    bool dragStarted = false;
    juce::Point<float> dragStart;
};

SampleList::SampleList()
{
    listBox.setModel (this);
    listBox.setRowHeight (kRowH);
    listBox.setColour (juce::ListBox::backgroundColourId, col::bg);
    listBox.setColour (juce::ListBox::outlineColourId, juce::Colours::transparentBlack);
    listBox.setOutlineThickness (0);
    addAndMakeVisible (listBox);
    setWantsKeyboardFocus (true);
}

SampleList::ColumnLayout SampleList::computeColumns (int totalW) const
{
    ColumnLayout c;
    c.simW  = ranked ? 44 : 0;
    c.lenW  = 52;
    c.noteW = 40;
    c.catW  = 42;
    c.packW = std::clamp (totalW / 4, 90, 220);
    c.nameX = 14;
    c.nameW = std::max (60, totalW - (c.simW + c.lenW + c.noteW + c.catW + c.packW + 20));
    c.packX = c.nameX + c.nameW + 6;
    c.catX  = c.packX + c.packW;
    c.noteX = c.catX  + c.catW;
    c.lenX  = c.noteX + c.noteW;
    c.simX  = c.lenX  + c.lenW;
    return c;
}

SampleList::SortColumn SampleList::hitTestHeader (juce::Point<int> pt) const
{
    if (pt.y < 0 || pt.y >= kHeaderH)
        return SortColumn::None;
    const auto c = computeColumns (getWidth() - 10);
    if (pt.x >= c.nameX && pt.x < c.packX) return SortColumn::Name;
    if (pt.x >= c.packX && pt.x < c.catX)  return SortColumn::Pack;
    if (pt.x >= c.catX  && pt.x < c.noteX) return SortColumn::Cat;
    if (pt.x >= c.noteX && pt.x < c.lenX)  return SortColumn::Note;
    if (pt.x >= c.lenX  && pt.x < c.lenX + c.lenW) return SortColumn::Len;
    if (c.simW > 0 && pt.x >= c.simX && pt.x < c.simX + c.simW) return SortColumn::Sim;
    return SortColumn::None;
}

void SampleList::applySort()
{
    if (! snap || results.size() <= 1 || sortCol == SortColumn::None)
        return;

    const bool asc = sortAsc;
    const auto colSel = sortCol;
    const auto& meta = snap->meta;

    std::stable_sort (results.begin(), results.end(), [&] (const SearchResult& a, const SearchResult& b) {
        const auto& ma = meta[static_cast<size_t> (a.index)];
        const auto& mb = meta[static_cast<size_t> (b.index)];

        auto tieBreak = [&] () {
            const int cmp = compareIgnoreCase (ma.filename, mb.filename);
            return cmp < 0;
        };

        switch (colSel)
        {
            case SortColumn::Name:
            {
                const int cmp = compareIgnoreCase (ma.filename, mb.filename);
                if (cmp != 0) return asc ? (cmp < 0) : (cmp > 0);
                return false;
            }
            case SortColumn::Pack:
            {
                const int cmp = compareIgnoreCase (samplePackName (ma.path, ma.folder),
                                                   samplePackName (mb.path, mb.folder));
                if (cmp != 0) return asc ? (cmp < 0) : (cmp > 0);
                return tieBreak();
            }
            case SortColumn::Cat:
            {
                if (ma.category != mb.category)
                {
                    const int cmp = compareIgnoreCase (categoryName (ma.category), categoryName (mb.category));
                    return asc ? (cmp < 0) : (cmp > 0);
                }
                return tieBreak();
            }
            case SortColumn::Note:
            {
                // Always keep unpitched (-1) at the bottom regardless of asc/desc.
                const bool aHas = ma.midiNote >= 0;
                const bool bHas = mb.midiNote >= 0;
                if (aHas != bHas) return aHas;
                if (ma.midiNote != mb.midiNote)
                    return asc ? (ma.midiNote < mb.midiNote) : (ma.midiNote > mb.midiNote);
                return tieBreak();
            }
            case SortColumn::Len:
            {
                if (ma.durationMs != mb.durationMs)
                    return asc ? (ma.durationMs < mb.durationMs) : (ma.durationMs > mb.durationMs);
                return tieBreak();
            }
            case SortColumn::Sim:
            {
                if (a.score != b.score)
                    return asc ? (a.score < b.score) : (a.score > b.score);
                return tieBreak();
            }
            case SortColumn::None:
            default:
                return false;
        }
    });
}

void SampleList::setData (std::shared_ptr<const db::IndexSnapshot> snapshot,
                          const std::vector<SearchResult>& visibleResults,
                          bool isRankedQuery)
{
    const bool switchedToRanked = isRankedQuery && ! ranked;
    const bool switchedToUnranked = ! isRankedQuery && ranked;

    snap = std::move (snapshot);
    results = visibleResults;
    ranked = isRankedQuery;

    if (switchedToRanked)
    {
        sortCol = SortColumn::Sim;
        sortAsc = false;
    }
    else if (switchedToUnranked && sortCol == SortColumn::Sim)
    {
        sortCol = SortColumn::Cat;
        sortAsc = true;
    }

    // Apply user's active column sort (skip if already sorted by Sim descending from VectorIndex).
    if (! (ranked && sortCol == SortColumn::Sim && ! sortAsc))
        applySort();

    suppressSelectionCallback = true;
    listBox.updateContent();
    int matchRow = -1;
    for (int i = 0; i < static_cast<int> (results.size()); ++i)
    {
        if (results[static_cast<size_t> (i)].id == selectedId)
        {
            matchRow = i;
            break;
        }
    }
    if (matchRow >= 0)
        listBox.selectRow (matchRow, true, true);
    else
        listBox.deselectAllRows();
    suppressSelectionCallback = false;

    repaint();
    listBox.repaint();
}

void SampleList::selectSampleId (int64_t id, bool scrollToRow)
{
    selectedId = id;
    suppressSelectionCallback = true;
    for (int i = 0; i < static_cast<int> (results.size()); ++i)
    {
        if (results[static_cast<size_t> (i)].id == id)
        {
            listBox.selectRow (i, ! scrollToRow, true);
            break;
        }
    }
    suppressSelectionCallback = false;
}

void SampleList::paint (juce::Graphics& g)
{
    g.fillAll (col::bg);

    const auto header = getLocalBounds().removeFromTop (kHeaderH);
    g.setColour (col::panel);
    g.fillRect (header);
    g.setColour (col::border);
    g.drawHorizontalLine (kHeaderH - 1, 0.0f, static_cast<float> (getWidth()));

    const auto c = computeColumns (getWidth() - 10);
    g.setFont (juce::FontOptions (11.0f));

    auto drawColHeader = [&] (SortColumn sc, const juce::String& label, int x, int w, juce::Justification just) {
        const bool active = (sortCol == sc);
        g.setColour (active ? col::text : col::muted);
        const juce::String txt = active ? (label + (sortAsc ? " ^" : " v")) : label;
        g.drawText (txt, x, 0, w, kHeaderH, just, true);
    };

    drawColHeader (SortColumn::Name, "Name", c.nameX, c.nameW, juce::Justification::centredLeft);
    drawColHeader (SortColumn::Pack, "Pack", c.packX, c.packW - 4, juce::Justification::centredLeft);
    drawColHeader (SortColumn::Cat,  "Cat",  c.catX,  c.catW,  juce::Justification::centredLeft);
    drawColHeader (SortColumn::Note, "Note", c.noteX, c.noteW, juce::Justification::centredLeft);
    drawColHeader (SortColumn::Len,  "Len",  c.lenX,  c.lenW,  juce::Justification::centredRight);
    if (c.simW > 0)
        drawColHeader (SortColumn::Sim, "Sim", c.simX, c.simW - 4, juce::Justification::centredRight);

    g.setColour (col::border);
    g.drawRect (getLocalBounds(), 1);
}

void SampleList::mouseMove (const juce::MouseEvent& e)
{
    setMouseCursor (hitTestHeader (e.getPosition()) != SortColumn::None
                        ? juce::MouseCursor::PointingHandCursor
                        : juce::MouseCursor::NormalCursor);
}

void SampleList::mouseDown (const juce::MouseEvent& e)
{
    const auto clicked = hitTestHeader (e.getPosition());
    if (clicked == SortColumn::None)
        return;

    if (sortCol == clicked)
    {
        sortAsc = ! sortAsc;
    }
    else
    {
        sortCol = clicked;
        sortAsc = (clicked != SortColumn::Sim); // Sim defaults descending; others ascending.
    }

    applySort();

    suppressSelectionCallback = true;
    listBox.updateContent();
    int matchRow = -1;
    for (int i = 0; i < static_cast<int> (results.size()); ++i)
    {
        if (results[static_cast<size_t> (i)].id == selectedId)
        {
            matchRow = i;
            break;
        }
    }
    if (matchRow >= 0)
        listBox.selectRow (matchRow, true, true);
    suppressSelectionCallback = false;

    repaint();
    listBox.repaint();
}

void SampleList::resized()
{
    auto area = getLocalBounds().reduced (1);
    area.removeFromTop (kHeaderH - 1);
    listBox.setBounds (area);
}

int SampleList::getNumRows()
{
    return static_cast<int> (results.size());
}

void SampleList::paintListBoxItem (int, juce::Graphics&, int, int, bool)
{
}

juce::Component* SampleList::refreshComponentForRow (int rowNumber, bool isRowSelected, juce::Component* existing)
{
    auto* comp = dynamic_cast<RowComponent*> (existing);
    if (! comp)
        comp = new RowComponent (*this);
    comp->update (rowNumber, isRowSelected);
    return comp;
}

void SampleList::selectedRowsChanged (int lastRowSelected)
{
    if (suppressSelectionCallback)
        return;
    if (lastRowSelected >= 0 && lastRowSelected < static_cast<int> (results.size()))
    {
        selectedId = results[static_cast<size_t> (lastRowSelected)].id;
        if (onSelectSample)
            onSelectSample (selectedId);
    }
}

void SampleList::listBoxItemDoubleClicked (int row, const juce::MouseEvent&)
{
    if (onFindSimilar && row >= 0 && row < static_cast<int> (results.size()))
        onFindSimilar (results[static_cast<size_t> (row)].id);
}

bool SampleList::keyPressed (const juce::KeyPress& key)
{
    if (key.getTextCharacter() == 's' || key.getTextCharacter() == 'S')
    {
        if (selectedId >= 0 && onFindSimilar)
        {
            onFindSimilar (selectedId);
            return true;
        }
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

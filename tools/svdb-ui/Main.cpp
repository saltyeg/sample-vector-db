// svdb-ui: renders SampleVectorDBEditor offscreen to a PNG for visual verification.
//
// Usage:
//   svdb-ui <out.png> [--size 980x580] [--query "punchy kick"] [--all|--oneshots|--loops] [--db <path>]

#include "PluginEditor.h"
#include "PluginProcessor.h"

#include <iostream>
#include <string>

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();

int main (int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI juceInit;

    if (argc < 2)
    {
        std::cerr << "usage: svdb-ui <out.png> [--size WxH] [--query text] [--all|--oneshots|--loops] [--db path]\n";
        return 1;
    }

    const juce::File outFile = juce::File::getCurrentWorkingDirectory().getChildFile (juce::String::fromUTF8 (argv[1]));
    int width = 980, height = 580;
    juce::String query;
    std::string dbPath;
    svdb::LengthFilter lenFilter = svdb::LengthFilter::Oneshots;
    bool selectFirst = true;

    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--size" && i + 1 < argc)
        {
            const juce::String v = argv[++i];
            width  = v.upToFirstOccurrenceOf ("x", false, false).getIntValue();
            height = v.fromFirstOccurrenceOf ("x", false, false).getIntValue();
        }
        else if (a == "--query" && i + 1 < argc) query = juce::String::fromUTF8 (argv[++i]);
        else if (a == "--db" && i + 1 < argc)    dbPath = argv[++i];
        else if (a == "--all")                   lenFilter = svdb::LengthFilter::All;
        else if (a == "--oneshots")              lenFilter = svdb::LengthFilter::Oneshots;
        else if (a == "--loops")                 lenFilter = svdb::LengthFilter::Loops;
        else if (a == "--no-select")             selectFirst = false;
    }

    std::unique_ptr<juce::AudioProcessor> procBase (createPluginFilter());
    auto& proc = dynamic_cast<svdb::SampleVectorDBProcessor&> (*procBase);
    if (! dbPath.empty())
        proc.openDatabase (dbPath);

    // Give background CLAP text warmup a short moment if models exist.
    juce::MessageManager::getInstance()->runDispatchLoopUntil (600);

    std::unique_ptr<juce::AudioProcessorEditor> edBase (proc.createEditorAndMakeActive());
    auto& ed = dynamic_cast<svdb::SampleVectorDBEditor&> (*edBase);
    ed.setSize (width, height);
    ed.setLengthFilter (lenFilter);
    if (query.isNotEmpty())
        ed.setSearchTextAndRun (query);
    if (selectFirst)
        ed.selectFirstResult();

    juce::MessageManager::getInstance()->runDispatchLoopUntil (150);

    const auto image = ed.createComponentSnapshot (ed.getLocalBounds(), true, 1.0f);
    outFile.getParentDirectory().createDirectory();
    outFile.deleteFile();
    juce::FileOutputStream stream (outFile);
    juce::PNGImageFormat png;
    if (! png.writeImageToStream (image, stream))
    {
        std::cerr << "error: cannot write " << outFile.getFullPathName().toStdString() << "\n";
        return 1;
    }
    std::cout << "wrote " << outFile.getFullPathName().toStdString() << "\n";
    edBase.reset();
    return 0;
}
